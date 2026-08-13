// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package gostate

import (
	"context"
	"errors"
	"fmt"
	"io"
	"runtime"
	"sync"

	"github.com/0xsoniclabs/carmen/go/backend/archive"
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/common/future"
	"github.com/0xsoniclabs/carmen/go/common/result"
	"github.com/0xsoniclabs/carmen/go/common/witness"
	"github.com/0xsoniclabs/carmen/go/state"
	"golang.org/x/crypto/sha3"
)

// GoState combines a LiveDB and optional Archive implementation into a common
// Carmen State implementation.
type GoState struct {
	live    state.LiveDB
	archive archive.Archive
	cleanup []func()

	// staged holds the blocks applied to the LiveDB whose fate is not yet decided,
	// oldest first. Commit consumes from the front, Rollback from the back. It is
	// guarded by its own lock rather than by the surrounding syncedState, because
	// the StagedBlock handles are operated on directly and so do not pass through
	// that wrapper.
	staged     []*stagedBlock
	stagedLock sync.Mutex

	stateError     error // collect errors occurred during operation
	stateErrorLock sync.RWMutex

	// Channels are only present if archive is enabled.
	archiveWriter          chan<- archiveUpdate
	archiveWriterFlushDone <-chan bool
	archiveWriterDone      <-chan bool
	archiveWriterError     <-chan error
}

func newGoState(live state.LiveDB, archive archive.Archive, cleanup []func()) state.State {

	res := &GoState{
		live:    live,
		archive: archive,
		cleanup: cleanup,
	}

	// If there is an archive, start an asynchronous archive writer routine.
	if archive != nil {
		in := make(chan archiveUpdate, 10)
		flush := make(chan bool)
		done := make(chan bool)
		err := make(chan error, 10)

		go func() {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			defer close(flush)
			defer close(done)
			// Process all incoming updates, no not stop on errors.
			for update := range in {
				// If there is no update, the state is asking for a flush signal.
				if update.update == nil {
					if issue := res.archive.Flush(); issue != nil {
						err <- issue
					}
					flush <- true
				} else {
					// Otherwise, process the update.
					issue := res.archive.Add(update.block, *update.update, update.updateHints)
					if issue != nil {
						err <- issue
					}
					if update.done != nil {
						if stateErrors := res.Check(); stateErrors != nil {
							update.done <- stateErrors
						}
						close(update.done)
					}
					if update.updateHints != nil {
						update.updateHints.Release()
					}
				}
			}
		}()

		res.archiveWriter = in
		res.archiveWriterDone = done
		res.archiveWriterFlushDone = flush
		res.archiveWriterError = err
	}

	return state.WrapIntoSyncedState(res)
}

var emptyCodeHash = common.GetHash(sha3.NewLegacyKeccak256(), []byte{})

type archiveUpdate = struct {
	block       uint64
	update      *common.Update  // nil to signal a flush
	updateHints common.Releaser // an optional field for passing update hints from the LiveDB to the Archive
	done        chan<- error    // a channel for the archive to signal when the update was processed
}

func (s *GoState) GetBalance(address common.Address) (amount.Amount, error) {
	if err := s.getStateError(); err != nil {
		return amount.New(), err
	}

	balance, err := s.live.GetBalance(address)
	if err != nil {
		s.addStateError(err)
	}

	return balance, s.getStateError()
}

func (s *GoState) GetNonce(address common.Address) (common.Nonce, error) {
	if err := s.getStateError(); err != nil {
		return common.Nonce{}, err
	}

	nonce, err := s.live.GetNonce(address)
	if err != nil {
		s.addStateError(err)
	}
	return nonce, s.getStateError()
}

func (s *GoState) GetStorage(address common.Address, key common.Key) (common.Value, error) {
	if err := s.getStateError(); err != nil {
		return common.Value{}, err
	}

	val, err := s.live.GetStorage(address, key)
	if err != nil {
		s.addStateError(err)
	}
	return val, s.getStateError()
}

func (s *GoState) GetCode(address common.Address) ([]byte, error) {
	if err := s.getStateError(); err != nil {
		return []byte{}, err
	}

	code, err := s.live.GetCode(address)
	if err != nil {
		s.addStateError(err)
	}
	return code, s.getStateError()
}

func (s *GoState) GetCodeSize(address common.Address) (int, error) {
	if err := s.getStateError(); err != nil {
		return 0, err
	}

	size, err := s.live.GetCodeSize(address)
	if err != nil {
		s.addStateError(err)
	}
	return size, s.getStateError()
}

func (s *GoState) GetCodeHash(address common.Address) (common.Hash, error) {
	if err := s.getStateError(); err != nil {
		return common.Hash{}, err
	}

	h, err := s.live.GetCodeHash(address)
	if err != nil {
		s.addStateError(err)
	}
	return h, s.getStateError()
}

func (s *GoState) HasEmptyStorage(addr common.Address) (bool, error) {
	if err := s.getStateError(); err != nil {
		return false, err
	}

	empty, err := s.live.HasEmptyStorage(addr)
	if err != nil {
		s.addStateError(err)
	}
	return empty, s.getStateError()
}

func (s *GoState) GetHash() (common.Hash, error) {
	return s.GetCommitment().Await().Get()
}

func (s *GoState) GetCommitment() future.Future[result.Result[common.Hash]] {
	if err := s.getStateError(); err != nil {
		return future.Immediate(result.Err[common.Hash](err))
	}

	h, err := s.live.GetHash()
	if err != nil {
		s.addStateError(err)
		return future.Immediate(result.Err[common.Hash](err))
	}
	return future.Immediate(result.Ok(h))
}

// Apply applies the update to the LiveDB and stages the resulting block. The
// archive is deliberately left untouched: an archive is append-only, so a block
// may only reach it once it is certain to stay, which is what StagedBlock.Commit
// declares.
func (s *GoState) Apply(block uint64, update common.Update) (state.StagedBlock, error) {
	if err := s.getStateError(); err != nil {
		return nil, err
	}

	undoList, archiveUpdateHints, err := s.live.Apply(block, &update)
	if err != nil {
		s.addStateError(err)
		return nil, s.getStateError()
	}

	hash, err := s.live.GetHash()
	if err != nil {
		s.addStateError(err)
		return nil, s.getStateError()
	}

	staged := &stagedBlock{
		state:  s,
		block:  block,
		update: update,
		undo:   undoList,
		hints:  archiveUpdateHints,
		hash:   hash,
	}

	s.stagedLock.Lock()
	defer s.stagedLock.Unlock()
	s.staged = append(s.staged, staged)
	return staged, nil
}

// consumeOldest removes the given block from the front of the staged sequence,
// rejecting the call if it is not the oldest staged block or was already
// consumed.
func (s *GoState) consumeOldest(block *stagedBlock) error {
	s.stagedLock.Lock()
	defer s.stagedLock.Unlock()

	if block.status != stagedPending {
		return block.consumedError("commit")
	}
	if len(s.staged) == 0 || s.staged[0] != block {
		return fmt.Errorf("%w: cannot commit block %d: it is not the oldest staged block, %d block(s) before it are still staged", state.ErrStagedBlockMisuse, block.block, s.indexOf(block))
	}
	s.staged = s.staged[1:]
	block.status = stagedCommitted
	return nil
}

// consumeNewest removes the given block from the back of the staged sequence,
// rejecting the call if it is not the newest staged block or was already
// consumed.
func (s *GoState) consumeNewest(block *stagedBlock) error {
	s.stagedLock.Lock()
	defer s.stagedLock.Unlock()

	if block.status != stagedPending {
		return block.consumedError("roll back")
	}
	last := len(s.staged) - 1
	if last < 0 || s.staged[last] != block {
		return fmt.Errorf("%w: cannot roll back block %d: it is not the newest staged block, %d block(s) after it are still staged", state.ErrStagedBlockMisuse, block.block, last-s.indexOf(block))
	}
	s.staged = s.staged[:last]
	block.status = stagedRolledBack
	return nil
}

// indexOf reports the position of the block in the staged sequence, or -1. It
// must be called while holding stagedLock.
func (s *GoState) indexOf(block *stagedBlock) int {
	for i, candidate := range s.staged {
		if candidate == block {
			return i
		}
	}
	return -1
}

// drainArchiveErrors collects the errors the archive writer reported so far
// without waiting for any pending write.
func (s *GoState) drainArchiveErrors() {
	for {
		select {
		case err := <-s.archiveWriterError:
			s.addStateError(err)
		default:
			return
		}
	}
}

// stagedStatus tracks which of the two terminal operations a staged block has
// already seen, so that a second one reports an error rather than acting twice.
type stagedStatus int

const (
	stagedPending stagedStatus = iota
	stagedCommitted
	stagedRolledBack
)

// stagedBlock is a block applied to the LiveDB whose fate is not yet decided. It
// retains everything needed to take either decision: the update and its hints to
// hand to the archive on Commit, and the undo list to replay on Rollback.
type stagedBlock struct {
	state  *GoState
	block  uint64
	update common.Update
	undo   []func() error
	hints  common.Releaser
	hash   common.Hash

	status           stagedStatus
	archiveWriteDone chan error
}

func (b *stagedBlock) StateHash() common.Hash {
	return b.hash
}

func (b *stagedBlock) Commit() error {
	if err := b.state.consumeOldest(b); err != nil {
		return err
	}

	// From here on the archive writer owns the hints and releases them.
	hints := b.hints
	b.hints = nil

	if b.state.archive == nil {
		if hints != nil {
			hints.Release()
		}
		return b.state.getStateError()
	}

	b.archiveWriteDone = make(chan error, 1)
	b.state.archiveWriter <- archiveUpdate{b.block, &b.update, hints, b.archiveWriteDone}
	b.state.drainArchiveErrors()
	return b.state.getStateError()
}

func (b *stagedBlock) Wait() error {
	if b.status != stagedCommitted {
		return fmt.Errorf("%w: cannot wait for block %d: it has not been committed", state.ErrStagedBlockMisuse, b.block)
	}
	if b.archiveWriteDone == nil {
		return nil // no archive, nothing to wait for
	}
	return <-b.archiveWriteDone
}

func (b *stagedBlock) Rollback() error {
	if err := b.state.consumeNewest(b); err != nil {
		return err
	}

	if b.hints != nil {
		b.hints.Release()
		b.hints = nil
	}
	if err := b.state.live.RevertLastBlock(b.undo); err != nil {
		b.state.addStateError(err)
	}
	return b.state.getStateError()
}

// consumedError explains that the block has already been decided on.
func (b *stagedBlock) consumedError(operation string) error {
	decision := "committed"
	if b.status == stagedRolledBack {
		decision = "rolled back"
	}
	return fmt.Errorf("%w: cannot %s block %d: it has already been %s", state.ErrStagedBlockMisuse, operation, b.block, decision)
}

// GetMemoryFootprint provides sizes of individual components of the state in the memory
func (s *GoState) GetMemoryFootprint() *common.MemoryFootprint {
	mf := common.NewMemoryFootprint(0)
	mf.AddChild("live", s.live.GetMemoryFootprint())
	if s.archive != nil {
		mf.AddChild("archive", s.archive.GetMemoryFootprint())
	}
	return mf
}

func (s *GoState) Flush() error {
	if s.archiveWriter != nil {
		// Signal to the archive worker that a flush should be conducted.
		s.archiveWriter <- archiveUpdate{}
	}

	err := s.live.Flush()
	if err != nil {
		s.addStateError(err)
	}

	if s.archiveWriter != nil {
		// Wait until the flush was processed.
		<-s.archiveWriterFlushDone
	}

	return s.Check()
}

// rollbackUndecidedBlocks takes back every staged block whose fate was never
// decided, newest first. Closing is the last chance to do so: a block that has
// not been committed by then is never going to be, and flushing it into the
// live state would leave the database unopenable, since an archive-backed
// state only opens when the archive head matches the live root.
func (s *GoState) rollbackUndecidedBlocks() {
	for {
		s.stagedLock.Lock()
		if len(s.staged) == 0 {
			s.stagedLock.Unlock()
			return
		}
		newest := s.staged[len(s.staged)-1]
		s.stagedLock.Unlock()

		// Rollback consumes the block and collects any revert error in the
		// state error, which Close reports through its final Check.
		_ = newest.Rollback()
	}
}

func (s *GoState) Close() error {
	s.rollbackUndecidedBlocks()
	s.addStateError(errors.Join(
		s.Flush(),
		s.live.Close(),
	))

	// Shut down archive writer background worker.
	if s.archiveWriter != nil {
		// Close archive stream, signaling writer to shut down.
		close(s.archiveWriter)
		// Wait for the shutdown to be complete.
		<-s.archiveWriterDone
		s.archiveWriter = nil
	}

	// Close the archive.
	if s.archive != nil {
		if err := s.archive.Close(); err != nil {
			s.addStateError(err)
		}
	}

	if s.cleanup != nil {
		for _, clean := range s.cleanup {
			if clean != nil {
				clean()
			}
		}
	}
	return s.Check()
}

func (s *GoState) GetArchiveState(block uint64) (as state.State, err error) {
	if s.archive == nil {
		return nil, state.NoArchiveError
	}
	if err := s.getStateError(); err != nil {
		return nil, err
	}
	lastBlock, empty, err := s.archive.GetBlockHeight()
	if err != nil {
		s.addStateError(fmt.Errorf("failed to get last block in the archive: %w", err))
		return nil, s.getStateError()
	}
	if empty {
		return nil, fmt.Errorf("block %d is not present in the archive (archive is empty)", block)
	}
	if block > lastBlock {
		return nil, fmt.Errorf("block %d is not present in the archive (non-empty archive, last block %d)", block, lastBlock)
	}
	return &ArchiveState{
		archive: s.archive,
		block:   block,
	}, nil
}

func (s *GoState) GetArchiveBlockHeight() (uint64, bool, error) {
	if s.archive == nil {
		return 0, false, state.NoArchiveError
	}
	if err := s.getStateError(); err != nil {
		return 0, false, err
	}
	lastBlock, empty, err := s.archive.GetBlockHeight()
	if err != nil {
		s.addStateError(fmt.Errorf("failed to get last block in the archive: %w", err))
		return 0, false, s.getStateError()
	}
	return lastBlock, empty, nil
}

func (s *GoState) Check() error {
	// drain errors from archive if present
	// but does not wait
	if s.archive != nil {
		var done bool
		for !done {
			select {
			case err := <-s.archiveWriterError:
				s.addStateError(err)
			default:
				done = true
			}
		}
	}
	return s.getStateError()
}

func (s *GoState) Export(context.Context, io.Writer, string) (common.Hash, error) {
	return common.Hash{}, state.ExportNotSupported
}

func (s *GoState) CreateWitnessProof(address common.Address, keys ...common.Key) (witness.Proof, error) {
	panic("not implemented")
}

// addStateError adds the provided error to the state error, ensuring thread safety.
func (s *GoState) addStateError(err error) {
	s.stateErrorLock.Lock()
	defer s.stateErrorLock.Unlock()
	s.stateError = errors.Join(s.stateError, err)
}

// getStateError retrieves the current state error, ensuring thread safety.
func (s *GoState) getStateError() error {
	s.stateErrorLock.RLock()
	defer s.stateErrorLock.RUnlock()
	return s.stateError
}
