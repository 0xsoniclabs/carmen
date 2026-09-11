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
	"slices"
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
	// oldest first, with everything either decision needs. A commit consumes the
	// front, a rollback the back. A decision holds the lock from consuming
	// its block to acting on it, so that the order in which blocks leave the
	// queue is the order in which they reach the archive or are reverted.
	// NOTE: this is unbounded, and may cause memory pressure.
	staged     []*stagedBlock
	stagedLock sync.Mutex
	// nextStagedId hands out the identity of each staged block, see stagedBlock.id.
	// It is guarded by stagedLock.
	nextStagedId uint64

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

// Apply applies the update to the LiveDB, stages the resulting block, and
// commits it at once. Nothing can hold a staged block yet, so a block is still
// final the moment it is applied.
//
// The channel signals the completion of any spawned asynchronous operations
// like the update of the archive, if there is such.
// The channel may be nil if there are no asynchronous operations to be performed.
// If the asynchronous operations fail, the error is returned through the channel.
func (s *GoState) Apply(block uint64, update common.Update) (<-chan error, error) {
	if err := s.getStateError(); err != nil {
		return nil, err
	}

	undoList, archiveUpdateHints, err := s.live.Apply(block, &update)
	if err != nil {
		// A failed update may have partially mutated the live state, and its undo
		// list is the only way back (see LiveDB.Apply). Reverting it leaves the
		// live state at the last complete block. The hints are released as well:
		// an update that fails while hashing has already allocated them.
		if archiveUpdateHints != nil {
			archiveUpdateHints.Release()
		}
		if revertErr := s.live.RevertLastBlock(undoList); revertErr != nil {
			s.addStateError(revertErr)
		}
		s.addStateError(err)
		return nil, s.getStateError()
	}

	hash, err := s.live.GetHash()
	if err != nil {
		// Without its root the block cannot be staged, and a block that cannot be
		// staged can never be decided; take it back entirely.
		if archiveUpdateHints != nil {
			archiveUpdateHints.Release()
		}
		if revertErr := s.live.RevertLastBlock(undoList); revertErr != nil {
			s.addStateError(revertErr)
		}
		s.addStateError(err)
		return nil, s.getStateError()
	}

	handle := s.stageBlock(&stagedBlock{
		state:  s,
		block:  block,
		hash:   hash,
		update: update,
		undo:   undoList,
		hints:  archiveUpdateHints,
	})

	// The block is decided the moment it is staged: no caller can hold it, so
	// there is nobody to decide it later.
	archiveWriteDone, err := handle.Commit()
	if err != nil {
		return nil, err
	}
	if s.archive == nil {
		return nil, nil // no archive, nothing to wait for
	}
	// Apply still reports the outcome of the archive write on a channel. The
	// block was committed right here, so this is its only waiter.
	relay := make(chan error, 1)
	go func() {
		defer close(relay)
		if err := archiveWriteDone.Wait(); err != nil {
			relay <- err
		}
	}()
	return relay, nil
}

// stageBlock appends the block to the staged queue and returns the handle a
// caller decides it with.
func (s *GoState) stageBlock(block *stagedBlock) state.StagedBlock {
	s.stagedLock.Lock()
	defer s.stagedLock.Unlock()
	s.nextStagedId++
	block.id = s.nextStagedId
	s.staged = append(s.staged, block)
	return block.GetHandle()
}

// commitStaged promotes the block the handle stands for into the archive. It
// is rejected if the handle has been decided already or if its block is not the
// oldest staged one.
func (s *GoState) commitStaged(handle *stagedBlockHandle) (*state.WaitHandle, error) {
	s.stagedLock.Lock()
	defer s.stagedLock.Unlock()

	if handle.status != stagedPending {
		return nil, handle.decidedError("commit")
	}
	if len(s.staged) == 0 || !s.staged[0].isFor(handle) {
		return nil, s.misplacedError("commit", handle, "oldest")
	}
	block := s.staged[0]
	s.staged = s.staged[1:]
	handle.status = stagedCommitted

	// From here on the archive writer owns the hints and releases them.
	if s.archive == nil {
		if block.hints != nil {
			block.hints.Release()
		}
		return state.NewWaitHandle(nil), nil
	}

	done := make(chan error, 1)
	s.archiveWriter <- archiveUpdate{block.block, &block.update, block.hints, done}
	s.drainArchiveErrors()
	return state.NewWaitHandle(done), nil
}

// rollbackStaged takes the block the handle stands for back from the LiveDB. It
// is rejected if the handle has been decided already or if its block is not the
// newest staged one.
func (s *GoState) rollbackStaged(handle *stagedBlockHandle) error {
	s.stagedLock.Lock()
	defer s.stagedLock.Unlock()

	if handle.status != stagedPending {
		return handle.decidedError("roll back")
	}
	last := len(s.staged) - 1
	lastBlock := s.staged[last]
	if last < 0 || !lastBlock.isFor(handle) {
		return s.misplacedError("roll back", handle, "newest")
	}
	handle.status = stagedRolledBack
	s.staged = s.staged[:last]

	if lastBlock.hints != nil {
		lastBlock.hints.Release()
	}
	if err := s.live.RevertLastBlock(lastBlock.undo); err != nil {
		s.addStateError(err)
		return err
	}
	return nil
}

// misplacedError explains that the handle's block is not at the end of the
// queue the operation consumes from -- the "oldest" or the "newest" -- or not
// staged at all. It must be called while holding stagedLock.
func (s *GoState) misplacedError(operation string, handle *stagedBlockHandle, end string) error {
	index := s.indexOf(handle)
	if index < 0 {
		return fmt.Errorf("%w: cannot %s block %x: it is not staged", state.ErrStagedBlockMisuse, operation, handle.hash)
	}
	distance := index // < blocks between it and the oldest
	if end == "newest" {
		distance = len(s.staged) - 1 - index
	}
	return fmt.Errorf("%w: cannot %s block %d: it is not the %s staged block, %d block(s) are staged between them", state.ErrStagedBlockMisuse, operation, s.staged[index].block, end, distance)
}

// indexOf reports the position of the handle's block in the staged sequence,
// or -1. It must be called while holding stagedLock.
func (s *GoState) indexOf(handle *stagedBlockHandle) int {
	for i, candidate := range s.staged {
		if candidate.isFor(handle) {
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

// stagedBlock is what the state retains for a block applied to the LiveDB whose
// fate is not yet decided: the update and its hints to hand to the archive on
// commit, the undo operations to replay on rollback, and the identity the
// block's handle is matched by.
type stagedBlock struct {
	state  *GoState
	id     uint64
	block  uint64
	hash   common.Hash
	update common.Update
	undo   []func() error
	hints  common.Releaser
}

func (b *stagedBlock) GetHandle() *stagedBlockHandle {
	return &stagedBlockHandle{state: b.state, id: b.id, hash: b.hash}
}

// isFor reports whether the block is the one the handle stands for.
func (b *stagedBlock) isFor(handle *stagedBlockHandle) bool {
	return b.id == handle.id
}

// stagedStatus tracks which of the two terminal operations a handle has already
// seen, so that a second one reports an error rather than acting twice.
type stagedStatus int

const (
	stagedPending stagedStatus = iota
	stagedCommitted
	stagedRolledBack
)

// stagedBlockHandle is the StagedBlock a caller holds.
// It's an opaque object identifying a staged block.
type stagedBlockHandle struct {
	state *GoState
	id    uint64
	hash  common.Hash

	status stagedStatus // guarded by state.stagedLock
}

func (h *stagedBlockHandle) StateHash() common.Hash {
	return h.hash
}

func (h *stagedBlockHandle) Commit() (*state.WaitHandle, error) {
	return h.state.commitStaged(h)
}

func (h *stagedBlockHandle) Rollback() error {
	return h.state.rollbackStaged(h)
}

// decidedError explains that the handle has already been decided.
func (h *stagedBlockHandle) decidedError(operation string) error {
	decision := "committed"
	if h.status == stagedRolledBack {
		decision = "rolled back"
	}
	return fmt.Errorf("%w: cannot %s block %x: it has already been %s", state.ErrStagedBlockMisuse, operation, h.hash, decision)
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
// decided, newest first.
// The handles of these blocks stay undecided; a late decision on one of them
// finds its block gone and is rejected as misuse.
func (s *GoState) rollbackUndecidedBlocks() {
	s.stagedLock.Lock()
	defer s.stagedLock.Unlock()
	for _, block := range slices.Backward(s.staged) {
		// The error is ignored: it is already recorded in the state error, which
		// Close reports through its final Check.
		_ = s.rollbackStaged(block.GetHandle())
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
