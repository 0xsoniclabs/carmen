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

	stateError error // collect errors occurred during operation

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
	done        chan<- struct{}
}

func (s *GoState) Exists(address common.Address) (bool, error) {
	if err := s.stateError; err != nil {
		return false, err
	}

	exist, err := s.live.Exists(address)
	if err != nil {
		s.stateError = errors.Join(s.stateError, err)
	}
	return exist, s.stateError
}

func (s *GoState) GetBalance(address common.Address) (amount.Amount, error) {
	if err := s.stateError; err != nil {
		return amount.New(), err
	}

	balance, err := s.live.GetBalance(address)
	if err != nil {
		s.stateError = errors.Join(s.stateError, err)
	}

	return balance, s.stateError
}

func (s *GoState) GetNonce(address common.Address) (common.Nonce, error) {
	if err := s.stateError; err != nil {
		return common.Nonce{}, err
	}

	nonce, err := s.live.GetNonce(address)
	if err != nil {
		s.stateError = errors.Join(s.stateError, err)
	}
	return nonce, s.stateError
}

func (s *GoState) GetStorage(address common.Address, key common.Key) (common.Value, error) {
	if err := s.stateError; err != nil {
		return common.Value{}, err
	}

	val, err := s.live.GetStorage(address, key)
	if err != nil {
		s.stateError = errors.Join(s.stateError, err)
	}
	return val, s.stateError
}

func (s *GoState) GetCode(address common.Address) ([]byte, error) {
	if err := s.stateError; err != nil {
		return []byte{}, err
	}

	code, err := s.live.GetCode(address)
	if err != nil {
		s.stateError = errors.Join(s.stateError, err)
	}
	return code, s.stateError
}

func (s *GoState) GetCodeSize(address common.Address) (int, error) {
	if err := s.stateError; err != nil {
		return 0, err
	}

	size, err := s.live.GetCodeSize(address)
	if err != nil {
		s.stateError = errors.Join(s.stateError, err)
	}
	return size, s.stateError
}

func (s *GoState) GetCodeHash(address common.Address) (common.Hash, error) {
	if err := s.stateError; err != nil {
		return common.Hash{}, err
	}

	h, err := s.live.GetCodeHash(address)
	if err != nil {
		s.stateError = errors.Join(s.stateError, err)
	}
	return h, s.stateError
}

func (s *GoState) HasEmptyStorage(addr common.Address) (bool, error) {
	if err := s.stateError; err != nil {
		return false, err
	}

	empty, err := s.live.HasEmptyStorage(addr)
	if err != nil {
		s.stateError = errors.Join(s.stateError, err)
	}
	return empty, s.stateError
}

func (s *GoState) GetHash() (common.Hash, error) {
	return s.GetCommitment().Await().Get()
}

func (s *GoState) GetCommitment() future.Future[result.Result[common.Hash]] {
	if err := s.stateError; err != nil {
		return future.Immediate(result.Err[common.Hash](err))
	}

	h, err := s.live.GetHash()
	if err != nil {
		s.stateError = errors.Join(s.stateError, err)
		return future.Immediate(result.Err[common.Hash](err))
	}
	return future.Immediate(result.Ok(h))
}

func (s *GoState) Apply(block uint64, update common.Update) error {
	return s.internalApply(block, update, false)
}

func (s *GoState) ApplySync(block uint64, update common.Update) error {
	return s.internalApply(block, update, true)
}

func (s *GoState) internalApply(block uint64, update common.Update, waitForSync bool) error {
	if err := s.stateError; err != nil {
		return err
	}

	// Apply the changes to the LiveDB.
	archiveUpdateHints, err := s.live.Apply(block, &update)
	if err != nil {
		s.stateError = errors.Join(s.stateError, err)
		return s.stateError
	}

	if s.archive != nil {
		var doneChan chan struct{}
		if waitForSync {
			doneChan = make(chan struct{})
		}
		// Send the update to the writer to be processed asynchronously.
		s.archiveWriter <- archiveUpdate{block, &update, archiveUpdateHints, doneChan}
		if waitForSync {
			// Wait until the update is fully processed.
			<-doneChan
		}

		// Drain potential errors, but do not wait for them.
		done := false
		for !done {
			select {
			// In case there was an error, process it.
			case err := <-s.archiveWriterError:
				s.stateError = errors.Join(s.stateError, err)
			default:
				// all errors consumed, moving on
				done = true
			}
		}
		if err := s.stateError; err != nil {
			return err
		}
	} else if archiveUpdateHints != nil {
		archiveUpdateHints.Release()
	}
	return nil
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
		s.stateError = errors.Join(s.stateError, err)
	}

	if s.archiveWriter != nil {
		// Wait until the flush was processed.
		<-s.archiveWriterFlushDone
	}

	return s.Check()
}

func (s *GoState) Close() error {
	s.stateError = errors.Join(
		s.stateError,
		s.Flush(),
		s.live.Close(),
	)

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
			s.stateError = errors.Join(s.stateError, err)
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
	if err := s.stateError; err != nil {
		return nil, err
	}
	lastBlock, empty, err := s.archive.GetBlockHeight()
	if err != nil {
		s.stateError = errors.Join(s.stateError, errors.Join(fmt.Errorf("failed to get last block in the archive"), err))
		return nil, s.stateError
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

func (s *GoState) RootHash(block uint64) (common.Hash, error) {
	if s.archive == nil {
		return common.Hash{}, state.NoArchiveError
	}
	if err := s.stateError; err != nil {
		return common.Hash{}, err
	}
	root, err := s.archive.GetHash(block)
	if err != nil {
		s.stateError = errors.Join(s.stateError, errors.Join(fmt.Errorf("failed to get root hash for block %d from the archive", block), err))
		return common.Hash{}, s.stateError
	}
	return root, nil
}

func (s *GoState) GetArchiveBlockHeight() (uint64, bool, error) {
	if s.archive == nil {
		return 0, false, state.NoArchiveError
	}
	if err := s.stateError; err != nil {
		return 0, false, err
	}
	lastBlock, empty, err := s.archive.GetBlockHeight()
	if err != nil {
		s.stateError = errors.Join(s.stateError, errors.Join(fmt.Errorf("failed to get last block in the archive"), err))
		return 0, false, s.stateError
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
				s.stateError = errors.Join(s.stateError, err)
			default:
				done = true
			}
		}
	}
	return s.stateError
}

func (s *GoState) Export(context.Context, io.Writer) (common.Hash, error) {
	return common.Hash{}, state.ExportNotSupported
}

func (s *GoState) CreateWitnessProof(address common.Address, keys ...common.Key) (witness.Proof, error) {
	panic("not implemented")
}
