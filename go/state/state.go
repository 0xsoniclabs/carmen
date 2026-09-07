// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package state

//go:generate mockgen -source state.go -destination state_mock.go -package state

import (
	"context"
	"fmt"
	"io"
	"sync"
	"sync/atomic"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/common/future"
	"github.com/0xsoniclabs/carmen/go/common/result"
	"github.com/0xsoniclabs/carmen/go/common/witness"
)

// NoArchiveError is an error returned by implementation of the State interface
// for archive operations if no archive is maintained by this implementation.
const NoArchiveError = common.ConstError("state does not maintain archive data")

// ErrStagedBlockMisuse reports that a StagedBlock was used against its contract:
// decided twice, decided out of order, or waited for without being committed. It
// marks a mistake in the calling code rather than a failure of the state, so the
// state stays usable and does not collect it as an issue.
const ErrStagedBlockMisuse = common.ConstError("staged block used out of contract")

// State interfaces provides access to accounts and smart contract values memory.
type State interface {
	// GetBalance provides balance for the input account address.
	GetBalance(address common.Address) (amount.Amount, error)

	// GetNonce returns nonce of the account for the  input account address.
	GetNonce(address common.Address) (common.Nonce, error)

	// GetStorage returns the memory slot for the account address (i.e. the contract) and the memory location key.
	GetStorage(address common.Address, key common.Key) (common.Value, error)

	// GetCode returns code of the contract for the input contract address.
	GetCode(address common.Address) ([]byte, error)

	// GetCodeSize returns the length of the contract for the input contract address.
	GetCodeSize(address common.Address) (int, error)

	// GetCodeHash returns the hash of the code of the input contract address.
	GetCodeHash(address common.Address) (common.Hash, error)

	// HasEmptyStorage returns true if the contract has no storage attached to it.
	HasEmptyStorage(addr common.Address) (bool, error)

	// Apply applies the provided updates to the live state and returns the result
	// as a staged block: applied to the live state, but not yet promoted into the
	// archive. The caller decides its fate through the returned StagedBlock.
	//
	// Several blocks may be staged at once, which lets a caller execute ahead of a
	// decision it has not taken yet.
	Apply(block uint64, update common.Update) (StagedBlock, error)

	// GetHash hashes the state.
	// Deprecated: use GetCommitment instead.
	GetHash() (common.Hash, error)

	// GetCommitment computes a full-state commitment. The operation may be
	// performed asynchronously, allowing concurrent operations in the
	// meanwhile. The commitment, however, will be provided for the state as it
	// was when GetCommitment was called.
	GetCommitment() future.Future[result.Result[common.Hash]]

	// Flush writes all committed content to disk.
	Flush() error

	// Close flushes the store and closes it.
	Close() error

	// GetMemoryFootprint computes an approximation of the memory used by this state.
	GetMemoryFootprint() *common.MemoryFootprint

	// GetArchiveState provides a historical State view for given block.
	// An error is returned if the archive is not enabled or if it is empty.
	GetArchiveState(block uint64) (State, error)

	// GetArchiveBlockHeight provides the block height available in the archive. If
	// there is no block in the archive, the empty flag is returned.
	// An error is returned if the archive is not enabled or an IO issue occurred.
	GetArchiveBlockHeight() (height uint64, empty bool, err error)

	// Check checks the state of the DB and reports an error if issues have been
	// encountered.
	// Check should be called periodically to validate all interactions
	// with a State instance.
	// If an error is reported, all operations since the
	// last successful check need to be considered invalid.
	Check() error

	// CreateWitnessProof creates a witness proof for the given account and keys.
	// Error may be produced when it occurs in the underlying database;
	// otherwise, the proof is returned.
	CreateWitnessProof(address common.Address, keys ...common.Key) (witness.Proof, error)

	// Export writes data from LiveDB into out.
	// Temporary staging data is placed under scratchDir.
	// If successful, expected root hash is returned.
	Export(ctx context.Context, out io.Writer, scratchDir string) (common.Hash, error)
}

// StagedBlock is a block that has been applied to the live state but is not yet
// part of the archive. It is what lets a caller execute several blocks ahead of a
// decision it has not taken yet and then keep or discard each of them.
//
// Exactly one of Commit or Rollback must be called. Both invalidate the block, and
// a second call on it reports an error rather than acting twice.
//
// Ordering is enforced rather than merely documented, because each operation is
// only meaningful at one end of the staged sequence. Commit applies to the OLDEST
// staged block, since the archive is append-only and must receive blocks in order.
// Rollback applies to the NEWEST, since every undo operation restores a value read
// before its own block ran, and so reconstructs the intended state only once every
// later block has already been rolled back.
type StagedBlock interface {
	// StateHash returns the root of the live state as of this block.
	StateHash() common.Hash

	// Commit promotes this block into the archive. It returns as soon as the write
	// is under way, without waiting for it to complete; use Wait for that.
	//
	// It reports an error if this is not the oldest staged block, or if the block
	// has already been committed or rolled back.
	Commit() error

	// Wait blocks until the archive write triggered by Commit has completed and
	// reports its outcome. It must be called after Commit. It returns immediately
	// if the state maintains no archive, since then there is nothing to wait for.
	Wait() error

	// Rollback reverts this block from the live state, restoring the root its
	// predecessor left behind.
	//
	// It reports an error if this is not the newest staged block, or if the block
	// has already been committed or rolled back.
	Rollback() error
}

// NewIrreversibleBlock returns a StagedBlock for a state that applies a block the
// moment Apply is called and offers no way to take it back. Commit has nothing
// left to do, Wait waits for whatever asynchronous work the state started (pass a
// nil channel if there is none), and Rollback reports that this state does not
// support it.
//
// It serves the state implementations that neither maintain an archive nor stage:
// their staged sequence is always empty, so no ordering rule can be broken and
// every block is final the moment it is applied.
//
// The hash function must report the root of this block. Reading the live root when
// asked instead would make a handle kept across a later block report that block's
// root. A state that computes its root asynchronously can resolve the value when
// the function is first called.
//
// Wait does not require a preceding Commit: the asynchronous work is started by
// Apply rather than by the decision, so there is something to wait for from the
// moment the handle exists.
func NewIrreversibleBlock(block uint64, hash func() common.Hash, done <-chan error) StagedBlock {
	return &irreversibleBlock{block: block, hash: hash, done: done}
}

type irreversibleBlock struct {
	block uint64
	hash  func() common.Hash
	done  <-chan error

	// committed rejects a second decision. Committing this block does nothing, but
	// a caller that decides twice is making the same mistake it would be told about
	// on a staging state, and should hear about it on either.
	committed atomic.Bool

	// waitOnce makes sure the outcome of the asynchronous work is read from the
	// channel once: it is closed after the outcome is sent, so a second read would
	// report success for work that failed.
	waitOnce sync.Once
	waitErr  error
}

func (b *irreversibleBlock) StateHash() common.Hash {
	return b.hash()
}

func (b *irreversibleBlock) Commit() error {
	if !b.committed.CompareAndSwap(false, true) {
		return fmt.Errorf("%w: cannot commit block %d: it has already been committed", ErrStagedBlockMisuse, b.block)
	}
	return nil
}

func (b *irreversibleBlock) Wait() error {
	if b.done == nil {
		return nil
	}
	b.waitOnce.Do(func() { b.waitErr = <-b.done })
	return b.waitErr
}

func (b *irreversibleBlock) Rollback() error {
	return fmt.Errorf("%w: cannot roll back block %d: this state does not support rolling back blocks", ErrStagedBlockMisuse, b.block)
}

type LiveDB interface {
	GetBalance(address common.Address) (balance amount.Amount, err error)
	GetNonce(address common.Address) (nonce common.Nonce, err error)
	GetStorage(address common.Address, key common.Key) (value common.Value, err error)
	GetCode(address common.Address) (value []byte, err error)
	GetCodeSize(address common.Address) (size int, err error)
	GetCodeHash(address common.Address) (hash common.Hash, err error)
	HasEmptyStorage(addr common.Address) (bool, error)
	GetHash() (hash common.Hash, err error)
	Apply(block uint64, update *common.Update) (undoList []func() error, archiveUpdateHints common.Releaser, err error)
	RevertLastBlock(undo []func() error) error
	Flush() error
	Close() error
	common.MemoryFootprintProvider
}
