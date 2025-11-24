// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package flat

import (
	"context"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sync"
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/backend"
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/common/future"
	"github.com/0xsoniclabs/carmen/go/common/result"
	"github.com/0xsoniclabs/carmen/go/common/witness"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/0xsoniclabs/tracy"
	"github.com/ethereum/go-ethereum/core/types"
)

// State is an in-memory flat representation of the blockchain state, wrapping
// another state implementation for computing commitments in the background.
//
// Updates to the underlying state are applied asynchronously in a background
// goroutine, allowing for non-blocking state modifications and commitment
// computations.
//
// This implementation is primarily intended for testing and development
// purposes, as it keeps the entire state in memory, which may not be feasible
// for large states in production environments. If beneficial, a future version
// using caching and on-demand loading could be implemented.
//
// NOTE: this implementation is NOT thread-safe. Concurrent access must be
// externally synchronized. Also, it is not intended for production use.
type State struct {
	// All-in-memory flat state representation.
	accounts map[common.Address]account
	storage  map[slotKey]common.Value
	codes    map[common.Hash][]byte

	// Backend storage for computing commits.
	backend state.State

	// Controls for interacting with the background worker keeping the backend
	// up to date, and computing commitments.
	commands chan<- command  // < commands to background worker
	syncs    <-chan struct{} // < signalled when syncing with background worker
	done     <-chan struct{} // < when background work is done

	issues issueCollector // < issues identified by background worker
}

// account holds the flat representation of an account's data.
type account struct {
	balance  amount.Amount
	nonce    common.Nonce
	codeSize int
	codeHash common.Hash
}

// slotKey uniquely identifies a storage slot for an account.
type slotKey struct {
	address common.Address
	key     common.Key
}

// command represents an operation to be performed by the background worker.
// There are three types of commands:
// 1. Update command: applies a state update for a specific block.
// 2. Commit command: requests the computation of the current state commitment.
// 3. Sync command: signals the worker to flush all pending updates and report any issues.
// Sync commands are represented by a command with both update and commit fields set to nil.
type command struct {
	update *update
	commit *future.Promise[result.Result[common.Hash]]
}

// update encapsulates a state update for a specific block.
type update struct {
	block uint64
	data  common.Update
}

// NewState creates a new flat State instance that wraps the provided backend state.
// The resulting state is wrapped into a synced state for thread-safe access.
func NewState(backend state.State) state.State {
	commands := make(chan command, 1024)
	syncs := make(chan struct{})
	done := make(chan struct{})

	res := &State{
		accounts: make(map[common.Address]account),
		storage:  make(map[slotKey]common.Value),
		codes:    make(map[common.Hash][]byte),
		backend:  backend,
		commands: commands,
		syncs:    syncs,
		done:     done,
	}

	go func() {
		defer close(done)
		processCommands(backend, commands, syncs, &res.issues)
	}()

	return state.WrapIntoSyncedState(res)
}

// WrapFactory wraps an existing state factory to produce flat State instances.
func WrapFactory(innerFactory state.StateFactory) state.StateFactory {
	return func(params state.Parameters) (state.State, error) {
		inner, err := innerFactory(params)
		if err != nil {
			return nil, err
		}
		return NewState(inner), nil
	}
}

// --- State Interface Implementation ---

func (s *State) Exists(address common.Address) (bool, error) {
	_, found := s.accounts[address]
	return found, nil
}

func (s *State) GetBalance(address common.Address) (amount.Amount, error) {
	return s.accounts[address].balance, nil
}

func (s *State) GetNonce(address common.Address) (common.Nonce, error) {
	return s.accounts[address].nonce, nil
}

func (s *State) GetStorage(address common.Address, key common.Key) (common.Value, error) {
	return s.storage[slotKey{address, key}], nil
}

func (s *State) GetCode(address common.Address) ([]byte, error) {
	hash := s.accounts[address].codeHash
	return s.codes[hash], nil
}

func (s *State) GetCodeSize(address common.Address) (int, error) {
	return s.accounts[address].codeSize, nil
}

func (s *State) GetCodeHash(address common.Address) (common.Hash, error) {
	return s.accounts[address].codeHash, nil
}

func (s *State) HasEmptyStorage(addr common.Address) (bool, error) {
	// TODO: eliminate this function entirely
	return true, nil
}

func (s *State) Apply(block uint64, data common.Update) error {

	zone := tracy.ZoneBegin("State.Apply")
	defer zone.End()

	for _, address := range data.DeletedAccounts {
		delete(s.accounts, address)
	}

	// init potentially empty accounts with empty code hash,
	for _, address := range data.CreatedAccounts {
		// empty account has empty code size, nonce, and balance
		s.accounts[address] = account{
			codeHash: common.Hash(types.EmptyCodeHash),
		}
	}

	for _, update := range data.Nonces {
		data := s.accounts[update.Account]
		data.nonce = update.Nonce
		s.accounts[update.Account] = data
	}

	for _, update := range data.Balances {
		data := s.accounts[update.Account]
		data.balance = update.Balance
		s.accounts[update.Account] = data
	}

	for _, update := range data.Slots {
		s.storage[slotKey{update.Account, update.Key}] = update.Value
	}

	for _, update := range data.Codes {
		data := s.accounts[update.Account]
		data.codeSize = len(update.Code)
		hash := common.Keccak256(update.Code)
		data.codeHash = hash
		s.accounts[update.Account] = data
		s.codes[hash] = update.Code
	}

	// Update the backend in the background.
	s.commands <- command{
		update: &update{
			block: block,
			data:  data,
		},
	}
	return nil
}

func (s *State) GetHash() (common.Hash, error) {
	return s.GetCommitment().Await().Get()
}

func (s *State) GetCommitment() future.Future[result.Result[common.Hash]] {
	promise, future := future.Create[result.Result[common.Hash]]()
	s.commands <- command{
		commit: &promise,
	}
	return future
}

func processCommands(
	backend state.State,
	commands <-chan command,
	syncs chan<- struct{},
	issues *issueCollector,
) {
	for command := range commands {
		if command.update != nil {
			zone := tracy.ZoneBegin("State.Update")
			issues.HandleIssue(backend.Apply(command.update.block, command.update.data))
			zone.End()
		} else if command.commit != nil {
			zone := tracy.ZoneBegin("State.Commit")
			result := backend.GetCommitment().Await()
			command.commit.Fulfill(result)
			zone.End()
		} else { // sync command
			zone := tracy.ZoneBegin("State.Sync")
			syncs <- struct{}{}
			zone.End()
		}
	}
}

func (s *State) sync() error {
	s.commands <- command{}
	<-s.syncs
	return s.issues.Collect()
}

// --- Operational Features ---

func (s *State) Check() error {
	if err := s.issues.Collect(); err != nil {
		return err
	}
	return s.backend.Check()
}

func (s *State) Flush() error {
	if err := s.sync(); err != nil {
		return err
	}
	return s.backend.Flush()
}

func (s *State) Close() error {
	if err := s.sync(); err != nil {
		return err
	}
	close(s.commands)
	<-s.done
	return s.backend.Close()
}

func (s *State) GetMemoryFootprint() *common.MemoryFootprint {
	res := common.NewMemoryFootprint(unsafe.Sizeof(*s))
	res.AddChild("accounts", memoryFootprintOfMap(s.accounts))
	res.AddChild("storage", memoryFootprintOfMap(s.storage))
	res.AddChild("codes", memoryFootprintOfMap(s.codes))
	res.AddChild("backend", s.backend.GetMemoryFootprint())
	return res
}

func (s *State) GetArchiveState(block uint64) (state.State, error) {
	return s.backend.GetArchiveState(block)
}

func (s *State) GetArchiveBlockHeight() (height uint64, empty bool, err error) {
	return s.backend.GetArchiveBlockHeight()
}

func (s *State) CreateWitnessProof(address common.Address, keys ...common.Key) (witness.Proof, error) {
	return s.backend.CreateWitnessProof(address, keys...)
}

func (s *State) Export(ctx context.Context, out io.Writer) (common.Hash, error) {
	if err := s.sync(); err != nil {
		return common.Hash{}, err
	}
	return s.backend.Export(ctx, out)
}

// Snapshot & Recovery
func (s *State) GetProof() (backend.Proof, error) {
	return s.backend.GetProof()
}

func (s *State) CreateSnapshot() (backend.Snapshot, error) {
	return s.backend.CreateSnapshot()
}
func (s *State) Restore(data backend.SnapshotData) error {
	return s.backend.Restore(data)
}
func (s *State) GetSnapshotVerifier(data []byte) (backend.SnapshotVerifier, error) {
	return s.backend.GetSnapshotVerifier(data)
}

// --- Helpers ---

// issueCollector collects issues encountered during background processing.
// It limits the number of stored issues to avoid excessive memory usage.
// Only the first 10 issues are stored; any additional issues are counted
// but not stored in detail.
type issueCollector struct {
	issues      []error // < collected issues
	extraIssues int     // < count of additional issues beyond stored ones
	mutex       sync.Mutex
}

func (c *issueCollector) HandleIssue(err error) {
	if err == nil {
		return
	}
	c.mutex.Lock()
	defer c.mutex.Unlock()
	if len(c.issues) < 10 {
		c.issues = append(c.issues, err)
	} else {
		c.extraIssues++
	}
}

func (c *issueCollector) Collect() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	if c.extraIssues > 0 {
		c.issues = append(c.issues, fmt.Errorf("%d additional errors truncated", c.extraIssues))
	}
	res := errors.Join(c.issues...)
	c.issues = c.issues[:0]
	c.extraIssues = 0
	return res
}

func memoryFootprintOfMap[A comparable, B any](m map[A]B) *common.MemoryFootprint {
	entrySize :=
		reflect.TypeFor[A]().Size() +
			reflect.TypeFor[B]().Size()
	return common.NewMemoryFootprint(uintptr(len(m)) * entrySize)
}
