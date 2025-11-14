package flat

import (
	"context"
	"errors"
	"fmt"
	"io"
	"reflect"
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/backend"
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/common/future"
	"github.com/0xsoniclabs/carmen/go/common/witness"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/ethereum/go-ethereum/core/types"
)

type State struct {
	// All-in-memory flat state representation.
	accounts map[common.Address]account
	storage  map[slotKey]common.Value
	codes    map[common.Hash][]byte

	// Backend storage for computing commits.
	backend  state.State
	commands chan<- command  // < commands to background worker
	syncs    <-chan error    // < signalled when syncing with background worker
	done     <-chan struct{} // < when background work is done
}

type account struct {
	balance  amount.Amount
	nonce    common.Nonce
	codeSize int
	codeHash common.Hash
}

type slotKey struct {
	address common.Address
	key     common.Key
}

type command struct {
	update *update
	commit future.Promise[common.Hash]
}

type update struct {
	block uint64
	data  common.Update
}

func NewState(backend state.State) *State {
	commands := make(chan command, 1024)
	syncs := make(chan error)
	done := make(chan struct{})

	go func() {
		defer close(done)
		var issues []error
		extraIssues := 0
		for command := range commands {
			if command.update != nil {
				err := backend.Apply(command.update.block, command.update.data)
				if err != nil {
					if len(issues) < 10 {
						issues = append(issues, fmt.Errorf("block %d: %w", command.update.block, err))
					} else {
						extraIssues++
					}
				}
			} else if command.commit != nil {
				result := backend.GetCommitment().Get()
				command.commit.Fulfill(result)
			} else { // sync command
				if extraIssues > 0 {
					issues = append(issues, fmt.Errorf("%d additional errors truncated", extraIssues))
					extraIssues = 0
				}
				syncs <- errors.Join(issues...)
				issues = issues[:0]
			}
		}
	}()

	return &State{
		accounts: make(map[common.Address]account),
		storage:  make(map[slotKey]common.Value),
		codes:    make(map[common.Hash][]byte),
		backend:  backend,
		commands: commands,
		syncs:    syncs,
		done:     done,
	}
}

func WrapFactory(innerFactory state.StateFactory) state.StateFactory {
	return func(params state.Parameters) (state.State, error) {
		inner, err := innerFactory(params)
		if err != nil {
			return nil, err
		}
		return NewState(inner), nil
	}
}

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
	return s.GetCommitment().Await()
}

func (s *State) GetCommitment() future.Future[common.Hash] {
	promise, future := future.Create[common.Hash]()
	s.commands <- command{
		commit: promise,
	}
	return future
}

func (s *State) sync() error {
	s.commands <- command{}
	return <-s.syncs
}

// --- Operational Features ---

func (s *State) Check() error {
	if err := s.sync(); err != nil {
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
	return nil, state.NoArchiveError
}

func (s *State) GetArchiveBlockHeight() (height uint64, empty bool, err error) {
	return 0, true, state.NoArchiveError
}

func (s *State) CreateWitnessProof(address common.Address, keys ...common.Key) (witness.Proof, error) {
	return nil, fmt.Errorf("witness proof not supported yet")
}

func (s *State) Export(ctx context.Context, out io.Writer) (common.Hash, error) {
	panic("not implemented")
}

// Snapshot & Recovery
func (s *State) GetProof() (backend.Proof, error) {
	return nil, backend.ErrSnapshotNotSupported
}

func (s *State) CreateSnapshot() (backend.Snapshot, error) {
	return nil, backend.ErrSnapshotNotSupported
}
func (s *State) Restore(backend.SnapshotData) error {
	return backend.ErrSnapshotNotSupported
}
func (s *State) GetSnapshotVerifier([]byte) (backend.SnapshotVerifier, error) {
	return nil, backend.ErrSnapshotNotSupported
}

func memoryFootprintOfMap[A comparable, B any](m map[A]B) *common.MemoryFootprint {
	entrySize :=
		reflect.TypeFor[A]().Size() +
			reflect.TypeFor[B]().Size()
	return common.NewMemoryFootprint(uintptr(len(m)) * entrySize)
}
