package vt

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"testing"

	"github.com/0xsoniclabs/carmen/go/backend"
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/common/immutable"
	"github.com/0xsoniclabs/carmen/go/common/witness"
	"github.com/0xsoniclabs/carmen/go/database/vt/geth"
	"github.com/0xsoniclabs/carmen/go/database/vt/memory"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/stretchr/testify/require"
)

func TestState_CreateAccount_Hash_Matches(t *testing.T) {
	state, err := initTestedStates(t)
	require.NoError(t, err, "failed to initialize tested states")
	defer func() {
		require.NoError(t, state.Close(), "failed to close state")
	}()

	addr := common.Address{1}

	update := common.Update{}
	update.CreatedAccounts = append(update.CreatedAccounts, addr)

	require.NoError(t, state.Apply(0, update), "failed to apply update")

	// check hash consistency
	hash, err := state.GetHash()
	require.NoError(t, err, "failed to get hash")
	require.NotEmpty(t, hash, "hash should not be empty")
}

func TestState_Insert_Single_Values_One_Account_One_Storage(t *testing.T) {
	state, err := initTestedStates(t)
	require.NoError(t, err, "failed to initialize tested states")
	defer func() {
		require.NoError(t, state.Close(), "failed to close state")
	}()

	update := common.Update{}
	addr := common.Address{1}
	key := common.Key{1}
	value := common.Value{1}

	update.CreatedAccounts = append(update.CreatedAccounts, addr)
	update.Nonces = append(update.Nonces, common.NonceUpdate{Account: addr, Nonce: common.ToNonce(1)})
	update.Balances = append(update.Balances, common.BalanceUpdate{Account: addr, Balance: amount.New(1)})
	update.Slots = append(update.Slots, common.SlotUpdate{Account: addr, Key: key, Value: value})

	require.NoError(t, state.Apply(0, update), "failed to apply update")

	// check hash consistency
	hash, err := state.GetHash()
	require.NoError(t, err, "failed to get hash")
	require.NotEmpty(t, hash, "hash should not be empty")

	// check basic properties of accounts
	nonce, err := state.GetNonce(addr)
	require.NoError(t, err, "failed to get nonce for account %x", addr)
	require.Equal(t, common.ToNonce(1), nonce, "nonce mismatch for account %x", addr)

	balance, err := state.GetBalance(addr)
	require.NoError(t, err, "failed to get balance for account %x", addr)
	require.Equal(t, amount.New(1), balance, "balance mismatch for account %x", addr)

	slot, err := state.GetStorage(addr, key)
	require.NoError(t, err, "failed to get storage for account %x and key %x", addr, key)
	require.Equal(t, value, slot, "storage value mismatch for account %x and key %x", addr, key)
}

func TestState_CreateAccounts_In_Blocks_Accounts_Updated(t *testing.T) {
	state, err := initTestedStates(t)
	require.NoError(t, err, "failed to initialize tested states")
	defer func() {
		require.NoError(t, state.Close(), "failed to close state")
	}()

	const numBlocks = 10
	const numInsertsPerBlock = 1000
	for i := 0; i < numBlocks; i++ {
		update := common.Update{}
		for j := 0; j < numInsertsPerBlock; j++ {
			addr := common.Address{byte(j), byte(j >> 8)}
			update.CreatedAccounts = append(update.CreatedAccounts, addr)
			update.Nonces = append(update.Nonces, common.NonceUpdate{Account: addr, Nonce: common.ToNonce(uint64(i * j))})
			update.Balances = append(update.Balances, common.BalanceUpdate{Account: addr, Balance: amount.New(uint64(i * j))})
		}
		require.NoError(t, state.Apply(uint64(i), update), "failed to apply block %d", i)

		// check hash consistency
		hash, err := state.GetHash()
		require.NoError(t, err, "failed to get hash for block %d", i)
		require.NotEmpty(t, hash, "hash should not be empty for block %d", i)

		// check basic properties of accounts
		for j := 0; j < numInsertsPerBlock; j++ {
			addr := common.Address{byte(j), byte(j >> 8)}

			nonce, err := state.GetNonce(addr)
			require.NoError(t, err, "failed to get nonce for account %x in block %d", addr, i)
			require.Equal(t, common.ToNonce(uint64(i*j)), nonce, "nonce mismatch for account %x in block %d", addr, i)

			balance, err := state.GetBalance(addr)
			require.NoError(t, err, "failed to get balance for account %x in block %d", addr, i)
			require.Equal(t, amount.New(uint64(i*j)), balance, "balance mismatch for account %x in block %d", addr, i)
		}
	}
}

func TestState_Storage_Can_Set_And_Receive(t *testing.T) {
	state, err := initTestedStates(t)
	require.NoError(t, err, "failed to initialize tested states")
	defer func() {
		require.NoError(t, state.Close(), "failed to close state")
	}()

	const numAddresses = 10
	const numKeysPerAddress = 1000
	update := common.Update{}
	for i := 0; i < numAddresses; i++ {
		addr := common.Address{byte(i), byte(i >> 8)}
		for j := 0; j < numKeysPerAddress; j++ {
			key := common.Key{byte(i), byte(i >> 8), byte(j), byte(j >> 8)}
			value := common.Value{byte(i), byte(i >> 8), byte(j), byte(j >> 8)}
			update.Codes = append(update.Codes, common.CodeUpdate{Account: addr, Code: []byte{}})
			update.Slots = append(update.Slots, common.SlotUpdate{Account: addr, Key: key, Value: value})
		}
	}
	require.NoError(t, state.Apply(uint64(0), update), "failed to apply block")

	// check hash consistency
	hash, err := state.GetHash()
	require.NoError(t, err, "failed to get hash for block")
	require.NotEmpty(t, hash, "hash should not be empty for block")

	// check storage properties
	for i := 0; i < numAddresses; i++ {
		addr := common.Address{byte(i), byte(i >> 8)}
		for j := 0; j < numKeysPerAddress; j++ {
			key := common.Key{byte(i), byte(i >> 8), byte(j), byte(j >> 8)}
			expectedValue := common.Value{byte(i), byte(i >> 8), byte(j), byte(j >> 8)}

			value, err := state.GetStorage(addr, key)
			require.NoError(t, err, "failed to get storage for account %x and key %x", addr, key)
			require.Equal(t, expectedValue, value, "storage value mismatch for account %x and key %x", addr, key)
		}
	}
}

func TestState_Code_Can_Set_And_Receive(t *testing.T) {
	state, err := initTestedStates(t)
	require.NoError(t, err, "failed to initialize tested states")
	defer func() {
		require.NoError(t, state.Close(), "failed to close state")
	}()

	code := make([]byte, 4096)
	for i := 0; i < len(code); i++ {
		code[i] = byte(i)
	}

	const numOfCodes = 1000
	expectedCodes := [numOfCodes][]byte{}
	update := common.Update{}
	for i := 0; i < numOfCodes; i++ {
		addr := common.Address{byte(i), byte(i >> 8)}
		update.Codes = append(update.Codes, common.CodeUpdate{
			Account: addr,
			Code:    code,
		})
		expectedCodes[i] = code
		code = append(code, byte(i))
	}

	require.NoError(t, state.Apply(0, update), "failed to apply update")

	// check hash consistency
	hash, err := state.GetHash()
	require.NoError(t, err, "failed to get hash for block")
	require.NotEmpty(t, hash, "hash should not be empty for block")

	for i := 0; i < numOfCodes; i++ {
		addr := common.Address{byte(i), byte(i >> 8)}

		code, err := state.GetCode(addr)
		require.NoError(t, err, "failed to get code for account %x", addr)
		require.True(t, bytes.Equal(code, expectedCodes[i]), "unexpected code for account %x: got %v, want %v", addr, code, expectedCodes[i])

		codeHash, err := state.GetCodeHash(addr)
		require.NoError(t, err, "failed to get code hash for account %x", addr)
		require.Equal(t, common.Keccak256(expectedCodes[i]), codeHash, "unexpected code hash for account %x", addr)

		codeLen, err := state.GetCodeSize(addr)
		require.NoError(t, err, "failed to get code size for account %x", addr)
		require.Equal(t, len(expectedCodes[i]), codeLen, "unexpected code size for account %x", addr)
	}
}

func TestState_Storage_Leading_Zeros_HashesMatch(t *testing.T) {
	addr1 := common.Address{0, 0, 0, 1}
	value := common.Value{0, 0, 0, 1}

	update := common.Update{
		CreatedAccounts: []common.Address{addr1},
		Balances: []common.BalanceUpdate{
			{Account: addr1, Balance: amount.New(1)},
		},
		Slots: []common.SlotUpdate{
			{Account: addr1, Key: common.Key{0, 0, 0, 1}, Value: value},
		},
	}

	state, err := initTestedStates(t)
	require.NoError(t, err, "failed to initialize tested states")
	defer func() {
		require.NoError(t, state.Close(), "failed to close state")
	}()

	require.NoError(t, state.Apply(0, update))

	hash, err := state.GetHash()
	require.NoError(t, err)
	require.NotEmpty(t, hash, "hash should not be empty")
}

// initTestedStates initializes a comparingState instance
// for all State instances under test.
func initTestedStates(t *testing.T) (*comparingState, error) {
	st := &comparingState{
		states: make(map[string]state.State),
		t:      t,
	}

	gethSt, err := geth.NewState(state.Parameters{})

	st.put("memory", memory.NewState())
	st.put("geth", gethSt)

	return st, err
}

// comparingState is a state.State implementation that compares multiple
// state.State instances. It is used to ensure that all operations
// performed on the state are consistent across all provided states.
// It implements the state.State interface and provides methods to
// manipulate accounts, storage, and code in a consistent manner across
// multiple state instances. The methods defined in this struct will
// call the corresponding methods on each state instance and return an
// error if any of the states return an error. This is useful for
// testing and debugging purposes, where you want to ensure that all
// state instances behave consistently with each other.
type comparingState struct {
	states map[string]state.State // map of state names to state instances
	t      *testing.T
}

// put adds a new state to the comparingState instance.
func (s *comparingState) put(name string, state state.State) {
	s.states[name] = state
}

func (s *comparingState) Exists(address common.Address) (bool, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(name string, state state.State) (bool, error) {
		return state.Exists(address)
	})
}

func (s *comparingState) GetNonce(address common.Address) (common.Nonce, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(name string, state state.State) (common.Nonce, error) {
		return state.GetNonce(address)
	})
}

func (s *comparingState) GetStorage(address common.Address, key common.Key) (common.Value, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(name string, state state.State) (common.Value, error) {
		return state.GetStorage(address, key)
	})
}

func (s *comparingState) GetCode(address common.Address) ([]byte, error) {
	s.t.Helper()
	b, err := getCmpStateValue(s, func(name string, state state.State) (immutable.Bytes, error) {
		b, err := state.GetCode(address)
		return immutable.NewBytes(b), err
	})
	return b.ToBytes(), err
}

func (s *comparingState) GetCodeSize(address common.Address) (int, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(name string, state state.State) (int, error) {
		return state.GetCodeSize(address)
	})
}

func (s *comparingState) GetCodeHash(address common.Address) (common.Hash, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(name string, state state.State) (common.Hash, error) {
		return state.GetCodeHash(address)
	})
}

func (s *comparingState) GetHash() (common.Hash, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(name string, state state.State) (common.Hash, error) {
		return state.GetHash()
	})
}

func (s *comparingState) HasEmptyStorage(addr common.Address) (bool, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(name string, state state.State) (bool, error) {
		return state.HasEmptyStorage(addr)
	})
}

func (s *comparingState) GetBalance(address common.Address) (amount.Amount, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(name string, state state.State) (amount.Amount, error) {
		return state.GetBalance(address)
	})
}

func (s *comparingState) Apply(block uint64, update common.Update) error {
	s.t.Helper()
	return s.action(func(name string, state state.State) error {
		return state.Apply(block, update)
	})
}

func (s *comparingState) GetArchiveState(block uint64) (state.State, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(name string, state state.State) (state.State, error) {
		return state.GetArchiveState(block)
	})
}

func (s *comparingState) GetArchiveBlockHeight() (height uint64, empty bool, err error) {
	s.t.Helper()
	return 0, true, state.NoArchiveError
}

//
//		Witness Proof features -- not supported at the moment
//

func (s *comparingState) CreateWitnessProof(address common.Address, keys ...common.Key) (witness.Proof, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(name string, state state.State) (witness.Proof, error) {
		return state.CreateWitnessProof(address, keys...)
	})
}

//
//		Snapshot features -- not supported in Verkle Trie
//

func (s *comparingState) GetProof() (backend.Proof, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(name string, state state.State) (backend.Proof, error) {
		return state.GetProof()
	})
}

func (s *comparingState) CreateSnapshot() (backend.Snapshot, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(name string, state state.State) (backend.Snapshot, error) {
		return state.CreateSnapshot()
	})
}

func (s *comparingState) Restore(data backend.SnapshotData) error {
	s.t.Helper()
	return s.action(func(name string, state state.State) error {
		return state.Restore(data)
	})
}

func (s *comparingState) GetSnapshotVerifier(metadata []byte) (backend.SnapshotVerifier, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(name string, state state.State) (backend.SnapshotVerifier, error) {
		return state.GetSnapshotVerifier(metadata)
	})
}

//
//	Operation features -- not supported
//

func (s *comparingState) Export(ctx context.Context, out io.Writer) (common.Hash, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(name string, state state.State) (common.Hash, error) {
		return state.Export(ctx, out)
	})
}

func (s *comparingState) Check() error {
	s.t.Helper()
	return s.action(func(name string, state state.State) error {
		return state.Check()
	})
}

func (s *comparingState) GetMemoryFootprint() *common.MemoryFootprint {
	s.t.Helper()
	return common.NewMemoryFootprint(uintptr(1))
}

//
//	I/O features
//

func (s *comparingState) Flush() error {
	s.t.Helper()
	return s.action(func(name string, state state.State) error {
		return state.Flush()
	})
}

func (s *comparingState) Close() error {
	s.t.Helper()
	return s.action(func(name string, state state.State) error {
		return state.Close()
	})
}

// action is a helper function that iterates over all states in the comparingState
// and applies the provided function to each state. It collects errors
// and returns a single error that combines all errors encountered during
// the execution of the provided function on each state.
func (s *comparingState) action(do func(name string, state state.State) error) error {
	s.t.Helper()
	var errs []error
	for name, state := range s.states {
		if err := do(name, state); err != nil {
			errs = append(errs, fmt.Errorf("state %s: %w", name, err))
		}
	}

	return errors.Join(errs...)
}

// getCmpStateValue is a helper function that iterates over all states in the comparingState
// and applies the provided function to each state. It collects errors and checks
// for value consistency across all states. If all states return the same value,
// it returns that value; otherwise, it returns an error indicating the mismatch.
func getCmpStateValue[T comparable](s *comparingState, do func(name string, state state.State) (T, error)) (T, error) {
	s.t.Helper()
	var errs []error
	var ref *T
	var refName string
	for name, state := range s.states {
		v, err := do(name, state)
		if err != nil {
			errs = append(errs, fmt.Errorf("state %s: %w", name, err))
		}

		if ref != nil && *ref != v {
			s.t.Errorf("state '%s', value mismatch: got %v, state '%s' was:  %v", name, v, refName, *ref)
		}

		if ref == nil {
			ref = new(T)
			*ref = v
			refName = name
		}
	}

	if ref == nil {
		ref = new(T)
	}

	return *ref, errors.Join(errs...)
}
