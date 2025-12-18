// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package vt

import (
	"bytes"
	"errors"
	"fmt"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/common/immutable"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/stretchr/testify/require"

	_ "github.com/0xsoniclabs/carmen/go/state/externalstate"
	_ "github.com/0xsoniclabs/carmen/go/state/gostate"
	_ "github.com/0xsoniclabs/carmen/go/state/gostate/experimental"
)

func TestState_CreateAccount_Hash_Matches(t *testing.T) {
	stateFactories, err := initTestedStates()
	require.NoError(t, err, "failed to initialize tested states")

	for _, factory := range stateFactories {
		t.Run(fmt.Sprintf("state_%d_%s", factory.config.Schema, factory.config.Variant), func(t *testing.T) {
			t.Parallel()
			state, err := factory.newState(t)
			require.NoError(t, err, "failed to initialize tested state")

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
		})
	}
}

func TestState_Insert_Single_Values_One_Account_One_Storage(t *testing.T) {
	stateFactories, err := initTestedStates()
	require.NoError(t, err, "failed to initialize tested states")

	for _, factory := range stateFactories {
		t.Run(fmt.Sprintf("state_%d_%s", factory.config.Schema, factory.config.Variant), func(t *testing.T) {
			t.Parallel()
			state, err := factory.newState(t)
			require.NoError(t, err, "failed to initialize tested state")

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
		})
	}

}

func TestState_CreateAccounts_In_Blocks_Accounts_Updated(t *testing.T) {
	stateFactories, err := initTestedStates()
	require.NoError(t, err, "failed to initialize tested states")

	for _, factory := range stateFactories {
		t.Run(fmt.Sprintf("state_%d_%s", factory.config.Schema, factory.config.Variant), func(t *testing.T) {
			t.Parallel()
			state, err := factory.newState(t)
			require.NoError(t, err, "failed to initialize tested state")

			defer func() {
				require.NoError(t, state.Close(), "failed to close state")
			}()

			const numBlocks = 10
			const numInsertsPerBlock = 10
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
		})
	}

}

func TestState_Storage_Can_Set_And_Receive(t *testing.T) {
	stateFactories, err := initTestedStates()
	require.NoError(t, err, "failed to initialize tested states")

	for _, factory := range stateFactories {
		t.Run(fmt.Sprintf("state_%d_%s", factory.config.Schema, factory.config.Variant), func(t *testing.T) {
			t.Parallel()
			state, err := factory.newState(t)
			require.NoError(t, err, "failed to initialize tested state")

			defer func() {
				require.NoError(t, state.Close(), "failed to close state")
			}()

			const numAddresses = 10
			const numKeysPerAddress = 10
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
		})
	}
}

func TestState_Code_Can_Set_And_Receive(t *testing.T) {
	stateFactories, err := initTestedStates()
	require.NoError(t, err, "failed to initialize tested states")

	for _, factory := range stateFactories {
		t.Run(fmt.Sprintf("state_%d_%s", factory.config.Schema, factory.config.Variant), func(t *testing.T) {
			t.Parallel()
			state, err := factory.newState(t)
			require.NoError(t, err, "failed to initialize tested state")

			defer func() {
				require.NoError(t, state.Close(), "failed to close state")
			}()

			// this method creates multiple codes of increasing size,
			// assigns them to different accounts, and then verifies
			// that the codes can be retrieved correctly along with
			// their hashes.
			// The codes are initialised with the input size, and
			// then each subsequent code is created by appending
			// a byte to the previous code, resulting in codes
			// of increasing size.
			testCodes := func(size int) {
				code := make([]byte, size)
				for i := 0; i < len(code); i++ {
					code[i] = byte(i)
				}

				const numOfCodes = 10
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

			testCodes(128)
			testCodes(64)   // smaller sets overrides larger ones
			testCodes(4096) // larger sets overrides smaller ones
		})
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

	stateFactories, err := initTestedStates()
	require.NoError(t, err, "failed to initialize tested states")

	for _, factory := range stateFactories {
		t.Run(fmt.Sprintf("state_%d_%s", factory.config.Schema, factory.config.Variant), func(t *testing.T) {
			t.Parallel()
			state, err := factory.newState(t)
			require.NoError(t, err, "failed to initialize tested state")

			defer func() {
				require.NoError(t, state.Close(), "failed to close state")
			}()

			require.NoError(t, state.Apply(0, update))

			hash, err := state.GetHash()
			require.NoError(t, err)
			require.NotEmpty(t, hash, "hash should not be empty")
		})
	}
}

// initTestedStates initializes one reference state factory for scheme 6
// and all other state factories with the same schema as this reference states.
func initTestedStates() ([]*testedStateFactory, error) {
	var refStateFactory state.StateFactory
	var refConfig state.Configuration

	for config, factory := range state.GetAllRegisteredStateFactories() {
		if config.Schema == 6 && config.Variant == "go-geth-memory" {
			refStateFactory = factory
			refConfig = config
		}
	}

	if refStateFactory == nil {
		return nil, fmt.Errorf("reference state not found among registered states")
	}

	var testedStates []*testedStateFactory
	for config, factory := range state.GetAllRegisteredStateFactories() {
		if config.Schema == refConfig.Schema && refConfig != config {
			st := &testedStateFactory{
				refStateFactory:    refStateFactory,
				testedStateFactory: factory,
				config:             config,
			}
			testedStates = append(testedStates, st)
		}
	}

	return testedStates, nil
}

// testedStateFactory is a helper struct that holds the reference state
// factory, configuration, and state factory for the tested state.
type testedStateFactory struct {
	config             state.Configuration // configuration of the tested state
	refStateFactory    state.StateFactory  // state factory for the reference state
	testedStateFactory state.StateFactory  // state factory for the tested state
}

// newState creates a new comparingState instance that wraps both
// the reference state and the tested state. It initializes the tested
// state and the reference state using the provided factories.
// If initialization fails, an error is returned.
func (s *testedStateFactory) newState(t *testing.T) (*comparingState, error) {
	testedState, err := s.testedStateFactory(state.Parameters{
		Directory: t.TempDir(),
		Variant:   s.config.Variant,
		Schema:    s.config.Schema,
		Archive:   s.config.Archive,
	})

	if err != nil {
		return nil, fmt.Errorf("failed to initialize tested state: %w", err)
	}

	refState, err := s.refStateFactory(state.Parameters{})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize reference state: %w", err)
	}

	return &comparingState{
		refState:    refState,
		testedState: testedState,
		config:      s.config,
		t:           t,
	}, nil
}

// comparingState is a state.State implementation that wraps
// two underlying states: a reference state and a tested state.
// It compares the results of operations on both states and
// reports discrepancies using the testing.T instance.
type comparingState struct {
	refState    state.State // reference state for comparison
	config      state.Configuration
	testedState state.State // tested state
	t           *testing.T
}

func (s *comparingState) Exists(address common.Address) (bool, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(state state.State) (bool, error) {
		return state.Exists(address)
	})
}

func (s *comparingState) GetNonce(address common.Address) (common.Nonce, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(state state.State) (common.Nonce, error) {
		return state.GetNonce(address)
	})
}

func (s *comparingState) GetStorage(address common.Address, key common.Key) (common.Value, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(state state.State) (common.Value, error) {
		return state.GetStorage(address, key)
	})
}

func (s *comparingState) GetCode(address common.Address) ([]byte, error) {
	s.t.Helper()
	b, err := getCmpStateValue(s, func(state state.State) (immutable.Bytes, error) {
		b, err := state.GetCode(address)
		return immutable.NewBytes(b), err
	})
	return b.ToBytes(), err
}

func (s *comparingState) GetCodeSize(address common.Address) (int, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(state state.State) (int, error) {
		return state.GetCodeSize(address)
	})
}

func (s *comparingState) GetCodeHash(address common.Address) (common.Hash, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(state state.State) (common.Hash, error) {
		return state.GetCodeHash(address)
	})
}

func (s *comparingState) GetHash() (common.Hash, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(state state.State) (common.Hash, error) {
		return state.GetHash()
	})
}

func (s *comparingState) GetBalance(address common.Address) (amount.Amount, error) {
	s.t.Helper()
	return getCmpStateValue(s, func(state state.State) (amount.Amount, error) {
		return state.GetBalance(address)
	})
}

func (s *comparingState) Apply(block uint64, update common.Update) error {
	s.t.Helper()
	return s.action(func(state state.State) error {
		return state.Apply(block, update)
	})
}

//
//	I/O features
//

func (s *comparingState) Flush() error {
	s.t.Helper()
	return s.action(func(state state.State) error {
		return state.Flush()
	})
}

func (s *comparingState) Close() error {
	s.t.Helper()
	return s.action(func(state state.State) error {
		return state.Close()
	})
}

// action is a helper function that iterates over all states in the comparingState
// and applies the provided function to each state. It collects errors
// and returns a single error that combines all errors encountered during
// the execution of the provided function on each state.
func (s *comparingState) action(do func(state state.State) error) error {
	s.t.Helper()

	if err := do(s.refState); err != nil {
		return err
	}

	if err := do(s.testedState); err != nil {
		return err
	}

	return nil
}

// getCmpStateValue is a helper function that iterates over all states in the comparingState
// and applies the provided function to each state. It collects errors and checks
// for value consistency across all states. If all states return the same value,
// it returns that value; otherwise, it returns an error indicating the mismatch.
func getCmpStateValue[T comparable](s *comparingState, do func(state state.State) (T, error)) (T, error) {
	s.t.Helper()
	var errs []error

	ref, err := do(s.refState)
	if err != nil {
		var empty T
		return empty, err
	}

	val, err := do(s.testedState)
	if err != nil {
		var empty T
		return empty, err
	}

	if ref != val {
		s.t.Errorf("value mismatch for %s: got %v, wanted:  %v", s.config.Variant, val, ref)
	}

	return ref, errors.Join(errs...)
}
