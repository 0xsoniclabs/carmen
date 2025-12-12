// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package geth

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/state"
	ethcommon "github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/stretchr/testify/require"
)

// initTestedState initializes the state for testing.
func initTestedState() map[string]func(param state.Parameters, t *testing.T) state.State {
	addClosing := func(st state.State, t *testing.T) {
		t.Cleanup(func() {
			require.NoError(t, st.Close(), "failed to close state")
		})
	}

	return map[string]func(param state.Parameters, t *testing.T) state.State{
		"memory": func(param state.Parameters, t *testing.T) state.State {
			st, err := NewState(param)
			require.NoError(t, err, "failed to create state")
			addClosing(st, t)
			return st
		},
		"memory source": func(param state.Parameters, t *testing.T) state.State {
			st, err := NewStateWithSource(param, newMemorySource())
			require.NoError(t, err, "failed to create memory state")
			addClosing(st, t)
			return st
		},
	}
}

func TestState_CreateAccounts_Many_Updates_Success(t *testing.T) {
	for name, stateInit := range initTestedState() {
		t.Run(name, func(t *testing.T) {
			state := stateInit(state.Parameters{}, t)
			const numBlocks = 10
			const numInsertsPerBlock = 100
			for i := 0; i < numBlocks; i++ {
				update := common.Update{}
				for j := 0; j < numInsertsPerBlock; j++ {
					addr := common.Address{byte(j), byte(i), byte(i >> 8)}
					update.CreatedAccounts = append(update.CreatedAccounts, addr)
					update.Nonces = append(update.Nonces, common.NonceUpdate{Account: addr, Nonce: common.ToNonce(1)})
					update.Balances = append(update.Balances, common.BalanceUpdate{Account: addr, Balance: amount.New(uint64(i * j))})
				}
				require.NoError(t, state.Apply(uint64(i), update), "failed to apply block %d", i)
			}
			for i := 0; i < numBlocks; i++ {
				for j := 0; j < numInsertsPerBlock; j++ {
					addr := common.Address{byte(j), byte(i), byte(i >> 8)}
					exists, err := state.Exists(addr)
					require.NoError(t, err, "failed to check existence of account %x", addr)
					require.True(t, exists, "account %x should exist but does not", addr)
					balance, err := state.GetBalance(addr)
					require.NoError(t, err, "failed to get balance for account %x", addr)
					expectedBalance := amount.New(uint64(i * j))
					require.Equal(t, 0, balance.ToBig().Cmp(expectedBalance.ToBig()), "unexpected balance for account %x: got %s, want %s", addr, balance, expectedBalance)
					nonce, err := state.GetNonce(addr)
					require.NoError(t, err, "failed to get nonce for account %x", addr)
					require.Equal(t, common.ToNonce(1), nonce, "unexpected nonce for account %x", addr)
				}
			}
		})
	}
}

func TestState_Update_And_Get_Code_Success(t *testing.T) {
	for name, stateInit := range initTestedState() {
		t.Run(name, func(t *testing.T) {
			state := stateInit(state.Parameters{}, t)
			code := make([]byte, 4096)
			for i := 0; i < len(code); i++ {
				code[i] = byte(i)
			}
			const numOfCodes = 100
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
		})
	}
}

func TestState_Set_And_Get_Storage_Success(t *testing.T) {
	for name, stateInit := range initTestedState() {
		t.Run(name, func(t *testing.T) {
			state := stateInit(state.Parameters{}, t)
			update := common.Update{}
			const numOfKeys = 100
			const numOfAddresses = 100
			for i := 0; i < numOfAddresses; i++ {
				addr := common.Address{byte(i), byte(i >> 8)}
				for j := 0; j < numOfKeys; j++ {
					key := common.Key{byte(j), byte(j >> 8)}
					value := common.Value{byte(i + j), byte((i + j) >> 8)}
					update.Slots = append(update.Slots, common.SlotUpdate{
						Account: addr,
						Key:     key,
						Value:   value,
					})
				}
			}
			require.NoError(t, state.Apply(0, update), "failed to apply")
			for i := 0; i < numOfAddresses; i++ {
				addr := common.Address{byte(i), byte(i >> 8)}
				for j := 0; j < numOfKeys; j++ {
					key := common.Key{byte(j), byte(j >> 8)}
					value, err := state.GetStorage(addr, key)
					require.NoError(t, err, "failed to get storage for account %x", addr)
					require.Equal(t, common.Value{byte(i + j), byte((i + j) >> 8)}, value, "unexpected storage value for account %x, key %x", addr, key)
				}
			}
		})
	}
}

func TestState_Set_And_Get_Storage_Success_Padded_Right(t *testing.T) {
	for name, stateInit := range initTestedState() {
		t.Run(name, func(t *testing.T) {
			state := stateInit(state.Parameters{}, t)
			update := common.Update{}
			addr := common.Address{0, 0, 1}
			key := common.Key{0, 0, 2}
			value := common.Value{0, 0, 3}
			update.Slots = append(update.Slots, common.SlotUpdate{
				Account: addr,
				Key:     key,
				Value:   value,
			})
			require.NoError(t, state.Apply(0, update), "failed to apply")
			gotValue, err := state.GetStorage(addr, key)
			require.NoError(t, err, "failed to get storage for account %x", addr)
			require.Equal(t, value, gotValue, "unexpected storage value for account %x, key %x", addr, key)
		})
	}
}

func TestState_GetBalance_InitialEmptyTrie_AllAccountPropertiesAreZero(t *testing.T) {
	for name, stateInit := range initTestedState() {
		t.Run(name, func(t *testing.T) {
			state := stateInit(state.Parameters{}, t)
			balance, err := state.GetBalance(common.Address{})
			require.NoError(t, err, "failed to get balance for empty account")
			require.True(t, balance.IsZero(), "expected zero balance for empty account, got %s", balance)
			nonce, err := state.GetNonce(common.Address{})
			require.NoError(t, err, "failed to get nonce for empty account")
			require.Equal(t, common.ToNonce(0), nonce, "expected nonce 0 for empty account")
			value, err := state.GetCodeHash(common.Address{})
			require.NoError(t, err, "failed to get code hash for empty account")
			require.Equal(t, common.Keccak256([]byte{}), value, "expected empty code hash for empty account")
			code, err := state.GetCode(common.Address{})
			require.NoError(t, err, "failed to get code for empty account")
			require.Equal(t, 0, len(code), "expected empty code for empty account, got %d bytes", len(code))
		})
	}
}

func TestState_GetStorage_Empty_Returns_Empty_Value(t *testing.T) {
	for name, stateInit := range initTestedState() {
		t.Run(name, func(t *testing.T) {
			state := stateInit(state.Parameters{}, t)
			value, err := state.GetStorage(common.Address{}, common.Key{})
			require.NoError(t, err, "failed to get storage for empty account")
			require.Equal(t, common.Value{}, value, "unexpected storage value for empty account")
		})
	}
}

func TestState_GetHash_Is_Updated_Each_Block(t *testing.T) {
	for name, stateInit := range initTestedState() {
		t.Run(name, func(t *testing.T) {
			state := stateInit(state.Parameters{}, t)
			var prevHash common.Hash
			const numBlocks = 10
			const numInsertsPerBlock = 100
			for i := 0; i < numBlocks; i++ {
				update := common.Update{}
				for j := 0; j < numInsertsPerBlock; j++ {
					addr := common.Address{byte(j), byte(i), byte(i >> 8)}
					update.CreatedAccounts = append(update.CreatedAccounts, addr)
					update.Nonces = append(update.Nonces, common.NonceUpdate{Account: addr, Nonce: common.ToNonce(1)})
					update.Balances = append(update.Balances, common.BalanceUpdate{Account: addr, Balance: amount.New(uint64(i * j))})
				}
				require.NoError(t, state.Apply(uint64(i), update), "failed to apply block %d", i)
				hash, err := state.GetHash()
				require.NoError(t, err, "failed to get hash for block %d", i)
				require.NotEqual(t, prevHash, hash, "hash did not change")
				prevHash = hash
			}
		})
	}
}

func TestState_GetArchiveState_ReturnsNoArchiveError(t *testing.T) {
	for name, stateInit := range initTestedState() {
		t.Run(name, func(t *testing.T) {
			st := stateInit(state.Parameters{}, t)
			_, err := st.GetArchiveState(0)
			require.ErrorIs(t, err, state.NoArchiveError, "expected noarchive error")
		})
	}
}

func TestState_GetArchiveBlockHeight_ReturnsNoArchiveError(t *testing.T) {
	for name, stateInit := range initTestedState() {
		t.Run(name, func(t *testing.T) {
			st := stateInit(state.Parameters{}, t)
			_, _, err := st.GetArchiveBlockHeight()
			require.ErrorIs(t, err, state.NoArchiveError, "expected noarchive error")
		})
	}
}

func TestState_DeletedAccount_Unsupported(t *testing.T) {
	for name, stateInit := range initTestedState() {
		t.Run(name, func(t *testing.T) {
			st := stateInit(state.Parameters{}, t)
			update := common.Update{}
			update.DeletedAccounts = append(update.DeletedAccounts, common.Address{})
			err := st.Apply(0, update)
			require.Error(t, err, "expected error when applying update with deleted accounts")
		})
	}
}

func TestState_HasEmptyStorage_Unsupported(t *testing.T) {
	for name, stateInit := range initTestedState() {
		t.Run(name, func(t *testing.T) {
			st := stateInit(state.Parameters{}, t)
			empty, err := st.HasEmptyStorage(common.Address{1})
			require.NoError(t, err)
			require.True(t, empty)
		})
	}
}

func TestState_Export_Unsupported(t *testing.T) {
	for name, stateInit := range initTestedState() {
		t.Run(name, func(t *testing.T) {
			st := stateInit(state.Parameters{}, t)
			_, err := st.Export(nil, nil)
			require.ErrorContains(t, err, "not supported", "expected error containing 'not supported'")
		})
	}
}

func TestState_Account_CodeHash_Initialised_With_Eth_Empty_Hash(t *testing.T) {
	require := require.New(t)

	state, err := NewState(state.Parameters{})
	require.NoError(err, "failed to create state")
	defer func() {
		require.NoError(state.Close(), "failed to close state")
	}()

	addr1 := common.Address{1}

	update := common.Update{
		CreatedAccounts: []common.Address{addr1}, // we expect the account must be explicitly created
		Balances: []common.BalanceUpdate{
			{Account: addr1, Balance: amount.New(1)},
		},
	}

	require.NoError(state.Apply(0, update))

	codeHash, err := state.GetCodeHash(addr1)
	require.NoError(err)
	require.Equal(common.Hash(types.EmptyCodeHash), codeHash)
}

func TestState_Account_CodeHash_NotEmptied_When_Recreated(t *testing.T) {
	require := require.New(t)

	state, err := NewState(state.Parameters{})
	require.NoError(err, "failed to create state")
	defer func() {
		require.NoError(state.Close(), "failed to close state")
	}()

	addr1 := common.Address{1}

	update := common.Update{
		Codes: []common.CodeUpdate{{Account: addr1, Code: []byte{1, 2, 3}}},
		Balances: []common.BalanceUpdate{
			{Account: addr1, Balance: amount.New(1)},
		},
	}

	require.NoError(state.Apply(0, update))

	codeHash, err := state.GetCodeHash(addr1)
	require.NoError(err)
	require.NotEqual(common.Hash(types.EmptyCodeHash), codeHash)

	// Recreate the account, which should not empty the code hash
	update2 := common.Update{
		CreatedAccounts: []common.Address{addr1},
	}

	require.NoError(state.Apply(0, update2))

	codeHash, err = state.GetCodeHash(addr1)
	require.NoError(err)
	require.NotEqual(common.Hash(types.EmptyCodeHash), codeHash)
}

func TestState_Account_Balance_NotEmptied_When_Recreated(t *testing.T) {
	require := require.New(t)

	state, err := NewState(state.Parameters{})
	require.NoError(err, "failed to create state")
	defer func() {
		require.NoError(state.Close(), "failed to close state")
	}()

	addr1 := common.Address{1}
	update := common.Update{
		CreatedAccounts: []common.Address{addr1}, // we expect the account must be explicitly created
		Balances: []common.BalanceUpdate{
			{Account: addr1, Balance: amount.New(1)},
		},
	}

	require.NoError(state.Apply(0, update))

	balance, err := state.GetBalance(addr1)
	require.NoError(err)
	require.Equal(amount.New(1), balance)

	// Recreate the account, which should not empty the code hash
	update2 := common.Update{
		CreatedAccounts: []common.Address{addr1},
	}

	require.NoError(state.Apply(0, update2))

	// The balance should remain the same
	balance, err = state.GetBalance(addr1)
	require.NoError(err)
	require.Equal(amount.New(1), balance)
}

func TestState_Account_Nonce_NotEmptied_When_Recreated(t *testing.T) {
	require := require.New(t)

	state, err := NewState(state.Parameters{})
	require.NoError(err, "failed to create state")
	defer func() {
		require.NoError(state.Close(), "failed to close state")
	}()

	addr1 := common.Address{1}

	update := common.Update{
		CreatedAccounts: []common.Address{addr1}, // we expect the account must be explicitly created
		Nonces: []common.NonceUpdate{
			{Account: addr1, Nonce: common.ToNonce(1)},
		},
	}

	require.NoError(state.Apply(0, update))

	nonce, err := state.GetNonce(addr1)
	require.NoError(err)
	require.Equal(common.ToNonce(1), nonce)

	// Recreate the account, which should not empty the nonce
	update2 := common.Update{
		CreatedAccounts: []common.Address{addr1},
	}

	require.NoError(state.Apply(0, update2))

	// The nonce should remain the same
	nonce, err = state.GetNonce(addr1)
	require.NoError(err)
	require.Equal(common.ToNonce(1), nonce)
}

func TestState_Error_from_Apply(t *testing.T) {
	injectedErr := fmt.Errorf("injected error")
	source := &errorInjectingNodeSource{parentSource: newMemorySource(), injectError: injectedErr, threshold: 1000}
	st, err := NewStateWithSource(state.Parameters{}, source)
	require.NoError(t, err, "failed to create state")

	update := common.Update{
		CreatedAccounts: []common.Address{{1}},
		Nonces: []common.NonceUpdate{
			{Account: common.Address{1}, Nonce: common.ToNonce(1)},
		},
	}

	err = st.Apply(0, update)
	require.NoError(t, err, "failed to apply update")

	for i := 0; i < source.count; i++ {
		source := &errorInjectingNodeSource{parentSource: newMemorySource(), injectError: injectedErr, threshold: i}
		st, err := NewStateWithSource(state.Parameters{}, source)
		require.NoError(t, err, "failed to create state")

		err = st.Apply(0, update)
		require.ErrorIs(t, err, injectedErr, "expected injected error at count %d", i)
	}
}

// errorInjectingNodeSource is a NodeSource that injects an error after a certain number of calls.
// It is used to test error handling in the state.
// It wraps another NodeSource and delegates calls to it until the threshold is reached.
// After the threshold is reached, it returns the injected error for all subsequent calls.
type errorInjectingNodeSource struct {
	parentSource NodeSource
	injectError  error
	threshold    int
	count        int
}

func (s *errorInjectingNodeSource) Node(owner ethcommon.Hash, path []byte, hash ethcommon.Hash) ([]byte, error) {
	if s.count >= s.threshold {
		return nil, s.injectError
	}

	s.count++
	return s.parentSource.Node(owner, path, hash)
}

func (s *errorInjectingNodeSource) set(path []byte, value []byte) error {
	if s.count >= s.threshold {
		return s.injectError
	}
	s.count++
	return s.parentSource.set(path, value)
}

func (s *errorInjectingNodeSource) Flush() error {
	if s.count >= s.threshold {
		return s.injectError
	}
	s.count++
	return s.parentSource.Flush()
}

func (s *errorInjectingNodeSource) Close() error {
	if s.count >= s.threshold {
		return s.injectError
	}
	s.count++
	return s.parentSource.Close()
}
