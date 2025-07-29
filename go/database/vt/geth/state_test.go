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
	"testing"

	"github.com/0xsoniclabs/carmen/go/backend"
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/stretchr/testify/require"
)

func TestState_CreateAccounts_Many_Updates_Success(t *testing.T) {
	state, err := NewState(state.Parameters{})
	require.NoError(t, err, "failed to create state")
	defer func() {
		require.NoError(t, state.Close(), "failed to close state")
	}()

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
}

func TestState_Update_And_Get_Code_Success(t *testing.T) {
	state, err := NewState(state.Parameters{})
	require.NoError(t, err, "failed to create state")
	defer func() {
		require.NoError(t, state.Close(), "failed to close state")
	}()

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
}

func TestState_Set_And_Get_Storage_Success(t *testing.T) {
	state, err := NewState(state.Parameters{})
	require.NoError(t, err, "failed to create state")
	defer func() {
		require.NoError(t, state.Close(), "failed to close state")
	}()

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
}

func TestState_GetBalance_InitialEmptyTrie_AllAccountPropertiesAreZero(t *testing.T) {
	state, err := NewState(state.Parameters{})
	require.NoError(t, err, "failed to create state")
	defer func() {
		require.NoError(t, state.Close(), "failed to close state")
	}()

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
}

func TestState_GetStorage_Empty_Returns_Empty_Value(t *testing.T) {
	state, err := NewState(state.Parameters{})
	require.NoError(t, err, "failed to create state")
	defer func() {
		require.NoError(t, state.Close(), "failed to close state")
	}()

	value, err := state.GetStorage(common.Address{}, common.Key{})
	require.NoError(t, err, "failed to get storage for empty account")
	require.Equal(t, common.Value{}, value, "unexpected storage value for empty account")
}

func TestState_GetHash_Is_Updated_Each_Block(t *testing.T) {
	state, err := NewState(state.Parameters{})
	require.NoError(t, err, "failed to create vt state")
	defer func() {
		require.NoError(t, state.Close(), "failed to close state")
	}()

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
}

func TestState_GetArchiveState_ReturnsNoArchiveError(t *testing.T) {
	st, err := NewState(state.Parameters{})
	require.NoError(t, err, "failed to create state")
	defer func() {
		require.NoError(t, st.Close(), "failed to close state")
	}()

	_, err = st.GetArchiveState(0)
	require.ErrorIs(t, err, state.NoArchiveError, "expected noarchive error")
}

func TestState_GetArchiveBlockHeight_ReturnsNoArchiveError(t *testing.T) {
	st, err := NewState(state.Parameters{})
	require.NoError(t, err, "failed to create state")
	defer func() {
		require.NoError(t, st.Close(), "failed to close state")
	}()

	_, _, err = st.GetArchiveBlockHeight()
	require.ErrorIs(t, err, state.NoArchiveError, "expected noarchive error")
}

func TestState_DeletedAccount_Unsupported(t *testing.T) {
	st, err := NewState(state.Parameters{})
	require.NoError(t, err, "failed to create vt state")
	defer func() {
		require.NoError(t, st.Close(), "failed to close state")
	}()

	update := common.Update{}
	update.DeletedAccounts = append(update.DeletedAccounts, common.Address{})

	err = st.Apply(0, update)
	require.Error(t, err, "expected error when applying update with deleted accounts")
}

func TestState_HasEmptyStorage_Unsupported(t *testing.T) {
	st, err := NewState(state.Parameters{})
	require.NoError(t, err, "failed to create state")
	defer func() {
		require.NoError(t, st.Close(), "failed to close state")
	}()

	_, err = st.HasEmptyStorage(common.Address{})
	require.ErrorContains(t, err, "not supported", "expected error containing 'not supported'")
}

func TestState_Snapshot_Unsupported(t *testing.T) {
	st, err := NewState(state.Parameters{})
	require.NoError(t, err, "failed to create state")
	defer func() {
		require.NoError(t, st.Close(), "failed to close state")
	}()

	_, err = st.GetProof()
	require.ErrorIs(t, err, backend.ErrSnapshotNotSupported, "expected snapshot not supported error")

	_, err = st.CreateSnapshot()
	require.ErrorIs(t, err, backend.ErrSnapshotNotSupported, "expected snapshot not supported error")

	err = st.Restore(nil)
	require.ErrorIs(t, err, backend.ErrSnapshotNotSupported, "expected snapshot not supported error")

	_, err = st.GetSnapshotVerifier(nil)
	require.ErrorIs(t, err, backend.ErrSnapshotNotSupported, "expected snapshot not supported error")
}

func TestState_Export_Unsupported(t *testing.T) {
	st, err := NewState(state.Parameters{})
	require.NoError(t, err, "failed to create state")
	defer func() {
		require.NoError(t, st.Close(), "failed to close state")
	}()

	_, err = st.Export(nil, nil)
	require.ErrorContains(t, err, "not supported", "expected error containing 'not supported'")
}
