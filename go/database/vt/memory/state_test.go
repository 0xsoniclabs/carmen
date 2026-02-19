// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package memory

import (
	"bytes"
	"crypto/rand"
	"testing"

	"github.com/0xsoniclabs/carmen/go/backend"
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/state"
	geth_common "github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	geth_trie "github.com/ethereum/go-ethereum/trie"
	"github.com/ethereum/go-ethereum/trie/utils"
	"github.com/ethereum/go-ethereum/triedb/database"
	"github.com/stretchr/testify/require"
)

func TestState_ImplementsState(t *testing.T) {
	var _ state.State = &State{}

	inst, _ := NewState(state.Parameters{})
	var _ state.State = inst
}

func TestState_NewState_CreatesEmptyState(t *testing.T) {
	require := require.New(t)
	state := newState()
	require.NotNil(state)
	require.Zero(state.GetHash())
}

func TestState_Exists(t *testing.T) {
	state := newState()
	exists, err := state.Exists(common.Address{1})
	require.NoError(t, err)
	require.False(t, exists, "Expected Exists to return false for non-existing address")
}

func TestState_CanStoreAndRestoreNonces(t *testing.T) {
	require := require.New(t)

	state := newState()

	address := common.Address{1}

	// Initially, the nonce should be zero
	nonce, err := state.GetNonce(address)
	require.NoError(err)
	require.Equal(common.ToNonce(0), nonce)

	// Set a nonce
	_, err = state.Apply(0, common.Update{
		Nonces: []common.NonceUpdate{{
			Account: address,
			Nonce:   common.ToNonce(42),
		}},
	})
	require.NoError(err)

	// Retrieve the nonce again
	nonce, err = state.GetNonce(address)
	require.NoError(err)
	require.Equal(common.ToNonce(42), nonce)

	// Set another nonce
	_, err = state.Apply(0, common.Update{
		Nonces: []common.NonceUpdate{{
			Account: address,
			Nonce:   common.ToNonce(123),
		}},
	})
	require.NoError(err)

	// Retrieve the updated nonce
	nonce, err = state.GetNonce(address)
	require.NoError(err)
	require.Equal(common.ToNonce(123), nonce)
}

func TestState_CanStoreAndRestoreBalances(t *testing.T) {
	require := require.New(t)

	state := newState()

	address := common.Address{1}

	// Initially, the balance should be zero
	balance, err := state.GetBalance(address)
	require.NoError(err)
	require.Equal(amount.New(0), balance)

	// Set a balance
	_, err = state.Apply(0, common.Update{
		Balances: []common.BalanceUpdate{{
			Account: address,
			Balance: amount.New(42),
		}},
	})
	require.NoError(err)

	// Retrieve the balance again
	balance, err = state.GetBalance(address)
	require.NoError(err)
	require.Equal(amount.New(42), balance)

	// Set another balance
	_, err = state.Apply(0, common.Update{
		Balances: []common.BalanceUpdate{{
			Account: address,
			Balance: amount.New(123),
		}},
	})
	require.NoError(err)

	// Retrieve the updated balance
	balance, err = state.GetBalance(address)
	require.NoError(err)
	require.Equal(amount.New(123), balance)
}

func TestState_CanStoreAndRestoreCodes(t *testing.T) {
	require := require.New(t)

	state := newState()

	address := common.Address{1}

	length, err := state.GetCodeSize(address)
	require.NoError(err)
	require.Equal(0, length)

	tests := map[string][]byte{
		"empty": {},
		"short": {1, 2, 3},
		"long":  {10_000: 1},
	}

	for name, code := range tests {
		t.Run(name, func(t *testing.T) {
			// Set a code.
			_, err := state.Apply(0, common.Update{
				Codes: []common.CodeUpdate{{
					Account: address,
					Code:    bytes.Clone(code),
				}},
			})
			require.NoError(err)

			// Retrieve the code size.
			length, err = state.GetCodeSize(address)
			require.NoError(err)
			require.Equal(len(code), length)

			// Retrieve the code hash.
			hash, err := state.GetCodeHash(address)
			require.NoError(err)
			require.Equal(common.Keccak256(code), hash)

			// Retrieve the code.
			restored, err := state.GetCode(address)
			require.NoError(err)
			require.Equal(code, restored)
		})
	}
}

func TestState_HasEmptyStorage_ReturnsError(t *testing.T) {
	state := newState()
	_, err := state.HasEmptyStorage(common.Address{1})
	require.ErrorContains(t, err, "not supported by Verkle Tries")
}

func TestState_CanStoreAndRestoreCodesOfArbitraryLength(t *testing.T) {
	require := require.New(t)
	state := newState()

	random := make([]byte, 1000)
	rand.Read(random)

	address := common.Address{1, 2, 3}
	for i := range len(random) {
		code := random[:i]

		// Set a code.
		_, err := state.Apply(0, common.Update{
			Codes: []common.CodeUpdate{{
				Account: address,
				Code:    bytes.Clone(code),
			}},
		})
		require.NoError(err)

		// Retrieve the code size.
		length, err := state.GetCodeSize(address)
		require.NoError(err)
		require.Equal(len(code), length)

		// Retrieve the code hash.
		hash, err := state.GetCodeHash(address)
		require.NoError(err)
		require.Equal(common.Keccak256(code), hash)

		// Retrieve the code.
		restored, err := state.GetCode(address)
		require.NoError(err)
		require.Equal(code, restored)
	}
}

func TestState_CanStoreAndRestoreStorageSlots(t *testing.T) {
	require := require.New(t)

	state := newState()

	address := common.Address{1}
	key := common.Key{2}

	// Initially, the balance should be zero
	value, err := state.GetStorage(address, key)
	require.NoError(err)
	require.Equal(common.Value{}, value)

	// Set a value
	_, err = state.Apply(0, common.Update{
		Slots: []common.SlotUpdate{{
			Account: address,
			Key:     key,
			Value:   common.Value{1, 2, 3},
		}},
	})
	require.NoError(err)

	// Retrieve the value again
	value, err = state.GetStorage(address, key)
	require.NoError(err)
	require.Equal(common.Value{1, 2, 3}, value)

	// Set another value
	_, err = state.Apply(0, common.Update{
		Slots: []common.SlotUpdate{{
			Account: address,
			Key:     key,
			Value:   common.Value{3, 2, 1},
		}},
	})
	require.NoError(err)

	// Retrieve the updated value
	value, err = state.GetStorage(address, key)
	require.NoError(err)
	require.Equal(common.Value{3, 2, 1}, value)
}

func TestState_EmptyStateHasZeroCommitment(t *testing.T) {
	require := require.New(t)

	state := newState()
	hash, err := state.GetHash()
	require.NoError(err)
	require.Equal(common.Hash(types.EmptyVerkleHash), hash)
}

func TestState_Check_ReturnsNoError(t *testing.T) {
	require.NoError(t, newState().Check())
}

func TestState_Flush_ReturnsNoError(t *testing.T) {
	require.NoError(t, newState().Flush())
}

func TestState_Close_ReturnsNoError(t *testing.T) {
	require.NoError(t, newState().Close())
}

func TestState_GetMemoryFootprint(t *testing.T) {
	require := require.New(t)
	state := newState()
	require.NotNil(state.GetMemoryFootprint())
}

func TestState_GetArchiveState_ReturnsNoArchiveError(t *testing.T) {
	_, err := newState().GetArchiveState(0)
	require.ErrorIs(t, err, state.NoArchiveError)
}

func TestState_GetArchiveBlockHeight_ReturnsNoArchiveError(t *testing.T) {
	_, _, err := newState().GetArchiveBlockHeight()
	require.ErrorIs(t, err, state.NoArchiveError)
}

func TestState_CreateWitnessProof_ReturnsNotSupportedError(t *testing.T) {
	_, err := newState().CreateWitnessProof(common.Address{1}, common.Key{2})
	require.ErrorContains(t, err, "witness proof not supported yet")
}

func TestState_Export_PanicsAsNotImplemented(t *testing.T) {
	require := require.New(t)
	state := newState()
	require.Panics(
		func() { state.Export(nil, nil) },
		"Export should panic as it is not implemented",
	)
}

func TestState_GetProof_ReturnsNotSupportedError(t *testing.T) {
	_, err := newState().GetProof()
	require.ErrorIs(t, err, backend.ErrSnapshotNotSupported)
}

func TestState_CreateSnapshot_ReturnsNotSupportedError(t *testing.T) {
	_, err := newState().CreateSnapshot()
	require.ErrorIs(t, err, backend.ErrSnapshotNotSupported)
}

func TestState_Restore_ReturnsNotSupportedError(t *testing.T) {
	var data backend.SnapshotData
	err := newState().Restore(data)
	require.ErrorIs(t, err, backend.ErrSnapshotNotSupported)
}

func TestState_GetSnapshotVerifier_ReturnsNotSupportedError(t *testing.T) {
	_, err := newState().GetSnapshotVerifier(nil)
	require.ErrorIs(t, err, backend.ErrSnapshotNotSupported)
}

// --- Tests comparing with Geth reference implementation ---

func TestState_StateWithContentHasExpectedCommitment(t *testing.T) {
	const PUSH32 = 0x7f
	require := require.New(t)

	addr1 := common.Address{1}
	addr2 := common.Address{2}
	addr3 := common.Address{3}

	update := common.Update{
		Balances: []common.BalanceUpdate{
			{Account: addr1, Balance: amount.New(100)},
			{Account: addr2, Balance: amount.New(200)},
			{Account: addr3, Balance: amount.New(300)},
		},
		Nonces: []common.NonceUpdate{
			{Account: addr1, Nonce: common.ToNonce(1)},
			{Account: addr2, Nonce: common.ToNonce(2)},
			{Account: addr3, Nonce: common.ToNonce(3)},
		},
		Codes: []common.CodeUpdate{
			{Account: addr1, Code: []byte{0x01, 0x02}},
			{Account: addr2, Code: []byte{0x03, 30: PUSH32, 31: 0x05}},           // truncated push data
			{Account: addr3, Code: []byte{0x06, 0x07, 0x08, 3 * 256 * 32: 0x09}}, // fills multiple leafs
		},
		Slots: []common.SlotUpdate{
			{Account: addr1, Key: common.Key{0x01}, Value: common.Value{0x05}},
			{Account: addr2, Key: common.Key{0x02}, Value: common.Value{0x06}},
		},
	}

	state := newState()
	state.Apply(0, update)

	hash, err := state.GetHash()
	require.NoError(err)

	reference, err := newRefState()
	require.NoError(err)
	reference.Apply(0, update)
	want, err := reference.GetHash()
	require.NoError(err)

	require.Equal(want, hash)
}

func TestState_IncrementalStateUpdatesResultInSameCommitments(t *testing.T) {
	const PUSH32 = 0x7f
	require := require.New(t)

	addr1 := common.Address{1}
	addr2 := common.Address{2}
	addr3 := common.Address{3}

	updates := []common.Update{
		// -- create data --
		{
			Balances: []common.BalanceUpdate{
				{Account: addr1, Balance: amount.New(100)},
				{Account: addr2, Balance: amount.New(200)},
			},
			Nonces: []common.NonceUpdate{
				{Account: addr1, Nonce: common.ToNonce(1)},
				{Account: addr2, Nonce: common.ToNonce(2)},
			},
			Codes: []common.CodeUpdate{
				{Account: addr1, Code: []byte{0x01, 0x02}},
				{Account: addr2, Code: []byte{0x03, 0x04, PUSH32, 32: 0x05}},
			},
			Slots: []common.SlotUpdate{
				{Account: addr1, Key: common.Key{0x01}, Value: common.Value{0x05}},
				{Account: addr2, Key: common.Key{0x02}, Value: common.Value{0x06}},
			},
		},
		// -- update data --
		{
			Balances: []common.BalanceUpdate{
				{Account: addr1, Balance: amount.New(150)},
				{Account: addr2, Balance: amount.New(250)},
				{Account: addr3, Balance: amount.New(350)},
			},
			Nonces: []common.NonceUpdate{
				{Account: addr1, Nonce: common.ToNonce(3)},
				{Account: addr2, Nonce: common.ToNonce(4)},
				{Account: addr3, Nonce: common.ToNonce(5)},
			},
			Codes: []common.CodeUpdate{
				{Account: addr1, Code: []byte{0x11, 0x12}},
				{Account: addr2, Code: []byte{0x13, 0x14, PUSH32, 32: 0x15}},
				{Account: addr3, Code: []byte{0x16, 0x17}},
			},
		},
		// -- set data to zero --
		{
			Balances: []common.BalanceUpdate{
				{Account: addr1, Balance: amount.New(0)},
			},
			Nonces: []common.NonceUpdate{
				{Account: addr1, Nonce: common.ToNonce(0)},
			},
			Codes: []common.CodeUpdate{
				{Account: addr1, Code: nil},
			},
			Slots: []common.SlotUpdate{
				{Account: addr1, Key: common.Key{0x01}, Value: common.Value{}},
			},
		},
		// -- grow code size --
		{
			Codes: []common.CodeUpdate{
				{Account: addr1, Code: []byte{10_000: 1}},
			},
		},
		// -- shrink code size --
		{
			Codes: []common.CodeUpdate{
				{Account: addr1, Code: []byte{1, 2, 3}},
			},
		},
	}

	state := newState()
	reference, err := newRefState()
	require.NoError(err)

	for _, update := range updates {
		state.Apply(0, update)
		hash, err := state.GetHash()
		require.NoError(err)

		reference.Apply(0, update)
		want, err := reference.GetHash()
		require.NoError(err)

		require.Equal(want, hash)
	}
}

func TestState_SingleAccountFittingInASingleNode_HasSameCommitmentAsReference(t *testing.T) {
	require := require.New(t)

	addr1 := common.Address{1}

	update := common.Update{
		CreatedAccounts: []common.Address{addr1}, // we expect the account must be explicitly created
		Balances: []common.BalanceUpdate{
			{Account: addr1, Balance: amount.New(1)},
		},
	}

	state := newState()
	_, err := state.Apply(0, update)
	require.NoError(err)

	hash, err := state.GetHash()
	require.NoError(err)

	reference, err := newRefState()
	require.NoError(err)
	require.NoError(reference.Apply(0, update))
	want, err := reference.GetHash()
	require.NoError(err)

	require.Equal(want, hash)
}

func TestState_Account_CodeHash_Initialised_With_Eth_Empty_Hash(t *testing.T) {
	require := require.New(t)

	addr1 := common.Address{1}

	update := common.Update{
		CreatedAccounts: []common.Address{addr1}, // we expect the account must be explicitly created
		Balances: []common.BalanceUpdate{
			{Account: addr1, Balance: amount.New(1)},
		},
	}

	state := newState()
	_, err := state.Apply(0, update)
	require.NoError(err)

	codeHash, err := state.GetCodeHash(addr1)
	require.NoError(err)
	require.Equal(common.Hash(types.EmptyCodeHash), codeHash)

	hash, err := state.GetHash()
	require.NoError(err)

	reference, err := newRefState()
	require.NoError(err)
	require.NoError(reference.Apply(0, update))
	want, err := reference.GetHash()
	require.NoError(err)

	require.Equal(want, hash)
}

func TestState_Account_CodeHash_NotEmptied_When_Recreated(t *testing.T) {
	require := require.New(t)

	addr1 := common.Address{1}

	update := common.Update{
		Codes: []common.CodeUpdate{{Account: addr1, Code: []byte{1, 2, 3}}},
		Balances: []common.BalanceUpdate{
			{Account: addr1, Balance: amount.New(1)},
		},
	}

	state := newState()
	_, err := state.Apply(0, update)
	require.NoError(err)

	codeHash, err := state.GetCodeHash(addr1)
	require.NoError(err)
	require.NotEqual(common.Hash(types.EmptyCodeHash), codeHash)

	hash, err := state.GetHash()
	require.NoError(err)

	reference, err := newRefState()
	require.NoError(err)
	require.NoError(reference.Apply(0, update))
	want, err := reference.GetHash()
	require.NoError(err)

	require.Equal(want, hash)

	// Recreate the account, which should not empty the code hash
	update2 := common.Update{
		CreatedAccounts: []common.Address{addr1},
	}

	_, err = state.Apply(0, update2)
	require.NoError(err)

	codeHash, err = state.GetCodeHash(addr1)
	require.NoError(err)
	require.NotEqual(common.Hash(types.EmptyCodeHash), codeHash)

	hash, err = state.GetHash()
	require.NoError(err)

	require.NoError(reference.Apply(0, update2))
	want, err = reference.GetHash()
	require.NoError(err)

	require.Equal(want, hash)
}

func TestState_Account_Balance_NotEmptied_When_Recreated(t *testing.T) {
	require := require.New(t)

	addr1 := common.Address{1}

	update := common.Update{
		CreatedAccounts: []common.Address{addr1}, // we expect the account must be explicitly created
		Balances: []common.BalanceUpdate{
			{Account: addr1, Balance: amount.New(1)},
		},
	}

	state := newState()
	_, err := state.Apply(0, update)
	require.NoError(err)

	balance, err := state.GetBalance(addr1)
	require.NoError(err)
	require.Equal(amount.New(1), balance)

	hash, err := state.GetHash()
	require.NoError(err)

	reference, err := newRefState()
	require.NoError(err)
	require.NoError(reference.Apply(0, update))
	want, err := reference.GetHash()
	require.NoError(err)

	require.Equal(want, hash)

	// Recreate the account, which should not empty the code hash
	update2 := common.Update{
		CreatedAccounts: []common.Address{addr1},
	}

	_, err = state.Apply(0, update2)
	require.NoError(err)

	// The balance should remain the same
	balance, err = state.GetBalance(addr1)
	require.NoError(err)
	require.Equal(amount.New(1), balance)

	hash, err = state.GetHash()
	require.NoError(err)

	require.NoError(reference.Apply(0, update2))
	want, err = reference.GetHash()
	require.NoError(err)

	require.Equal(want, hash)
}

func TestState_Account_Nonce_NotEmptied_When_Recreated(t *testing.T) {
	require := require.New(t)

	addr1 := common.Address{1}

	update := common.Update{
		CreatedAccounts: []common.Address{addr1}, // we expect the account must be explicitly created
		Nonces: []common.NonceUpdate{
			{Account: addr1, Nonce: common.ToNonce(1)},
		},
	}

	state := newState()
	_, err := state.Apply(0, update)
	require.NoError(err)

	nonce, err := state.GetNonce(addr1)
	require.NoError(err)
	require.Equal(common.ToNonce(1), nonce)

	hash, err := state.GetHash()
	require.NoError(err)

	reference, err := newRefState()
	require.NoError(err)
	require.NoError(reference.Apply(0, update))
	want, err := reference.GetHash()
	require.NoError(err)

	require.Equal(want, hash)

	// Recreate the account, which should not empty the nonce
	update2 := common.Update{
		CreatedAccounts: []common.Address{addr1},
	}

	_, err = state.Apply(0, update2)
	require.NoError(err)

	// The nonce should remain the same
	nonce, err = state.GetNonce(addr1)
	require.NoError(err)
	require.Equal(common.ToNonce(1), nonce)

	hash, err = state.GetHash()
	require.NoError(err)

	require.NoError(reference.Apply(0, update2))
	want, err = reference.GetHash()
	require.NoError(err)

	require.Equal(want, hash)
}

// --- reference implementation from geth ---

type refState struct {
	trie *geth_trie.VerkleTrie
}

func newRefState() (*refState, error) {
	trie, err := geth_trie.NewVerkleTrie(
		types.EmptyVerkleHash,
		&refTestDb{},
		utils.NewPointCache(1_000),
	)
	if err != nil {
		return nil, err
	}
	return &refState{trie: trie}, nil
}

func (s *refState) Apply(block uint64, update common.Update) error {
	accountStates := map[geth_common.Address]*types.StateAccount{}

	getAccountState := func(addr geth_common.Address) *types.StateAccount {
		state, ok := accountStates[addr]
		if !ok {
			s, err := s.trie.GetAccount(addr)
			if err != nil {
				panic(err)
			}
			if s == nil {
				s = types.NewEmptyStateAccount()
			}
			accountStates[addr] = s
			state = s
		}
		return state
	}

	for _, update := range update.Balances {
		addr := geth_common.Address(update.Account)
		balance := update.Balance.Uint256()
		state := getAccountState(addr)
		state.Balance = &balance
	}

	for _, update := range update.Nonces {
		addr := geth_common.Address(update.Account)
		state := getAccountState(addr)
		state.Nonce = update.Nonce.ToUint64()
	}

	codes := make(map[geth_common.Address][]byte)
	for _, update := range update.Codes {
		addr := geth_common.Address(update.Account)
		state := getAccountState(addr)
		codeHash := common.Keccak256(update.Code)
		state.CodeHash = codeHash[:]
		codes[addr] = update.Code
	}

	for addr, state := range accountStates {
		s.trie.UpdateAccount(addr, state, len(codes[addr]))
	}

	for addr, code := range codes {
		hash := geth_common.Hash(common.Keccak256(code))
		s.trie.UpdateContractCode(addr, hash, code)
	}

	for _, update := range update.Slots {
		addr := geth_common.Address(update.Account)
		key := geth_common.BytesToHash(update.Key[:])
		value := geth_common.BytesToHash(update.Value[:])
		s.trie.UpdateStorage(addr, key[:], value[:])
	}

	return nil
}

func (s *refState) GetHash() (common.Hash, error) {
	hash, _ := s.trie.Commit(false)
	return common.Hash(hash), nil
}

type refTestDb struct {
}

func (db *refTestDb) NodeReader(stateRoot geth_common.Hash) (database.NodeReader, error) {
	panic("NodeReader not implemented")
}
