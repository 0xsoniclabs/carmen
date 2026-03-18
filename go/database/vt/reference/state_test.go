// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package reference

import (
	"bytes"
	"crypto/rand"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/common/future"
	"github.com/0xsoniclabs/carmen/go/common/result"
	geth "github.com/0xsoniclabs/carmen/go/database/vt/geth"
	"github.com/0xsoniclabs/carmen/go/database/vt/reference/trie"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/ethereum/go-ethereum/core/types"

	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
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

func TestState_TrieConfig_ReturnsTheUnderlyingTrieConfig(t *testing.T) {
	require := require.New(t)
	ctrl := gomock.NewController(t)
	trie := NewMockTrie(ctrl)

	want := "my-trie-config"
	trie.EXPECT().Config().Return(want)

	state := &State{trie: trie}
	got := state.TrieConfig()
	require.Equal(want, got)
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
			_, err = state.Apply(0, common.Update{
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

func TestState_HasEmptyStorage_ReturnsTrue(t *testing.T) {
	require := require.New(t)
	state := newState()
	empty, err := state.HasEmptyStorage(common.Address{1})
	require.NoError(err)
	require.True(empty)
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
	hash, err := state.GetCommitment().Await().Get()
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

	hash, err := state.GetCommitment().Await().Get()
	require.NoError(err)

	reference, err := newRefState(t)
	require.NoError(err)
	reference.Apply(0, update)
	want, err := reference.GetCommitment().Await().Get()
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
	reference, err := newRefState(t)
	require.NoError(err)

	for _, update := range updates {
		state.Apply(0, update)
		hash, err := state.GetCommitment().Await().Get()
		require.NoError(err)

		reference.Apply(0, update)
		want, err := reference.GetCommitment().Await().Get()
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

	hash, err := state.GetCommitment().Await().Get()
	require.NoError(err)

	reference, err := newRefState(t)
	require.NoError(err)
	_, err = reference.Apply(0, update)
	require.NoError(err)

	want, err := reference.GetCommitment().Await().Get()
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

	hash, err := state.GetCommitment().Await().Get()
	require.NoError(err)

	reference, err := newRefState(t)
	require.NoError(err)
	_, err = reference.Apply(0, update)
	require.NoError(err)
	want, err := reference.GetCommitment().Await().Get()
	require.NoError(err)

	require.Equal(want, hash)
}

// --- Use geth leveldb as reference implementation ---

type refState struct {
	trie state.State
}

func newRefState(t *testing.T) (*refState, error) {
	testDir := t.TempDir()
	trie, err := geth.NewState(
		state.Parameters{
			Directory: testDir,
			Archive:   state.NoArchive,
		},
	)
	if err != nil {
		return nil, err
	}
	return &refState{trie: trie}, nil
}

func (s *refState) Apply(block uint64, update common.Update) (<-chan error, error) {
	return s.trie.Apply(block, update)
}

func (s *refState) GetCommitment() future.Future[result.Result[common.Hash]] {
	return s.trie.GetCommitment()
}

// newState creates a new, empty in-memory state instance based on a reference
// version of the Verkle trie.
func newState() state.State {
	return NewStateUsing(&trie.Trie{})
}
