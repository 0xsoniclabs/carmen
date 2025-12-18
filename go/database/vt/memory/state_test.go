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
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/database/vt/memory/trie"
	"github.com/0xsoniclabs/carmen/go/database/vt/reference"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/stretchr/testify/require"
)

func TestNewState_DefaultsToParallelCommit(t *testing.T) {
	state, err := NewState(state.Parameters{Variant: "memory"})
	require.NoError(t, err)

	memoryState, ok := state.(*reference.State)
	require.True(t, ok)

	config, ok := memoryState.TrieConfig().(trie.TrieConfig)
	require.True(t, ok)

	require.True(t, config.ParallelCommit)
}

func TestNewState_SeqSuffixDisablesParallelCommits(t *testing.T) {
	variants := map[state.Variant]bool{
		"memory-seq":    false,
		"memory-par":    true,
		"memory":        true,
		"something-seq": false,
		"something-par": true,
		"something":     true,
	}

	for variant, wantParallel := range variants {
		t.Run(string(variant), func(t *testing.T) {
			state, err := NewState(state.Parameters{Variant: variant})
			require.NoError(t, err)

			memoryState, ok := state.(*reference.State)
			require.True(t, ok)

			config, ok := memoryState.TrieConfig().(trie.TrieConfig)
			require.True(t, ok)

			require.Equal(t, wantParallel, config.ParallelCommit)
		})
	}
}

func TestState_CanStoreAndRestoreNonces(t *testing.T) {
	variants := map[string]state.Parameters{
		"sequential": {Variant: "memory-seq"},
		"parallel":   {Variant: "memory-par"},
	}

	for name, params := range variants {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)

			state, err := NewState(params)
			require.NoError(err)

			address := common.Address{1}

			// Initially, the nonce should be zero
			nonce, err := state.GetNonce(address)
			require.NoError(err)
			require.Equal(common.ToNonce(0), nonce)

			// Set a nonce
			require.NoError(state.Apply(0, common.Update{
				Nonces: []common.NonceUpdate{{
					Account: address,
					Nonce:   common.ToNonce(42),
				}},
			}))

			// Retrieve the nonce again
			nonce, err = state.GetNonce(address)
			require.NoError(err)
			require.Equal(common.ToNonce(42), nonce)

			// Set another nonce
			require.NoError(state.Apply(0, common.Update{
				Nonces: []common.NonceUpdate{{
					Account: address,
					Nonce:   common.ToNonce(123),
				}},
			}))

			// Retrieve the updated nonce
			nonce, err = state.GetNonce(address)
			require.NoError(err)
			require.Equal(common.ToNonce(123), nonce)
		})
	}
}

func TestState_StateWithContentHasExpectedCommitment(t *testing.T) {
	// This is a smoke test to verify whether the in-memory state
	// produces the same commitment hash as the reference state
	// implementation for a given set of updates.
	variants := map[string]state.Parameters{
		"sequential": {Variant: "memory-seq"},
		"parallel":   {Variant: "memory-par"},
	}
	for name, params := range variants {
		t.Run(name, func(t *testing.T) {
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

			state, err := NewState(params)
			require.NoError(err)
			require.NoError(state.Apply(0, update))

			hash, err := state.GetCommitment().Await().Get()
			require.NoError(err)

			reference, err := reference.NewState(params)
			require.NoError(err)
			require.NoError(reference.Apply(0, update))
			want, err := reference.GetCommitment().Await().Get()
			require.NoError(err)

			require.Equal(want, hash)
		})
	}
}
