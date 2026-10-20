// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package mpt

import (
	"math/rand"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/stretchr/testify/require"
)

func TestMptState_TotalSupply_EmptyStateHasZeroSupply(t *testing.T) {
	state, err := OpenGoMemoryState(t.TempDir(), S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	require.NoError(t, err)
	defer state.Close()

	total, err := state.TotalSupply()
	require.NoError(t, err)
	require.Equal(t, amount.New(), total)
}

func TestMptState_TotalSupply_SumsBalancesOfAllAccountsIgnoringStorage(t *testing.T) {
	state, err := OpenGoMemoryState(t.TempDir(), S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	require.NoError(t, err)
	defer state.Close()

	want := amount.New()
	for i := range 100 {
		addr := common.Address{byte(i), byte(i >> 4), 1}
		balance := amount.New(uint64(i + 1))
		want = amount.Add(want, balance)
		require.NoError(t, state.SetBalance(addr, balance))
		require.NoError(t, state.SetNonce(addr, common.ToNonce(1)))
		require.NoError(t, state.SetStorage(addr, common.Key{1}, common.Value{2}))
	}

	total, err := state.TotalSupply()
	require.NoError(t, err)
	require.Equal(t, want, total)
}

func TestMptState_TotalSupply_OverflowIsReported(t *testing.T) {
	state, err := OpenGoMemoryState(t.TempDir(), S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	require.NoError(t, err)
	defer state.Close()

	require.NoError(t, state.SetBalance(common.Address{1}, amount.Max()))
	require.NoError(t, state.SetBalance(common.Address{2}, amount.New(1)))

	_, err = state.TotalSupply()
	require.ErrorIs(t, err, errTotalSupplyOverflow)
}

func TestMptState_TotalSupply_IsMaintainedIncrementally(t *testing.T) {
	state, err := OpenGoMemoryState(t.TempDir(), S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	require.NoError(t, err)
	defer state.Close()

	// The first query computes and starts tracking the supply.
	total, err := state.TotalSupply()
	require.NoError(t, err)
	require.Equal(t, amount.New(), total)

	_, known := state.trie.getTotalSupply()
	require.True(t, known)

	rnd := rand.New(rand.NewSource(42))
	for block := range 50 {
		update := common.Update{}
		for range 20 {
			addr := common.Address{byte(rnd.Intn(30))}
			switch rnd.Intn(4) {
			case 0:
				update.AppendDeleteAccount(addr)
			case 1:
				update.AppendCreateAccount(addr)
			default:
				update.AppendBalanceUpdate(addr, amount.New(uint64(rnd.Intn(1000))))
			}
		}
		// The update is not normalized, so ensure that it is applied as is.
		_, err := state.Apply(uint64(block), &update)
		require.NoError(t, err)

		got, err := state.TotalSupply()
		require.NoError(t, err)
		want, err := state.computeTotalSupply()
		require.NoError(t, err)
		require.Equal(t, want, got, "block %d", block)
	}
}

func TestMptState_TotalSupply_DeletingAccountReducesSupply(t *testing.T) {
	state, err := OpenGoMemoryState(t.TempDir(), S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	require.NoError(t, err)
	defer state.Close()

	_, err = state.TotalSupply()
	require.NoError(t, err)
	require.NoError(t, state.SetBalance(common.Address{1}, amount.New(10)))
	require.NoError(t, state.SetBalance(common.Address{2}, amount.New(5)))
	require.NoError(t, state.DeleteAccount(common.Address{1}))

	total, err := state.TotalSupply()
	require.NoError(t, err)
	require.Equal(t, amount.New(5), total)
}

func TestMptState_TotalSupply_IsPersisted(t *testing.T) {
	dir := t.TempDir()
	state, err := OpenGoFileState(dir, S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	require.NoError(t, err)
	require.NoError(t, state.SetBalance(common.Address{1}, amount.New(10)))
	_, err = state.TotalSupply()
	require.NoError(t, err)
	require.NoError(t, state.SetBalance(common.Address{2}, amount.New(5)))
	require.NoError(t, state.Close())

	state, err = OpenGoFileState(dir, S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	require.NoError(t, err)
	defer state.Close()

	// The value is available without traversing the trie.
	total, known := state.trie.getTotalSupply()
	require.True(t, known)
	require.Equal(t, amount.New(15), total)
}

func TestMptState_TotalSupply_UnknownValueIsComputedForExistingState(t *testing.T) {
	dir := t.TempDir()
	state, err := OpenGoFileState(dir, S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	require.NoError(t, err)
	// Balances set before the supply is tracked.
	require.NoError(t, state.SetBalance(common.Address{1}, amount.New(10)))
	require.NoError(t, state.Close())

	state, err = OpenGoFileState(dir, S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	require.NoError(t, err)
	defer state.Close()

	_, known := state.trie.getTotalSupply()
	require.False(t, known)

	require.NoError(t, state.SetBalance(common.Address{2}, amount.New(5)))
	total, err := state.TotalSupply()
	require.NoError(t, err)
	require.Equal(t, amount.New(15), total)
}

func TestLiveTrie_AdjustTotalSupply_InconsistentUpdatesInvalidateValue(t *testing.T) {
	tests := map[string]struct{ initial, old, new amount.Amount }{
		"underflow": {amount.New(1), amount.New(2), amount.New(0)},
		"overflow":  {amount.Max(), amount.New(0), amount.New(1)},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			trie := &LiveTrie{}
			trie.setTotalSupply(test.initial)
			trie.adjustTotalSupply(test.old, test.new)
			_, known := trie.getTotalSupply()
			require.False(t, known)
		})
	}
}

func TestLiveTrie_AdjustTotalSupply_UnknownValueStaysUnknown(t *testing.T) {
	trie := &LiveTrie{}
	trie.adjustTotalSupply(amount.New(1), amount.New(2))
	_, known := trie.getTotalSupply()
	require.False(t, known)
}

func TestMetadata_TotalSupply_InvalidValueIsRejected(t *testing.T) {
	_, err := (&metadata{TotalSupply: "abc"}).getTotalSupply()
	require.Error(t, err)
}
