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
