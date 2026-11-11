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
	"errors"
	"path/filepath"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
)

func TestMptState_TotalSupply_EmptyStateHasZeroSupply(t *testing.T) {
	state, err := OpenGoMemoryState(t.TempDir(), S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	require.NoError(t, err)
	defer func() { require.NoError(t, state.Close()) }()

	total, err := state.TotalSupply()
	require.NoError(t, err)
	require.Equal(t, amount.New(), total)
}

func TestMptState_TotalSupply_SumsBalancesOfAllAccountsIgnoringStorage(t *testing.T) {
	state, err := OpenGoMemoryState(t.TempDir(), S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	require.NoError(t, err)
	defer func() { require.NoError(t, state.Close()) }()

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
	defer func() { require.NoError(t, state.Close()) }()

	require.NoError(t, state.SetBalance(common.Address{1}, amount.Max()))
	require.NoError(t, state.SetBalance(common.Address{2}, amount.New(1)))

	_, err = state.TotalSupply()
	require.ErrorIs(t, err, errTotalSupplyOverflow)
}

func TestMptState_TotalSupply_AvailableValueIsNotRecomputed(t *testing.T) {
	require := require.New(t)
	db := NewMockDatabase(gomock.NewController(t))
	db.EXPECT().updateHashesFor(gomock.Any()).AnyTimes()
	db.EXPECT().getConfig().Return(S5LiveConfig).AnyTimes()
	db.EXPECT().Flush().Return(nil).Times(1)
	state := newMptStateWithEmptyNodeFiles(t, db)

	for range 2 {
		total, err := state.TotalSupply()
		require.NoError(err)
		require.Equal(amount.New(), total)
	}
}

func TestMptState_TotalSupply_FailedComputationIsReportedAndRetried(t *testing.T) {
	require := require.New(t)
	injectedErr := errors.New("injected error")
	db := NewMockDatabase(gomock.NewController(t))
	db.EXPECT().updateHashesFor(gomock.Any()).AnyTimes()
	db.EXPECT().getConfig().Return(S5LiveConfig).AnyTimes()
	gomock.InOrder(
		db.EXPECT().Flush().Return(injectedErr),
		db.EXPECT().Flush().Return(nil),
	)
	state := newMptStateWithEmptyNodeFiles(t, db)

	_, err := state.TotalSupply()
	require.ErrorIs(err, injectedErr)
	total, err := state.TotalSupply()
	require.NoError(err)
	require.Equal(amount.New(), total)
}

func TestMptState_TotalSupply_CachedValueIsDroppedByBalanceUpdate(t *testing.T) {
	require := require.New(t)
	state, err := OpenGoMemoryState(t.TempDir(), S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	require.NoError(err)
	defer func() { require.NoError(state.Close()) }()

	require.NoError(state.SetBalance(common.Address{1}, amount.New(10)))
	total, err := state.TotalSupply()
	require.NoError(err)
	require.Equal(amount.New(10), total)

	require.NoError(state.SetBalance(common.Address{1}, amount.New(3)))
	total, err = state.TotalSupply()
	require.NoError(err)
	require.Equal(amount.New(3), total)
}

func TestMptState_TotalSupply_IsRestoredByRevertLastBlock(t *testing.T) {
	require := require.New(t)
	state, err := OpenGoMemoryState(t.TempDir(), S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	require.NoError(err)
	defer func() { require.NoError(state.Close()) }()

	setup := common.Update{
		Balances: []common.BalanceUpdate{{Account: common.Address{1}, Balance: amount.New(100)}},
	}
	_, _, err = state.Apply(0, &setup)
	require.NoError(err)
	want, err := state.TotalSupply()
	require.NoError(err)
	require.Equal(amount.New(100), want)

	// The update modifies an existing account and creates a new one.
	update := common.Update{
		Balances: []common.BalanceUpdate{
			{Account: common.Address{1}, Balance: amount.New(30)},
			{Account: common.Address{2}, Balance: amount.New(50)},
		},
	}
	undo, _, err := state.Apply(1, &update)
	require.NoError(err)
	total, err := state.TotalSupply()
	require.NoError(err)
	require.Equal(amount.New(80), total)

	require.NoError(state.RevertLastBlock(undo))

	total, err = state.TotalSupply()
	require.NoError(err)
	require.Equal(want, total)
}

// newMptStateWithEmptyNodeFiles creates a state on the given forest whose node
// files hold an empty trie.
func newMptStateWithEmptyNodeFiles(t *testing.T, db Database) *MptState {
	dir := t.TempDir()
	files, err := OpenGoFileState(dir, S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	require.NoError(t, err)
	require.NoError(t, files.Close())
	return &MptState{directory: dir, trie: &LiveTrie{forest: db, metaDataFile: filepath.Join(dir, "meta.json")}}
}
