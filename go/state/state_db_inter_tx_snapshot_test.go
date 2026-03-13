// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package state

import (
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
)

func TestStateDB_InterTxSnapshot_ReturnsSnapshotID(t *testing.T) {
	t.Parallel()
	require := require.New(t)
	ctrl := gomock.NewController(t)
	db := NewMockState(ctrl)
	db.EXPECT().Exists(gomock.Any()).Return(false, nil).AnyTimes()
	state := createStateDBWith(db, 1, true)

	snapshotID := state.InterTxSnapshot()
	require.Equal(snapshotID, InterTxSnapshotID(0), "unexpected snapshot ID: want %d, got %d", 0, snapshotID)

	// Empty transaction does not increment snapshot ID
	state.BeginTransaction()
	state.EndTransaction()
	snapshotID2 := state.InterTxSnapshot()
	require.Equal(snapshotID2, InterTxSnapshotID(0), "unexpected snapshot ID: want %d, got %d", 1, snapshotID2)

	state.BeginTransaction()
	state.SetCode(common.Address{}, []byte{0x1})
	state.EndTransaction()
	snapshotID3 := state.InterTxSnapshot()
	require.Equal(snapshotID3, InterTxSnapshotID(3), "unexpected snapshot ID: want %d, got %d", 1, snapshotID2)
}

func TestStateDB_RevertToInterTxSnapshot_ReturnsErrorIfInvalidSnapshotID(t *testing.T) {
	t.Parallel()
	require := require.New(t)
	ctrl := gomock.NewController(t)
	db := NewMockState(ctrl)
	db.EXPECT().Exists(gomock.Any()).Return(false, nil).AnyTimes()
	state := createStateDBWith(db, 1, true)

	err := state.RevertToInterTxSnapshot(1)
	require.EqualError(err, "cannot revert to inter-transaction snapshot with value 1, only 0 snapshots in the current block")

	state.BeginTransaction()
	state.SetCode(common.Address{}, []byte{0x1})
	state.EndTransaction()
	err = state.RevertToInterTxSnapshot(10)
	require.EqualError(err, "cannot revert to inter-transaction snapshot with value 10, only 3 snapshots in the current block")
}

func TestStateDB_RevertToInterTxSnapshot_RevertsClearedAccountsFromCreateAccount(t *testing.T) {
	t.Parallel()
	ctrl := gomock.NewController(t)
	require := require.New(t)

	db := NewMockState(ctrl)
	state := createStateDBWith(db, 1, true)
	addr := common.Address{0x1}
	db.EXPECT().Check().Return(nil).AnyTimes()
	db.EXPECT().Flush().Return(nil).AnyTimes()
	db.EXPECT().Close().Return(nil).AnyTimes()
	db.EXPECT().GetBalance(gomock.Any()).Return(amount.New(0), nil).AnyTimes()
	db.EXPECT().GetNonce(gomock.Any()).Return(common.Nonce{0}, nil).AnyTimes()
	db.EXPECT().GetCode(gomock.Any()).Return(nil, nil).AnyTimes()
	db.EXPECT().GetCodeSize(gomock.Any()).Return(0, nil).AnyTimes()
	db.EXPECT().Exists(gomock.Any()).Return(false, nil).AnyTimes()

	val, exists := state.clearedAccounts[addr]
	require.False(exists)
	require.Equal(noClearing, val)

	snapshot := state.InterTxSnapshot()
	state.BeginTransaction()
	state.CreateAccount(addr)
	state.Suicide(addr)
	state.CreateAccount(addr)
	state.EndTransaction()

	if err := db.Check(); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	val, exists = state.clearedAccounts[addr]
	require.True(exists)
	require.Equal(cleared, val)

	err := state.RevertToInterTxSnapshot(snapshot)
	if err != nil {
		t.Fatalf("Error while reverting to inter-transaction snapshot: %v", err)
	}
	val, exists = state.clearedAccounts[addr]
	require.False(exists)
	require.Equal(noClearing, val)
}

func TestStateDB_RevertToInterTxSnapshot_RevertsClearedAccountsFromSuicide(t *testing.T) {
	t.Parallel()
	ctrl := gomock.NewController(t)
	require := require.New(t)

	db := NewMockState(ctrl)
	addr := common.Address{0x1}
	db.EXPECT().Exists(addr).Return(true, nil).AnyTimes()
	db.EXPECT().Check().Return(nil).AnyTimes()
	state := createStateDBWith(db, 1, true)

	val, exists := state.clearedAccounts[addr]
	require.False(exists)
	require.Equal(noClearing, val)

	snapshot := state.InterTxSnapshot()
	state.BeginTransaction()
	state.Suicide(addr)
	state.CreateAccount(addr)
	state.clearedAccounts[addr] = noClearing
	state.Suicide(addr)
	state.EndTransaction()

	if err := db.Check(); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	val, exists = state.clearedAccounts[addr]
	require.True(exists)
	require.Equal(cleared, val)

	err := state.RevertToInterTxSnapshot(snapshot)
	if err != nil {
		t.Fatalf("Error while reverting to inter-transaction snapshot: %v", err)
	}
	val, exists = state.clearedAccounts[addr]
	require.False(exists)
	require.Equal(noClearing, val)
}

func TestStateDB_RevertToInterTxSnapshot_RevertsStateFromEndTransaction(t *testing.T) {
	t.Parallel()
	ctrl := gomock.NewController(t)
	require := require.New(t)
	addr := common.Address{0x1}

	db := NewMockState(ctrl)
	db.EXPECT().Check().Return(nil).AnyTimes()
	db.EXPECT().Exists(addr).Return(true, nil).AnyTimes()
	db.EXPECT().GetBalance(gomock.Any()).Return(amount.New(0), nil).AnyTimes()
	db.EXPECT().GetNonce(gomock.Any()).Return(common.Nonce{0}, nil).AnyTimes()
	db.EXPECT().GetCode(gomock.Any()).Return(nil, nil).AnyTimes()
	db.EXPECT().GetCodeSize(gomock.Any()).Return(0, nil).AnyTimes()
	state := createStateDBWith(db, 1, true)

	// To Revert:
	// - Empty account: pending clearing (delete)
	// - Cleared account (existing)
	val, exists := state.clearedAccounts[addr]
	require.False(exists)
	require.Equal(noClearing, val)
	snapshot := state.InterTxSnapshot()
	state.BeginTransaction()
	state.emptyCandidates = append(state.emptyCandidates, addr)
	state.accountsToDelete = append(state.accountsToDelete, addr)
	state.EndTransaction()
	val, exists = state.clearedAccounts[addr]
	require.True(exists)
	require.Equal(cleared, val)

	err := state.RevertToInterTxSnapshot(snapshot)
	require.NoError(err)
	val, exists = state.clearedAccounts[addr]
	require.False(exists)
	require.Equal(noClearing, val)

	// To Revert:
	// - Cleared account (delete)
	clearedAccountValue, clearedAccountExists := state.clearedAccounts[addr]
	require.False(clearedAccountExists)
	require.Equal(noClearing, clearedAccountValue)
	snapshot = state.InterTxSnapshot()
	state.BeginTransaction()
	state.accountsToDelete = append(state.accountsToDelete, addr)
	state.EndTransaction()
	err = state.Check()

	require.NoError(err)
	clearedAccountValue, clearedAccountExists = state.clearedAccounts[addr]
	require.True(clearedAccountExists)
	require.Equal(cleared, clearedAccountValue)

	err = state.RevertToInterTxSnapshot(snapshot)
	require.NoError(err)
	clearedAccountValue, clearedAccountExists = state.clearedAccounts[addr]
	require.False(clearedAccountExists)
	require.Equal(noClearing, clearedAccountValue)

	// To Revert:
	// - Written slots
	// - Empty account: pending clearing (existing)
	// - Has suicided: pending clearing
	// - Data
	key := common.Key{0x2}
	clearedAccountVal, clearedAccountExists := state.clearedAccounts[addr]
	require.False(clearedAccountExists)
	require.Equal(noClearing, clearedAccountVal)

	// Set data to test data undo
	valueToWrite := common.Value{0x3}
	slotId := slotId{addr, key}
	_, slotExists := state.data.Get(slotId)
	require.False(slotExists)
	dataSnapshot := state.InterTxSnapshot()
	state.BeginTransaction()
	state.SetState(addr, key, valueToWrite)
	state.EndTransaction()
	slotValue, slotExists := state.data.Get(slotId)
	require.True(slotExists)
	require.Equal(valueToWrite, slotValue.committed)

	snapshot = state.InterTxSnapshot()
	state.BeginTransaction()
	state.emptyCandidates = append(state.emptyCandidates, addr)
	state.Suicide(addr)
	state.EndTransaction()
	require.NoError(state.Check())

	slotValue, slotExists = state.data.Get(slotId)
	require.True(slotExists)
	require.Equal(slotValue.committed, common.Value{}) // Value reset by suicide
	clearedAccountVal, clearedAccountExists = state.clearedAccounts[addr]
	require.True(clearedAccountExists)
	require.Equal(cleared, clearedAccountVal)

	err = state.RevertToInterTxSnapshot(snapshot)
	require.NoError(err)
	slotValue, slotExists = state.data.Get(slotId)
	require.True(slotExists)
	require.Equal(valueToWrite, slotValue.committed)

	// Back to initial state
	err = state.RevertToInterTxSnapshot(dataSnapshot)
	require.NoError(err)
	_, slotExists = state.data.Get(slotId)
	require.False(slotExists)
}
