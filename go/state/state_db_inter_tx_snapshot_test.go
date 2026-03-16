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
	db.EXPECT().Check().Return(nil).AnyTimes()
	state := createStateDBWith(db, 1, true)

	snapshotID := state.InterTxSnapshot()
	require.Equal(InterTxSnapshotID(0), snapshotID)

	// Empty transaction does not increment snapshot ID
	state.BeginTransaction()
	state.EndTransaction()
	snapshotID2 := state.InterTxSnapshot()
	require.Equal(InterTxSnapshotID(0), snapshotID2)

	state.BeginTransaction()
	state.SetCode(common.Address{}, []byte{0x1})
	state.EndTransaction()
	snapshotID3 := state.InterTxSnapshot()
	require.Equal(InterTxSnapshotID(3), snapshotID3)
}

func TestStateDB_RevertToInterTxSnapshot_ReturnsErrorIfInvalidSnapshotID(t *testing.T) {
	t.Parallel()
	require := require.New(t)
	ctrl := gomock.NewController(t)
	db := NewMockState(ctrl)
	db.EXPECT().Exists(gomock.Any()).Return(false, nil).AnyTimes()
	db.EXPECT().Check().Return(nil).AnyTimes()
	state := createStateDBWith(db, 1, true)

	state.RevertToInterTxSnapshot(1)
	require.EqualError(state.Check(), "cannot revert to inter-transaction snapshot with value 1, only 0 snapshots in the current block")
	state.errors = state.errors[:0]

	state.BeginTransaction()
	state.SetCode(common.Address{}, []byte{0x1})
	state.EndTransaction()
	state.RevertToInterTxSnapshot(10)
	require.EqualError(state.Check(), "cannot revert to inter-transaction snapshot with value 10, only 3 snapshots in the current block")
}

func TestStateDB_EndTransaction_AddsUndoForRestoreWhen(t *testing.T) {
	runTests := func(t *testing.T, beginOp func(state *stateDB)) {
		testCases := map[string](func(t *testing.T, state *stateDB, beginOp func(state *stateDB))){
			"UpdatingWrittenSlots":                UpdatingWrittenSlots,
			"AddingEmptyAccountWithoutState":      AddingEmptyAccountWithState,
			"AddingEmptyAccountWithState":         AddingEmptyAccountWithState,
			"DeletingSuicidedAccountWithoutState": DeletingSuicidedAccountWithoutState,
			"DeletingSuicidedAccountWithState":    DeletingSuicidedAccountWithState,
			"DeletingAccountWithData":             DeletingAccountWithData,
			"DeletingAccountWithoutState":         DeletingAccountWithoutState,
			"DeletingAccountWithState":            DeletingAccountWithState,
		}

		for name, testFunc := range testCases {
			t.Run(name, func(t *testing.T) {
				ctrl := gomock.NewController(t)
				db := NewMockState(ctrl)
				db.EXPECT().Check().Return(nil).AnyTimes()
				db.EXPECT().Check().Return(nil).AnyTimes()
				db.EXPECT().Exists(gomock.Any()).Return(true, nil).AnyTimes()
				db.EXPECT().GetBalance(gomock.Any()).Return(amount.New(0), nil).AnyTimes()
				db.EXPECT().GetNonce(gomock.Any()).Return(common.Nonce{0}, nil).AnyTimes()
				db.EXPECT().GetCode(gomock.Any()).Return(nil, nil).AnyTimes()
				db.EXPECT().GetCodeSize(gomock.Any()).Return(0, nil).AnyTimes()
				state := createStateDBWith(db, 1, true)

				testFunc(t, state, beginOp)
			})
		}
	}

	t.Run("WithBeginTransaction", func(t *testing.T) {
		runTests(t, func(state *stateDB) {
			state.BeginTransaction()
		})
	})

	t.Run("WithoutBeginTransaction", func(t *testing.T) {
		runTests(t, func(state *stateDB) {
			// No-op
		})
	})
}

func UpdatingWrittenSlots(t *testing.T, state *stateDB, beginOp func(state *stateDB)) {
	t.Parallel()
	require := require.New(t)

	backup := state.InterTxSnapshot()

	beginOp(state)
	entry := slotValue{committed: common.Value{0x1}, current: common.Value{0x2}}
	state.writtenSlots[&entry] = true
	state.EndTransaction()

	require.Equal(entry.committed, common.Value{0x2})

	state.RevertToInterTxSnapshot(backup)
	require.Equal(entry.committed, common.Value{0x1})
	require.Equal(entry.current, common.Value{0x2})
}

func AddingEmptyAccountWithoutState(t *testing.T, state *stateDB, beginOp func(state *stateDB)) {
	t.Parallel()
	require := require.New(t)

	addr := common.Address{0x1}
	backup := state.InterTxSnapshot()
	beginOp(state)
	state.emptyCandidates = append(state.emptyCandidates, addr)
	state.EndTransaction()
	require.Equal(state.clearedAccounts[addr], cleared)

	state.RevertToInterTxSnapshot(backup)
	require.Empty(state.clearedAccounts)

}

func AddingEmptyAccountWithState(t *testing.T, state *stateDB, beginOp func(state *stateDB)) {
	t.Parallel()
	require := require.New(t)

	addr := common.Address{0x1}
	backup := state.InterTxSnapshot()

	beginOp(state)
	state.clearedAccounts[addr] = noClearing
	state.emptyCandidates = append(state.emptyCandidates, addr)
	state.EndTransaction()
	require.Equal(state.clearedAccounts[addr], cleared)

	state.RevertToInterTxSnapshot(backup)
	require.Equal(state.clearedAccounts[addr], noClearing)
}

func DeletingSuicidedAccountWithoutState(t *testing.T, state *stateDB, beginOp func(state *stateDB)) {
	t.Parallel()
	require := require.New(t)

	addr := common.Address{0x1}
	backup := state.InterTxSnapshot()

	beginOp(state)
	state.accounts[addr] = &accountState{current: accountSelfDestructed}
	state.accountsToDelete = append(state.accountsToDelete, addr)
	state.EndTransaction()
	require.Equal(state.clearedAccounts[addr], cleared)

	state.RevertToInterTxSnapshot(backup)
	require.Empty(state.clearedAccounts)
}

func DeletingSuicidedAccountWithState(t *testing.T, state *stateDB, beginOp func(state *stateDB)) {
	t.Parallel()
	require := require.New(t)

	addr := common.Address{0x1}
	backup := state.InterTxSnapshot()

	beginOp(state)
	state.clearedAccounts[addr] = noClearing
	state.accounts[addr] = &accountState{current: accountSelfDestructed}
	state.accountsToDelete = append(state.accountsToDelete, addr)
	state.EndTransaction()
	require.Equal(state.clearedAccounts[addr], cleared)

	state.RevertToInterTxSnapshot(backup)
	require.Equal(state.clearedAccounts[addr], noClearing)
}

func DeletingAccountWithData(t *testing.T, state *stateDB, beginOp func(state *stateDB)) {
	t.Parallel()
	require := require.New(t)
	addr := common.Address{0x1}
	key := common.Key{0x2}

	defaultSlotValue := slotValue{
		stored:         common.Value{0x1},
		storedKnown:    false,
		committed:      common.Value{0x1},
		committedKnown: false,
		current:        common.Value{0x1},
	}
	testSlotValue := defaultSlotValue

	state.data.Put(slotId{addr, key}, &testSlotValue)
	backup := state.InterTxSnapshot()

	beginOp(state)
	state.accountsToDelete = append(state.accountsToDelete, addr)
	state.EndTransaction()

	require.Equal(state.clearedAccounts[addr], cleared)
	require.Equal(testSlotValue.stored, common.Value{})
	require.Equal(testSlotValue.storedKnown, true)
	require.Equal(testSlotValue.committed, common.Value{})
	require.Equal(testSlotValue.committedKnown, true)
	require.Equal(testSlotValue.current, common.Value{})

	state.RevertToInterTxSnapshot(backup)
	require.Empty(state.clearedAccounts)
	require.Equal(testSlotValue, defaultSlotValue)
}

func DeletingAccountWithoutState(t *testing.T, state *stateDB, beginOp func(state *stateDB)) {
	t.Parallel()
	require := require.New(t)
	addr := common.Address{0x1}

	backup := state.InterTxSnapshot()

	beginOp(state)
	state.accountsToDelete = append(state.accountsToDelete, addr)
	state.EndTransaction()
	require.Equal(state.clearedAccounts[addr], cleared)

	state.RevertToInterTxSnapshot(backup)
	require.Empty(state.clearedAccounts)
}

func DeletingAccountWithState(t *testing.T, state *stateDB, beginOp func(state *stateDB)) {
	t.Parallel()
	require := require.New(t)
	addr := common.Address{0x1}

	backup := state.InterTxSnapshot()

	beginOp(state)
	state.clearedAccounts[addr] = noClearing
	state.accountsToDelete = append(state.accountsToDelete, addr)
	state.EndTransaction()
	require.Equal(state.clearedAccounts[addr], cleared)

	state.RevertToInterTxSnapshot(backup)
	require.Equal(state.clearedAccounts[addr], noClearing)
}
