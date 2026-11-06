// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

// This collects tests of stateDB functions affected by revision changes.

package state

import (
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
)

func TestStateDB_Suicide_EndTransaction(t *testing.T) {
	balance := amount.New(10)
	tests := map[string]struct {
		revision            Revision
		balanceAfterSuicide amount.Amount
		accountExists       bool
	}{
		"BeforeAmsterdam/BalanceIsBurnt": {
			revision:            AMSTERDAM - 1, // Latest before
			balanceAfterSuicide: amount.New(),
			accountExists:       false,
		},
		"AfterAmsterdam/BalanceIsRetained": {
			revision:            AMSTERDAM,
			balanceAfterSuicide: balance,
			accountExists:       true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			ctrl := gomock.NewController(t)
			mock := NewMockState(ctrl)
			db := CreateStateDBUsing(mock)

			mock.EXPECT().GetBalance(address1).Return(balance, nil)

			require.True(db.Suicide(address1))
			require.Equal(balance, db.GetBalance(address1))

			db.EndTransaction(test.revision)

			require.Equal(test.accountExists, db.Exist(address1))
			require.Equal(test.balanceAfterSuicide, db.GetBalance(address1))
		})
	}
}

func TestStateDB_Suicide_RevertToInterTxSnapshot(t *testing.T) {
	balance := amount.New(12)
	tests := map[string]struct {
		revision            Revision
		balanceAfterSuicide amount.Amount
		accountExists       bool
	}{
		"BeforeAmsterdam/BalanceIsBurnt": {
			revision:            AMSTERDAM - 1, // Latest before
			balanceAfterSuicide: amount.New(),
			accountExists:       false,
		},
		"AfterAmsterdam/BalanceIsRetained": {
			revision:            AMSTERDAM,
			balanceAfterSuicide: balance,
			accountExists:       true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			ctrl := gomock.NewController(t)
			mock := NewMockState(ctrl)
			db := CreateStateDBUsing(mock)

			setExpectationForEmptyAccount(t, mock, address1)

			db.CreateAccount(address1)
			db.SetNonce(address1, 1)
			db.SetCode(address1, []byte{1, 2, 3})
			db.SetState(address1, key1, val1)
			db.AddBalance(address1, balance)
			db.EndTransaction(test.revision)

			snapshot := db.InterTxSnapshot()
			require.True(db.Suicide(address1))
			db.EndTransaction(test.revision)

			require.Equal(test.accountExists, db.Exist(address1))
			require.Equal(test.balanceAfterSuicide, db.GetBalance(address1))

			db.RevertToInterTxSnapshot(snapshot)

			require.True(db.Exist(address1))
			require.False(db.HasSuicided(address1))
			require.Equal(balance, db.GetBalance(address1))
			require.Equal(uint64(1), db.GetNonce(address1))
			require.Equal([]byte{1, 2, 3}, db.GetCode(address1))
			require.Equal(val1, db.GetState(address1, key1))
		})
	}
}

func TestStateDB_Suicide_ClearsSubsequentWritesByRevision(t *testing.T) {
	tests := map[string]struct {
		revision            Revision
		balanceAfterSuicide amount.Amount
		accountExists       bool
	}{
		"BeforeAmsterdam/WritesAfterSuicideAreCleared": {
			revision:            AMSTERDAM - 1,
			balanceAfterSuicide: amount.New(),
			accountExists:       false,
		},
		"AfterAmsterdam/OnlyBalanceSurvivesSubsequentWrites": {
			revision:            AMSTERDAM,
			balanceAfterSuicide: amount.New(17),
			accountExists:       true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			ctrl := gomock.NewController(t)
			mock := NewMockState(ctrl)
			db := CreateStateDBUsing(mock)
			mock.EXPECT().GetBalance(address1).Return(amount.New(12), nil)
			mock.EXPECT().GetStorage(address1, key1).Return(val1, nil)
			db.SetNonce(address1, 1)
			db.SetCode(address1, []byte{1, 2, 3})
			require.Equal(val1, db.GetState(address1, key1))
			require.True(db.Suicide(address1))
			db.AddBalance(address1, amount.New(5))
			db.SetNonce(address1, 4321)
			db.SetCode(address1, []byte{0x12, 0x34})
			db.SetState(address1, key1, val2)
			require.True(db.HasSuicided(address1))
			db.EndTransaction(test.revision)
			require.Equal(test.accountExists, db.Exist(address1))
			require.False(db.HasSuicided(address1))
			require.Equal(test.balanceAfterSuicide, db.GetBalance(address1))
			require.Zero(db.GetNonce(address1))
			require.Empty(db.GetCode(address1))
			require.Equal(common.Value{}, db.GetState(address1, key1))
			require.True(db.HasEmptyStorage(address1))
		})
	}
}

func TestStateDB_SuicideAndCreateAccount_RecreatedAccountRetainsBalanceInAllRevisions(t *testing.T) {
	tests := map[string]struct {
		revision            Revision
		balanceAfterSuicide amount.Amount
		accountExists       bool
	}{
		"BeforeAmsterdam/RecreatedAccountRetainsBalance": {
			revision:            AMSTERDAM - 1,
			balanceAfterSuicide: amount.New(17),
			accountExists:       true,
		},
		"AfterAmsterdam/RecreatedAccountRetainsBalance": {
			revision:            AMSTERDAM,
			balanceAfterSuicide: amount.New(17),
			accountExists:       true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			ctrl := gomock.NewController(t)
			mock := NewMockState(ctrl)
			db := CreateStateDBUsing(mock)
			mock.EXPECT().GetBalance(address1).Return(amount.New(12), nil)
			mock.EXPECT().GetStorage(address1, key1).Return(val1, nil)
			db.SetNonce(address1, 1)
			db.SetCode(address1, []byte{1, 2, 3})
			require.Equal(val1, db.GetState(address1, key1))
			require.True(db.Suicide(address1))
			db.CreateAccount(address1)
			require.False(db.HasSuicided(address1))
			require.Equal(amount.New(12), db.GetBalance(address1))
			db.AddBalance(address1, amount.New(5))
			db.EndTransaction(test.revision)
			// The account is no longer suicided at the end of the transaction, so its balance is not burned.
			require.Equal(test.accountExists, db.Exist(address1))
			require.Equal(test.balanceAfterSuicide, db.GetBalance(address1))
			require.Zero(db.GetNonce(address1))
			require.Empty(db.GetCode(address1))
			require.Equal(common.Value{}, db.GetState(address1, key1))
			require.True(db.HasEmptyStorage(address1))
		})
	}
}

func TestStateDB_SuicideAndCreateAccountAndSuicide_BurnsBalanceByRevision(t *testing.T) {
	tests := map[string]struct {
		revision            Revision
		balanceAfterSuicide amount.Amount
		accountExists       bool
	}{
		"BeforeAmsterdam/BalanceBurnedAndAccountDeleted": {
			revision:            AMSTERDAM - 1,
			balanceAfterSuicide: amount.New(),
			accountExists:       false,
		},
		"AfterAmsterdam/BalanceRetainedAndAccountExists": {
			revision:            AMSTERDAM,
			balanceAfterSuicide: amount.New(17),
			accountExists:       true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			ctrl := gomock.NewController(t)
			mock := NewMockState(ctrl)
			db := CreateStateDBUsing(mock)
			mock.EXPECT().GetBalance(address1).Return(amount.New(12), nil)
			mock.EXPECT().GetStorage(address1, key1).Return(val1, nil)
			db.SetNonce(address1, 1)
			db.SetCode(address1, []byte{1, 2, 3})
			require.Equal(val1, db.GetState(address1, key1))
			require.True(db.Suicide(address1))
			db.CreateAccount(address1)
			db.AddBalance(address1, amount.New(5))
			require.True(db.Suicide(address1))
			require.True(db.HasSuicided(address1))
			db.EndTransaction(test.revision)
			require.Equal(test.accountExists, db.Exist(address1))
			require.Equal(test.balanceAfterSuicide, db.GetBalance(address1))
			require.Zero(db.GetNonce(address1))
			require.Empty(db.GetCode(address1))
			require.Equal(common.Value{}, db.GetState(address1, key1))
			require.True(db.HasEmptyStorage(address1))
		})
	}
}
