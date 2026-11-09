// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package state_test

import (
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/stretchr/testify/require"
)

func TestCarmen_Suicide_EndTransaction(t *testing.T) {
	balance := amount.New(12)
	tests := map[string]struct {
		revision            state.Revision
		balanceAfterSuicide amount.Amount
		accountExists       bool
		// equivalentState produces the state expected after the suicide.
		equivalentState func(state.StateDB)
	}{
		"BeforeAmsterdam/BalanceIsBurnt": {
			revision:            state.AMSTERDAM - 1, // Latest before
			balanceAfterSuicide: amount.New(),
			accountExists:       false,
			equivalentState:     func(state.StateDB) {},
		},
		"AfterAmsterdam/BalanceIsRetained": {
			revision:            state.AMSTERDAM,
			balanceAfterSuicide: balance,
			accountExists:       true,
			equivalentState: func(s state.StateDB) {
				// CreateAccount sets empty code, which is required.
				s.CreateAccount(address1)
				s.AddBalance(address1, balance)
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			for _, config := range initStates() {
				t.Run(config.name(), func(t *testing.T) {
					t.Parallel()
					require := require.New(t)
					s, err := config.createState(t.TempDir())
					require.NoError(err)
					db := state.CreateStateDBUsing(s)
					defer func() {
						require.NoError(db.Close())
					}()

					db.BeginBlock()
					db.CreateAccount(address1)
					db.CreateContract(address1)
					db.SetNonce(address1, 1)
					db.SetCode(address1, []byte{1, 2, 3})
					db.SetState(address1, key1, val1)
					db.AddBalance(address1, balance)
					require.True(db.Suicide(address1))
					require.Equal(balance, db.GetBalance(address1))

					db.EndTransaction(test.revision)
					endAndCommitBlock(t, db, 0)

					require.Equal(test.accountExists, db.Exist(address1))
					require.Equal(test.balanceAfterSuicide, db.GetBalance(address1))
					require.Zero(db.GetNonce(address1))
					require.Empty(db.GetCode(address1))
					require.Equal(common.Value{}, db.GetState(address1, key1))
					// Verkle tries keep cleared storage slots, so the hash differs in schema 6.
					if config.config.Schema != 6 {
						require.Equal(executeOpOnCleanStateDB(t, config, test.equivalentState), db.GetHash())
					}
				})
			}
		})
	}
}
