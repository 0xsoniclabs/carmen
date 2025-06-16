// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package geth

import (
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/state"
	"testing"
)

func TestCreateAccounts_Many_Updates_Success(t *testing.T) {
	state, err := NewState(state.Parameters{})
	if err != nil {
		t.Fatalf("failed to create vt state: %v", err)
	}
	defer func() {
		if err := state.Close(); err != nil {
			t.Errorf("failed to close state: %v", err)
		}
	}()

	const numBlocks = 10
	const numInsertsPerBlock = 100
	var counter int
	for i := 0; i < numBlocks; i++ {
		update := common.Update{}
		update.CreatedAccounts = make([]common.Address, 0, numInsertsPerBlock)
		for j := 0; j < numInsertsPerBlock; j++ {
			addr := common.Address{byte(counter), byte(counter >> 8), byte(counter >> 16), byte(counter >> 24), byte(counter >> 32)}
			update.CreatedAccounts = append(update.CreatedAccounts, addr)
			update.Nonces = append(update.Nonces, common.NonceUpdate{Account: addr, Nonce: common.ToNonce(1)})
			update.Balances = append(update.Balances, common.BalanceUpdate{Account: addr, Balance: amount.New(uint64(counter))})
			counter++
		}
		if err := state.Apply(uint64(i), update); err != nil {
			t.Errorf("failed to apply block %d: %v", i, err)
		}
	}

	// read values
	for i := 0; i < numBlocks*numInsertsPerBlock; i++ {
		addr := common.Address{byte(i), byte(i >> 8), byte(i >> 16), byte(i >> 24), byte(i >> 32)}
		exists, err := state.Exists(addr)
		if err != nil {
			t.Errorf("failed to check existence of account %s: %v", addr, err)
		}
		if !exists {
			t.Fatalf("account %s should exist but does not", addr)
		}

		balance, err := state.GetBalance(addr)
		if err != nil {
			t.Errorf("failed to get balance for account %s: %v", addr, err)
		}
		expectedBalance := amount.New(uint64(i))
		if balance.ToBig().Cmp(expectedBalance.ToBig()) != 0 {
			t.Errorf("unexpected balance for account %s: got %s, want %s", addr, balance, expectedBalance)
		}

		nonce, err := state.GetNonce(addr)
		if err != nil {
			t.Errorf("failed to get nonce for account %s: %v", addr, err)
		}
		if nonce != common.ToNonce(1) {
			t.Errorf("unexpected nonce for account %s: got %d, want %d", addr, nonce, common.ToNonce(1))
		}
	}
}
