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
	"bytes"
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
	for i := 0; i < numBlocks; i++ {
		update := common.Update{}
		for j := 0; j < numInsertsPerBlock; j++ {
			addr := common.Address{byte(j), byte(i), byte(i >> 8)}
			update.CreatedAccounts = append(update.CreatedAccounts, addr)
			update.Nonces = append(update.Nonces, common.NonceUpdate{Account: addr, Nonce: common.ToNonce(1)})
			update.Balances = append(update.Balances, common.BalanceUpdate{Account: addr, Balance: amount.New(uint64(i * j))})
		}
		if err := state.Apply(uint64(i), update); err != nil {
			t.Errorf("failed to apply block %d: %v", i, err)
		}
	}

	// read values
	for i := 0; i < numBlocks; i++ {
		for j := 0; j < numInsertsPerBlock; j++ {
			addr := common.Address{byte(j), byte(i), byte(i >> 8)}
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
			expectedBalance := amount.New(uint64(i * j))
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

}

func TestUpdate_And_Get_Code_Success(t *testing.T) {
	state, err := NewState(state.Parameters{})
	if err != nil {
		t.Fatalf("failed to create vt state: %v", err)
	}
	defer func() {
		if err := state.Close(); err != nil {
			t.Errorf("failed to close state: %v", err)
		}
	}()

	code := make([]byte, 4096)
	for i := 0; i < len(code); i++ {
		code[i] = byte(i)
	}

	const numOfCodes = 100
	expectedCodes := [numOfCodes][]byte{}
	update := common.Update{}
	for i := 0; i < numOfCodes; i++ {
		addr := common.Address{byte(i), byte(i >> 8)}
		update.Codes = append(update.Codes, common.CodeUpdate{
			Account: addr,
			Code:    code,
		})
		expectedCodes[i] = code
		code = append(code, byte(i))
	}

	if err := state.Apply(0, update); err != nil {
		t.Errorf("failed to apply update: %v", err)
	}

	// read codes
	for i := 0; i < numOfCodes; i++ {
		addr := common.Address{byte(i), byte(i >> 8)}

		code, err := state.GetCode(addr)
		if err != nil {
			t.Fatalf("failed to get code for account %s: %v", addr, err)
		}

		if got, want := code, expectedCodes[i]; !bytes.Equal(got, want) {
			t.Errorf("unexpected code for account %s: got %v, want %v", addr, got, want)
		}

		codeHash, err := state.GetCodeHash(addr)
		if err != nil {
			t.Fatalf("failed to get code hash for account %s: %v", addr, err)
		}

		if got, want := codeHash, common.Keccak256(expectedCodes[i]); got != want {
			t.Errorf("unexpected code hash for account %s: got %s, want %s", addr, got, want)
		}

		codeLen, err := state.GetCodeSize(addr)
		if err != nil {
			t.Fatalf("failed to get code size for account %s: %v", addr, err)
		}

		if got, want := codeLen, len(expectedCodes[i]); got != want {
			t.Errorf("unexpected code size for account %s: got %d, want %d", addr, got, want)
		}
	}
}

func TestSet_And_Get_Storage_Success(t *testing.T) {
	state, err := NewState(state.Parameters{})
	if err != nil {
		t.Fatalf("failed to create vt state: %v", err)
	}
	defer func() {
		if err := state.Close(); err != nil {
			t.Errorf("failed to close state: %v", err)
		}
	}()

	update := common.Update{}
	const numOfKeys = 100
	const numOfAddresses = 100
	for i := 0; i < numOfAddresses; i++ {
		addr := common.Address{byte(i), byte(i >> 8)}
		for j := 0; j < numOfKeys; j++ {
			key := common.Key{byte(j), byte(j >> 8)}
			value := common.Value{byte(i + j), byte((i + j) >> 8)}
			update.Slots = append(update.Slots, common.SlotUpdate{
				Account: addr,
				Key:     key,
				Value:   value,
			})
		}
	}
	if err := state.Apply(0, update); err != nil {
		t.Errorf("failed to apply: %v", err)
	}

	// read values
	for i := 0; i < numOfAddresses; i++ {
		addr := common.Address{byte(i), byte(i >> 8)}
		for j := 0; j < numOfKeys; j++ {
			key := common.Key{byte(j), byte(j >> 8)}
			value, err := state.GetStorage(addr, key)
			if err != nil {
				t.Fatalf("failed to get storage for account %s: %v", addr, err)
			}

			if got, want := value, (common.Value{byte(i + j), byte((i + j) >> 8)}); got != want {
				t.Errorf("unexpected storage value for account %s, key %s: got %v, want %v", addr, key, got, want)
			}
		}
	}
}

func TestGetBalance_Account_Empty(t *testing.T) {
	state, err := NewState(state.Parameters{})
	if err != nil {
		t.Fatalf("failed to create vt state: %v", err)
	}
	defer func() {
		if err := state.Close(); err != nil {
			t.Errorf("failed to close state: %v", err)
		}
	}()

	balance, err := state.GetBalance(common.Address{})
	if err != nil {
		t.Fatalf("failed to get balance for empty account: %v", err)
	}

	if !balance.IsZero() {
		t.Errorf("expected zero balance for empty account, got %s", balance)
	}
}

func TestGetStorage_Empty(t *testing.T) {
	state, err := NewState(state.Parameters{})
	if err != nil {
		t.Fatalf("failed to create vt state: %v", err)
	}
	defer func() {
		if err := state.Close(); err != nil {
			t.Errorf("failed to close state: %v", err)
		}
	}()

	value, err := state.GetStorage(common.Address{}, common.Key{})
	if err != nil {
		t.Fatalf("failed to get storage for empty account: %v", err)
	}

	if got, want := value, (common.Value{}); got != want {
		t.Errorf("unexpected storage value for empty account: got %v, want %v", got, want)
	}
}

func TestGetHash_Is_Updated_Each_Block(t *testing.T) {
	state, err := NewState(state.Parameters{})
	if err != nil {
		t.Fatalf("failed to create vt state: %v", err)
	}
	defer func() {
		if err := state.Close(); err != nil {
			t.Errorf("failed to close state: %v", err)
		}
	}()

	var prevHash common.Hash

	const numBlocks = 10
	const numInsertsPerBlock = 100
	for i := 0; i < numBlocks; i++ {
		update := common.Update{}
		for j := 0; j < numInsertsPerBlock; j++ {
			addr := common.Address{byte(j), byte(i), byte(i >> 8)}
			update.CreatedAccounts = append(update.CreatedAccounts, addr)
			update.Nonces = append(update.Nonces, common.NonceUpdate{Account: addr, Nonce: common.ToNonce(1)})
			update.Balances = append(update.Balances, common.BalanceUpdate{Account: addr, Balance: amount.New(uint64(i * j))})
		}
		if err := state.Apply(uint64(i), update); err != nil {
			t.Errorf("failed to apply block %d: %v", i, err)
		}

		hash, err := state.GetHash()
		if err != nil {
			t.Fatalf("failed to get block %d: %v", i, err)
		}

		if hash == prevHash {
			t.Errorf("hash did not changed")
		}

		prevHash = hash
	}
}
