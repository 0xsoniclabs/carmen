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
	"math/rand"
	"testing"

	"github.com/Fantom-foundation/Carmen/go/common"
	"github.com/Fantom-foundation/Carmen/go/common/amount"
)

func TestMptState_TotalSupply_EmptyStateHasZeroSupply(t *testing.T) {
	state, err := OpenGoMemoryState(t.TempDir(), S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer state.Close()

	total, err := state.TotalSupply()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if amount.New() != total {
		t.Fatalf("want %v, got %v", amount.New(), total)
	}
}

func TestMptState_TotalSupply_SumsBalancesOfAllAccountsIgnoringStorage(t *testing.T) {
	state, err := OpenGoMemoryState(t.TempDir(), S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer state.Close()

	want := amount.New()
	for i := 0; i < 100; i++ {
		addr := common.Address{byte(i), byte(i >> 4), 1}
		balance := amount.New(uint64(i + 1))
		want = amount.Add(want, balance)
		if err := state.SetBalance(addr, balance); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if err := state.SetNonce(addr, common.ToNonce(1)); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if err := state.SetStorage(addr, common.Key{1}, common.Value{2}); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	total, err := state.TotalSupply()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want != total {
		t.Fatalf("want %v, got %v", want, total)
	}
}

func TestMptState_TotalSupply_OverflowIsReported(t *testing.T) {
	state, err := OpenGoMemoryState(t.TempDir(), S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer state.Close()

	if err := state.SetBalance(common.Address{1}, amount.Max()); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := state.SetBalance(common.Address{2}, amount.New(1)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, err = state.TotalSupply()
	if !errors.Is(err, errTotalSupplyOverflow) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestMptState_TotalSupply_IsMaintainedIncrementally(t *testing.T) {
	state, err := OpenGoMemoryState(t.TempDir(), S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer state.Close()

	// The first query computes and starts tracking the supply.
	total, err := state.TotalSupply()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if amount.New() != total {
		t.Fatalf("want %v, got %v", amount.New(), total)
	}

	_, known := state.trie.getTotalSupply()
	if !known {
		t.Fatal("expected known to be true")
	}

	rnd := rand.New(rand.NewSource(42))
	for block := 0; block < 50; block++ {
		update := common.Update{}
		for i := 0; i < 20; i++ {
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
		_, err := state.Apply(uint64(block), update)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		got, err := state.TotalSupply()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want, err := state.computeTotalSupply()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if want != got {
			t.Fatalf("block %d: want %v, got %v", block, want, got)
		}
	}
}

func TestMptState_TotalSupply_DeletingAccountReducesSupply(t *testing.T) {
	state, err := OpenGoMemoryState(t.TempDir(), S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer state.Close()

	_, err = state.TotalSupply()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := state.SetBalance(common.Address{1}, amount.New(10)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := state.SetBalance(common.Address{2}, amount.New(5)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := state.DeleteAccount(common.Address{1}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	total, err := state.TotalSupply()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if amount.New(5) != total {
		t.Fatalf("want %v, got %v", amount.New(5), total)
	}
}

func TestMptState_TotalSupply_IsPersisted(t *testing.T) {
	dir := t.TempDir()
	state, err := OpenGoFileState(dir, S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := state.SetBalance(common.Address{1}, amount.New(10)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, err = state.TotalSupply()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := state.SetBalance(common.Address{2}, amount.New(5)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := state.Close(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	state, err = OpenGoFileState(dir, S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer state.Close()

	// The value is available without traversing the trie.
	total, known := state.trie.getTotalSupply()
	if !known {
		t.Fatal("expected known to be true")
	}
	if amount.New(15) != total {
		t.Fatalf("want %v, got %v", amount.New(15), total)
	}
}

func TestMptState_TotalSupply_UnknownValueIsComputedForExistingState(t *testing.T) {
	dir := t.TempDir()
	state, err := OpenGoFileState(dir, S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Balances set before the supply is tracked.
	if err := state.SetBalance(common.Address{1}, amount.New(10)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := state.Close(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	state, err = OpenGoFileState(dir, S5LiveConfig, NodeCacheConfig{Capacity: 1024})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer state.Close()

	_, known := state.trie.getTotalSupply()
	if known {
		t.Fatal("expected known to be false")
	}

	if err := state.SetBalance(common.Address{2}, amount.New(5)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	total, err := state.TotalSupply()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if amount.New(15) != total {
		t.Fatalf("want %v, got %v", amount.New(15), total)
	}
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
			if known {
				t.Fatal("expected known to be false")
			}
		})
	}
}

func TestLiveTrie_AdjustTotalSupply_UnknownValueStaysUnknown(t *testing.T) {
	trie := &LiveTrie{}
	trie.adjustTotalSupply(amount.New(1), amount.New(2))
	_, known := trie.getTotalSupply()
	if known {
		t.Fatal("expected known to be false")
	}
}

func TestMetadata_TotalSupply_InvalidValueIsRejected(t *testing.T) {
	_, err := (&metadata{TotalSupply: "abc"}).getTotalSupply()
	if err == nil {
		t.Fatal("expected an error")
	}
}
