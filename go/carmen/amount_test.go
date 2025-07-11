// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package carmen

import (
	"math/big"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/holiman/uint256"
)

func TestAmount_NewAmount(t *testing.T) {
	testCases := []struct {
		args []uint64
	}{
		{args: []uint64{}},
		{args: []uint64{1}},
		{args: []uint64{1, 2}},
		{args: []uint64{1, 2, 3}},
		{args: []uint64{1, 2, 3, 4}},
	}
	for _, tc := range testCases {
		if want, got := amount.New(tc.args...), NewAmount(tc.args...); want != got {
			t.Errorf("New(%v): expected %v, got %v", tc.args, want, got)
		}
	}
}

func TestAmount_NewAmountFromUint256(t *testing.T) {
	testCases := []struct {
		arg *uint256.Int
	}{
		{arg: uint256.NewInt(0)},
		{arg: uint256.NewInt(1)},
		{arg: uint256.NewInt(256)},
		{arg: new(uint256.Int).Lsh(uint256.NewInt(1), 64)},
		{arg: new(uint256.Int).Lsh(uint256.NewInt(1), 128)},
		{arg: new(uint256.Int).Lsh(uint256.NewInt(1), 192)},
	}
	for _, tc := range testCases {
		if want, got := amount.NewFromUint256(tc.arg), NewAmountFromUint256(tc.arg); want != got {
			t.Errorf("NewFromUint256(%v): expected %v, got %v", tc.arg, want, got)
		}
	}
}

func TestAmount_NewAmountFromBytes(t *testing.T) {
	testCases := []struct {
		args []byte
	}{
		{args: []byte{}},
		{args: []byte{1}},
		{args: []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
	}
	for _, tc := range testCases {
		if want, got := amount.NewFromBytes(tc.args...), NewAmountFromBytes(tc.args...); want != got {
			t.Errorf("NewFromBytes(%v): expected %v, got %v", tc.args, want, got)
		}
	}
}

func TestAmount_NewAmountFromBigInt(t *testing.T) {
	number := big.NewInt(100)
	want, err := amount.NewFromBigInt(number)
	if err != nil {
		t.Errorf("NewFromBigInt(%v): unexpected error: %v", number, err)
	}

	got, err := NewAmountFromBigInt(number)
	if err != nil {
		t.Errorf("NewFromBigInt(%v): unexpected error: %v", number, err)
	}

	if want != got {
		t.Errorf("NewFromBigInt(%v): expected %v, got %v", number, want, got)
	}
}
