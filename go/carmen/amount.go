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

	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/holiman/uint256"
)

// Amount is a 256-bit unsigned integer used for token values like balances.
type Amount = amount.Amount

// NewAmount creates a new U256 Amount from up to 4 uint64 arguments. The
// arguments are given in the Big Endian order. No argument results in a value of zero.
// The constructor panics if more than 4 arguments are given.
func NewAmount(args ...uint64) Amount {
	return amount.New(args...)
}

// NewAmountFromUint256 creates a new amount from an uint256.
func NewAmountFromUint256(value *uint256.Int) Amount {
	return amount.NewFromUint256(value)
}

// NewAmountFromBytes creates a new Amount instance from up to 32 byte arguments.
// The arguments are given in the Big Endian order. No argument results in a
// value of zero. The constructor panics if more than 32 arguments are given.
func NewAmountFromBytes(bytes ...byte) Amount {
	return amount.NewFromBytes(bytes...)
}

// NewAmountFromBigInt creates a new Amount instance from a big.Int.
func NewAmountFromBigInt(b *big.Int) (Amount, error) {
	return amount.NewFromBigInt(b)
}
