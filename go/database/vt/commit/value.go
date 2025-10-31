// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package commit

import (
	"github.com/crate-crypto/go-ipa/banderwagon"
)

// Value is a numeric value the commitment can be made of. A commit is a
// commitment to a vector of these values. The value range is approximately
// 253 bits, which is just under 32 bytes. Thus, it is not possible to
// represent all 32 byte values, but it is possible to represent all 31 byte
// values.
//
// Background: The value is stored as a scalar in the Banderwagon curve field.
type Value struct {
	scalar banderwagon.Fr
}

// NewValue creates a new value from a uint64 value. Any 64-bit value is a valid
// value.
func NewValue(value uint64) Value {
	var scalar banderwagon.Fr
	scalar.SetUint64(value)
	return Value{scalar: scalar}
}

// NewValuesFromLittleEndianBytes creates a new value from a 32-byte little-endian
// byte slice. The value is expanded with zeros to 32 bytes if the input is
// shorter. Inputs longer than 32 bytes are truncated to 32 bytes.
func NewValueFromLittleEndianBytes(data []byte) Value {
	var padded [32]byte
	copy(padded[:], data)
	var scalar banderwagon.Fr
	scalar.SetBytesLE(padded[:])
	return Value{scalar: scalar}
}

// SetBit128 sets the 128th bit of the value to 1. This is a special bit used
// in the Verkle trie to indicate that the value is used. Thus, this operation
// is provided as a convenience method.
func (v *Value) SetBit128() {
	bytes := v.scalar.Bytes()
	bytes[15] = bytes[15] | 0x01
	v.scalar.SetBytes(bytes[:])
}

func (v *Value) Sub(other Value) *Value {
	var result Value
	result.scalar.Sub(&v.scalar, &other.scalar)
	return &result
}
