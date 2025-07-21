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
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestValue_DefaultIsZero(t *testing.T) {
	require := require.New(t)
	zero := NewValue(0)
	require.Equal(zero, Value{})
}

func TestNewValue_ProducesValueMatchingInput(t *testing.T) {
	tests := []uint64{0, 1, 2, 42, math.MaxInt64, math.MaxInt64 + 1, math.MaxUint64}

	for _, value := range tests {
		t.Run(fmt.Sprintf("%d", value), func(t *testing.T) {
			require := require.New(t)
			v := NewValue(value)
			have := v.scalar.Bytes()
			want := [32]byte{}
			binary.BigEndian.PutUint64(want[24:], value)
			require.Equal(want, have)
		})
	}
}

func TestNewValueFromBytes_UsesLittleEndianAndRightZeroPadding(t *testing.T) {
	tests := map[string]struct {
		input []byte
		want  uint64
	}{
		"empty": {
			input: []byte{},
			want:  0,
		},
		"one byte": {
			input: []byte{0x01},
			want:  1,
		},
		"two bytes": {
			input: []byte{0x01, 0x02},
			want:  0x0201,
		},
		"three bytes": {
			input: []byte{0x01, 0x02, 0x03},
			want:  0x030201,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			have := NewValueFromLittleEndianBytes(test.input)
			want := NewValue(test.want)
			require.Equal(want, have)
		})
	}
}

func TestValue_SetBit128(t *testing.T) {
	require := require.New(t)

	// - starting from zero
	v := NewValue(0)
	require.Equal([32]byte{}, v.scalar.Bytes(), "initial value should be zero")
	v.SetBit128()
	require.Equal([32]byte{15: 1}, v.scalar.Bytes(), "128th bit should be set")
	v.SetBit128()
	require.Equal([32]byte{15: 1}, v.scalar.Bytes(), "128th bit should be set")

	// - starting from a non-zero value
	v = NewValue(uint64(0xf1f2f3f4f5f6f7f8))
	require.Equal([32]byte{24: 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8}, v.scalar.Bytes())
	v.SetBit128()
	require.Equal([32]byte{15: 1, 24: 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8},
		v.scalar.Bytes(), "128th bit should be set without changing other bits")
	v.SetBit128()
	require.Equal([32]byte{15: 1, 24: 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8},
		v.scalar.Bytes(), "128th bit should be set without changing other bits")
}
