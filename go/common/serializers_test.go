// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package common_test

import (
	"slices"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"golang.org/x/exp/rand"
)

func TestAddressSerializer(t *testing.T) {
	var s common.AddressSerializer
	var _ common.Serializer[common.Address] = s
}

func TestKeySerializer(t *testing.T) {
	var s common.KeySerializer
	var _ common.Serializer[common.Key] = s
}

func TestValueSerializer(t *testing.T) {
	var s common.ValueSerializer
	var _ common.Serializer[common.Value] = s
}

func TestHashSerializer(t *testing.T) {
	var s common.HashSerializer
	var _ common.Serializer[common.Hash] = s
}

func TestNonceSerializer(t *testing.T) {
	var s common.NonceSerializer
	var _ common.Serializer[common.Nonce] = s
}
func TestAmountSerializer(t *testing.T) {
	var s common.AmountSerializer
	var _ common.Serializer[amount.Amount] = s
}

func TestSerializers(t *testing.T) {
	loops := rand.Intn(10_000)

	t.Run("TestSerializers_Address", func(t *testing.T) {
		var a common.Address
		const size = 20
		for i := 1; i < loops; i++ {
			a[i%size]++
		}
		testSerializer[common.Address](t, a, size, common.AddressSerializer{})
	})

	t.Run("TestSerializers_Key", func(t *testing.T) {
		var a common.Key
		const size = 32
		for i := 1; i < loops; i++ {
			a[i%size]++
		}
		testSerializer[common.Key](t, a, size, common.KeySerializer{})
	})

	t.Run("TestSerializers_Value", func(t *testing.T) {
		var a common.Value
		const size = 32
		for i := 1; i < loops; i++ {
			a[i%size]++
		}
		testSerializer[common.Value](t, a, size, common.ValueSerializer{})
	})

	t.Run("TestSerializers_Hash", func(t *testing.T) {
		var a common.Hash
		const size = 32
		for i := 1; i < loops; i++ {
			a[i%size]++
		}
		testSerializer[common.Hash](t, a, size, common.HashSerializer{})
	})

	t.Run("TestSerializers_AccountState", func(t *testing.T) {
		var a common.AccountState = 253
		testSerializer[common.AccountState](t, a, 1, common.AccountStateSerializer{})
	})

	t.Run("TestSerializers_Amount", func(t *testing.T) {
		var a [amount.BytesLength]byte
		for i := 1; i < loops; i++ {
			a[i%amount.BytesLength]++
		}
		testSerializer[amount.Amount](t, amount.NewFromBytes(a[:]...), amount.BytesLength, common.AmountSerializer{})
	})

	t.Run("TestSerializers_Nonce", func(t *testing.T) {
		var a common.Nonce
		const size = common.NonceSize
		for i := 1; i < loops; i++ {
			a[i%size]++
		}
		testSerializer[common.Nonce](t, a, size, common.NonceSerializer{})
	})

	t.Run("TestSerializers_Identifier32", func(t *testing.T) {
		var a = uint32(loops)
		const size = 4
		testSerializer[uint32](t, a, size, common.Identifier32Serializer{})
	})

	t.Run("TestSerializers_Identifier64", func(t *testing.T) {
		var a = uint64(loops)
		const size = 8
		testSerializer[uint64](t, a, size, common.Identifier64Serializer{})
	})
}

func testSerializer[T comparable](t *testing.T, val T, size int, serializer common.Serializer[T]) {
	t.Helper()
	serialized := serializer.ToBytes(val)

	if got, want := serializer.FromBytes(serialized), val; got != want {
		t.Errorf("recovered value do not match: %v != %v", got, want)
	}

	got := make([]byte, size)
	serializer.CopyBytes(val, got)
	if !slices.Equal(got, serialized) {
		t.Errorf("recovered value do not match: %v != %v", got, serialized)
	}

	if got, want := serializer.Size(), size; got != want {
		t.Errorf("sizes do not match: %v != %v", got, want)
	}
}
