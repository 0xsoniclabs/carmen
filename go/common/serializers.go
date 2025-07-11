// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package common

import (
	"encoding/binary"

	"github.com/0xsoniclabs/carmen/go/common/amount"
)

// AddressSerializer is a Serializer of the Address type
type AddressSerializer struct{}

func (a AddressSerializer) ToBytes(address Address) []byte {
	return address[:]
}
func (a AddressSerializer) CopyBytes(address Address, out []byte) {
	copy(out, address[:])
}
func (a AddressSerializer) FromBytes(bytes []byte) Address {
	return *(*Address)(bytes)
}

func (a AddressSerializer) Size() int {
	return AddressSize
}

// KeySerializer is a Serializer of the Key type
type KeySerializer struct{}

func (a KeySerializer) ToBytes(key Key) []byte {
	return key[:]
}
func (a KeySerializer) CopyBytes(key Key, out []byte) {
	copy(out, key[:])
}
func (a KeySerializer) FromBytes(bytes []byte) Key {
	return *(*Key)(bytes)
}
func (a KeySerializer) Size() int {
	return KeySize
}

// ValueSerializer is a Serializer of the Value type
type ValueSerializer struct{}

func (a ValueSerializer) ToBytes(value Value) []byte {
	return value[:]
}
func (a ValueSerializer) CopyBytes(value Value, out []byte) {
	copy(out, value[:])
}
func (a ValueSerializer) FromBytes(bytes []byte) Value {
	return *(*Value)(bytes)
}
func (a ValueSerializer) Size() int {
	return ValueSize
}

// HashSerializer is a Serializer of the Hash type
type HashSerializer struct{}

func (a HashSerializer) ToBytes(hash Hash) []byte {
	return hash[:]
}
func (a HashSerializer) CopyBytes(hash Hash, out []byte) {
	copy(out, hash[:])
}
func (a HashSerializer) FromBytes(bytes []byte) Hash {
	return *(*Hash)(bytes)
}
func (a HashSerializer) Size() int {
	return HashSize
}

// AccountStateSerializer is a Serializer of the AccountState type
type AccountStateSerializer struct{}

func (a AccountStateSerializer) ToBytes(value AccountState) []byte {
	return []byte{byte(value)}
}
func (a AccountStateSerializer) CopyBytes(value AccountState, out []byte) {
	out[0] = byte(value)
}
func (a AccountStateSerializer) FromBytes(bytes []byte) AccountState {
	return AccountState(bytes[0])
}
func (a AccountStateSerializer) Size() int {
	return 1
}

// AmountSerializer is a Serializer of the amount.Amount type
type AmountSerializer struct{}

func (a AmountSerializer) ToBytes(value amount.Amount) []byte {
	b := value.Bytes32()
	return b[:]
}
func (a AmountSerializer) CopyBytes(value amount.Amount, out []byte) {
	b := value.Bytes32()
	copy(out, b[:])
}
func (a AmountSerializer) FromBytes(bytes []byte) amount.Amount {
	return amount.NewFromBytes(bytes...)
}
func (a AmountSerializer) Size() int {
	return amount.BytesLength
}

// NonceSerializer is a Serializer of the Nonce type
type NonceSerializer struct{}

func (a NonceSerializer) ToBytes(value Nonce) []byte {
	return value[:]
}
func (a NonceSerializer) CopyBytes(value Nonce, out []byte) {
	copy(out, value[:])
}
func (a NonceSerializer) FromBytes(bytes []byte) Nonce {
	return *(*Nonce)(bytes)
}
func (a NonceSerializer) Size() int {
	return NonceSize
}

// Identifier32Serializer is a Serializer of the uint32 Identifier type
type Identifier32Serializer struct{}

func (a Identifier32Serializer) ToBytes(value uint32) []byte {
	return binary.BigEndian.AppendUint32([]byte{}, value)
}
func (a Identifier32Serializer) CopyBytes(value uint32, out []byte) {
	binary.BigEndian.PutUint32(out, value)
}
func (a Identifier32Serializer) FromBytes(bytes []byte) uint32 {
	return binary.BigEndian.Uint32(bytes)
}
func (a Identifier32Serializer) Size() int {
	return 4
}

// Identifier64Serializer is a Serializer of the uint64 Identifier type
type Identifier64Serializer struct{}

func (a Identifier64Serializer) ToBytes(value uint64) []byte {
	return binary.BigEndian.AppendUint64([]byte{}, value)
}
func (a Identifier64Serializer) CopyBytes(value uint64, out []byte) {
	binary.BigEndian.PutUint64(out, value)
}
func (a Identifier64Serializer) FromBytes(bytes []byte) uint64 {
	return binary.BigEndian.Uint64(bytes)
}
func (a Identifier64Serializer) Size() int {
	return 8
}
