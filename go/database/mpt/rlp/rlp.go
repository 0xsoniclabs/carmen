// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package rlp

import (
	"encoding/binary"
	"fmt"
	"math/big"

	"github.com/0xsoniclabs/carmen/go/common"
)

// The definition of the RLP encoding can be found here:
// https://ethereum.org/en/developers/docs/data-structures-and-encoding/rlp
//
// Based on Appendix B of https://ethereum.github.io/yellowpaper/paper.pdf
//
// Recursive-Length Prefix (RLP) serialization is based on a recursive
// structure definition of an `item`. An item is defined as
//   - a string of bytes
//   - a list of items
// Note the recursive definition in the second line. This recursive step
// allows arbitrarily nested structures to be encoded. This package provides
// RLP encoding support for Items and a few convenience utilities for encoding
// frequently utilized types.

// Item is an interface for everything that can be RLP encoded by this package.
type Item interface {
	// write writes the RLP encoding of this item to the given writer.
	write(writer) writer

	// getEncodedLength computes the encoded length of this item in bytes.
	getEncodedLength() int
}

// Encode is a convenience function for serializing an item structure.
func Encode(item Item) []byte {
	return EncodeInto(make([]byte, 0, 1024), item)
}

func EncodeInto(dst []byte, item Item) []byte {
	writer := writer(dst)
	return item.write(writer)
}

// Decode decodes an RLP stream into an item.
func Decode(rlp []byte) (Item, error) {
	item, consumed, err := decode(rlp)
	if consumed < uint64(len(rlp)) {
		// trailing bytes are considered as a corrupted RLP stream
		// with an exception to trailing zeros.
		// It is in particular for embedded nodes that are laid out in a fixed size
		// byte array with the payload from the start of the array padded with zeros.
		for _, b := range rlp[consumed:] {
			if b != 0 {
				return nil, fmt.Errorf("unexpected trailing data: %x", rlp[consumed:])
			}
		}
	}
	return item, err
}

// decode decodes an RLP stream into an item.
func decode(rlp []byte) (Item, uint64, error) {
	if len(rlp) == 0 {
		return nil, 0, fmt.Errorf("input RLP is empty")
	}

	// here it checks first byte of the RLP stream to determine the type of the item.
	// Based on the type, it decodes the type.
	l := rlp[0]
	if l < 0x80 { // single byte RLP
		return String{Str: rlp[0:1]}, 1, nil
	}

	if l >= 0x80 && l <= 0xb7 { // short string
		length := int(l - 0x80)
		if len(rlp) < length+1 {
			return nil, 0, fmt.Errorf("expected %d bytes, got: %d", length+1, len(rlp))
		}

		return String{Str: rlp[1 : length+1]}, uint64(length + 1), nil
	}

	if l > 0xb7 && l < 0xc0 { // long string
		bytesLength := uint64(l - 0xb7)
		length, err := readNumber(rlp[1:], byte(bytesLength))
		if err != nil {
			return nil, 0, err
		}

		offset := bytesLength + 1
		return String{Str: rlp[offset : offset+length]}, offset + length, nil
	}

	if l >= 0xc0 && l <= 0xf7 { // short list
		length := int(l - 0xc0)
		if len(rlp) < length+1 {
			return nil, 0, fmt.Errorf("expected %d bytes, got: %d", length+1, len(rlp))
		}

		items, err := decodeList(rlp[1 : length+1])
		return List{Items: items}, uint64(length + 1), err
	}

	// l > 0xf7 - long list
	bytesLength := uint64(l - 0xf7)
	length, err := readNumber(rlp[1:], byte(bytesLength))
	if err != nil {
		return nil, 0, err
	}
	offset := bytesLength + 1
	items, err := decodeList(rlp[offset : offset+length])
	return List{Items: items}, offset + length, err
}

// decodeList decodes a list of items from the given RLP stream.
// The function expects an RLP stream with possibly multiple items encoded
// while the prefix with the length is already cut out.
// The consumes chunks of input RLP by passing it to the decoder
// until the input is empty.
func decodeList(rlp []byte) ([]Item, error) {
	items := make([]Item, 0, 17)
	buf := rlp
	for len(buf) > 0 {
		item, offset, err := decode(buf)
		if err != nil {
			return nil, err
		}

		items = append(items, item)
		buf = buf[offset:]
	}

	return items, nil
}

// writer is a specialized writer for this package writing encoded RLP
// content in a pre-allocated buffer.
type writer []byte

func (w writer) Write(data []byte) writer {
	return append(w, data...)
}

func (w writer) Put(c byte) writer {
	return append(w, c)
}

// ----------------------------------------------------------------------------
//                           Core Item Types
// ----------------------------------------------------------------------------

// String is the atomic ground type of an RLP input structure representing a
// (potentially empty) string of bytes.
type String struct {
	Str []byte
}

func (s String) write(writer writer) writer {
	l := len(s.Str)
	// Single-element strings are encoded as a single byte if the
	// value is small enough.
	if l == 1 && s.Str[0] < 0x80 {
		return writer.Write(s.Str)
	}
	// For the rest, the length is encoded, followed by the string itself.
	writer = encodeLength(l, 0x80, writer)
	return writer.Write(s.Str)
}

func (s String) getEncodedLength() int {
	l := len(s.Str)
	if l == 1 && s.Str[0] < 0x80 {
		return 1
	}
	return l + getEncodedLengthLength(l)
}

// BigInt converts the RLP encoded string to a big.Int.
// It interprets the string as a big-endian integer and returns a new big.Int
// set to this value.
//
// Returns:
// A pointer to a big.Int that represents the value of the RLP encoded string.
func (s String) BigInt() *big.Int {
	return new(big.Int).SetBytes(s.Str)
}

// Uint64 attempts to convert the RLP encoded string to a uint64.
// It interprets the string as a big-endian integer and returns the resulting uint64.
//
// If the length of the string is greater than 8 (the maximum byte size of a uint64),
// an error is returned indicating that the string is too long to be converted.
//
// Returns:
// The uint64 representation of the RLP encoded string if the conversion is successful,
// and an error if the string is too long to be converted to a uint64.
func (s String) Uint64() (uint64, error) {
	if len(s.Str) > 8 {
		return 0, fmt.Errorf("string is too long to be converted to uint64: %d", len(s.Str))
	}
	return readNumber(s.Str, byte(len(s.Str)))
}

// Hash is a used specifically to hold a pointer to hash.
// Its usage is similar to rlp.String, but this type should be used for performance reasons.
// In particular, conversion of common.Hash to rlp.String requires conversion of array
// to slice, which executes runtime.convTSlice() many times.
// Especially on ARM architecture it was detected to take considerable runtime.
type Hash struct {
	Hash *common.Hash
}

func (s Hash) write(writer writer) writer {
	writer = encodeLength(32, 0x80, writer)
	return writer.Write(s.Hash[:])
}

func (s Hash) getEncodedLength() int {
	// 32 bytes of hash + one byte to store length
	return 32 + 1
}

// List composes a list of items into a new item to be serialized.
type List struct {
	Items []Item
}

func (l List) write(writer writer) writer {
	length := 0
	for i := 0; i < len(l.Items); i++ {
		length += l.Items[i].getEncodedLength()
	}
	writer = encodeLength(length, 0xc0, writer)
	for i := 0; i < len(l.Items); i++ {
		writer = l.Items[i].write(writer)
	}
	return writer
}

func (l List) getEncodedLength() int {
	sum := 0
	for _, item := range l.Items {
		sum += item.getEncodedLength()
	}
	return sum + getEncodedLengthLength(sum)
}

// encodeLength is utility function used by String and List structures to
// encode the length of the string or list in the output stream.
func encodeLength(length int, offset byte, writer writer) writer {
	if length < 56 {
		return writer.Put(offset + byte(length))
	}
	numBytesForLength := getNumBytes(uint64(length))
	writer = writer.Put(offset + 55 + numBytesForLength)
	for i := byte(0); i < numBytesForLength; i++ {
		writer = writer.Put(byte(length >> (8 * (numBytesForLength - i - 1))))
	}
	return writer
}

// getNumBytes computes the minimum number of bytes required to represent
// the given value in big-endian encoding.
func getNumBytes(value uint64) byte {
	if value == 0 {
		return 0
	}
	for res := byte(1); ; res++ {
		if value >>= 8; value == 0 {
			return res
		}
	}
}

func getEncodedLengthLength(length int) int {
	if length < 56 {
		return 1
	}
	return int(getNumBytes(uint64(length))) + 1
}

// Encoded allows for embedding an already RLP encoded data fragment in a new RLP encoding.
type Encoded struct {
	Data []byte
}

func (e Encoded) write(writer writer) writer {
	return writer.Write(e.Data)
}

func (e Encoded) getEncodedLength() int {
	return len(e.Data)
}

// ----------------------------------------------------------------------------
//                           Utility Item Types
// ----------------------------------------------------------------------------

// Uint64 is an Item encoding unsigned integers into RLP by interpreting them
// as a string of bytes. The bytes are derived from the integer value by
// encoding it in big-endian byte order and removing leading zero-bytes.
type Uint64 struct {
	Value uint64
}

func (u Uint64) write(writer writer) writer {
	// Uint64 values are encoded using their non-zero big-endian encoding suffix.
	if u.Value == 0 {
		return writer.Put(0x80)
	}
	var buffer = make([]byte, 8)
	binary.BigEndian.PutUint64(buffer, u.Value)
	for buffer[0] == 0 {
		buffer = buffer[1:]
	}
	return String{Str: buffer}.write(writer)
}

func (u Uint64) getEncodedLength() int {
	if u.Value < 0x80 {
		return 1
	}
	return 1 + int(getNumBytes(u.Value))
}

// BigInt is an Item encoding big.Int values into RLP by interpreting them
// as a string of bytes. The encoding schema is implemented analogous to the
// Uint64 encoder above.
type BigInt struct {
	Value *big.Int
}

func (i BigInt) write(writer writer) writer {
	// Based on: https://github.com/ethereum/go-ethereum/blob/v1.12.0/rlp/encbuffer.go#L152
	// Values that fit in 64 bit are encoded using the uint64 encoder.
	bitlen := i.Value.BitLen()
	if bitlen <= 64 {
		return Uint64{Value: i.Value.Uint64()}.write(writer)
	}
	// Integer is larger than 64 bits, encode from BigInt's Bits()
	// using big-endian order.
	const wordBytes = (32 << (uint64(^big.Word(0)) >> 63)) / 8
	length := ((bitlen + 7) & -8) >> 3
	index := length
	var buffer = make([]byte, length)
	for _, d := range i.Value.Bits() {
		for j := 0; j < wordBytes && index > 0; j++ {
			index--
			buffer[index] = byte(d)
			d >>= 8
		}
	}
	writer = encodeLength(length, 0x80, writer)
	return writer.Write(buffer)
}

func (i BigInt) getEncodedLength() int {
	bitlen := i.Value.BitLen()
	if bitlen <= 64 {
		return Uint64{Value: i.Value.Uint64()}.getEncodedLength()
	}
	length := ((bitlen + 7) & -8) >> 3
	return getEncodedLengthLength(length) + length
}

// readNumber interprets the first 'slen' bytes of 'b' as a big-endian integer.
// It returns the resulting integer and any error encountered.
//
// The function expects 'slen' to be the size of the integer to be read from 'b'.
// If 'slen' is greater than the length of 'b', an error is returned.
//
// The function supports reading integers of up to 8 bytes (64 bits).
//
// Parameters:
// b:    The byte slice from which to read the integer.
// slen: The size of the integer to be read from 'b'.
//
// Returns:
// The integer read from 'b' and an error if 'slen' is greater than the length of 'b'.
func readNumber(b []byte, slen byte) (uint64, error) {
	if int(slen) > len(b) {
		return 0, fmt.Errorf("expected %d bytes, got: %d", slen, len(b))
	}
	var s uint64
	for i := byte(0); i < slen; i++ {
		s = s<<8 | uint64(b[i])
	}
	return s, nil
}
