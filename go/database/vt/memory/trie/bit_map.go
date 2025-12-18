// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package trie

import "math/bits"

// bitMap is a simple bitmap implementation for 256 bits.
type bitMap [256 / 64]uint64

// get returns true if the bit at the specified index is set.
func (b *bitMap) get(index byte) bool {
	return (b[index/64] & (1 << (index % 64))) != 0
}

// set sets the bit at the specified index.
func (b *bitMap) set(index byte) {
	b[index/64] |= 1 << (index % 64)
}

// clear clears all bits in the bitmap.
func (b *bitMap) clear() {
	b[0] = 0
	b[1] = 0
	b[2] = 0
	b[3] = 0
}

// any returns true if any bit in the bitmap is set.
func (b *bitMap) any() bool {
	return b[0]|b[1]|b[2]|b[3] != 0
}

// popCount returns the number of bits set in the bitmap.
func (b *bitMap) popCount() int {
	count := 0
	for _, v := range b {
		count += bits.OnesCount64(v)
	}
	return count
}
