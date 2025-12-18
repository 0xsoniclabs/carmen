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

import (
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBitMap_SetAndGet(t *testing.T) {
	require := require.New(t)
	indices := []byte{0, 1, 2, 3, 4, 5, 63, 64, 127, 128, 191, 192, 255}

	var bm bitMap
	for i, index := range indices {
		for j := range indices {
			require.Equal(j < i, bm.get(indices[j]), "Before setting: i=%d,j=%d", i, j)
		}
		bm.set(index)
		for j := range indices {
			require.Equal(j <= i, bm.get(indices[j]), "After setting: i=%d,j=%d", i, j)
		}
	}
}

func TestBitMap_Any_TrueIfAnyFlagIsSet(t *testing.T) {
	require := require.New(t)
	indices := []byte{0, 1, 2, 3, 4, 5, 63, 64, 127, 128, 191, 192, 255}

	for _, index := range indices {
		var bm bitMap
		require.False(bm.any(), "Bitmap should be empty initially")
		bm.set(index)
		require.True(bm.any(), "Bitmap should report non-empty after setting index %d", index)
	}

	// also multiple bits are fine
	var bm bitMap
	bm.set(1)
	bm.set(128)
	require.True(bm.any())
}

func TestBitMap_Clear_RemovesAllBits(t *testing.T) {
	require := require.New(t)
	indices := []byte{0, 1, 2, 3, 4, 5, 63, 64, 127, 128, 191, 192, 255}

	var bm bitMap
	for _, index := range indices {
		bm.set(index)
	}
	for i := range 256 {
		require.Equal(
			slices.Contains(indices, byte(i)),
			bm.get(byte(i)),
		)
	}

	bm.clear()
	require.False(bm.any(), "Bitmap should be empty after clear")
	for i := range 256 {
		require.False(bm.get(byte(i)), "All bits should be cleared after clear")
	}
}

func TestBitMap_PopCount_CountsTheNumberOfOneBits(t *testing.T) {
	require := require.New(t)

	bm := bitMap{}
	for i := range 256 {
		require.Equal(i, bm.popCount(), "Pop count before setting index %d", i)
		bm.set(byte(i))
		require.Equal(i+1, bm.popCount(), "Pop count after setting index %d", i)
	}
}
