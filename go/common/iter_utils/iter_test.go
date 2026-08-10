// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package iter_utils

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestIter_Map2_ReturnsMappedSeq2(t *testing.T) {
	require := require.New(t)

	seq := func(yield func(int, string) bool) {
		pairs := []struct {
			key   int
			value string
		}{
			{1, "one"},
			{2, "two"},
			{3, "three"},
		}

		for _, pair := range pairs {
			if !yield(pair.key, pair.value) {
				return
			}
		}
	}
	mapFunc := func(k int, v string) (string, int) {
		return v, k * 10
	}

	mappedSeq := Map2(seq, mapFunc)

	results := make(map[string]int)
	mappedSeq(func(k string, v int) bool {
		results[k] = v
		return true // Continue iteration
	})

	// Define the expected results after mapping
	expectedResults := map[string]int{
		"one":   10,
		"two":   20,
		"three": 30,
	}

	require.Equal(expectedResults, results)
}

func TestIter_FromSliceWith_ReturnsSeq2(t *testing.T) {
	require := require.New(t)

	slice := []string{"a", "ab", "abc"}
	keyFunc := func(v string) int {
		return len(v) // Use the length of the string as the key
	}

	seq := FromSliceWith(slice, keyFunc)

	results := make(map[int]string)
	seq(func(k int, v string) bool {
		results[k] = v
		return true // Continue iteration
	})

	// Define the expected results after mapping
	expectedResults := map[int]string{
		1: "a",
		2: "ab",
		3: "abc",
	}

	require.Equal(expectedResults, results)
}
