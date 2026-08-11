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
	"maps"
	"slices"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestIter_Map_ReturnsMappedSeq(t *testing.T) {
	input := []int{1, 2, 3}
	expected := []string{"1", "2", "3"}

	mapFunc := func(v int) string {
		return strconv.Itoa(v)
	}
	seq := Map(slices.Values(input), mapFunc)
	require.Equal(t, expected, slices.Collect(seq))
}

func TestIter_DropKeys_ReturnsSeqOfValues(t *testing.T) {
	input := map[int]string{1: "A", 2: "B", 3: "C"}
	expected := []string{"A", "B", "C"}

	seq := DropKeys(maps.All(input))
	got := slices.Collect(seq)
	slices.Sort(got)
	require.Equal(t, expected, got)
}

func TestIter_FromSliceWith_ReturnsSeq(t *testing.T) {
	require := require.New(t)

	input := []string{"a", "ab", "abc"}

	seq := FromSlice(input)
	require.Equal(input, slices.Collect(seq))
}
