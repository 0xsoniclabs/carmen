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
	"fmt"
	"maps"
	"slices"
	"strconv"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common/result"
	"github.com/stretchr/testify/require"
)

func TestIter_Unpack_ReturnsKeyAndValue(t *testing.T) {
	require := require.New(t)

	key, value := Pair[int, string]{Key: 1, Value: "one"}.Unpack()

	require.Equal(1, key)
	require.Equal("one", value)
}

func TestIter_OkSeq_YieldsAllValuesAsSuccesses(t *testing.T) {
	require := require.New(t)

	seq := OkSeq(slices.Values([]string{"a", "b"}))

	values, seqErr := Unwrap(seq)
	require.Equal([]string{"a", "b"}, slices.Collect(values))
	require.NoError(seqErr())
}

func TestIter_OkSeq2_YieldsAllPairsAsSuccesses(t *testing.T) {
	require := require.New(t)

	seq := OkSeq2(Enumerate([]string{"a", "ab"}))

	pairs, seqErr := Unwrap2(seq)
	require.Equal(map[uint64]string{0: "a", 1: "ab"}, maps.Collect(pairs))
	require.NoError(seqErr())
}

func TestIter_Unwrap_StopsAtFirstFailureAndReportsIt(t *testing.T) {
	require := require.New(t)

	injected := fmt.Errorf("injected failure")
	seq, seqErr := Unwrap(func(yield func(result.Result[string]) bool) {
		if !yield(result.Ok("one")) {
			return
		}
		if !yield(result.Err[string](injected)) {
			return
		}
		yield(result.Ok("three"))
	})

	require.Equal([]string{"one"}, slices.Collect(seq))
	require.ErrorIs(seqErr(), injected)
}

func TestIter_Unwrap2_YieldsAllPairsWithoutError(t *testing.T) {
	require := require.New(t)

	seq, seqErr := Unwrap2(OkSeq2(Enumerate([]string{"a", "b"})))

	require.Equal(map[uint64]string{0: "a", 1: "b"}, maps.Collect(seq))
	require.NoError(seqErr())
}

func TestIter_Unwrap2_StopsAtFirstFailureAndReportsIt(t *testing.T) {
	require := require.New(t)

	injected := fmt.Errorf("injected failure")
	seq, seqErr := Unwrap2(func(yield func(result.Result[Pair[int, string]]) bool) {
		if !yield(result.Ok(Pair[int, string]{Key: 1, Value: "one"})) {
			return
		}
		if !yield(result.Err[Pair[int, string]](injected)) {
			return
		}
		yield(result.Ok(Pair[int, string]{Key: 3, Value: "three"}))
	})

	results := maps.Collect(seq)

	require.ErrorIs(seqErr(), injected)
	require.Equal(map[int]string{1: "one"}, results)
}

func TestIter_Map_ReturnsMappedSeq(t *testing.T) {
	input := []int{1, 2, 3}
	expected := []string{"1", "2", "3"}

	mapFunc := func(v int) string {
		return strconv.Itoa(v)
	}
	seq := Map(slices.Values(input), mapFunc)
	require.Equal(t, expected, slices.Collect(seq))
}

func TestIter_MapOk_MapsSuccessfulValues(t *testing.T) {
	require := require.New(t)

	seq := MapOk(OkSeq(slices.Values([]int{1, 2, 3})), strconv.Itoa)

	values, seqErr := Unwrap(seq)
	require.Equal([]string{"1", "2", "3"}, slices.Collect(values))
	require.NoError(seqErr())
}

func TestIter_MapOk_ForwardsFailures(t *testing.T) {
	require := require.New(t)

	injected := fmt.Errorf("injected failure")
	seq := MapOk(
		func(yield func(result.Result[int]) bool) {
			yield(result.Err[int](injected))
		},
		func(v int) string {
			t.Fatal("the mapping function must not be applied to a failure")
			return ""
		},
	)

	for r := range seq {
		_, err := r.Get()
		require.ErrorIs(err, injected)
	}
}

func TestIter_MapOk2_MapsSuccessfulPairs(t *testing.T) {
	require := require.New(t)

	seq := MapOk2(OkSeq2(Enumerate([]string{"a", "ab"})), func(k uint64, v string) (string, int) {
		return v, int(k) + len(v)
	})

	pairs, seqErr := Unwrap2(seq)
	require.Equal(map[string]int{"a": 1, "ab": 3}, maps.Collect(pairs))
	require.NoError(seqErr())
}

func TestIter_MapOk2_ForwardsFailures(t *testing.T) {
	require := require.New(t)

	injected := fmt.Errorf("injected failure")
	seq := MapOk2(
		func(yield func(result.Result[Pair[int, string]]) bool) {
			yield(result.Err[Pair[int, string]](injected))
		},
		func(k int, v string) (int, string) {
			t.Fatal("the mapping function must not be applied to a failure")
			return k, v
		},
	)

	for r := range seq {
		_, err := r.Get()
		require.ErrorIs(err, injected)
	}
}

func TestIter_DropKeysOk2_YieldsValuesWithoutKeys(t *testing.T) {
	require := require.New(t)

	seq := DropKeysOk2(OkSeq2(Enumerate([]string{"a", "b"})))

	values, seqErr := Unwrap(seq)
	require.Equal([]string{"a", "b"}, slices.Collect(values))
	require.NoError(seqErr())
}

func TestIter_DropKeysOk2_ForwardsFailures(t *testing.T) {
	require := require.New(t)

	injected := fmt.Errorf("injected failure")
	seq := DropKeysOk2(func(yield func(result.Result[Pair[int, string]]) bool) {
		yield(result.Err[Pair[int, string]](injected))
	})

	for r := range seq {
		_, err := r.Get()
		require.ErrorIs(err, injected)
	}
}

func TestIter_Enumerate_PairsValuesWithTheirIndices(t *testing.T) {
	require := require.New(t)

	seq := Enumerate([]string{"a", "b", "c"})

	require.Equal(map[uint64]string{0: "a", 1: "b", 2: "c"}, maps.Collect(seq))
}
