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
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

func TestLruCache_NewLruCache_RoundsUpCapacityIfLessThanTwo(t *testing.T) {
	tests := map[string]struct {
		capacity int
	}{
		"zero capacity":     {capacity: 0},
		"capacity of one":   {capacity: 1},
		"negative capacity": {capacity: -100},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			cache := NewLruCache[int, int](test.capacity)
			require.Equal(2, cache.capacity)
		})
	}
}

func TestLruCache_Get_MovesEntryToHead(t *testing.T) {
	require := require.New(t)
	cache := NewLruCache[int, int](3)
	cache.Set(1, 11)
	cache.Set(2, 22)
	cache.Set(3, 33)

	value, exists := cache.Get(1)
	require.True(exists)
	require.Equal(11, value)
	requireLruOrder(require, cache, []int{1, 3, 2})
}

func TestLruCache_Set_EvictsLeastRecentlyUsedWhenCapacityExceeded(t *testing.T) {
	require := require.New(t)
	cache := NewLruCache[int, int](3)

	cache.Set(1, 11)
	cache.Set(2, 22)
	_, _, evicted := cache.Set(3, 33)
	require.False(evicted)

	_, exists := cache.Get(1) // refresh 1 so that 2 becomes the least recently used
	require.True(exists)

	evictedKey, evictedValue, evicted := cache.Set(4, 44)
	require.True(evicted)
	require.Equal(2, evictedKey)
	require.Equal(22, evictedValue)

	_, exists = cache.Get(2)
	require.False(exists)
	requireLruOrder(require, cache, []int{4, 1, 3})
}

func TestLruCache_Set_UpdatesExistingValueAndMovesEntryToHead(t *testing.T) {
	require := require.New(t)
	cache := NewLruCache[int, int](3)
	cache.Set(1, 11)
	cache.Set(2, 22)
	cache.Set(3, 33)

	_, _, evicted := cache.Set(2, 222)
	require.False(evicted)

	value, exists := cache.Get(2)
	require.True(exists)
	require.Equal(222, value)
	requireLruOrder(require, cache, []int{2, 3, 1})
}

func TestLruCache_GetOrSet_ReturnsPresentValueOrSetsNew(t *testing.T) {
	require := require.New(t)
	cache := NewLruCache[int, int](2)

	_, present, _, _, evicted := cache.GetOrSet(1, 11)
	require.False(present)
	require.False(evicted)

	current, present, _, _, _ := cache.GetOrSet(1, 12)
	require.True(present)
	require.Equal(11, current)

	cache.GetOrSet(2, 22)
	_, present, evictedKey, evictedValue, evicted := cache.GetOrSet(3, 33)
	require.False(present)
	require.True(evicted)
	require.Equal(1, evictedKey)
	require.Equal(11, evictedValue)
}

func TestLruCache_Remove_KeepsListConsistent(t *testing.T) {
	init := func() *LruCache[int, int] {
		cache := NewLruCache[int, int](3)
		cache.Set(1, 11)
		cache.Set(2, 22)
		cache.Set(3, 33)
		return cache
	}

	tests := map[string]struct {
		key        int
		wantValue  int
		wantExists bool
		wantOrder  []int
	}{
		"head":        {key: 3, wantValue: 33, wantExists: true, wantOrder: []int{2, 1}},
		"middle":      {key: 2, wantValue: 22, wantExists: true, wantOrder: []int{3, 1}},
		"tail":        {key: 1, wantValue: 11, wantExists: true, wantOrder: []int{3, 2}},
		"missing key": {key: 4, wantOrder: []int{3, 2, 1}},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			cache := init()

			value, exists := cache.Remove(test.key)
			require.Equal(test.wantExists, exists)
			require.Equal(test.wantValue, value)
			requireLruOrder(require, cache, test.wantOrder)
		})
	}
}

func TestLruCache_Remove_EmptiesSingleEntryCache(t *testing.T) {
	require := require.New(t)
	cache := NewLruCache[int, int](3)
	cache.Set(1, 11)

	value, exists := cache.Remove(1)
	require.True(exists)
	require.Equal(11, value)
	requireLruOrder(require, cache, nil)
}

func TestLruCache_Clear_RemovesAllElementsAndKeepsCacheUsable(t *testing.T) {
	require := require.New(t)
	cache := NewLruCache[int, int](4)
	cache.Set(1, 11)
	cache.Set(2, 22)
	cache.Set(3, 33)
	cache.Set(4, 44)

	cache.Clear()
	requireLruOrder(require, cache, nil)

	cache.Set(5, 55)
	value, exists := cache.Get(5)
	require.True(exists)
	require.Equal(55, value)
	requireLruOrder(require, cache, []int{5})
}

func TestLruCache_dropLast_RemovesTailAndKeepsListConsistent(t *testing.T) {
	tests := map[string]struct {
		keys      []int
		wantKey   int
		wantOrder []int
	}{
		"multiple values": {keys: []int{1, 2, 3}, wantKey: 1, wantOrder: []int{3, 2}},
		"two values":      {keys: []int{1, 2}, wantKey: 1, wantOrder: []int{2}},
		"single value":    {keys: []int{1}, wantKey: 1, wantOrder: nil},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			cache := NewLruCache[int, int](4)
			for _, key := range test.keys {
				cache.Set(key, key*10)
			}

			dropped := cache.dropLast()
			require.NotNil(dropped)
			require.Equal(test.wantKey, dropped.key)
			require.Equal(test.wantKey*10, dropped.val)
			requireLruOrder(require, cache, test.wantOrder)
		})
	}
}

func TestLruCache_dropLast_ReturnsNilOnEmptyCache(t *testing.T) {
	require := require.New(t)
	cache := NewLruCache[int, int](4)

	require.Nil(cache.dropLast())
	requireLruOrder(require, cache, nil)
}

func TestLruCache_GetMemoryFootprint_ReturnsCorrectSize(t *testing.T) {
	require := require.New(t)
	cache := NewLruCache[int, int](3)
	cache.Set(1, 11)
	cache.Set(2, 22)

	expectedSize := unsafe.Sizeof(*cache) +
		3*(unsafe.Sizeof(int(0))+unsafe.Sizeof(&entry[int, int]{})) +
		2*(unsafe.Sizeof(entry[int, int]{})+unsafe.Sizeof(int(0)))

	require.Equal(expectedSize, cache.GetMemoryFootprint(unsafe.Sizeof(int(0))).Total())
}

func TestLruCache_GetDynamicMemoryFootprint_ReturnsCorrectSize(t *testing.T) {
	require := require.New(t)
	sizes := map[int]uintptr{
		1: unsafe.Sizeof(int16(0)),
		2: unsafe.Sizeof(int32(0)),
		3: unsafe.Sizeof(int64(0)),
	}

	cache := NewLruCache[int, int](3)

	expectedSize := unsafe.Sizeof(*cache) +
		3*(unsafe.Sizeof(int(0))+unsafe.Sizeof(&entry[int, int]{}))
	for key, size := range sizes {
		cache.Set(key, key)
		expectedSize += unsafe.Sizeof(entry[int, int]{}) + size
	}

	require.Equal(expectedSize, cache.GetDynamicMemoryFootprint(func(v int) uintptr {
		return sizes[v]
	}).Total())
}

func TestEntry_String_ReturnsKeyValueDescription(t *testing.T) {
	require := require.New(t)
	e := entry[int, int]{10, 20, nil, nil}
	require.Equal("Entry: 10 -> 20", e.String())
}

// requireLruOrder checks that the cache holds exactly the given keys,
// linked from head to tail in both directions.
func requireLruOrder[K comparable, V any](require *require.Assertions, cache *LruCache[K, V], keys []K) {
	require.Len(cache.cache, len(keys))

	item := cache.head
	for _, key := range keys {
		require.NotNil(item)
		require.Equal(key, item.key)
		item = item.next
	}
	require.Nil(item)

	item = cache.tail
	for i := len(keys) - 1; i >= 0; i-- {
		require.NotNil(item)
		require.Equal(keys[i], item.key)
		item = item.prev
	}
	require.Nil(item)
}
