// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package memory

import (
	"fmt"
	"github.com/0xsoniclabs/carmen/go/common"
	"unsafe"
)

// LinearHashMap is a structure mapping a list of key/value pairs.
// It stores keys in buckets based on the hash computed ouf of the keys. The keys associate the values.
// The number of buckets increases on insert when the number of stored keys overflows the capacity of this map.
// In contrast to simple HashMaps, this structure grows by splitting one bucket into two when the capacity is exceeded,
// so the map does not have to be fully copied to a new bigger structure.
// The capacity is verified on each insert and potentially the split is triggered.
// It is inspired by: https://hackthology.com/linear-hashing.html#fn-5
type LinearHashMap[K comparable, V comparable] struct {
	list []*BlockList[K, V]

	records       uint // current total number of records in the whole table
	blockCapacity int  //maximal number of elements per block

	comparator common.Comparator[K]
	common.LinearHashBase[K, V]
}

// NewLinearHashMap creates a new instance with the initial number of buckets and constant bucket size.
// The number of buckets will grow as this table grows
func NewLinearHashMap[K comparable, V comparable](blockItems, numBuckets int, hasher common.Hasher[K], comparator common.Comparator[K]) *LinearHashMap[K, V] {
	list := make([]*BlockList[K, V], numBuckets)
	for i := 0; i < numBuckets; i++ {
		list[i] = NewBlockList[K, V](blockItems, comparator)
	}

	return &LinearHashMap[K, V]{
		list:           list,
		blockCapacity:  blockItems,
		comparator:     comparator,
		LinearHashBase: common.NewLinearHashBase[K, V](numBuckets, hasher, comparator),
	}
}

// Put assigns the value to the input key.
func (h *LinearHashMap[K, V]) Put(key K, value V) {
	bucketId := h.GetBucketId(&key)
	bucket := h.list[bucketId]
	beforeSize := bucket.Size()

	bucket.Put(key, value)

	if beforeSize < bucket.Size() {
		h.records += 1
	}

	// when the number of buckets overflows, split one bucket into two
	h.checkSplit()
}

// Get returns value associated to the input key.
func (h *LinearHashMap[K, V]) Get(key K) (value V, exists bool) {
	bucket := h.GetBucketId(&key)
	value, exists = h.list[bucket].Get(key)
	return
}

// GetOrAdd either returns a value stored under input key, or it associates the input value
// when the key is not stored yet.
// It returns true if the key was present, or false otherwise.
func (h *LinearHashMap[K, V]) GetOrAdd(key K, val V) (value V, exists bool) {
	bucket := h.GetBucketId(&key)
	value, exists = h.list[bucket].GetOrAdd(key, val)
	if !exists {
		h.records += 1
		h.checkSplit()
	}
	return
}

// ForEach iterates all stored key/value pairs.
func (h *LinearHashMap[K, V]) ForEach(callback func(K, V)) {
	for _, v := range h.list {
		v.ForEach(callback)
	}
}

// Remove deletes the key from the map and returns whether an element was removed.
func (h *LinearHashMap[K, V]) Remove(key K) bool {
	bucket := h.GetBucketId(&key)
	exists := h.list[bucket].Remove(key)
	if exists {
		h.records -= 1
	}

	return exists
}

func (h *LinearHashMap[K, V]) Size() int {
	return int(h.records)
}

func (h *LinearHashMap[K, V]) Clear() {
	h.records = 0
	for _, v := range h.list {
		v.Clear()
	}
}

// GetBuckets returns the number of buckets.
func (h *LinearHashMap[K, V]) GetBuckets() int {
	return len(h.list)
}

func (h *LinearHashMap[K, V]) checkSplit() {
	// when the number of buckets overflows, split one bucket into two
	if h.records > uint(len(h.list))*uint(h.blockCapacity) {
		h.split()
	}
}

// split creates a new bucket and extends the total number of buckets.
// It locates a bucket to split, extends the bit mask by adding one more bit
// and re-distribute keys between the old bucket and the new bucket.
func (h *LinearHashMap[K, V]) split() {
	bucketId := h.NextBucketId()
	oldBucket := h.list[bucketId]

	oldEntries := oldBucket.GetEntries()
	entriesA, entriesB := h.SplitEntries(bucketId, oldEntries)

	bucketA := InitBlockList[K, V](h.blockCapacity, entriesA, h.comparator)
	bucketB := InitBlockList[K, V](h.blockCapacity, entriesB, h.comparator)

	// append the new one
	h.list[bucketId] = bucketA
	h.list = append(h.list, bucketB)
}

func (h *LinearHashMap[K, V]) PrintDump() {
	for i, v := range h.list {
		fmt.Printf("Bucket: %d\n", i)
		v.ForEach(func(k K, v V) {
			fmt.Printf("%s \n", h.ToString(k, v))
		})
	}
}

func (h *LinearHashMap[K, V]) GetMemoryFootprint() *common.MemoryFootprint {
	selfSize := unsafe.Sizeof(*h)
	var entrySize uintptr
	for _, item := range h.list {
		entrySize += item.GetMemoryFootprint().Value()
	}
	footprint := common.NewMemoryFootprint(selfSize)
	footprint.AddChild("buckets", common.NewMemoryFootprint(entrySize))
	return footprint
}
