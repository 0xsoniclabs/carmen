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
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/backend/index"
	"github.com/0xsoniclabs/carmen/go/backend/index/indexhash"
	"github.com/0xsoniclabs/carmen/go/common"
)

const initCapacity = 10_000

// Index is an in-memory implementation of index.Index.
type Index[K comparable, I common.Identifier] struct {
	data          map[K]I
	keySerializer common.Serializer[K]
	hashIndex     *indexhash.IndexHash[K]
}

// NewIndex constructs a new Index instance.
func NewIndex[K comparable, I common.Identifier](serializer common.Serializer[K]) *Index[K, I] {
	memory := Index[K, I]{
		data:          make(map[K]I, initCapacity),
		keySerializer: serializer,
		hashIndex:     indexhash.NewIndexHash[K](serializer),
	}
	return &memory
}

// Size returns the number of registered keys.
func (m *Index[K, I]) Size() I {
	return I(len(m.data))
}

// GetOrAdd returns an index mapping for the key, or creates the new index.
func (m *Index[K, I]) GetOrAdd(key K) (I, error) {
	idx, exists := m.data[key]
	if !exists {
		size := len(m.data)

		idx = I(size)
		m.data[key] = idx
		m.hashIndex.AddKey(key)
	}
	return idx, nil
}

// Get returns an index mapping for the key, returns index.ErrNotFound if not exists.
func (m *Index[K, I]) Get(key K) (I, error) {
	idx, exists := m.data[key]
	if !exists {
		return idx, index.ErrNotFound
	}
	return idx, nil
}

// Contains returns whether the key exists in the mapping or not.
func (m *Index[K, I]) Contains(key K) bool {
	_, exists := m.data[key]
	return exists
}

// GetStateHash returns the index hash.
func (m *Index[K, I]) GetStateHash() (common.Hash, error) {
	return m.hashIndex.Commit()
}

// Flush does nothing.
func (m *Index[K, I]) Flush() error {
	return nil
}

// Close closes the storage and clean-ups all possible dirty values.
func (m *Index[K, I]) Close() error {
	return nil
}

// GetMemoryFootprint provides the size of the index in memory in bytes.
func (m *Index[K, I]) GetMemoryFootprint() *common.MemoryFootprint {
	dataMapItemSize := unsafe.Sizeof(struct {
		key K
		idx I
	}{})
	mf := common.NewMemoryFootprint(unsafe.Sizeof(*m) + uintptr(len(m.data))*dataMapItemSize)
	mf.AddChild("hashIndex", m.hashIndex.GetMemoryFootprint())
	mf.SetNote(fmt.Sprintf("(items: %d)", len(m.data)))
	return mf
}
