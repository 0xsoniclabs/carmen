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

	"github.com/0xsoniclabs/carmen/go/backend/hashtree"
	"github.com/0xsoniclabs/carmen/go/common"
)

// Store is an in-memory store.Store implementation - it maps IDs to values
type Store[I common.Identifier, V any] struct {
	data         [][]byte // data of pages [page][byte of page]
	hashTree     hashtree.HashTree
	serializer   common.Serializer[V]
	pageSize     int // the amount of bytes of one page
	pageItems    int // the amount of items stored in one page
	pageDataSize int // the amount of bytes in one page used by data items (without padding)
	itemSize     int // the amount of bytes per one value
}

// NewStore constructs a new instance of Store.
// It needs a serializer of data items and the default value for a not-set item.
func NewStore[I common.Identifier, V any](serializer common.Serializer[V], pageSize int, hashtreeFactory hashtree.Factory) (*Store[I, V], error) {
	if pageSize < serializer.Size() {
		return nil, fmt.Errorf("memory store pageSize too small (minimum %d)", serializer.Size())
	}

	itemSize := serializer.Size()
	memory := &Store[I, V]{
		data:         [][]byte{},
		serializer:   serializer,
		pageSize:     pageSize,
		pageItems:    pageSize / itemSize,
		pageDataSize: pageSize / itemSize * itemSize,
		itemSize:     itemSize,
	}
	memory.hashTree = hashtreeFactory.Create(memory)
	return memory, nil
}

// itemPosition provides the position of an item in data pages
func (m *Store[I, V]) itemPosition(id I) (page int, position int64) {
	// casting to I for division in proper bit width
	return int(id / I(m.pageItems)), (int64(id) % int64(m.pageItems)) * int64(m.itemSize)
}

// GetPage provides the hashing page data
func (m *Store[I, V]) GetPage(pageNum int) ([]byte, error) {
	return m.data[pageNum][0:m.pageDataSize], nil
}

// GetHash provides a hash of the page (in the latest state)
func (m *Store[I, V]) GetHash(partNum int) (hash common.Hash, err error) {
	return m.hashTree.GetPageHash(partNum)
}

// Set a value of an item
func (m *Store[I, V]) Set(id I, value V) error {
	pageNum, itemPosition := m.itemPosition(id)
	for pageNum >= len(m.data) {
		m.data = append(m.data, make([]byte, m.pageSize))
	}

	copy(m.data[pageNum][itemPosition:itemPosition+int64(m.itemSize)], m.serializer.ToBytes(value))
	m.hashTree.MarkUpdated(pageNum)
	return nil
}

// Get a value of the item (or the itemDefault, if not defined)
func (m *Store[I, V]) Get(id I) (item V, err error) {
	page, itemPosition := m.itemPosition(id)
	if page < len(m.data) {
		item = m.serializer.FromBytes(m.data[page][itemPosition : itemPosition+int64(m.itemSize)])
	}
	return item, nil
}

// GetStateHash computes and returns a cryptographical hash of the stored data
func (m *Store[I, V]) GetStateHash() (common.Hash, error) {
	return m.hashTree.HashRoot()
}

// Flush the store
func (m *Store[I, V]) Flush() error {
	return nil // no-op for in-memory database
}

// Close the store
func (m *Store[I, V]) Close() error {
	return nil // no-op for in-memory database
}

// GetMemoryFootprint provides the size of the store in memory in bytes
func (m *Store[I, V]) GetMemoryFootprint() *common.MemoryFootprint {
	dataSize := uintptr(len(m.data) * m.pageSize)
	mf := common.NewMemoryFootprint(unsafe.Sizeof(*m) + dataSize)
	mf.AddChild("hashTree", m.hashTree.GetMemoryFootprint())
	return mf
}
