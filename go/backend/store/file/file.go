// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package file

import (
	"errors"
	"fmt"
	"io"
	"os"
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/backend/hashtree"
	"github.com/0xsoniclabs/carmen/go/common"
)

// Store is a filesystem-based store.Store implementation - it stores mapping of ID to value in binary files.
type Store[I common.Identifier, V any] struct {
	file           *os.File
	hashTree       hashtree.HashTree
	serializer     common.Serializer[V]
	pageSize       int // the amount of bytes of one page
	pageItems      int // the amount of items stored in one page
	hashedPageSize int // the amount of the page bytes to be passed into the hashing function - rounded to whole items
	itemSize       int // the amount of bytes per one value
	pagesCount     int // the amount of store pages
}

// NewStore constructs a new instance of FileStore.
// It needs a serializer of data items and the default value for a not-set item.
func NewStore[I common.Identifier, V any](path string, serializer common.Serializer[V], pageSize int, hashtreeFactory hashtree.Factory) (*Store[I, V], error) {
	if pageSize < serializer.Size() {
		return nil, fmt.Errorf("file store pageSize too small (minimum %d)", serializer.Size())
	}

	file, err := os.OpenFile(path+"/data", os.O_RDWR|os.O_CREATE, 0600)
	if err != nil {
		return nil, fmt.Errorf("failed to open/create data file; %s", err)
	}

	itemSize := serializer.Size()
	s := &Store[I, V]{
		file:           file,
		serializer:     serializer,
		pageSize:       pageSize,
		pageItems:      pageSize / itemSize,
		hashedPageSize: pageSize / itemSize * itemSize,
		itemSize:       itemSize,
	}
	s.pagesCount, err = s.getPagesCount()
	if err != nil {
		return nil, err
	}
	s.hashTree = hashtreeFactory.Create(s)
	return s, nil
}

func (m *Store[I, V]) getPagesCount() (count int, err error) {
	fi, err := m.file.Stat()
	if err != nil {
		return 0, err
	}
	dataFileSize := fi.Size()
	pagesCount := int(dataFileSize / int64(m.pageSize))
	if dataFileSize%int64(m.pageSize) != 0 {
		pagesCount++
	}
	return pagesCount, nil
}

// itemPosition provides the position of an item in data pages
func (m *Store[I, V]) itemPosition(id I) (page int, position int64) {
	page = int(id / I(m.pageItems)) // casting to I for division in proper bit width
	pageStart := int64(page) * int64(m.pageSize)
	inPageStart := (int64(id) % int64(m.pageItems)) * int64(m.itemSize)
	position = pageStart + inPageStart
	return
}

// GetPage provides a page bytes for needs of the hash obtaining
func (m *Store[I, V]) GetPage(page int) ([]byte, error) {
	buffer := make([]byte, m.hashedPageSize)

	_, err := m.file.ReadAt(buffer, int64(page)*int64(m.pageSize))
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, err // the page does not exist in the data file yet
	}
	return buffer, nil
}

// GetHash provides a hash of the page (in the latest state)
func (m *Store[I, V]) GetHash(partNum int) (hash common.Hash, err error) {
	return m.hashTree.GetPageHash(partNum)
}

// Set a value of an item
func (m *Store[I, V]) Set(id I, value V) error {
	pageNum, itemPosition := m.itemPosition(id)

	_, err := m.file.WriteAt(m.serializer.ToBytes(value), itemPosition)
	if err != nil {
		return fmt.Errorf("failed to write into data file; %s", err)
	}

	if pageNum >= m.pagesCount {
		m.pagesCount = pageNum + 1
	}

	m.hashTree.MarkUpdated(pageNum)
	return nil
}

// Get a value of the item (or the itemDefault, if not defined)
func (m *Store[I, V]) Get(id I) (value V, err error) {
	_, itemPosition := m.itemPosition(id)

	bytes := make([]byte, m.itemSize)
	n, err := m.file.ReadAt(bytes, itemPosition)
	if err != nil {
		if errors.Is(err, io.EOF) {
			return value, nil // the item does not exist in the page file (the file is shorter)
		}
		return value, err
	}
	if n != m.itemSize {
		return value, fmt.Errorf("unable to read - page file is corrupted")
	}
	return m.serializer.FromBytes(bytes), nil
}

// GetStateHash computes and returns a cryptographical hash of the stored data
func (m *Store[I, V]) GetStateHash() (common.Hash, error) {
	return m.hashTree.HashRoot()
}

// Flush the store
func (m *Store[I, V]) Flush() error {
	return m.file.Sync()
}

// Close the store
func (m *Store[I, V]) Close() error {
	return m.file.Close()
}

// GetMemoryFootprint provides the size of the store in memory in bytes
func (m *Store[I, V]) GetMemoryFootprint() *common.MemoryFootprint {
	mf := common.NewMemoryFootprint(unsafe.Sizeof(*m))
	mf.AddChild("hashTree", m.hashTree.GetMemoryFootprint())
	return mf
}
