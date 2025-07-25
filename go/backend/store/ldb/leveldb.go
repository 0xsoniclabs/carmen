// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package ldb

import (
	"encoding/binary"
	"fmt"
	"github.com/0xsoniclabs/carmen/go/backend"
	"github.com/0xsoniclabs/carmen/go/backend/store"
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/backend/hashtree"
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/syndtr/goleveldb/leveldb"
	"github.com/syndtr/goleveldb/leveldb/util"
)

// Store is a database-based store.Store implementation. It stores items in a key-value database.
type Store[I common.Identifier, V any] struct {
	db              backend.LevelDB
	hashTree        hashtree.HashTree
	valueSerializer common.Serializer[V]
	indexSerializer common.Serializer[I]
	pageSize        int // the amount of items stored in one database page
	itemSize        int // the amount of bytes per one value
	pagesCount      int // the amount of store pages
	table           backend.TableSpace
}

// NewStore constructs a new instance of the Store.
func NewStore[I common.Identifier, V any](
	db backend.LevelDB,
	table backend.TableSpace,
	serializer common.Serializer[V],
	indexSerializer common.Serializer[I],
	hashTreeFactory hashtree.Factory,
	pageSize int) (store *Store[I, V], err error) {

	if pageSize < serializer.Size() {
		return nil, fmt.Errorf("leveldb store pageSize too small (minimum %d)", serializer.Size())
	}

	store = &Store[I, V]{
		db:              db,
		valueSerializer: serializer,
		indexSerializer: indexSerializer,
		pageSize:        pageSize / serializer.Size(),
		itemSize:        serializer.Size(),
		table:           table,
	}
	if err = store.loadPagesCount(); err != nil {
		return nil, err
	}
	store.hashTree = hashTreeFactory.Create(store)
	return store, nil
}

// itemPosition provides the position of an item in the page as well as the page number
func (m *Store[I, V]) itemPosition(id I) (page int, position int) {
	return int(id) / m.pageSize, (int(id) % m.pageSize) * m.valueSerializer.Size()
}

// GetPage provides the hashing page data
func (m *Store[I, V]) GetPage(page int) (pageData []byte, err error) {
	return m.getPageFromLdbReader(page, m.db)
}

// getPageFromLdbReader provides the hashing page from given LevelDB reader (snapshot or database)
func (m *Store[I, V]) getPageFromLdbReader(page int, db backend.LevelDBReader) (pageData []byte, err error) {
	pageStartKey := page * m.pageSize
	pageEndKey := pageStartKey + m.pageSize
	startDbKey := m.convertKey(I(pageStartKey)).ToBytes()
	endDbKey := m.convertKey(I(pageEndKey)).ToBytes()
	r := util.Range{Start: startDbKey, Limit: endDbKey}
	iter := db.NewIterator(&r, nil)
	defer iter.Release()

	// create the page first
	pageData = make([]byte, m.pageSize*m.itemSize)
	// inject values at the right positions
	for iter.Next() {
		key := m.indexSerializer.FromBytes(iter.Key()[1:]) // strip first byte (table space) and get the idx only
		_, position := m.itemPosition(key)
		copy(pageData[position:], iter.Value())
	}
	return pageData, iter.Error()
}

func (m *Store[I, V]) Set(id I, value V) (err error) {
	// index is mapped in the database directly
	dbKey := m.convertKey(id).ToBytes()
	if err = m.db.Put(dbKey, m.valueSerializer.ToBytes(value), nil); err == nil {
		page, _ := m.itemPosition(id)
		if page >= m.pagesCount {
			if err := m.setPagesCount(page + 1); err != nil {
				return err
			}
		}
		m.hashTree.MarkUpdated(page)
		if page >= m.pagesCount {
			m.pagesCount = page + 1
		}
	}
	return
}

func (m *Store[I, V]) Get(id I) (v V, err error) {
	// index is mapped in the database directly
	var val []byte
	dbKey := m.convertKey(id).ToBytes()
	if val, err = m.db.Get(dbKey, nil); err != nil {
		if err == leveldb.ErrNotFound {
			return v, nil
		}
	} else {
		v = m.valueSerializer.FromBytes(val)
	}
	return
}

// getPagesCountDbKey provides a database key, where should be the amount of pages in the depot stored
func getPagesCountDbKey(table backend.TableSpace) []byte {
	return []byte{byte(table)} // suppose the key serializer will never produce empty slice
}

// loadPagesCount loads the amount of pages in the depot into the pagesCount field
func (m *Store[I, V]) loadPagesCount() error {
	val, err := m.db.Get(getPagesCountDbKey(m.table), nil)
	if err == leveldb.ErrNotFound {
		m.pagesCount = 0
		return nil
	}
	if err != nil {
		return err
	}
	m.pagesCount = int(binary.LittleEndian.Uint32(val))
	return nil
}

// setPagesCount allows to set pages count - to be called when a new page is created
func (m *Store[I, V]) setPagesCount(count int) error {
	m.pagesCount = count
	val := binary.LittleEndian.AppendUint32(nil, uint32(count))
	return m.db.Put(getPagesCountDbKey(m.table), val, nil)
}

// GetStateHash computes and returns a cryptographical hash of the stored data
func (m *Store[I, V]) GetStateHash() (common.Hash, error) {
	return m.hashTree.HashRoot()
}

// GetProof returns a proof the snapshot exhibits if it is created
// for the current state of the data structure.
func (m *Store[I, V]) GetProof() (backend.Proof, error) {
	hash, err := m.GetStateHash()
	if err != nil {
		return nil, err
	}
	return store.NewProof(hash), nil
}

// CreateSnapshot creates a snapshot of the current state of the data
// structure. The snapshot should be shielded from subsequent modifications
// and be accessible until released.
func (m *Store[I, V]) CreateSnapshot() (backend.Snapshot, error) {
	branchingFactor := m.hashTree.GetBranchingFactor()
	hash, err := m.hashTree.HashRoot()
	if err != nil {
		return nil, err
	}
	snap, err := m.db.GetSnapshot()
	if err != nil {
		return nil, err
	}

	newSnap := &SnapshotSource[I, V]{
		snap:  snap,
		store: m,
	}

	snapshot := store.CreateStoreSnapshotFromStore[V](m.valueSerializer, branchingFactor, hash, m.pagesCount, newSnap)
	return snapshot, nil
}

// Restore restores the data structure to the given snapshot state. This
// may invalidate any former snapshots created on the data structure. In
// particular, it is not required to be able to synchronize to a former
// snapshot derived from the targeted data structure.
func (m *Store[I, V]) Restore(snapshotData backend.SnapshotData) error {
	snapshot, err := store.CreateStoreSnapshotFromData[V](m.valueSerializer, snapshotData)
	if err != nil {
		return fmt.Errorf("unable to restore snapshot; %s", err)
	}
	if snapshot.GetBranchingFactor() != m.hashTree.GetBranchingFactor() {
		return fmt.Errorf("unable to restore snapshot - unexpected branching factor")
	}

	err = m.hashTree.Reset()
	if err != nil {
		return fmt.Errorf("unable to restore snapshot - failed to remove old hashTree; %s", err)
	}

	var id I
	partsNum := snapshot.GetNumParts()
	for partNum := 0; partNum < partsNum; partNum++ {
		data, err := snapshot.GetPartData(partNum)
		if err != nil {
			return err
		}
		if len(data) != m.pageSize*m.itemSize {
			return fmt.Errorf("unable to restore snapshot - unexpected length of store part")
		}
		for i := 0; i < m.pageSize && len(data) != 0; i++ {
			err := m.Set(id, m.valueSerializer.FromBytes(data[0:m.itemSize]))
			if err != nil {
				return err
			}
			data = data[m.itemSize:]
			id++
		}
		m.hashTree.MarkUpdated(partNum)
	}
	return nil
}

func (m *Store[I, V]) GetSnapshotVerifier([]byte) (backend.SnapshotVerifier, error) {
	return store.CreateStoreSnapshotVerifier[V](m.valueSerializer), nil
}

func (m *Store[I, V]) Flush() error {
	// commit state hash root
	_, err := m.GetStateHash()
	return err
}

func (m *Store[I, V]) Close() error {
	return m.Flush()
}

// convertKey translates the Index representation of the key into a database key.
// The database key is prepended with the table space prefix, furthermore the input key is converted to bytes
// by the key serializer
func (m *Store[I, V]) convertKey(idx I) backend.DbKey {
	return backend.ToDBKey(m.table, m.indexSerializer.ToBytes(idx))
}

// GetMemoryFootprint provides the size of the store in memory in bytes
func (m *Store[I, V]) GetMemoryFootprint() *common.MemoryFootprint {
	mf := common.NewMemoryFootprint(unsafe.Sizeof(*m))
	mf.AddChild("hashTree", m.hashTree.GetMemoryFootprint())
	mf.AddChild("levelDb", m.db.GetMemoryFootprint())
	return mf
}
