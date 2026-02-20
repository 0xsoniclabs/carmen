// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package geth2

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/syndtr/goleveldb/leveldb"
	"github.com/syndtr/goleveldb/leveldb/opt"
)

const (
	ErrNotFound  = common.ConstError("not found")
	ErrNoArchive = state.NoArchiveError
)

// Store is an interface for storing and retrieving Verkle Trie nodes, as well
// as keeping track of the current block number. It serves as the interface to
// the underlying storage layer (e.g., LevelDB) for the Verkle Trie state.
type Store interface {
	// NextBlock returns the block number of the next block to be added to the store.
	NextBlock() uint64

	// HeadState returns a NodeSource for the head state of the store (i.e., the
	// state at the latest block).
	HeadState() NodeSource

	// AddBlock adds a new block with the given block number and changes to the
	// store. The changes should cover the delta between the previous head state
	// and the new head state introduced by this block.
	AddBlock(block uint64, changes []Entry) error

	// HistoricState returns a NodeSource for the state at the given block number.
	// If the store does not support historical states, it returns ErrNoArchive.
	HistoricState(block uint64) (NodeSource, error)

	// Flush flushes any pending changes to the underlying storage.
	Flush() error

	// Close closes the store and releases any resources it holds.
	Close() error
}

type NodeSource interface {
	GetNode(path []byte) ([]byte, error)
}

type Entry struct {
	Path []byte
	Blob []byte
}

// --- Factories ---

// newLevelDbLiveStore creates a LevelDB-backed store that only keeps the latest
// state (head) in the DB. It does not support historical states, and returns
// ErrNoArchive if a historic state is requested.
func newLevelDbLiveStore(path string) (*levelDbStore, error) {
	// This implementation uses the path of a node as a key in the DB directly.
	// When nodes are updated, old versions of the node are overwritten.
	return newLevelDbStore(
		path,
		func(_ uint64, path []byte) []byte {
			return path
		},
		func(db *leveldb.DB, block uint64, path []byte) ([]byte, error) {
			data, err := db.Get(path, &opt.ReadOptions{})
			if err == leveldb.ErrNotFound {
				return nil, ErrNotFound
			}
			return data, err
		},
		false,
	)
}

// newLevelDbArchiveStore creates a LevelDB-backed store that keeps all
// historical states in the DB.
func newLevelDbArchiveStore(path string) (*levelDbStore, error) {
	// This implementation uses a composite key of the path and the block number
	// for each node. When nodes are updated, new entries are added to the DB with
	// the same path but different block numbers. When a historic state is requested,
	// the store looks for the entry with the largest block number that is smaller
	// or equal to the requested block number.
	return newLevelDbStore(
		path,
		toArchivePath,
		func(db *leveldb.DB, block uint64, path []byte) ([]byte, error) {
			// In the archive, we need to find the last block that updated this
			// path before or at the given block, and read the node from that
			// block.
			key := toArchivePath(block, path)
			iter := db.NewIterator(nil, &opt.ReadOptions{})
			defer iter.Release()
			if iter.Seek(key) {
				if bytes.Equal(iter.Key(), key) {
					// There is a perfect match for this block and path, return it.
					return iter.Value(), nil
				}
			}
			// No perfect match, we need to find the last update for this path
			// before this block.
			if iter.Prev() {
				// Check if the previous entry is for the same path.
				prevKey := iter.Key()
				if areArchiveKeysForSamePath(prevKey, key) {
					return iter.Value(), nil
				}
			}
			return nil, ErrNotFound
		},
		true,
	)
}

// --- Implementations ---

type keyFactoryFn func(block uint64, path []byte) []byte
type nodeFinderFn func(db *leveldb.DB, block uint64, path []byte) ([]byte, error)

// levelDbStore is a generic store implementation instantiated by factory
// functions to realize different storage strategies (live vs archive).
type levelDbStore struct {
	db              *leveldb.DB
	nextBlock       uint64
	keyFactory      keyFactoryFn
	findNode        nodeFinderFn
	supportsHistory bool
}

func newLevelDbStore(
	path string,
	keyFactory keyFactoryFn,
	findNode nodeFinderFn,
	supportsHistory bool,
) (*levelDbStore, error) {
	db, err := leveldb.OpenFile(path, nil)
	if err != nil {
		return nil, err
	}

	// Load the next block from the DB, if it exists.
	nextBlock, err := loadNextBlockFromDb(db)
	if err != nil {
		return nil, errors.Join(err, db.Close())
	}

	return &levelDbStore{
		db:              db,
		nextBlock:       nextBlock,
		keyFactory:      keyFactory,
		findNode:        findNode,
		supportsHistory: supportsHistory,
	}, nil
}

func (s *levelDbStore) NextBlock() uint64 {
	return s.nextBlock
}

func (s *levelDbStore) HeadState() NodeSource {
	return &levelDbNodeSource{db: s.db, block: &s.nextBlock, find: s.findNode}
}

func (s *levelDbStore) AddBlock(block uint64, changes []Entry) error {

	batch := new(leveldb.Batch)
	for _, change := range changes {
		key := s.keyFactory(block, change.Path)
		batch.Put(key, change.Blob)
	}
	err := s.db.Write(batch, &opt.WriteOptions{})
	if err != nil {
		return err
	}

	s.nextBlock = block + 1
	return storeNextBlockToDb(s.db, block+1)
}

func (s *levelDbStore) HistoricState(block uint64) (NodeSource, error) {
	if !s.supportsHistory {
		return nil, ErrNoArchive
	}
	if block >= s.nextBlock {
		return nil, fmt.Errorf("requested block %d is in the future (next block: %d)", block, s.nextBlock)
	}
	return &levelDbNodeSource{db: s.db, block: &block, find: s.findNode}, nil
}

func (s *levelDbStore) Flush() error {
	// no-op, as flushes are managed by LevelDB internally.
	return nil
}

func (s *levelDbStore) Close() error {
	return s.db.Close()
}

type levelDbNodeSource struct {
	db    *leveldb.DB
	block *uint64
	find  nodeFinderFn
}

func (s *levelDbNodeSource) GetNode(path []byte) ([]byte, error) {
	block := *s.block
	return s.find(s.db, block, path)
}

func toArchivePath(block uint64, path []byte) []byte {
	res := make([]byte, 1+len(path)+4)
	res[0] = byte(len(path))
	copy(res[1:], path)
	binary.BigEndian.PutUint32(res[1+len(path):], uint32(block))
	return res
}

func areArchiveKeysForSamePath(key1, key2 []byte) bool {
	if len(key1) != len(key2) {
		return false
	}
	if len(key1) < 4 {
		return false
	}
	// Mask out the last 4 byte (block number) and compare the rest of the key,
	// which corresponds to the path.
	key1 = key1[:len(key1)-4]
	key2 = key2[:len(key2)-4]
	return bytes.Equal(key1, key2)
}

// --- Utilities ---

var (
	nextBlockKey = func() []byte {
		key := make([]byte, 64) // longer than any other key, to avoid collisions
		for i := range key {
			key[i] = 255
		}
		return key
	}()
)

func loadNextBlockFromDb(db *leveldb.DB) (uint64, error) {
	key := nextBlockKey
	if data, err := db.Get(key, &opt.ReadOptions{}); err == nil {
		if len(data) != 8 {
			return 0, fmt.Errorf("invalid next block data")
		}
		nextBlock := binary.BigEndian.Uint64(data)
		return nextBlock, nil
	} else if err != leveldb.ErrNotFound {
		return 0, err
	}
	return 0, nil // default to 0 if not found
}

func storeNextBlockToDb(db *leveldb.DB, nextBlock uint64) error {
	key := nextBlockKey
	data := make([]byte, 8)
	binary.BigEndian.PutUint64(data, nextBlock)
	return db.Put(key, data, &opt.WriteOptions{})
}
