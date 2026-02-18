package geth2

import (
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/syndtr/goleveldb/leveldb"
	"github.com/syndtr/goleveldb/leveldb/opt"
)

const (
	ErrNotFound = common.ConstError("not found")
)

// NodeStore is an interface for a key-value store used to persist Verkle nodes.
type NodeStore interface {
	Get(key []byte) ([]byte, error)
	Set(key []byte, value []byte) error
	Close() error
}

// levelDbStore is a simple implementation of NodeStore using LevelDB.
type levelDbStore struct {
	db *leveldb.DB
}

func newLevelDbStore(path string) (*levelDbStore, error) {
	db, err := leveldb.OpenFile(path, nil)
	if err != nil {
		return nil, err
	}
	return &levelDbStore{db: db}, nil
}

func (s *levelDbStore) Get(key []byte) ([]byte, error) {
	data, err := s.db.Get(key, &opt.ReadOptions{})
	if err == leveldb.ErrNotFound {
		return nil, ErrNotFound
	}
	return data, err
}

func (s *levelDbStore) Set(key []byte, value []byte) error {
	return s.db.Put(key, value, &opt.WriteOptions{})
}

func (s *levelDbStore) Close() error {
	return s.db.Close()
}

// memoryDbStore is a simple in-memory implementation of NodeStore for testing purposes.
type memoryDbStore struct {
	store map[string][]byte
}

func newMemoryDbStore() *memoryDbStore {
	return &memoryDbStore{store: make(map[string][]byte)}
}

func (s *memoryDbStore) Get(key []byte) ([]byte, error) {
	value, ok := s.store[string(key)]
	if !ok {
		return nil, ErrNotFound
	}
	return value, nil
}

func (s *memoryDbStore) Set(key []byte, value []byte) error {
	s.store[string(key)] = value
	return nil
}

func (s *memoryDbStore) Close() error {
	// No resources to clean up for in-memory store.
	return nil
}
