// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package index_test

import (
	"fmt"
	"testing"

	"github.com/0xsoniclabs/carmen/go/backend"
	"github.com/0xsoniclabs/carmen/go/backend/index"
	"github.com/0xsoniclabs/carmen/go/backend/index/cache"
	"github.com/0xsoniclabs/carmen/go/backend/index/file"
	"github.com/0xsoniclabs/carmen/go/backend/index/ldb"
	"github.com/0xsoniclabs/carmen/go/backend/index/memory"
	"github.com/0xsoniclabs/carmen/go/common"
)

func initIndexesMap() map[string]func(t *testing.T) index.Index[common.Address, uint32] {

	keySerializer := common.AddressSerializer{}
	idSerializer := common.Identifier32Serializer{}

	return map[string]func(t *testing.T) index.Index[common.Address, uint32]{
		"memindex": func(t *testing.T) index.Index[common.Address, uint32] {
			return memory.NewIndex[common.Address, uint32](keySerializer)
		},
		"memLinearHashIndex": func(t *testing.T) index.Index[common.Address, uint32] {
			return memory.NewLinearHashIndex[common.Address, uint32](keySerializer, idSerializer, common.AddressHasher{}, common.AddressComparator{})
		},
		"cachedMemoryIndex": func(t *testing.T) index.Index[common.Address, uint32] {
			return cache.NewIndex[common.Address, uint32](memory.NewIndex[common.Address, uint32](keySerializer), 10)
		},
		"ldbindex": func(t *testing.T) index.Index[common.Address, uint32] {
			db, err := backend.OpenLevelDb(t.TempDir(), nil)
			if err != nil {
				t.Fatalf("failed to init leveldb; %s", err)
			}
			ldbindex, err := ldb.NewIndex[common.Address, uint32](db, backend.BalanceStoreKey, keySerializer, idSerializer)
			if err != nil {
				t.Fatalf("failed to init leveldb; %s", err)
			}
			t.Cleanup(func() {
				_ = ldbindex.Close()
				_ = db.Close()
			})
			return ldbindex
		},
		"fileIndex": func(t *testing.T) index.Index[common.Address, uint32] {
			fileIndex, err := file.NewIndex[common.Address, uint32](t.TempDir(), keySerializer, idSerializer, common.AddressHasher{}, common.AddressComparator{})
			if err != nil {
				t.Fatalf("failed to init leveldb; %s", err)
			}
			t.Cleanup(func() {
				_ = fileIndex.Close()
			})
			return fileIndex
		},
		"cachedFileIndex": func(t *testing.T) index.Index[common.Address, uint32] {
			fileIndex, err := file.NewIndex[common.Address, uint32](t.TempDir(), keySerializer, idSerializer, common.AddressHasher{}, common.AddressComparator{})
			if err != nil {
				t.Fatalf("failed to init leveldb; %s", err)
			}
			t.Cleanup(func() {
				_ = fileIndex.Close()
			})
			return cache.NewIndex[common.Address, uint32](fileIndex, 10)
		},
		"cachedLdbIndex": func(t *testing.T) index.Index[common.Address, uint32] {
			db, err := backend.OpenLevelDb(t.TempDir(), nil)
			if err != nil {
				t.Fatalf("failed to init leveldb; %s", err)
			}
			ldbindex, err := ldb.NewIndex[common.Address, uint32](db, backend.BalanceStoreKey, keySerializer, idSerializer)
			if err != nil {
				t.Fatalf("failed to init leveldb; %s", err)
			}
			t.Cleanup(func() {
				_ = ldbindex.Close()
				_ = db.Close()
			})
			return cache.NewIndex[common.Address, uint32](ldbindex, 10)
		},
	}
}

func TestIndex_SizeIsAccuratelyReported(t *testing.T) {
	for name, idx := range initIndexesMap() {
		for _, size := range []int{0, 1, 5, 1000, 12345} {
			t.Run(fmt.Sprintf("index %s size %d", name, size), func(t *testing.T) {

				index := idx(t)
				if got, want := index.Size(), uint32(0); got != want {
					t.Errorf("wrong size of index, wanted %v, got %v", want, got)
				}

				if id, err := index.GetOrAdd(common.Address{1}); err != nil || id != 0 {
					t.Errorf("failed to register new key: %v / %v", id, err)
				}

				if got, want := index.Size(), uint32(1); got != want {
					t.Errorf("wrong size of index, wanted %v, got %v", want, got)
				}

				// Registering the same does not change the size.
				if id, err := index.GetOrAdd(common.Address{1}); err != nil || id != 0 {
					t.Errorf("failed to register new key: %v / %v", id, err)
				}

				if got, want := index.Size(), uint32(1); got != want {
					t.Errorf("wrong size of index, wanted %v, got %v", want, got)
				}

				// Registering a new key does.
				if id, err := index.GetOrAdd(common.Address{2}); err != nil || id != 1 {
					t.Errorf("failed to register new key: %v / %v", id, err)
				}

				if got, want := index.Size(), uint32(2); got != want {
					t.Errorf("wrong size of index, wanted %v, got %v", want, got)
				}
			})
		}
	}
}

func TestIndexesInitialHash(t *testing.T) {
	indexes := initIndexesMap()

	for _, idx := range indexes {
		hash, err := idx(t).GetStateHash()
		if err != nil {
			t.Fatalf("failed to hash empty index; %s", err)
		}
		if hash != (common.Hash{}) {
			t.Errorf("invalid hash of empty index: %x (expected zeros)", hash)
		}
	}
}

func TestIndexesHashingByComparison(t *testing.T) {
	indexes := initIndexesMap()
	for i := 0; i < 10; i++ {
		ids := make([]uint32, len(indexes))
		indexInstances := make([]index.Index[common.Address, uint32], 0, len(indexes))
		var indexId int
		for _, idx := range indexes {
			indexInstance := idx(t)
			idx, err := indexInstance.GetOrAdd(common.Address{byte(0x20 + i)})
			ids[indexId] = idx
			indexId += 1
			indexInstances = append(indexInstances, indexInstance)
			if err != nil {
				t.Fatalf("failed to set index item %d; %s", i, err)
			}
		}
		if err := compareIds(ids); err != nil {
			t.Errorf("ids for item %d does not match: %s", i, err)
		}
		if err := compareHashes(indexInstances); err != nil {
			t.Errorf("hashes does not match after inserting item %d: %s", i, err)
		}
	}
}

func TestIndexesHashesAgainstReferenceOutput(t *testing.T) {
	indexes := initIndexesMap()

	// Tests the hashes for keys 0x01, 0x02 inserted in sequence.
	// reference hashes from the C++ implementation
	expectedHashes := []string{
		"ff9226e320b1deb7fabecff9ac800cd8eb1e3fb7709c003e2effcce37eec68ed",
		"c28553369c52e217564d3f5a783e2643186064498d1b3071568408d49eae6cbe",
	}

	indexInstances := make([]index.Index[common.Address, uint32], 0, len(indexes))
	for _, idx := range indexes {
		indexInstances = append(indexInstances, idx(t))
	}

	for i, expectedHash := range expectedHashes {
		for _, indexInstance := range indexInstances {
			_, err := indexInstance.GetOrAdd(common.Address{byte(i + 1)}) // 0x01 - 0x02
			if err != nil {
				t.Fatalf("failed to set index item; %s", err)
			}
			hash, err := indexInstance.GetStateHash()
			if err != nil {
				t.Fatalf("failed to hash index; %s", err)
			}
			if expectedHash != fmt.Sprintf("%x", hash) {
				t.Fatalf("invalid hash: %x (expected %s)", hash, expectedHash)
			}
		}
	}
}

func compareHashes(indexes []index.Index[common.Address, uint32]) error {
	var firstHash common.Hash
	for i, index := range indexes {
		hash, err := index.GetStateHash()
		if err != nil {
			return err
		}
		if i == 0 {
			firstHash = hash
		} else if firstHash != hash {
			return fmt.Errorf("different hashes: %x != %x", firstHash, hash)
		}
	}
	return nil
}

func compareIds(ids []uint32) error {
	var firstId uint32
	for i, id := range ids {
		if i == 0 {
			firstId = id
		} else if firstId != id {
			return fmt.Errorf("different ids: %d != %d", firstId, id)
		}
	}
	return nil
}

func fillIndex(t *testing.T, index index.Index[common.Address, uint32], size int) {
	for i := 0; i < size; i++ {
		addr := common.AddressFromNumber(i)
		if idx, err := index.GetOrAdd(addr); idx != uint32(i) || err != nil {
			t.Errorf("failed to add address %d", i)
		}
	}
}

func checkIndexContent(t *testing.T, index index.Index[common.Address, uint32], size int) {
	for i := 0; i < size; i++ {
		addr := common.AddressFromNumber(i)
		if idx, err := index.GetOrAdd(addr); idx != uint32(i) || err != nil {
			t.Errorf("failed to locate address %d", i)
		}
	}
}
