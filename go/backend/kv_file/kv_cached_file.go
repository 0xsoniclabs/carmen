// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package kv_file

import (
	"fmt"
	"maps"
	"sync"
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/common"
)

// KVCachedFile wraps a KVFile and provides an in-memory cache for key-value pairs.
// Writes to the file are buffered in memory and flushed to disk when a threshold is reached.
//
// The cache keeps track of which cached entries are "dirty" (i.e. have not yet
// been persisted to the underlying file). Only dirty entries are written on
// flush, so repeated Flush calls with no intervening Set do not rewrite the
// same values to disk.
type KVCachedFile[K comparable, V any] struct {
	cache       *common.LruCache[K, V]
	flushBuffer map[K]V
	dirty       map[K]bool
	file        KVFileWithMemoryFootprint[K, V]
	fromDisk    map[K]bool

	mutex                sync.Mutex
	flushBufferThreshold int
}

// OpenKVCachedFile wraps the given KVFile with a cache of the specified size and a flush buffer threshold.
func OpenKVCachedFile[K comparable, V any](file KVFileWithMemoryFootprint[K, V], cacheSize int, flushBufferThreshold int) (*KVCachedFile[K, V], error) {
	if file == nil {
		return nil, fmt.Errorf("file cannot be nil")
	}

	if cacheSize <= 0 || flushBufferThreshold <= 0 {
		return nil, fmt.Errorf("cacheSize and flushBufferThreshold must be greater than 0, got %d and %d", cacheSize, flushBufferThreshold)
	}

	return &KVCachedFile[K, V]{
		cache:                common.NewLruCache[K, V](cacheSize),
		flushBuffer:          make(map[K]V),
		dirty:                make(map[K]bool),
		fromDisk:             make(map[K]bool),
		file:                 file,
		flushBufferThreshold: flushBufferThreshold,
	}, nil
}

func (c *KVCachedFile[K, V]) Set(key K, value V) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.handleCacheSet(&key, &value, true)
}

func (c *KVCachedFile[K, V]) SetBatch(entries map[K]V) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	for key, value := range entries {
		if err := c.handleCacheSet(&key, &value, true); err != nil {
			return err
		}
	}

	return nil
}

// Get retrieves a value from the cache or disk.
// Entries found in the flushBuffer are promoted to the cache.
func (c *KVCachedFile[K, V]) Get(key K) (*V, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	if val, inCache := c.cache.Get(key); inCache {
		return &val, nil
	}
	if val, inFlushBuffer := c.flushBuffer[key]; inFlushBuffer {
		delete(c.flushBuffer, key)
		if err := c.handleCacheSet(&key, &val, false); err != nil {
			return nil, err
		}
		return &val, nil
	}
	value, err := c.file.Get(key)
	if err != nil {
		return nil, err
	}
	err = c.handleCacheSet(&key, value, false)
	if err != nil {
		return nil, err
	}
	c.fromDisk[key] = true

	return value, nil
}

// Flush writes all pending (dirty) key-value pairs to the disk.
func (c *KVCachedFile[K, V]) Flush() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.flushLocked()
}

// flushLocked moves all dirty cache entries into the flush buffer and then
// writes the buffer to the underlying file. Clean cache entries are left in
// place so that they are not re-written to disk. This is intended to be
// called with the mutex locked.
func (c *KVCachedFile[K, V]) flushLocked() error {
	c.cache.Iterate(func(key K, value V) bool {
		if _, isDirty := c.dirty[key]; isDirty {
			c.flushBuffer[key] = value
		}
		return true
	})

	return c.flushPending()
}

// handleCacheSet caches the given key/value pair, marking it dirty. If the
// insertion evicts another entry from the cache, that entry is moved into the
// flush buffer; when the buffer reaches the flush threshold the pending writes
// are persisted to the underlying file.
func (c *KVCachedFile[K, V]) handleCacheSet(key *K, value *V, dirty bool) error {
	if key == nil || value == nil {
		return nil // No-op
	}
	if dirty {
		c.dirty[*key] = true
	}
	evictedKey, evictedValue, evicted := c.cache.Set(*key, *value)
	if evicted {
		if _, isDirty := c.dirty[evictedKey]; isDirty {
			c.flushBuffer[evictedKey] = evictedValue
			if len(c.flushBuffer) >= c.flushBufferThreshold {
				return c.flushPending()
			}
		}
	}
	return nil
}

// flushPending empties the flushBuffer by writing its contents to the
// underlying file via SetBatch and then calling Flush on it. Keys that have
// been persisted are removed from the dirty set. This is intended to be
// called with the mutex locked.
func (c *KVCachedFile[K, V]) flushPending() error {
	if len(c.flushBuffer) == 0 {
		return nil
	}

	if err := c.file.SetBatch(c.flushBuffer); err != nil {
		return err
	}
	if err := c.file.Flush(); err != nil {
		return err
	}
	for k := range c.flushBuffer {
		delete(c.dirty, k)
		c.fromDisk[k] = true
	}
	c.flushBuffer = make(map[K]V)
	return nil
}

// GetAll returns a map of all keys and values handled by the KVCachedFile.
func (c *KVCachedFile[K, V]) GetAll() (map[K]V, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	all, err := c.file.GetAll()
	if err != nil {
		return nil, err
	}
	// Defensively initialize the map in case the underlying file returns
	// a nil map for empty content.
	if all == nil {
		all = make(map[K]V)
	}

	// Mark all keys from the underlying file as on disk
	for key := range all {
		c.fromDisk[key] = true
	}

	// Add all values from the flush buffer
	maps.Copy(all, c.flushBuffer)

	// Add all values from the cache
	c.cache.Iterate(func(key K, value V) bool {
		all[key] = value
		return true
	})

	return all, nil
}

// Size returns the number of keys handled by the CachedFile.
func (c *KVCachedFile[K, V]) Size() (uint64, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	uniqueKeys := uint64(0)
	var err error
	c.cache.Iterate(func(key K, value V) bool {
		if _, inFlushBuffer := c.flushBuffer[key]; !inFlushBuffer {
			if _, isFromDisk := c.fromDisk[key]; !isFromDisk {
				v, err2 := c.file.Get(key)
				if err2 != nil {
					err = err2
					return false
				}
				if v == nil {
					uniqueKeys++
				}
			}
		}
		return true
	})
	if err != nil {
		return 0, err
	}

	for key := range c.flushBuffer {
		if _, isFromDisk := c.fromDisk[key]; !isFromDisk {
			v, err := c.file.Get(key)
			if err != nil {
				return 0, err
			}
			if v == nil {
				uniqueKeys++
			}
		}
	}

	fileSize, err := c.file.Size()
	if err != nil {
		return 0, err
	}
	return uniqueKeys + fileSize, nil
}

func (c *KVCachedFile[K, V]) FileSize() (uint64, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.file.FileSize()
}

func (c *KVCachedFile[K, V]) Close() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	if err := c.flushLocked(); err != nil {
		return err
	}
	return c.file.Close()
}

func (c *KVCachedFile[K, V]) GetMemoryFootprint() *common.MemoryFootprint {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	fileFootprint := c.file.GetMemoryFootprint()
	valueSize := func(v V) uintptr {
		switch vv := any(v).(type) {
		case []byte:
			return uintptr(cap(vv))
		case string:
			return uintptr(len(vv))
		default:
			return uintptr(unsafe.Sizeof(v))
		}
	}
	mf := c.cache.GetDynamicMemoryFootprint(valueSize)
	var sizeValues uintptr
	for k, v := range c.flushBuffer {
		sizeValues += unsafe.Sizeof(k) + valueSize(v)
	}
	return common.NewMemoryFootprint(unsafe.Sizeof(*c) + sizeValues + mf.Total() + fileFootprint.Total())
}
