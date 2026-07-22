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
	"iter"
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
		file:                 file,
		flushBufferThreshold: flushBufferThreshold,
	}, nil
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
		if err := c.handleCacheSet(&key, &val, true); err != nil {
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

	return value, nil
}

func (c *KVCachedFile[K, V]) Set(key K, value V) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.handleCacheSet(&key, &value, true)
}

// Has checks if a key exists in the cache, the pending writes, or the
// underlying file. Unlike Get, it does not load the value into the cache.
func (c *KVCachedFile[K, V]) Has(key K) (bool, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	if _, inCache := c.cache.Get(key); inCache {
		return true, nil
	}
	if _, inFlushBuffer := c.flushBuffer[key]; inFlushBuffer {
		return true, nil
	}
	return c.file.Has(key)
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

// Flush writes all pending (dirty) key-value pairs to the disk.
func (c *KVCachedFile[K, V]) Flush() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.flushLocked()
}

// Size returns the number of keys handled by the CachedFile.
func (c *KVCachedFile[K, V]) Size() (uint64, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	uniqueKeys := uint64(0)

	countIfNotInFile := func(key K) error {
		has, err := c.file.Has(key)
		if err != nil {
			return err
		}
		if !has {
			uniqueKeys++
		}
		return nil
	}

	var err error
	c.cache.Iterate(func(key K, value V) bool {
		if _, inFlushBuffer := c.flushBuffer[key]; !inFlushBuffer {
			err2 := countIfNotInFile(key)
			if err2 != nil {
				err = err2
				return false
			}
		}
		return true
	})
	if err != nil {
		return 0, err
	}

	for key := range c.flushBuffer {
		err = countIfNotInFile(key)
		if err != nil {
			return 0, err
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

// Iterate returns an iterator over all key-value pairs handled by the
// KVCachedFile. Any pending writes are flushed to disk first so that the
// underlying file iterator observes a complete, up-to-date view.
func (c *KVCachedFile[K, V]) Iterate() (iter.Seq2[K, V], error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Flush any pending writes to ensure the underlying file is up-to-date.
	if err := c.flushLocked(); err != nil {
		return nil, err
	}

	// After flushing, every key/value pair lives on disk, so we can
	// delegate to the underlying file's iterator.
	return c.file.Iterate()
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
	var dirtySize uintptr
	for k := range c.dirty {
		dirtySize += unsafe.Sizeof(k) + unsafe.Sizeof(true)
	}
	return common.NewMemoryFootprint(unsafe.Sizeof(*c) + sizeValues + dirtySize + mf.Total() + fileFootprint.Total())
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
	}
	c.flushBuffer = make(map[K]V)
	return nil
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
