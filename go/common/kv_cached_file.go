// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package common

import (
	"fmt"
	"sync"
	"unsafe"
)

// KVCachedFile wraps a KVFile and provides an in-memory cache for key-value pairs.
// Writes to the file are buffered in memory and flushed to disk when a threshold is reached.
type KVCachedFile[K comparable, V any] struct {
	cache       *LruCache[K, V]
	flushBuffer map[K]V
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
		cache:                NewLruCache[K, V](cacheSize),
		flushBuffer:          make(map[K]V),
		file:                 file,
		flushBufferThreshold: flushBufferThreshold,
	}, nil
}

func (c *KVCachedFile[K, V]) Set(key K, value V) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	err := c.handleCacheSet(&key, &value)
	if err != nil {
		return err
	}
	return nil
}

func (c *KVCachedFile[K, V]) SetBatch(entries map[K]V) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	for key, value := range entries {
		err := c.handleCacheSet(&key, &value)
		if err != nil {
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
		err := c.handleCacheSet(&key, &val)
		if err != nil {
			return nil, err
		}
		return &val, nil
	}
	value, err := c.file.Get(key)
	if err != nil {
		return nil, err
	}
	return value, nil
}

// Flush writes all key-value pairs to the disk.
func (c *KVCachedFile[K, V]) Flush() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.flushLocked()
}

// flushLocked writes all cached and pending key-value pairs to the underlying
// file. This is intended to be called with the mutex locked.
func (c *KVCachedFile[K, V]) flushLocked() error {
	c.cache.Iterate(func(key K, value V) bool {
		c.flushBuffer[key] = value
		return true
	})

	return c.flushPending()
}

func (c *KVCachedFile[K, V]) handleCacheSet(key *K, value *V) error {
	evictedKey, evictedValue, evicted := c.cache.Set(*key, *value)
	if evicted {
		c.flushBuffer[evictedKey] = evictedValue
		if len(c.flushBuffer) >= c.flushBufferThreshold {
			return c.flushPending()
		}
	}
	return nil
}

// flushPending empties the flushBuffer by writing its contents to the
// underlying file via SetBatch and then calling Flush on it.
// This is intended to be called with the mutex locked.
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
	// Note: cannot use the builtin clear() because a test file in this
	// package declares a `clear` type that shadows the builtin.
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

	// Add all values from the flush buffer
	for key, value := range c.flushBuffer {
		all[key] = value
	}

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
			v, err2 := c.file.Get(key)
			if err2 != nil {
				err = err2
				return false
			}
			if v == nil {
				uniqueKeys++
			}
		}
		return true
	})
	if err != nil {
		return 0, err
	}
	for key := range c.flushBuffer {
		v, err := c.file.Get(key)
		if err != nil {
			return 0, err
		}
		if v == nil {
			uniqueKeys++
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

func (c *KVCachedFile[K, V]) GetMemoryFootprint() *MemoryFootprint {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	fileFootprint := c.file.GetMemoryFootprint()
	mf := c.cache.GetDynamicMemoryFootprint(func(v V) uintptr {
		return uintptr(unsafe.Sizeof(v))
	})
	var sizeValues uintptr
	for k, v := range c.flushBuffer {
		sizeValues += unsafe.Sizeof(k) + unsafe.Sizeof(v)
	}
	return NewMemoryFootprint(unsafe.Sizeof(*c) + sizeValues + mf.Total() + fileFootprint.Total())
}
