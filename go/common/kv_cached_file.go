package common

import (
	"errors"
	"io"
	"os"
	"sync"
	"unsafe"
)

type readValueFn[K comparable, V any] func(reader io.ReadSeeker) (K, V, error)
type writeValueFn[K comparable, V any] func(writer io.Writer, key K, value V) error

// KVCachedFile is a KV-store that caches values in memory and persists them to disk.
// Writes to the file are buffered in memory and flushed to disk when a threshold is reached.
type KVCachedFile[K comparable, V any] struct {
	cache        *LruCache[K, V]
	flushBuffer  map[K]V
	offsets      map[K]uint64
	filePath     string
	fileSize     uint64
	readValueFn  readValueFn[K, V]
	writeValueFn writeValueFn[K, V]

	mutex                sync.Mutex
	flushBufferThreshold int
}

func OpenKVCachedFile[K comparable, V any](path string, headerSize uint64, cacheSize int, flushBufferThreshold int, readValueFn readValueFn[K, V],
	writeValueFn writeValueFn[K, V]) (*KVCachedFile[K, V], error) {

	// Create the file if it does not exist.
	if _, err := os.Stat(path); os.IsNotExist(err) {
		if err := os.WriteFile(path, []byte{}, 0600); err != nil {
			return nil, err
		}
	}

	offsets, size, err := readFileOffsets[K](path, headerSize, readValueFn)
	if err != nil {
		return nil, err
	}

	return InitKVCachedFileWith[K, V](path, size, offsets, cacheSize, flushBufferThreshold, readValueFn, writeValueFn)
}

func InitKVCachedFileWith[K comparable, V any](path string, size uint64, offsets map[K]uint64, cacheSize int, flushBufferThreshold int, readValueFn readValueFn[K, V],
	writeValueFn writeValueFn[K, V]) (*KVCachedFile[K, V], error) {
	return &KVCachedFile[K, V]{
		filePath:             path,
		fileSize:             size,
		cache:                NewLruCache[K, V](cacheSize),
		flushBuffer:          make(map[K]V),
		offsets:              offsets,
		readValueFn:          readValueFn,
		writeValueFn:         writeValueFn,
		flushBufferThreshold: flushBufferThreshold,
	}, nil
}

func (c *KVCachedFile[K, V]) Set(key K, value V) error {
	c.mutex.Lock()
	if _, inCache := c.cache.Get(key); !inCache {
		if _, inFlushBuffer := c.flushBuffer[key]; !inFlushBuffer {
			if _, onDisk := c.offsets[key]; !onDisk {
				c.handleCacheSet(&key, &value)
			}
		}
	}
	c.mutex.Unlock()
	return nil
}

func (c *KVCachedFile[K, V]) handleCacheSet(key *K, value *V) error {
	evictedKey, evictedValue, evicted := c.cache.Set(*key, *value)
	if evicted {
		if _, onDisk := c.offsets[evictedKey]; !onDisk {
			if _, found := c.flushBuffer[evictedKey]; !found {
				c.flushBuffer[evictedKey] = evictedValue
			}
			if len(c.flushBuffer) >= c.flushBufferThreshold {
				return c.flushPending()
			}
		}
	}
	return nil
}

func (c *KVCachedFile[K, V]) Get(key K) (*V, error) {
	c.mutex.Lock()
	if val, inCache := c.cache.Get(key); inCache {
		c.mutex.Unlock()
		return &val, nil
	}
	if val, inFlushBuffer := c.flushBuffer[key]; inFlushBuffer {
		c.mutex.Unlock()
		delete(c.flushBuffer, key)
		err := c.handleCacheSet(&key, &val)
		if err != nil {
			return nil, err
		}
		return &val, nil
	}
	if offset, found := c.offsets[key]; found {
		key, val, err := c.readFromDiskAtOffset(offset)
		if err != nil {
			c.mutex.Unlock()
			return nil, err
		}
		c.handleCacheSet(key, val)
		c.mutex.Unlock()
		return val, nil
	}
	c.mutex.Unlock()
	return nil, nil
}

// readFromDiskAtOffset reads a key-value pair from the file at the given offset.
// It must be called with the mutex locked.
func (c *KVCachedFile[K, V]) readFromDiskAtOffset(offset uint64) (*K, *V, error) {
	return readFromDiskAtOffset[K, V](c.filePath, offset, c.readValueFn)
}

func (c *KVCachedFile[K, V]) Flush() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.cache.Iterate(func(key K, value V) bool {
		if _, onDisk := c.offsets[key]; !onDisk {
			c.flushBuffer[key] = value
		}
		return true
	})

	return c.flushPending()
}

// flushPending empties the flushBuffer and writes the values to disk.
// The offsets and file size are updated.
// This is intended to be called with the mutex locked.
func (c *KVCachedFile[K, V]) flushPending() error {
	if len(c.flushBuffer) == 0 {
		return nil
	}

	offsets, size, err := appendToFile[K, V](c.flushBuffer, c.filePath, c.writeValueFn)
	if err != nil {
		return err
	}
	c.fileSize = size
	c.flushBuffer = make(map[K]V)
	for hash, offset := range offsets {
		c.offsets[hash] = offset
	}
	return nil
}

// GetAll returns a map of all keys and values handled by the KVCachedFile.
func (c *KVCachedFile[K, V]) GetAll() (map[K]V, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	all := make(map[K]V)

	// Add all values from the cache
	c.cache.Iterate(func(key K, value V) bool {
		all[key] = value
		return true
	})

	// Add all values from the flush buffer
	for key, value := range c.flushBuffer {
		all[key] = value
	}

	// Add all values from the file on disk
	file, err := os.Open(c.filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	for key, offset := range c.offsets {
		_, err := file.Seek(int64(offset), io.SeekStart)
		if err != nil {
			return nil, err
		}
		readKey, value, err := c.readValueFn(file)
		if err != nil {
			return nil, err
		}
		if readKey != key {
			return nil, errors.New("key mismatch when reading from file")
		}
		all[key] = value
	}

	return all, nil
}

func (c *KVCachedFile[K, V]) FileSize() uint64 {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.fileSize
}

// Size returns the number of keys handled by the CachedFile.
func (c *KVCachedFile[K, V]) Size() int {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	uniqueKeys := 0
	c.cache.Iterate(func(key K, value V) bool {
		if _, inFlushBuffer := c.flushBuffer[key]; !inFlushBuffer {
			if _, onDisk := c.offsets[key]; !onDisk {
				uniqueKeys++
			}
		}
		return true
	})
	return uniqueKeys + len(c.flushBuffer) + len(c.offsets)
}

func (c *KVCachedFile[K, V]) GetMemoryFootprint() *MemoryFootprint {
	var sizeCodes uint
	c.mutex.Lock()
	for k, v := range c.offsets {
		sizeCodes += uint(uint(unsafe.Sizeof(k)) + uint(unsafe.Sizeof(v)))
	}
	mf := c.cache.GetDynamicMemoryFootprint(func(v V) uintptr {
		return uintptr(unsafe.Sizeof(v))
	})
	for k, v := range c.flushBuffer {
		sizeCodes += uint(unsafe.Sizeof(k) + unsafe.Sizeof(v))
	}
	c.mutex.Unlock()
	return NewMemoryFootprint(unsafe.Sizeof(*c) + uintptr(sizeCodes) + mf.Total())
}

func readFileOffsets[K comparable, V any](path string, headerSize uint64, readValueFn func(reader io.ReadSeeker) (K, V, error)) (map[K]uint64, uint64, error) {
	// If there is no file, initialize and return an empty code collection.
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return map[K]uint64{}, 0, nil
	}
	if err != nil {
		return nil, 0, err
	}

	file, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer file.Close()
	if headerSize > 0 {
		_, err := file.Seek(int64(headerSize), io.SeekStart)
		if err != nil {
			return nil, 0, err
		}
	}
	data, err := parseOffets(file, readValueFn)
	return data, uint64(info.Size()), err
}

func parseOffets[K comparable, V any](reader io.ReadSeeker, readValueFn func(reader io.ReadSeeker) (K, V, error)) (map[K]uint64, error) {
	res := map[K]uint64{}
	for {
		offset, err := reader.Seek(0, io.SeekCurrent)
		if err != nil {
			return nil, err
		}
		key, _, err := readValueFn(reader)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		res[key] = uint64(offset)
	}
	return res, nil
}

// appendCodes appends the given map of codes to the given file.
func appendToFile[K comparable, V any](pending map[K]V, filename string, writeValueFn writeValueFn[K, V]) (offsets map[K]uint64, fileSize uint64, err error) {
	file, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return nil, 0, err
	}
	offsets = make(map[K]uint64)
	for key, value := range pending {
		curOffset, err := file.Seek(0, io.SeekEnd)
		if err != nil {
			return nil, 0, err
		}
		err = writeValueFn(file, key, value)
		if err != nil {
			return nil, 0, err
		}
		offsets[key] = uint64(curOffset)
	}
	size, err2 := file.Seek(0, io.SeekCurrent)
	return offsets, uint64(size), errors.Join(err2, file.Close())
}

func readFromDiskAtOffset[K comparable, V any](path string, offset uint64, readValueFn readValueFn[K, V]) (*K, *V, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	_, err = file.Seek(int64(offset), io.SeekStart)
	if err != nil {
		return nil, nil, err
	}

	key, value, err := readValueFn(file)
	if err != nil {
		return nil, nil, err
	}
	return &key, &value, nil
}
