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
	"errors"
	"fmt"
	"io"
	"os"
	"sync"
	"unsafe"
)

// OffsetFile a KVFile that keeps track of the offsets of each key in the file.
type OffsetFile[K comparable, V any] struct {
	offsets map[K]uint64

	filePath     string
	fileSize     uint64
	readValueFn  readValueFn[K, V]
	writeValueFn writeValueFn[K, V]

	mutex sync.Mutex
}

func OpenOffsetFile[K comparable, V any](path string, readValueFn readValueFn[K, V], writeValueFn writeValueFn[K, V]) (*OffsetFile[K, V], error) {
	// Create the file if it does not exist.
	if _, err := os.Stat(path); os.IsNotExist(err) {
		if err := os.WriteFile(path, []byte{}, 0600); err != nil {
			return nil, err
		}
	}

	offsets, size, err := readFileOffsets[K](path, readValueFn)
	if err != nil {
		return nil, err
	}
	return &OffsetFile[K, V]{
		offsets:      offsets,
		filePath:     path,
		fileSize:     size,
		readValueFn:  readValueFn,
		writeValueFn: writeValueFn,
	}, nil
}

func (o *OffsetFile[K, V]) Set(key K, value V) (err error) {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	file, err := os.OpenFile(o.filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(err, file.Close())
	}()

	if err := o.writeEntryLocked(file, key, value); err != nil {
		return err
	}
	return o.updateFileSizeLocked(file)
}

func (o *OffsetFile[K, V]) SetBatch(pending map[K]V) (err error) {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	file, err := os.OpenFile(o.filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(err, file.Close())
	}()

	for key, value := range pending {
		if err := o.writeEntryLocked(file, key, value); err != nil {
			return err
		}
	}
	return o.updateFileSizeLocked(file)
}

func (c *OffsetFile[K, V]) Get(key K) (*V, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	offset, exists := c.offsets[key]
	if !exists {
		return nil, nil
	}
	k, v, err := readFromDiskAtOffset(c.filePath, offset, c.readValueFn)
	if err != nil {
		return nil, err
	}
	if *k != key {
		return nil, fmt.Errorf("key mismatch: expected %v, got %v", key, *k)
	}
	return v, nil
}

func (c *OffsetFile[K, V]) Flush() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	// No-op for OffsetFile, as writes are immediately persisted to disk.
	return nil
}

func (c *OffsetFile[K, V]) Close() error {
	// No-op for OffsetFile, as writes are immediately persisted to disk.
	return nil
}

func (c *OffsetFile[K, V]) Size() (uint64, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return uint64(len(c.offsets)), nil
}

func (c *OffsetFile[K, V]) FileSize() (uint64, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.fileSize, nil
}

// writeEntryLocked appends a single (key, value) entry to the given file and
// updates the in-memory offset map. The caller must hold o.mutex and must
// have opened the file in append mode.
func (o *OffsetFile[K, V]) writeEntryLocked(file *os.File, key K, value V) error {
	curOffset, err := file.Seek(0, io.SeekEnd)
	if err != nil {
		return err
	}
	if err := o.writeValueFn(file, key, value); err != nil {
		return err
	}
	o.offsets[key] = uint64(curOffset)
	return nil
}

// updateFileSizeLocked refreshes the cached file size from the given file.
// The caller must hold o.mutex.
func (o *OffsetFile[K, V]) updateFileSizeLocked(file *os.File) error {
	size, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	o.fileSize = uint64(size)
	return nil
}

func (c *OffsetFile[K, V]) GetAll() (map[K]V, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Add all values from the file on disk
	file, err := os.Open(c.filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	all := make(map[K]V)
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

func readFileOffsets[K comparable, V any](path string, readValueFn func(reader io.ReadSeeker) (K, V, error)) (map[K]uint64, uint64, error) {
	// If there is no file, initialize and return an empty value collection.
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
	data, err := parseOffsets(file, readValueFn)
	return data, uint64(info.Size()), err
}

func parseOffsets[K comparable, V any](reader io.ReadSeeker, readValueFn func(reader io.ReadSeeker) (K, V, error)) (map[K]uint64, error) {
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

func (c *OffsetFile[K, V]) GetMemoryFootprint() *MemoryFootprint {
	// The memory footprint of the OffsetFile is the size of the struct itself plus the size of the offsets map.
	c.mutex.Lock()
	defer c.mutex.Unlock()
	sizeOffsets := uint(0)
	for k, v := range c.offsets {
		sizeOffsets += uint(unsafe.Sizeof(k) + unsafe.Sizeof(v))
	}
	return NewMemoryFootprint(unsafe.Sizeof(*c) + uintptr(sizeOffsets))
}
