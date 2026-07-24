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
	"errors"
	"fmt"
	"io"
	"iter"
	"os"
	"sync"
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/common"
)

// OrderedFile is a KVFile that stores fixed-size values in a single file,
// addressed by their index: the value for key k occupies the k-th slot of the
// file. It is suited for dense, sequentially numbered data. Safe for
// concurrent use.
type OrderedFile[V any] struct {
	file     *os.File
	filepath string

	itemSize     uint64 // item size in bytes
	readValueFn  readValueFn[uint64, V]
	writeValueFn writeValueFn[uint64, V]

	// mutex guards all file access. File access needs no explicit close-state
	// tracking: os.File refcounts its descriptor internally, so any operation
	// racing with (or following) Close fails with os.ErrClosed instead of
	// touching a recycled descriptor.
	mutex sync.Mutex
}

// OpenOrderedFile opens an OrderedFile at the given path, creating the file if
// it does not exist.
// The key passed to `readValueFn` and `writeValueFn` is ignored, as the key is implicit in the file offset.
func OpenOrderedFile[V any](path string, itemSize uint64, readValueFn readValueFn[uint64, V], writeValueFn writeValueFn[uint64, V]) (*OrderedFile[V], error) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		if err := os.WriteFile(path, []byte{}, 0600); err != nil {
			return nil, err
		}
	}

	file, err := os.OpenFile(path, os.O_RDWR, 0600)
	if err != nil {
		return nil, err
	}

	// The file must be a whole number of fixed-size records.
	fileSize, err := file.Stat()
	if err != nil {
		return nil, errors.Join(err, file.Close())
	}
	if fileSize.Size()%int64(itemSize) != 0 {
		return nil, errors.Join(
			fmt.Errorf("invalid root file format: size %d is not a multiple of entry size %d", fileSize.Size(), itemSize),
			file.Close(),
		)
	}

	return &OrderedFile[V]{
		file:         file,
		filepath:     path,
		itemSize:     itemSize,
		readValueFn:  readValueFn,
		writeValueFn: writeValueFn,
	}, nil
}

func (o *OrderedFile[V]) Get(key uint64) (*V, error) {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	// Keys beyond the end of the file are treated as missing, per the KVFile
	// contract (Get returns (nil, nil) for keys that do not exist).
	size, err := o.sizeLocked()
	if err != nil {
		return nil, err
	}
	if key >= size {
		return nil, nil
	}

	value, err := o.readAtLocked(key)
	if err != nil {
		return nil, err
	}
	return &value, nil
}

// Has reports whether a value is stored for the given key.
func (o *OrderedFile[V]) Has(key uint64) (bool, error) {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	size, err := o.sizeLocked()
	if err != nil {
		return false, err
	}
	return key < size, nil
}

func (o *OrderedFile[V]) Set(key uint64, value V) error {
	return o.SetBatch(map[uint64]V{key: value})
}

func (o *OrderedFile[V]) SetBatch(entries map[uint64]V) error {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	for key, value := range entries {
		_, err := o.file.Seek(int64(key*o.itemSize), io.SeekStart)
		if err != nil {
			return err
		}

		err = o.writeValueFn(o.file, key, value)
		if err != nil {
			return err
		}
	}

	return nil
}

func (o *OrderedFile[V]) Flush() error {
	// No-op: writes are persisted immediately.
	return nil
}

func (o *OrderedFile[V]) FileSize() (uint64, error) {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	info, err := o.file.Stat()
	if err != nil {
		return 0, err
	}

	return uint64(info.Size()), nil
}

// Iterate returns an iterator over all stored key-value pairs in ascending key
// order. The set of keys visited is fixed when Iterate is called.
func (o *OrderedFile[V]) Iterate() (iter.Seq2[uint64, V], error) {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	size, err := o.sizeLocked()
	if err != nil {
		return nil, err
	}

	return func(yield func(uint64, V) bool) {
		for key := range size {
			o.mutex.Lock()
			value, err := o.readAtLocked(key)
			o.mutex.Unlock()
			if err != nil {
				// Includes os.ErrClosed when the file is closed mid-iteration.
				return
			}
			if !yield(key, value) {
				return
			}
		}
	}, nil
}

// Close closes the underlying file. It is idempotent.
func (o *OrderedFile[V]) Close() error {
	// A concurrent close surfaces as os.ErrClosed to in-flight readers and to
	// a repeated Close; both are reported as success.
	if err := o.file.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
		return err
	}
	return nil
}

// GetMemoryFootprint returns the memory footprint of the OrderedFile.
func (o *OrderedFile[V]) GetMemoryFootprint() *common.MemoryFootprint {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	return common.NewMemoryFootprint(unsafe.Sizeof(*o))
}

// readAtLocked reads the record for the given key. The caller must hold
// o.mutex, since records are overwritten in place and a read racing a write to
// the same record could observe a torn value.
func (o *OrderedFile[V]) readAtLocked(key uint64) (V, error) {
	// A positioned read bounded to one record so a decoder cannot overrun into
	// a neighbouring record.
	section := io.NewSectionReader(o.file, int64(key*o.itemSize), int64(o.itemSize))
	_, value, err := o.readValueFn(section)
	return value, err
}

// sizeLocked returns the number of items in the file. The caller must hold
// o.mutex.
func (o *OrderedFile[V]) sizeLocked() (uint64, error) {
	info, err := o.file.Stat()
	if err != nil {
		return 0, err
	}

	return uint64(info.Size()) / o.itemSize, nil
}
