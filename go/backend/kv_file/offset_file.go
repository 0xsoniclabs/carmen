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
	"maps"
	"os"
	"sync"
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/common"
)

// OffsetFile is a KVFile backed by a single append-only file with an in-memory
// index mapping each key to the offset of its most recent record. Updating a
// key appends a new record and repoints the index; existing records are never
// modified.
type OffsetFile[K comparable, V any] struct {
	offsets map[K]uint64

	file         *os.File
	filePath     string
	fileSize     uint64
	readValueFn  readValueFn[K, V]
	writeValueFn writeValueFn[K, V]

	// mutex guards offsets and fileSize. File access needs no explicit
	// close-state tracking: os.File refcounts its descriptor internally, so
	// any operation racing with (or following) Close fails with os.ErrClosed
	// instead of touching a recycled descriptor.
	mutex sync.Mutex
}

func OpenOffsetFile[K comparable, V any](path string, readValueFn readValueFn[K, V], writeValueFn writeValueFn[K, V]) (*OffsetFile[K, V], error) {
	file, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0600)
	if err != nil {
		return nil, err
	}

	// Scan the existing content to build the in-memory offset index. The
	// file descriptor is retained for the lifetime of the OffsetFile so
	// that subsequent reads and writes can reuse it.
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return nil, errors.Join(err, file.Close())
	}
	offsets, err := parseOffsets(file, readValueFn)
	if err != nil {
		return nil, errors.Join(err, file.Close())
	}
	info, err := file.Stat()
	if err != nil {
		return nil, errors.Join(err, file.Close())
	}

	return &OffsetFile[K, V]{
		offsets:      offsets,
		file:         file,
		filePath:     path,
		fileSize:     uint64(info.Size()),
		readValueFn:  readValueFn,
		writeValueFn: writeValueFn,
	}, nil
}

func (o *OffsetFile[K, V]) Set(key K, value V) error {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	if err := o.writeEntryLocked(key, value); err != nil {
		return err
	}
	return o.updateFileSizeLocked()
}

func (o *OffsetFile[K, V]) SetBatch(pending map[K]V) error {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	for key, value := range pending {
		if err := o.writeEntryLocked(key, value); err != nil {
			return err
		}
	}
	return o.updateFileSizeLocked()
}

func (o *OffsetFile[K, V]) Get(key K) (*V, error) {
	o.mutex.Lock()
	offset, exists := o.offsets[key]
	limit := o.fileSize
	o.mutex.Unlock()
	if !exists {
		return nil, nil
	}
	// The record at the resolved offset is immutable (the file is
	// append-only), so the read itself needs no synchronisation with
	// concurrent writers; positioned reads do not touch the shared seek
	// position. A concurrent Close makes the read fail with os.ErrClosed.
	k, v, err := readFromDiskAtOffset(o.file, offset, limit, o.readValueFn)
	if err != nil {
		return nil, err
	}
	if *k != key {
		return nil, fmt.Errorf("key mismatch: expected %v, got %v", key, *k)
	}
	return v, nil
}

func (o *OffsetFile[K, V]) Flush() error {
	o.mutex.Lock()
	defer o.mutex.Unlock()
	// No-op for OffsetFile, as writes are immediately persisted to disk.
	return nil
}

// Close closes the underlying file descriptor. Closing is idempotent: repeated
// calls report success without any effect. In-flight readers and iterators
// observe the close as os.ErrClosed on their next file access; os.File
// guarantees that a concurrent Close never lets an access touch a recycled
// descriptor.
func (o *OffsetFile[K, V]) Close() error {
	if err := o.file.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
		return err
	}
	return nil
}

func (o *OffsetFile[K, V]) Size() (uint64, error) {
	o.mutex.Lock()
	defer o.mutex.Unlock()
	return uint64(len(o.offsets)), nil
}

func (o *OffsetFile[K, V]) FileSize() (uint64, error) {
	o.mutex.Lock()
	defer o.mutex.Unlock()
	return o.fileSize, nil
}

// writeEntryLocked appends a single (key, value) entry to the retained file
// and updates the in-memory offset map. The caller must hold o.mutex and
// must have verified that the file is not closed.
func (o *OffsetFile[K, V]) writeEntryLocked(key K, value V) error {
	curOffset, err := o.file.Seek(0, io.SeekEnd)
	if err != nil {
		return err
	}
	if err := o.writeValueFn(o.file, key, value); err != nil {
		return err
	}
	o.offsets[key] = uint64(curOffset)
	return nil
}

// updateFileSizeLocked refreshes the cached file size from the retained file.
// The caller must hold o.mutex.
func (o *OffsetFile[K, V]) updateFileSizeLocked() error {
	size, err := o.file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	o.fileSize = uint64(size)
	return nil
}

func (o *OffsetFile[K, V]) Has(key K) (bool, error) {
	o.mutex.Lock()
	defer o.mutex.Unlock()
	_, exists := o.offsets[key]
	return exists, nil
}

func (o *OffsetFile[K, V]) Iterate() (iter.Seq2[K, V], error) {
	o.mutex.Lock()
	defer o.mutex.Unlock()

	// The iterator operates on a snapshot: the offsets are cloned here and the
	// records they point to are immutable because the file is append-only.
	// Reads use positioned reads that do not touch the shared seek position,
	// so the iterator does not need to hold the mutex. If the file is closed
	// mid-iteration, the next read fails with os.ErrClosed and the iteration
	// terminates.
	offsets := maps.Clone(o.offsets)
	limit := o.fileSize
	return func(yield func(K, V) bool) {
		for key, offset := range offsets {
			_, value, err := readFromDiskAtOffset(o.file, offset, limit, o.readValueFn)
			if err != nil {
				return
			}
			if !yield(key, *value) {
				return
			}
		}
	}, nil
}

func (o *OffsetFile[K, V]) GetMemoryFootprint() *common.MemoryFootprint {
	// The memory footprint of the OffsetFile is the size of the struct itself plus the size of the offsets map.
	o.mutex.Lock()
	defer o.mutex.Unlock()
	sizeOffsets := uint(0)
	for k, v := range o.offsets {
		sizeOffsets += uint(unsafe.Sizeof(k) + unsafe.Sizeof(v))
	}
	return common.NewMemoryFootprint(unsafe.Sizeof(*o) + uintptr(sizeOffsets))
}

// parseOffsets scans a reader from its current position to EOF, calling
// readValueFn to determine each record boundary, and returns a map from the
// key of each record to its starting offset in the reader.
func parseOffsets[K comparable, V any](reader io.ReadSeeker, readValueFn readValueFn[K, V]) (map[K]uint64, error) {
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

// readFromDiskAtOffset reads a single (key, value) record starting at `offset`
// using readValueFn. The record must end at or before `limit` (the file size at
// the time the offset was resolved). Reads go through io.SectionReader and thus
// use positioned reads (pread) that do not affect the file's seek position, so
// no synchronisation with concurrent writers is required; reading from a file
// that was closed concurrently fails with os.ErrClosed. On error the returned
// key and value pointers are nil so that callers can distinguish a read
// failure from a legitimate zero-valued record.
func readFromDiskAtOffset[K comparable, V any](file io.ReaderAt, offset, limit uint64, readValueFn readValueFn[K, V]) (*K, *V, error) {
	section := io.NewSectionReader(file, int64(offset), int64(limit-offset))
	k, v, err := readValueFn(section)
	if err != nil {
		return nil, nil, err
	}
	return &k, &v, nil
}
