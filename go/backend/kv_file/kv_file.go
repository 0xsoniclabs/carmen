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
	"io"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/iter_utils"
)

//go:generate mockgen -source kv_file.go -destination kv_file_mocks.go -package kv_file

// KVFile is a file that supports key-value operations.
// Key-value pairs can be updated but not deleted.
type KVFile[K comparable, V any] interface {
	// Get retrieves a value from the file by key.
	// Returns nil if the key does not exist.
	Get(key K) (*V, error)

	// Has checks if a key exists in the file.
	Has(key K) (bool, error)

	// Set adds a key-value pair to the file.
	Set(key K, value V) error

	// SetBatch adds multiple key-value pairs to the file.
	SetBatch(entries map[K]V) error

	// Flush ensures that all buffered data is written to disk.
	Flush() error

	// FileSize returns the size of the file in bytes.
	FileSize() (uint64, error)

	// Iterate returns an iterator over the key-value pairs in the file. A pair
	// that cannot be read is yielded as an error element which ends the
	// iteration, so consumers must inspect every element to tell a complete
	// iteration from one cut short by a failure.
	Iterate() (iter_utils.ResultSeq2[K, V], error)

	// Close closes the file and releases any resources associated with it.
	Close() error
}

// KVFileWithMemoryFootprint extends KVFile with the ability to report its
// current in-memory footprint.
type KVFileWithMemoryFootprint[K comparable, V any] interface {
	KVFile[K, V]
	GetMemoryFootprint() *common.MemoryFootprint
}

// readValueFn is a utility type for reading a key-value pair from an io.Reader.
// The reader is already positioned at the start of the record; implementations
// must consume exactly one record.
//
//nolint:unused
type readValueFn[K comparable, V any] func(reader io.Reader) (K, V, error)

// writeValueFn is a utility type for writing a key-value pair to an io.Writer.
//
//nolint:unused
type writeValueFn[K comparable, V any] func(writer io.Writer, key K, value V) error
