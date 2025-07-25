// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package multimap

import "github.com/0xsoniclabs/carmen/go/common"

// MultiMap defines the interface for mapping keys to sets of multiple values.
type MultiMap[K common.Identifier, V common.Identifier] interface {
	// Add adds the given key/value pair.
	Add(key K, value V) error

	// Remove removes a single key/value entry.
	Remove(key K, value V) error

	// RemoveAll removes all entries with the given key.
	RemoveAll(key K) error

	// GetAll provides all values associated with the given key.
	GetAll(key K) ([]V, error)

	// provides the size of the store in memory in bytes
	common.MemoryFootprintProvider

	// needs to be flush and closable
	common.FlushAndCloser
}
