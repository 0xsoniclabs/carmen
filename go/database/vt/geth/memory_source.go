// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package geth

import (
	"github.com/0xsoniclabs/carmen/go/common/immutable"
	"github.com/ethereum/go-ethereum/common"
)

// NodeSource is an interface for retrieving and storing nodes in memory.
// It serves as a base for different node storage implementations, such as
// in-memory storage, cached storage, or persistent storage.
// It is itself not intended for real-life usage, as the amount of memory
// would quickly grow to an unmanageable size.
type memorySource struct {
	nodes map[immutable.Bytes][]byte
}

func NewMemorySource() NodeSource {
	return &memorySource{
		nodes: make(map[immutable.Bytes][]byte),
	}
}

func (s *memorySource) Node(owner common.Hash, path []byte, hash common.Hash) ([]byte, error) {
	key := immutable.NewBytes(path)
	return s.nodes[key], nil
}

func (s *memorySource) Set(path []byte, value []byte) error {
	s.nodes[immutable.NewBytes(path)] = value

	return nil
}

func (s *memorySource) Flush() error {
	return nil
}

func (s *memorySource) Close() error {
	return s.Flush()
}
