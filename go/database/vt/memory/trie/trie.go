// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package trie

import (
	"github.com/0xsoniclabs/carmen/go/database/vt/commit"
)

// Key is a fixed-size byte array used to address values in the trie.
type Key [32]byte

// Value is a fixed-size byte array used to represent data stored in the trie.
type Value [32]byte

// Trie implements a all-in-memory version of a Verkle trie as specified by
// Ethereum. It provides a basic key-value store with fixed-length keys and
// values and the ability to provide a cryptographic commitment of the trie's
// state using Pedersen commitments.
//
// This implementation is not optimized for performance or storage efficiency,
// but serves as a reference for the trie structure and operations. It is
// not intended for production use.
//
// For an overview of the Verkle trie structure, see
// https://blog.ethereum.org/2021/12/02/verkle-tree-structure
type Trie struct {
	root node
}

// Get retrieves the value associated with the given key from the trie. All keys
// that have not been set will return the zero value.
func (t *Trie) Get(key Key) Value {
	if t.root == nil {
		return Value{}
	}
	return t.root.get(key, 0)
}

// Set associates the given key with the specified value in the trie. If the key
// already exists, its value will be updated.
func (t *Trie) Set(key Key, value Value) {
	if t.root == nil {
		t.root = newLeaf(key)
	}
	t.root = t.root.set(key, 0, value)
}

// Commit returns the cryptographic commitment of the current state of the trie.
func (t *Trie) Commit() commit.Commitment {
	if t.root == nil {
		return commit.Identity()
	}
	return t.root.commit()
}
