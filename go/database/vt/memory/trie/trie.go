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
	"github.com/0xsoniclabs/carmen/go/database/vt/reference/trie"
	"github.com/0xsoniclabs/tracy"
)

// Key is a fixed-size byte array used to address values in the trie.
type Key = trie.Key

// Value is a fixed-size byte array used to represent data stored in the trie.
type Value = trie.Value

// Trie implements an all-in-memory version of a Verkle trie as specified by
// Ethereum. It provides a basic key-value store with fixed-length keys and
// values and the ability to provide a cryptographic commitment of the trie's
// state using Pedersen commitments.
//
// This implementation is optimized for performance, but does not offer
// persistence. It is not intended for production use.
//
// For an overview of the Verkle trie structure, see
// https://blog.ethereum.org/2021/12/02/verkle-tree-structure
type Trie struct {
	config TrieConfig
	root   node
}

type TrieConfig struct {
	ParallelCommit bool
}

func NewTrie(config TrieConfig) *Trie {
	return &Trie{
		config: config,
	}
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
		t.root = &inner{}
	}
	t.root = t.root.set(key, 0, value)
}

// Commit returns the cryptographic commitment of the current state of the trie.
func (t *Trie) Commit() commit.Commitment {
	if t.root == nil {
		return commit.Identity()
	}
	if t.config.ParallelCommit {
		return t.commit_parallel()
	}
	return t.commit_sequential()
}

func (t *Trie) commit_sequential() commit.Commitment {
	return t.root.commit()
}

func (t *Trie) commit_parallel() commit.Commitment {
	// Phase 1: collect tasks to be done in parallel
	tasks := make([]*task, 0, 1024)
	zone := tracy.ZoneBegin("trie::commit_parallel::collect_tasks")
	t.root.collectCommitTasks(&tasks)
	zone.End()

	// Phase 2: run tasks in parallel
	zone = tracy.ZoneBegin("trie::commit_parallel::run_tasks")
	runTasks(tasks)
	zone.End()

	// Phase 3: fetch new root commitment
	zone2 := tracy.ZoneBegin("trie::commit_parallel::fetch_root_commitment")
	defer zone2.End()
	return t.root.commit()
}
