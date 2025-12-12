// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package memory

import (
	"strings"

	"github.com/0xsoniclabs/carmen/go/database/vt/memory/trie"
	"github.com/0xsoniclabs/carmen/go/database/vt/reference"
	"github.com/0xsoniclabs/carmen/go/state"
)

// TODO: test support for sequential and parallel modes of tries;

// NewState creates a new, empty in-memory state instance.
func NewState(params state.Parameters) (state.State, error) {
	config := trie.TrieConfig{
		ParallelCommit: !strings.HasSuffix(string(params.Variant), "-seq"),
	}
	return reference.NewCustomState(
		params, trie.NewTrie(config), newEmbedding(),
	)
}
