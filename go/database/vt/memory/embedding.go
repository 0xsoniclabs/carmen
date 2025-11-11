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
	"sync"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/database/vt/memory/trie"
	"github.com/0xsoniclabs/carmen/go/database/vt/reference"
	"github.com/holiman/uint256"
)

func newEmbedding() reference.Embedding {
	return reference.NewEmbeddingWithIndexer(&cachedIndexer{})
}

type cachedIndexer struct {
	cache map[trieKeyCacheKey]trie.Key
	lock  sync.Mutex
}

func newCachedIndexer() *cachedIndexer {
	return &cachedIndexer{
		cache: make(map[trieKeyCacheKey]trie.Key),
	}
}

func (i *cachedIndexer) GetTrieKey(
	address common.Address,
	index uint256.Int,
	subIndex byte,
) trie.Key {
	i.lock.Lock()
	defer i.lock.Unlock()
	cacheKey := trieKeyCacheKey{
		address: address,
		index:   index,
	}
	if key, exists := i.cache[cacheKey]; exists {
		key[31] = subIndex
		return key
	}
	embedding := reference.Embedding{}
	key := embedding.GetTrieKey(address, index, subIndex)
	i.cache[cacheKey] = key
	return key
}

type trieKeyCacheKey struct {
	address common.Address
	index   uint256.Int
}
