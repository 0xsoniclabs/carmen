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
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/database/vt/reference"
	"github.com/holiman/uint256"
	"github.com/stretchr/testify/require"
)

func TestCachedIndexer_ProducesSameKeyAsReference(t *testing.T) {
	require := require.New(t)

	indexer := newCachedIndexer()
	reference := reference.Embedding{}

	address := common.Address{10, 20, 30, 40, 50}
	index := uint256.NewInt(123456789)
	subIndex := byte(15)

	key1 := indexer.GetTrieKey(address, *index, subIndex)
	key2 := reference.GetTrieKey(address, *index, subIndex)
	require.Equal(key1, key2)
}

func TestCachedIndexer_CachesResults(t *testing.T) {
	require := require.New(t)

	indexer := newCachedIndexer()
	require.Empty(indexer.cache)

	address := common.Address{1, 2, 3, 4, 5}
	index := uint256.NewInt(42)
	subIndex := byte(7)

	cacheKey := trieKeyCacheKey{
		address: address,
		index:   *index,
	}

	_, found := indexer.cache[cacheKey]
	require.False(found)

	indexer.GetTrieKey(address, *index, subIndex)
	_, found = indexer.cache[cacheKey]
	require.True(found, "Key should be cached after first retrieval")
}

func TestCacheIndexer_IsThreadSafe(t *testing.T) {
	// This test fails if data races are detected when run with the --race flag.
	indexer := newCachedIndexer()

	address := common.Address{5, 4, 3, 2, 1}
	index := uint256.NewInt(987654321)
	subIndex := byte(3)

	const N = 5
	var wg sync.WaitGroup
	wg.Add(N)
	for range N {
		go func() {
			defer wg.Done()
			for range 10 {
				indexer.GetTrieKey(address, *index, subIndex)
			}
		}()
	}
	wg.Wait()
}
