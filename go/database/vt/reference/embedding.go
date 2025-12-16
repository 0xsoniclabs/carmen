// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package reference

import (
	"slices"
	"sync"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/database/vt/commit"
	"github.com/0xsoniclabs/carmen/go/database/vt/reference/trie"
	"github.com/holiman/uint256"
)

var (
	zero                                = uint256.NewInt(0)
	headerStorageOffset                 = uint256.NewInt(64)
	codeOffset                          = uint256.NewInt(128)
	codeStorageDelta                    = new(uint256.Int).Sub(codeOffset, headerStorageOffset)
	verkleNodeWidth                     = uint256.NewInt(256)
	verkleNodeWidthLog2                 = 8
	mainStorageOffsetLshVerkleNodeWidth = new(uint256.Int).Lsh(uint256.NewInt(1), 248-uint(verkleNodeWidthLog2))
)

// embedding provides methods to compute Verkle trie keys for various
// account-related data fields.
type embedding struct {
	cache keyCache
}

// getBasicDataKey returns the Verkle trie key of the basic data field for
// the specified account. The basic fields cover the code size, nonce, and
// balance of the account.
func (e *embedding) getBasicDataKey(address common.Address) trie.Key {
	return e.getTrieKey(address, *uint256.NewInt(0), 0)
}

// getCodeHashKey returns the Verkle trie key of the code hash field for
// the specified account.
func (e *embedding) getCodeHashKey(address common.Address) trie.Key {
	return e.getTrieKey(address, *uint256.NewInt(0), 1)
}

func (e *embedding) getCodeChunkKey(address common.Address, chunkNumber int) trie.Key {
	// Derived from
	// https://github.com/0xsoniclabs/go-ethereum/blob/e563918a84b4104e44935ddc6850f11738dcc3f5/trie/utils/verkle.go#L188
	var (
		chunk                  = uint256.NewInt(uint64(chunkNumber))
		chunkOffset            = new(uint256.Int).Add(codeOffset, chunk)
		treeIndex, subIndexMod = new(uint256.Int).DivMod(chunkOffset, verkleNodeWidth, new(uint256.Int))
		subIndex               = byte(subIndexMod.Uint64())
	)
	return e.getTrieKey(address, *treeIndex, subIndex)
}

// getStorageKey returns the Verkle trie key of the storage slot addressed an
// address/key pair.
func (e *embedding) getStorageKey(address common.Address, key common.Key) trie.Key {
	// Derived from
	// https://github.com/0xsoniclabs/go-ethereum/blob/e563918a84b4104e44935ddc6850f11738dcc3f5/trie/utils/verkle.go#L203

	var suffix byte
	var treeIndex uint256.Int
	treeIndex.SetBytes(key[:])
	if treeIndex.Cmp(codeStorageDelta) < 0 {
		treeIndex.Add(headerStorageOffset, &treeIndex)
		suffix = byte(treeIndex[0] & 0xFF)
		treeIndex = *zero
	} else {
		suffix = key[len(key)-1]
		treeIndex.Rsh(&treeIndex, 8)
		treeIndex.Add(&treeIndex, mainStorageOffsetLshVerkleNodeWidth)
	}

	return e.getTrieKey(address, treeIndex, suffix)
}

func (e *embedding) getTrieKey(
	address common.Address,
	index uint256.Int,
	subIndex byte,
) trie.Key {
	return e.cache.getTrieKey(address, index, subIndex)
}

// keyCache is a naive, never-evicting cache for computed trie keys.
// TODO: Replace with a proper cache with eviction policy.
type keyCache struct {
	mu    sync.Mutex
	cache map[keyCacheKey]trie.Key
}

type keyCacheKey struct {
	address common.Address
	index   uint256.Int
}

// getTrieKey is a helper function for hashing information to obtain trie keys.
func (c *keyCache) getTrieKey(
	address common.Address,
	treeIndex uint256.Int,
	subIndex byte,
) trie.Key {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check the cache first.
	key := keyCacheKey{
		address: address,
		index:   treeIndex,
	}
	if res, ok := c.cache[key]; ok {
		res[31] = subIndex
		return res
	}

	res := _getTrieKey(address, treeIndex, subIndex)
	if c.cache == nil {
		c.cache = make(map[keyCacheKey]trie.Key)
	}
	c.cache[key] = res
	return res
}

// _getTrieKey is the uncached version of getTrieKey.
func _getTrieKey(
	address common.Address,
	treeIndex uint256.Int,
	subIndex byte,
) trie.Key {
	// Compute the key for the given address and index.
	// Inspired by https://github.com/0xsoniclabs/go-ethereum/blob/e563918a84b4104e44935ddc6850f11738dcc3f5/trie/utils/verkle.go#L116

	// The key is computed by:
	//   C = Commit([2+256*64,address_low,address_high,tree_index_low,tree_index_high])
	//   H = Hash(C)
	//   K = append(H[:31], subIndex)

	expanded := [32]byte{}
	copy(expanded[12:], address[:])

	values := [256]commit.Value{}
	values[0] = commit.NewValue(2 + 256*64)
	values[1] = commit.NewValueFromLittleEndianBytes(expanded[:16])
	values[2] = commit.NewValueFromLittleEndianBytes(expanded[16:])

	index := treeIndex.Bytes32() // < produces result in big-endian order
	slices.Reverse(index[:])     // < reverse to little-endian order
	values[3] = commit.NewValueFromLittleEndianBytes(index[:16])
	values[4] = commit.NewValueFromLittleEndianBytes(index[16:])

	// Compute the the Pedersen Hash for the values.
	hash := commit.Commit(values).Hash()

	// Compose the trie key.
	res := trie.Key{}
	copy(res[:31], hash[:31])
	res[31] = subIndex
	return res
}
