package memory

import (
	"slices"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/database/vt/memory/commit"
	"github.com/0xsoniclabs/carmen/go/database/vt/memory/trie"
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

// getBasicDataKey returns the Verkle trie key of the basic data field for
// the specified account. The basic fields cover the code size, nonce, and
// balance of the account.
func getBasicDataKey(address common.Address) trie.Key {
	return getTrieKey(address, *uint256.NewInt(0), 0)
}

// getCodeHashKey returns the Verkle trie key of the code hash field for
// the specified account.
func getCodeHashKey(address common.Address) trie.Key {
	return getTrieKey(address, *uint256.NewInt(0), 1)
}

func getCodeChunkKey(address common.Address, chunkNumber int) trie.Key {
	// Derived from
	// https://github.com/0xsoniclabs/go-ethereum/blob/e563918a84b4104e44935ddc6850f11738dcc3f5/trie/utils/verkle.go#L188
	var (
		chunk                  = uint256.NewInt(uint64(chunkNumber))
		chunkOffset            = new(uint256.Int).Add(codeOffset, chunk)
		treeIndex, subIndexMod = new(uint256.Int).DivMod(chunkOffset, verkleNodeWidth, new(uint256.Int))
		subIndex               = byte(subIndexMod.Uint64())
	)
	return getTrieKey(address, *treeIndex, subIndex)
}

// getStorageKey returns the Verkle trie key of the storage slot addressed an
// address/key pair.
func getStorageKey(address common.Address, key common.Key) trie.Key {
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

	return getTrieKey(address, treeIndex, suffix)
}

// getTrieKey is a helper function for hashing information to obtain trie keys.
func getTrieKey(
	address common.Address,
	treeIndex uint256.Int,
	subIndex byte,
) trie.Key {
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

	// Compute the the Pederson Hash for the values.
	hash := commit.Commit(values).Hash()

	// Compose the trie key.
	res := trie.Key{}
	copy(res[:31], hash[:31])
	res[31] = subIndex
	return res
}
