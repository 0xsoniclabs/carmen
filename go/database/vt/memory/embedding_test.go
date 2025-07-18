package memory

import (
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/database/vt/memory/trie"
	"github.com/ethereum/go-ethereum/trie/utils"
	"github.com/holiman/uint256"
	"github.com/stretchr/testify/require"
)

func TestGetTreeKey_CompareWithGethResults(t *testing.T) {

	getKey := func(address int, index int, subIndex byte) trie.Key {
		return getTrieKey(
			common.Address{byte(address)},
			*uint256.NewInt(uint64(index)),
			subIndex,
		)
	}

	getRefKey := func(address int, index int, subIndex byte) trie.Key {
		res := utils.GetTreeKey(
			[]byte{byte(address), 19: 0},
			uint256.NewInt(uint64(index)),
			subIndex,
		)
		return trie.Key(res[:])
	}

	const N = 10
	for i := range N {
		for j := range N {
			for k := range N {
				have := getKey(i, j, byte(k))
				want := getRefKey(i, j, byte(k))
				require.Equal(t, want, have, "i=%d,j=%d,k=%d", i, j, k)
			}
		}
	}
}

func TestGetStorageKey_CompareWithGethResults(t *testing.T) {

	getKey := func(address int, key int) trie.Key {
		return getStorageKey(
			common.Address{byte(address)},
			common.Key{byte(key)},
		)
	}

	getRefKey := func(address int, key int) trie.Key {
		res := utils.StorageSlotKey(
			[]byte{byte(address), 19: 0},
			[]byte{byte(key), 31: 0},
		)
		return trie.Key(res[:])
	}

	const N = 10
	for i := range N {
		for j := range N {
			have := getKey(i, j)
			want := getRefKey(i, j)
			require.Equal(t, want, have, "i=%d,j=%d", i, j)
		}
	}
}
