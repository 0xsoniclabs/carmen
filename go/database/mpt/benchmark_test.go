// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package mpt

import (
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"testing"
)

var s5PlainKeysBenchmarkConfig = MptConfig{
	Name:                          "S5-Live-PlainKeys-Benchmark",
	UseHashedPaths:                false,
	TrackSuffixLengthsInLeafNodes: true,
	Hashing:                       EthereumLikeHashing, // requires tracking of suffix lengths
	HashStorageLocation:           HashStoredWithParent,
}

var hashSink common.Hash

func Benchmark_Mpt_Hash_BranchNode_All_Leaves_Updated(b *testing.B) {
	live, err := OpenInMemoryLiveTrie(b.TempDir(), s5PlainKeysBenchmarkConfig, NodeCacheConfig{Capacity: 1024})
	if err != nil {
		b.Fatalf("failed to create live trie: %v", err)
	}

	if err := createTestNode_One_BranchNode_With_Full_Leaves_Space(live); err != nil {
		b.Fatalf("failed to create test nodes: %v", err)
	}

	// start with a tree with all commitments computed
	if _, _, err := live.UpdateHashes(); err != nil {
		b.Fatalf("failed to update hashes: %v", err)
	}

	var counter uint64
	for i := 0; i < b.N; i++ {
		// modify all values in one leaf
		for j := 0; j < 16; j++ {
			key := common.Address{byte(i) << 4} // set first byte to insert at one branch at each iteration, updating all leaves
			value := AccountInfo{Balance: amount.New(counter + 1)}
			counter++
			if err := live.SetAccountInfo(key, value); err != nil {
				b.Fatalf("failed to set account info: %v", err)
			}
		}

		hashSink, _, err = live.UpdateHashes()
		if err != nil {
			b.Fatalf("failed to update hashes: %v", err)
		}
	}
}

func Benchmark_Mpt_Hash_BranchNode_Single_Leaf_Updated(b *testing.B) {
	live, err := OpenInMemoryLiveTrie(b.TempDir(), s5PlainKeysBenchmarkConfig, NodeCacheConfig{Capacity: 1024})
	if err != nil {
		b.Fatalf("failed to create live trie: %v", err)
	}

	if err := createTestNode_One_BranchNode_With_Full_Leaves_Space(live); err != nil {
		b.Fatalf("failed to create test nodes: %v", err)
	}

	// start with a tree with all commitments computed
	if _, _, err := live.UpdateHashes(); err != nil {
		b.Fatalf("failed to update hashes: %v", err)
	}

	var counter uint64
	for i := 0; i < b.N; i++ {
		// update just one leaf at time
		key := common.Address{byte(i) << 4} // set first byte to insert at one branch
		value := AccountInfo{Balance: amount.New(counter + 1)}
		counter++
		if err := live.SetAccountInfo(key, value); err != nil {
			b.Fatalf("failed to set account info: %v", err)
		}

		hashSink, _, err = live.UpdateHashes()
		if err != nil {
			b.Fatalf("failed to update hashes: %v", err)
		}
	}
}

var nibblesSink []Nibble

func Benchmark_Mpt_Hash_Key(b *testing.B) {
	for i := 0; i < b.N; i++ {
		address := common.Address{byte(i), byte(i >> 8), byte(i >> 16), byte(i >> 24), 0x2}
		nibblesSink = addressToHashedNibbles(address)
	}
}

func createTestNode_One_BranchNode_With_Full_Leaves_Space(live *LiveTrie) error {
	for i := 0; i < 16; i++ {
		key := common.Address{byte(i) << 4} // set first byte to insert at different branch at each iteration
		value := AccountInfo{Nonce: common.Nonce{byte(i + 1)}}

		if err := live.SetAccountInfo(key, value); err != nil {
			return err
		}
	}

	return nil
}
