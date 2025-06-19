package verkle

import (
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/database/verkle/utils"
	"github.com/ethereum/go-verkle"
	"testing"
)

var emptyNodeResolverBenchmarkFn = func(path []byte) ([]byte, error) {
	return nil, nil // no-op for in-memory tree
}

func Benchmark_VerkleTree_Commit_To_InnerNode_All_Leaves_Updated(b *testing.B) {
	root, err := createTestNode_One_Inner_Node_With_Full_Leaves_Space()
	if err != nil {
		b.Fatalf("failed to create test node: %v", err)
	}

	// start with a tree with all commitments computed
	root.Commit()

	var counter int
	for i := 0; i < b.N; i++ {
		// modify all values in one leaf
		for j := 0; j < verkle.NodeWidth; j++ {
			var key common.Key
			key[0] = byte(j) // set first byte to insert at one branch at each iteration, updating all leaves
			value := common.Value{byte(counter), byte(counter >> 8), byte(counter >> 16), byte(counter >> 24), 0x1}
			counter++
			if err := root.Insert(key[:], value[:], emptyNodeResolverBenchmarkFn); err != nil {
				b.Fatalf("failed to insert key %v: %v", key, err)
			}
		}

		root.Commit() // measurement time
	}
}

func Benchmark_VerkleTree_Commit_To_InnerNode_Single_Leaf_Updated(b *testing.B) {
	root, err := createTestNode_One_Inner_Node_With_Full_Leaves_Space()
	if err != nil {
		b.Fatalf("failed to create test node: %v", err)
	}

	// start with a tree with all commitments computed
	root.Commit()

	var counter int
	var key common.Key
	for i := 0; i < b.N; i++ {
		// update just one leaf at time
		key[0] = byte(i % verkle.NodeWidth) // set first byte to insert at one branch updating only one leaf
		value := common.Value{byte(counter), byte(counter >> 8), byte(counter >> 16), byte(counter >> 24), 0x1}
		counter++
		if err := root.Insert(key[:], value[:], emptyNodeResolverBenchmarkFn); err != nil {
			b.Fatalf("failed to insert key %v: %v", key, err)
		}

		root.Commit() // measurement time - only inner node is included in the measurement
	}
}

func Benchmark_VerkleTree_Commit_To_LeafNode_Update_All_Values(b *testing.B) {
	root, err := createTestNode_One_Inner_Node_With_Full_Leaves_Space()
	if err != nil {
		b.Fatalf("failed to create test node: %v", err)
	}

	// tree is commited at the beginning
	root.Commit()

	var counter int
	for i := 0; i < b.N; i++ {
		var key common.Key
		key[31] = byte(i % verkle.NodeWidth) // set the last byte to insert at a one value of the first leaf
		value := common.Value{byte(counter), byte(counter >> 8), byte(counter >> 16), byte(counter >> 24), 0x1}
		counter++
		if err := root.Insert(key[:], value[:], emptyNodeResolverBenchmarkFn); err != nil {
			b.Fatalf("failed to insert key %v: %v", key, err)
		}

		root.Commit()
	}
}

func Benchmark_VerkleTree_Commit_To_LeafNode_Update_Single_Value(b *testing.B) {
	root, err := createTestNode_One_Inner_Node_With_Full_Leaves_Space()
	if err != nil {
		b.Fatalf("failed to create test node: %v", err)
	}

	// tree is commited at the beginning
	root.Commit()

	var counter int
	for i := 0; i < b.N; i++ {
		for j := 0; j < verkle.NodeWidth; j++ {
			var key common.Key
			key[31] = byte(j) // set the last byte to insert at one value of the first leaf
			value := common.Value{byte(counter), byte(counter >> 8), byte(counter >> 16), byte(counter >> 24), 0x1}
			counter++
			if err := root.Insert(key[:], value[:], emptyNodeResolverBenchmarkFn); err != nil {
				b.Fatalf("failed to insert key %v: %v", key, err)
			}
		}

		root.Commit()
	}
}

var pointHashSink []byte

func Benchmark_VerkleTree_Hash_Key_No_Cache(b *testing.B) {

	for i := 0; i < b.N; i++ {
		key := common.Value{byte(i), byte(i >> 8), byte(i >> 16), byte(i >> 24), 0x1}
		address := common.Address{byte(i), byte(i >> 8), byte(i >> 16), byte(i >> 24), 0x2}
		pointHashSink = utils.StorageSlotKey(key[:], address[:])
	}
}

func createTestNode_One_Inner_Node_With_Full_Leaves_Space() (*verkle.InternalNode, error) {
	root := verkle.New().(*verkle.InternalNode)
	for i := 0; i < verkle.NodeWidth; i++ {
		key := common.Key{byte(i)} // set first byte to insert at different branch at each iteration
		value := common.Value{byte(i)}

		if err := root.Insert(key[:], value[:], emptyNodeResolverBenchmarkFn); err != nil {
			return nil, err
		}
	}

	return root, nil
}
