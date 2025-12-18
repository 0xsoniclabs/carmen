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
	"fmt"
	"testing"

	"github.com/0xsoniclabs/carmen/go/database/vt/commit"
	"github.com/stretchr/testify/require"
)

func TestInnerNode_Get_ReturnsZeroIfThereIsNoNextNode(t *testing.T) {
	require := require.New(t)
	innerNode := &inner{}
	require.Zero(innerNode.get(Key{}, 0))
}

func TestInnerNode_Get_ReturnsValueFromNextNode(t *testing.T) {
	require := require.New(t)

	key1 := Key{1, 2, 3}
	key2 := Key{1, 2, 4}

	var root node = newLeaf(key1)
	root = root.set(key1, 2, Value{42})
	root = root.set(key2, 2, Value{84})

	inner, ok := root.(*inner)
	require.True(ok, "Root should be an inner node")

	require.Equal(Value{42}, inner.get(key1, 2))
	require.Equal(Value{84}, inner.get(key2, 2))
}

func TestInnerNode_Set_CreatesNewLeafIfThereIsNoNextNode(t *testing.T) {
	require := require.New(t)

	key := Key{1, 2, 3}

	innerNode := &inner{}
	require.Nil(innerNode.children[key[2]])

	res, ok := innerNode.set(key, 2, Value{42}).(*inner)
	require.True(ok, "Setting a new key not result in a leaf node")
	require.Equal(innerNode, res, "Setting a new key should not change the inner node")

	require.NotNil(innerNode.children[key[2]])
}

func TestInnerNode_DirtyChildCommitmentsAreTracked(t *testing.T) {
	require := require.New(t)

	// Initially, the inner node should have a clean commitment state.
	innerNode := &inner{}
	require.False(innerNode.dirtyChildCommitments.any())

	// Setting a value should mark corresponding child as dirty.
	key := Key{1, 2, 3}
	innerNode.set(key, 0, Value{42})
	require.True(innerNode.dirtyChildCommitments.any())
	for i := range 256 {
		require.Equal(i == int(key[0]), innerNode.dirtyChildCommitments.get(byte(i)))
	}

	// Committing should clean the state.
	firstCommit := innerNode.commit()
	require.False(innerNode.dirtyChildCommitments.any())

	// Committing again should return the same commitment.
	secondCommit := innerNode.commit()
	require.False(innerNode.dirtyChildCommitments.any())
	require.True(firstCommit.Equal(secondCommit))

	// Setting another value should mark the commitment as dirty again.
	innerNode.set(Key{1, 2, 4}, 0, Value{84})
	require.True(innerNode.dirtyChildCommitments.any())
}

func TestInnerNode_Commit_ComputesCommitmentFromChildren(t *testing.T) {
	require := require.New(t)

	innerNode := &inner{}
	key1 := Key{1, 2, 3}
	key2 := Key{1, 2, 4}

	// Set two values in the inner node.
	innerNode.set(key1, 2, Value{42})
	innerNode.set(key2, 2, Value{84})

	// Compute the commitment.
	commitment := innerNode.commit()

	require.NotNil(commitment)
	require.True(commitment.IsValid())

	// The commitment should be computed from the values of the children.
	expectedCommitment := commit.Commit([256]commit.Value{
		3: innerNode.children[key1[2]].commit().ToValue(),
		4: innerNode.children[key2[2]].commit().ToValue(),
	})

	require.True(commitment.Equal(expectedCommitment))
}

func TestInnerNode_collectCommitTasks_GathersTasksFromDirtyChildren(t *testing.T) {
	require := require.New(t)

	innerNode := &inner{}
	key1 := Key{1, 2, 3}
	key2 := Key{1, 2, 4}

	// Set two values in the inner node.
	innerNode.set(key1, 2, Value{42})
	innerNode.set(key2, 2, Value{84})

	// Collect commitment tasks.
	tasks := []*task{}
	innerNode.collectCommitTasks(&tasks)

	// There should be some tasks.
	require.NotEmpty(tasks)

	// Running those tasks should compute all commitments.
	require.True(innerNode.dirtyChildCommitments.any())
	for _, task := range tasks {
		task.run()
	}
	require.False(innerNode.dirtyChildCommitments.any())

	// The commitment should be computed from the values of the children.
	expectedCommitment := commit.Commit([256]commit.Value{
		3: innerNode.children[key1[2]].commit().ToValue(),
		4: innerNode.children[key2[2]].commit().ToValue(),
	})
	commitment := innerNode.commit()
	require.NotNil(commitment)
	require.True(commitment.IsValid())
	require.True(commitment.Equal(expectedCommitment))
}

func TestInnerNode_collectCommitTasks_CleanChildren_ResultsInNoTasks(t *testing.T) {
	require := require.New(t)

	innerNode := &inner{}
	key := Key{1, 2, 3}
	innerNode.set(key, 2, Value{42})

	// Commit to clean the state.
	innerNode.commit()
	require.False(innerNode.dirtyChildCommitments.any())

	// In the clean case, there should be no tasks.
	tasks := []*task{}
	innerNode.collectCommitTasks(&tasks)
	require.Empty(tasks)
}

func TestLeafNode_NewLeaf_ProducesEmptyLeafWithStem(t *testing.T) {
	require := require.New(t)

	// Create a new leaf node with a specific key.
	key := Key{1, 2, 3, 4, 5}
	leafNode := newLeaf(key)

	// Check that the stem is set correctly.
	require.Equal(key[:31], leafNode.stem[:], "Stem should match the first 31 bytes of the key")

	// Check that all values are initialized to zero.
	require.Equal([256]Value{}, leafNode.values, "All values should be initialized to zero")

	// Check that the used bitmap is empty.
	require.False(leafNode.used.any(), "Used bitmap should be empty")
}

func TestLeafNode_Get_ReturnsValueForMatchingStem(t *testing.T) {
	require := require.New(t)

	key := Key{1, 2, 3, 31: 1}
	leaf := newLeaf(key)

	// Initially, the value for the key should be zero.
	require.Zero(leaf.get(key, 0), "Value for the key should be zero initially")

	// Set a value for the key.
	leaf.set(key, 0, Value{42})

	// Now retrieving the value should return the set value.
	require.Equal(Value{42}, leaf.get(key, 0), "Value for the key should match the set value")
}

func TestLeafNode_Get_ReturnsZeroForNonMatchingStem(t *testing.T) {
	require := require.New(t)

	key1 := Key{1, 2, 3}
	key2 := Key{4, 5, 6}

	leaf := newLeaf(key1)
	leaf.set(key1, 0, Value{42})

	require.Zero(leaf.get(key2, 0), "Value for non-matching key should be zero")
}

func TestLeafNode_Set_SplitsLeafIfStemDoesNotMatch(t *testing.T) {
	require := require.New(t)

	key1 := Key{1, 2, 3}
	key2 := Key{1, 2, 4}

	leafNode := newLeaf(key1)
	leafNode.set(key1, 0, Value{42})

	// Setting a different key should split the leaf.
	newNode := leafNode.set(key2, 2, Value{84})

	// The new node should be an inner node now.
	innerNode, ok := newNode.(*inner)
	require.True(ok, "Setting a different key should create an inner node")

	// The inner node should have the leaf as one of its children.
	require.Equal(leafNode, innerNode.children[key1[2]].(*leaf))
	require.NotNil(innerNode.children[key2[2]])
}

func TestLeafNode_Set_TracksDirtyState(t *testing.T) {
	require := require.New(t)

	stem := Key{1, 2, 3}
	leaf := newLeaf(stem)

	// initially, the leaf is clean
	require.False(leaf.lowDirty)
	require.False(leaf.highDirty)

	// setting a value in the low range marks the low part as dirty
	leaf.set(Key{1, 2, 3, 31: 1}, 0, Value{10})
	require.True(leaf.lowDirty)
	require.False(leaf.highDirty)

	// setting a value in the high range marks the high part as dirty
	leaf.set(Key{1, 2, 3, 31: 130}, 0, Value{20})
	require.True(leaf.lowDirty)
	require.True(leaf.highDirty)

	// committing clears the dirty state
	leaf.commit()
	require.False(leaf.lowDirty)
	require.False(leaf.highDirty)

	// setting the same value does not change the dirty state
	leaf.set(Key{1, 2, 3, 31: 1}, 0, Value{10})
	require.False(leaf.lowDirty)
	require.False(leaf.highDirty)

	leaf.set(Key{1, 2, 3, 31: 130}, 0, Value{20})
	require.False(leaf.lowDirty)
	require.False(leaf.highDirty)
}

func TestLeafNode_CanSetAndGetValues(t *testing.T) {
	require := require.New(t)

	key1 := Key{1, 2, 3, 31: 1}
	key2 := Key{1, 2, 3, 31: 2}
	key3 := Key{1, 2, 3, 31: 3}

	leaf := newLeaf(key1)

	require.False(leaf.isUsed(key1[31]))
	require.False(leaf.isUsed(key2[31]))
	require.False(leaf.isUsed(key3[31]))

	require.Zero(leaf.get(key1, 0))
	require.Zero(leaf.get(key2, 0))
	require.Zero(leaf.get(key3, 0))

	// Setting a value for key 1 makes the value retrievable and marks the
	// suffix as used.
	leaf.set(key1, 0, Value{10})

	require.True(leaf.isUsed(key1[31]))
	require.False(leaf.isUsed(key2[31]))
	require.False(leaf.isUsed(key3[31]))

	require.Equal(Value{10}, leaf.get(key1, 0))
	require.Zero(leaf.get(key2, 0))
	require.Zero(leaf.get(key3, 0))

	// Setting the value for key 2 to zero does not change the value but marks
	// the suffix as used.
	leaf.set(key2, 0, Value{})

	require.True(leaf.isUsed(key1[31]))
	require.True(leaf.isUsed(key2[31]))
	require.False(leaf.isUsed(key3[31]))

	require.Equal(Value{10}, leaf.get(key1, 0))
	require.Zero(leaf.get(key2, 0))
	require.Zero(leaf.get(key3, 0))

	// Resetting the value for key 1 to zero does not change the used bitmap.
	leaf.set(key1, 0, Value{})
	require.True(leaf.isUsed(key1[31]))
	require.True(leaf.isUsed(key2[31]))
	require.False(leaf.isUsed(key3[31]))

	require.Zero(leaf.get(key1, 0))
	require.Zero(leaf.get(key2, 0))
	require.Zero(leaf.get(key3, 0))
}

func TestLeafNode_CanComputeCommitment(t *testing.T) {
	require := require.New(t)

	key1 := Key{1, 2, 3, 31: 1}
	key2 := Key{1, 2, 3, 31: 130}

	val1 := Value{8: 1, 20: 10}
	val2 := Value{8: 2, 20: 20}

	leaf := newLeaf(key1)
	leaf.set(key1, 0, val1)
	leaf.set(key2, 0, val2)

	have := leaf.commit()

	require.NotNil(have)
	require.True(have.IsValid())

	low1 := commit.NewValueFromLittleEndianBytes(val1[:16])
	low2 := commit.NewValueFromLittleEndianBytes(val2[:16])
	high1 := commit.NewValueFromLittleEndianBytes(val1[16:])
	high2 := commit.NewValueFromLittleEndianBytes(val2[16:])

	low1.SetBit128()
	low2.SetBit128()

	c1 := commit.Commit([256]commit.Value{2: low1, 3: high1})
	c2 := commit.Commit([256]commit.Value{4: low2, 5: high2})

	want := commit.Commit([256]commit.Value{
		commit.NewValue(1),
		commit.NewValueFromLittleEndianBytes(key1[:31]),
		c1.ToValue(),
		c2.ToValue(),
	})
	require.True(have.Equal(want))
}

func TestLeafNode_CommitmentDirtyStateIsTracked(t *testing.T) {
	require := require.New(t)

	key1 := Key{1, 2, 3, 31: 1}
	key2 := Key{1, 2, 3, 31: 130}

	leaf := newLeaf(key1)
	require.False(leaf.lowDirty)
	require.False(leaf.highDirty)

	leaf.set(key1, 0, Value{10})
	require.True(leaf.lowDirty)
	require.False(leaf.highDirty)

	leaf.set(key2, 0, Value{20})
	require.True(leaf.lowDirty)
	require.True(leaf.highDirty)

	first := leaf.commit()
	require.False(leaf.lowDirty)
	require.False(leaf.highDirty)

	second := leaf.commit()
	require.True(first.Equal(second))

	leaf.set(key1, 0, Value{30})
	require.True(leaf.lowDirty)
	require.False(leaf.highDirty)

	third := leaf.commit()
	require.False(leaf.lowDirty)
	require.False(leaf.highDirty)

	require.False(first.Equal(third))
}

func (l *leaf) commit_seq() commit.Commitment {
	return l.commit()
}

func (l *leaf) commit_par() commit.Commitment {
	tasks := []*task{}
	l.collectCommitTasks(&tasks)
	for _, task := range tasks {
		task.run()
	}
	return l.commitment
}

func TestLeafNode_Commit_SequentialAndParallelProduceSameResults(t *testing.T) {
	seq := newLeaf(Key{1, 2, 3, 31: 1})
	par := newLeaf(Key{1, 2, 3, 31: 1})

	// --- empty ---
	s := seq.commit_seq()
	p := par.commit_par()
	require.Equal(t, s.Hash(), p.Hash(), "Commitments of empty leaf should match")

	// --- set low value ---
	seq.set(Key{1, 2, 3, 31: 1}, 0, Value{10})
	par.set(Key{1, 2, 3, 31: 1}, 0, Value{10})
	s = seq.commit_seq()
	p = par.commit_par()
	require.Equal(t, s.Hash(), p.Hash(), "Commitments after setting low value should match")

	// --- set high value ---
	seq.set(Key{1, 2, 3, 31: 160}, 0, Value{10})
	par.set(Key{1, 2, 3, 31: 160}, 0, Value{10})
	s = seq.commit_seq()
	p = par.commit_par()
	require.Equal(t, s.Hash(), p.Hash(), "Commitments after setting high value should match")

	// --- update low value ---
	seq.set(Key{1, 2, 3, 31: 1}, 0, Value{20})
	par.set(Key{1, 2, 3, 31: 1}, 0, Value{20})
	s = seq.commit_seq()
	p = par.commit_par()
	require.Equal(t, s.Hash(), p.Hash(), "Commitments after updating low value should match")

	// --- update high value ---
	seq.set(Key{1, 2, 3, 31: 160}, 0, Value{20})
	par.set(Key{1, 2, 3, 31: 160}, 0, Value{20})
	s = seq.commit_seq()
	p = par.commit_par()
	require.Equal(t, s.Hash(), p.Hash(), "Commitments after updating high value should match")
}

func Benchmark_AllocateSmallArray(b *testing.B) {
	for _, size := range []int{0, 1, 2, 4, 8, 16, 32, 64, 128, 256} {
		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			benchmark_AllocateArray(b, size)
		})
	}
}

func benchmark_AllocateArray(b *testing.B, size int) {
	last := make([]*task, 0, size)
	for b.Loop() {
		if cap(last) != size {
			b.Fail()
		}
		last = make([]*task, 0, size)
	}
}
