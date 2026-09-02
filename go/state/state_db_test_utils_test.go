// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package state

import (
	"iter"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/stretchr/testify/require"
)

// Helpers shared by the tests of this package.

// stateDBOperation is a helper struct representing an operation to be performed on the StateDB, with its name and arguments.
type stateDBOperation struct {
	Op   func(db StateDB, rng *rand.Rand, args opArgs)
	Name string
	Args opArgs
}

// opArgs is a struct containing an address and a key, to be used as arguments for operations performed on the StateDB.
type opArgs struct {
	Address common.Address
	Key     common.Key
}

func setStateOp(db StateDB, rng *rand.Rand, args opArgs) {
	randomValue := common.Value(randomByteArrayWithPrefix(rng, 32, []byte{0x2}))
	db.SetState(args.Address, args.Key, randomValue)
}

func setNonceOp(db StateDB, rng *rand.Rand, args opArgs) {
	db.SetNonce(args.Address, rng.Uint64N(math.MaxUint64)+1)
}

func setCodeOp(db StateDB, rng *rand.Rand, args opArgs) {
	randomCode := randomByteArrayWithPrefix(rng, 8, []byte{0x2})
	db.SetCode(args.Address, randomCode)
}

func addBalanceOp(db StateDB, rng *rand.Rand, args opArgs) {
	db.AddBalance(args.Address, amount.New(10))
}

func subBalanceOp(db StateDB, rng *rand.Rand, args opArgs) {
	if db.GetBalance(args.Address).Uint64() < 10 {
		// Avoid underflow.
		return
	}
	db.SubBalance(args.Address, amount.New(10))
}

func createAccountOp(db StateDB, rng *rand.Rand, args opArgs) {
	db.CreateAccount(args.Address)
}

func suicideOp(db StateDB, rng *rand.Rand, args opArgs) {
	db.Suicide(args.Address)
}

func addLogOp(db StateDB, rng *rand.Rand, args opArgs) {
	db.AddLog(&common.Log{
		Address: args.Address,
		Topics:  []common.Hash{common.Hash(randomByteArrayWithPrefix(rng, 32, []byte{}))},
		Data:    randomByteArrayWithPrefix(rng, 3, []byte{}),
	})
}

// randomByteArrayWithPrefix generates a random byte array of the given size, where the first bytes are the given prefix.
// If len(prefix) > size, the prefix is truncated.
func randomByteArrayWithPrefix(rng *rand.Rand, size int, prefix []byte) []byte {
	b := make([]byte, size)
	copy(b, prefix)
	for i := len(prefix); i < size; i++ {
		b[i] = byte(rng.Uint64N(256))
	}
	return b
}

// cartesianProductTriple generates the cartesian product of a slice with itself three times,
// yielding the result as a sequence of tuples containing the values.
func cartesianProductTriple[T any](slice []T) iter.Seq[[]T] {
	return func(yield func([]T) bool) {
		for _, a := range slice {
			for _, b := range slice {
				for _, c := range slice {
					if !yield([]T{a, b, c}) {
						return
					}
				}
			}
		}
	}
}

// orderedPartitions yields each ordered partition of the input slice.
func orderedPartitions[T any](input []T) iter.Seq[[][]T] {
	return func(yield func([][]T) bool) {
		n := len(input)
		if n == 0 {
			return
		}

		numCombinations := 1 << (n - 1)
		for i := range numCombinations {
			var result [][]T
			currentGroup := []T{input[0]}

			for j := 0; j < n-1; j++ {
				// If the j-th bit of i is set, we start a new group;
				// otherwise, we continue adding to the current group.
				if (i>>j)&1 == 1 {
					result = append(result, currentGroup)
					currentGroup = []T{input[j+1]}
				} else {
					currentGroup = append(currentGroup, input[j+1])
				}
			}
			result = append(result, currentGroup)
			if !yield(result) {
				return
			}
		}
	}
}

func Test_CartesianTriple(t *testing.T) {
	t.Parallel()

	input := []string{"a", "b"}
	var triples [][]string
	for t := range cartesianProductTriple(input) {
		triples = append(triples, t)
	}

	expected := [][]string{
		{"a", "a", "a"},
		{"a", "a", "b"},
		{"a", "b", "a"},
		{"a", "b", "b"},
		{"b", "a", "a"},
		{"b", "a", "b"},
		{"b", "b", "a"},
		{"b", "b", "b"},
	}

	require.Equal(t, expected, triples)
}

func Test_OrderedPartitions(t *testing.T) {
	t.Parallel()

	input := []int{1, 2, 3}
	var partitions [][][]int
	for p := range orderedPartitions(input) {
		partitions = append(partitions, p)
	}

	expected := [][][]int{
		{{1, 2, 3}},
		{{1}, {2, 3}},
		{{1, 2}, {3}},
		{{1}, {2}, {3}},
	}

	require.Equal(t, expected, partitions)
}
