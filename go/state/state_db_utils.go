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
	"slices"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
)

// cloneMapWith clones `m` using `cloneFunc` to clone the values. If `m` is nil, it returns nil.
func cloneMapWith[K comparable, V any](m map[K]V, cloneFunc func(V) V) map[K]V {
	if m == nil {
		return nil
	}
	cloned := make(map[K]V, len(m))
	for k, v := range m {
		cloned[k] = cloneFunc(v)
	}
	return cloned
}

func cloneBalanceValue(bv *balanceValue) *balanceValue {
	if bv == nil {
		return nil
	}
	cloned := *bv
	if bv.original != nil {
		originalCopy := *bv.original
		cloned.original = &originalCopy
	}
	return &cloned
}

func cloneNonceValue(nv *nonceValue) *nonceValue {
	if nv == nil {
		return nil
	}
	cloned := *nv
	if nv.original != nil {
		originalCopy := *nv.original
		cloned.original = &originalCopy
	}
	return &cloned
}

func cloneCodeValue(cv *codeValue) *codeValue {
	if cv == nil {
		return nil
	}
	cloned := *cv
	if cv.hash != nil {
		hashCopy := *cv.hash
		cloned.hash = &hashCopy
	}
	cloned.code = slices.Clone(cv.code)
	return &cloned
}

func cloneValue[V any](v *V) *V {
	if v == nil {
		return nil
	}
	cloned := *v
	return &cloned
}

// Test utils

// StateDBOperation is an helper struct representing an operation to be performed on the StateDB, with its name and arguments.
type StateDBOperation struct {
	Op   func(db StateDB, rng *rand.Rand, args OpArgs)
	Name string
	Args OpArgs
}

// OpArgs is a struct containing an address and a key, to be used as arguments for operations performed on the StateDB.
type OpArgs struct {
	Address *common.Address
	Key     *common.Key
}

func SetStateOp(db StateDB, rng *rand.Rand, args OpArgs) {
	randomValue := common.Value(RandomByteArrayWithPrefix(rng, 32, []byte{0x2}))
	db.SetState(*args.Address, *args.Key, randomValue)
}

func SetNonceOp(db StateDB, rng *rand.Rand, args OpArgs) {
	db.SetNonce(*args.Address, rng.Uint64N(math.MaxUint64)+1)
}

func SetCodeOp(db StateDB, rng *rand.Rand, args OpArgs) {
	randomCode := RandomByteArrayWithPrefix(rng, 8, []byte{0x2})
	db.SetCode(*args.Address, randomCode)
}

func AddBalanceOp(db StateDB, rng *rand.Rand, args OpArgs) {
	db.AddBalance(*args.Address, amount.New(10))
}

func SubBalanceOp(db StateDB, rng *rand.Rand, args OpArgs) {
	if db.GetBalance(*args.Address).Uint64() < 10 {
		// Avoid underflow.
		return
	}
	db.SubBalance(*args.Address, amount.New(10))
}

func CreateAccountOp(db StateDB, rng *rand.Rand, args OpArgs) {
	db.CreateAccount(*args.Address)
}

func SuicideOp(db StateDB, rng *rand.Rand, args OpArgs) {
	db.Suicide(*args.Address)
}

func AddLogOp(db StateDB, rng *rand.Rand, args OpArgs) {
	db.AddLog(&common.Log{
		Address: *args.Address,
		Topics:  []common.Hash{common.Hash(RandomByteArrayWithPrefix(rng, 32, []byte{}))},
		Data:    RandomByteArrayWithPrefix(rng, 3, []byte{}),
	})
}

// RandomByteArrayWithPrefix generates a random byte array of the given size, where the first bytes are the given prefix.
func RandomByteArrayWithPrefix(rng *rand.Rand, size int, prefix []byte) []byte {
	b := make([]byte, size)
	copy(b, prefix)
	for i := len(prefix); i < size; i++ {
		b[i] = byte(rng.Uint64N(256))
	}
	return b
}

// CartesianProductTriple generates the cartesian product of a slice with itself three times,
// yielding the result as a sequence of tuples containing the values.
func CartesianProductTriple[T any](slice []T) iter.Seq[[]T] {
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

// OrderedPartitions yields each ordered partition of the input slice.
func OrderedPartitions[T any](input []T) iter.Seq[[][]T] {
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
