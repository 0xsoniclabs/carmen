// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package iter_utils

import (
	"iter"
	"slices"
)

// Map maps an `iter.Seq` of values to another `iter.Seq` of values using the provided mapping function.
func Map[VIn, VOut any](
	seq iter.Seq[VIn],
	f func(VIn) VOut,
) iter.Seq[VOut] {
	return func(yield func(VOut) bool) {
		for v := range seq {
			if !yield(f(v)) {
				return
			}
		}
	}
}

// DropKeys drops the keys from an `iter.Seq2` of key-value pairs and returns an `iter.Seq` of values.
func DropKeys[K, V any](seq iter.Seq2[K, V]) iter.Seq[V] {
	return func(yield func(V) bool) {
		for _, v := range seq {
			if !yield(v) {
				return
			}
		}
	}
}

// FromSlice creates an `iter.Seq` from a slice of values.
// Original ordering is preserved.
func FromSlice[V any](slice []V) iter.Seq[V] {
	return DropKeys(slices.All(slice))
}
