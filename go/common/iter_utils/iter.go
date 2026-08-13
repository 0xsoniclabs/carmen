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
	"maps"
	"slices"

	"github.com/0xsoniclabs/carmen/go/common/result"
)

// Pair combines a key and a value into a single value. It allows key-value
// pairs to be transported through the single-element `iter.Seq`, which is
// required whenever the elements need to be wrapped, for instance in a
// `result.Result`.
type Pair[K, V any] struct {
	Key   K
	Value V
}

// Unpack splits the pair into its key and value.
func (p Pair[K, V]) Unpack() (K, V) {
	return p.Key, p.Value
}

// ResultSeq is a sequence of values whose production may fail. Producers report
// a failure by yielding an error element and stopping the iteration; consumers
// must inspect every element, which `Unwrap` facilitates.
type ResultSeq[V any] = iter.Seq[result.Result[V]]

// ResultSeq2 is the key-value counterpart of `ResultSeq`. It is the fallible
// equivalent of an `iter.Seq2`.
type ResultSeq2[K, V any] = ResultSeq[Pair[K, V]]

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

// Map2 maps an `iter.Seq2` of key-value pairs to another `iter.Seq2` of key-value pairs using the provided mapping function.
func Map2[KIn, VIn, KOut, VOut any](
	seq iter.Seq2[KIn, VIn],
	f func(KIn, VIn) (KOut, VOut),
) iter.Seq2[KOut, VOut] {
	return func(yield func(KOut, VOut) bool) {
		for k, v := range seq {
			k2, v2 := f(k, v)
			if !yield(k2, v2) {
				return
			}
		}
	}
}

// Unwrap converts a `ResultSeq` into a plain `iter.Seq` that stops at the first
// failure. The returned function reports that failure and must only be
// consulted once the iteration has ended.
func Unwrap[V any](seq ResultSeq[V]) (iter.Seq[V], func() error) {
	var err error
	return func(yield func(V) bool) {
		err = nil
		for r := range seq {
			v, e := r.Get()
			if e != nil {
				err = e
				return
			}
			if !yield(v) {
				return
			}
		}
	}, func() error { return err }
}

// Unwrap2 is the key-value counterpart of `Unwrap`.
func Unwrap2[K, V any](seq ResultSeq2[K, V]) (iter.Seq2[K, V], func() error) {
	pairs, seqErr := Unwrap(seq)
	return func(yield func(K, V) bool) {
		for pair := range pairs {
			if !yield(pair.Unpack()) {
				return
			}
		}
	}, seqErr
}

// OkSeq lifts an `iter.Seq` into a `ResultSeq` that never fails.
func OkSeq[V any](seq iter.Seq[V]) ResultSeq[V] {
	return func(yield func(result.Result[V]) bool) {
		for v := range seq {
			if !yield(result.Ok(v)) {
				return
			}
		}
	}
}

// OkSeq2 lifts an `iter.Seq2` into a `ResultSeq2` that never fails.
func OkSeq2[K, V any](seq iter.Seq2[K, V]) ResultSeq2[K, V] {
	return func(yield func(result.Result[Pair[K, V]]) bool) {
		for k, v := range seq {
			if !yield(result.Ok(Pair[K, V]{Key: k, Value: v})) {
				return
			}
		}
	}
}

// MapOk maps the successful values of a `ResultSeq` using the provided mapping
// function. Failures are forwarded unchanged.
func MapOk[VIn, VOut any](
	seq ResultSeq[VIn],
	f func(VIn) VOut,
) ResultSeq[VOut] {
	return Map(seq, func(r result.Result[VIn]) result.Result[VOut] {
		return result.Map(r, f)
	})
}

// MapOk2 maps the successful key-value pairs of a `ResultSeq2` using the
// provided mapping function. Failures are forwarded unchanged.
func MapOk2[KIn, VIn, KOut, VOut any](
	seq ResultSeq2[KIn, VIn],
	f func(KIn, VIn) (KOut, VOut),
) ResultSeq2[KOut, VOut] {
	return MapOk(seq, func(in Pair[KIn, VIn]) Pair[KOut, VOut] {
		key, value := f(in.Unpack())
		return Pair[KOut, VOut]{Key: key, Value: value}
	})
}

// CollectOk collects the successful values of a `ResultSeq` into a slice.
// If the sequence reports a failure, it is returned alongside the values
// collected before the failure.
func CollectOk[V any](seq ResultSeq[V]) ([]V, error) {
	pairs, seqErr := Unwrap(seq)
	entries := slices.Collect(pairs)
	return entries, seqErr()
}

// CollectOk2 collects the successful key-value pairs of a `ResultSeq2` into a map.
// If the sequence reports a failure, it is returned alongside the pairs
// collected before the failure.
func CollectOk2[K comparable, V any](seq ResultSeq2[K, V]) (map[K]V, error) {
	pairs, seqErr := Unwrap2(seq)
	entries := maps.Collect(pairs)
	return entries, seqErr()
}

// DropKeysOk2 drops the keys of a `ResultSeq2`, yielding a `ResultSeq` of the values.
// Failures are forwarded unchanged.
func DropKeysOk2[K, V any](seq ResultSeq2[K, V]) ResultSeq[V] {
	return MapOk(seq, func(pair Pair[K, V]) V { return pair.Value })
}

// Enumerate creates an `iter.Seq2` from a slice of values, pairing each value
// with its `uint64` index.
func Enumerate[V any](slice []V) iter.Seq2[uint64, V] {
	return Map2(slices.All(slice), func(i int, v V) (uint64, V) { return uint64(i), v })
}
