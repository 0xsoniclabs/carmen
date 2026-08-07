package common

import "iter"

// Map2 maps an `iter.Seq2` of key-value pairs to another `iter.Seq2`
// of key-value pairs using the provided mapping function.
func Map2[KIn, VIn, KOut, VOut any](
	seq iter.Seq2[KIn, VIn],
	f func(KIn, VIn) (KOut, VOut),
) iter.Seq2[KOut, VOut] {
	return func(yield func(KOut, VOut) bool) {
		for k, v := range seq {
			if !yield(f(k, v)) {
				return
			}
		}
	}
}
