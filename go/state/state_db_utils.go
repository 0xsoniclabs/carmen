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

import "slices"

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
