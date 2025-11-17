// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package result

// Result encapsulates a value along with an error. It is intended to be used
// in scenarios where a single type is needed to represent the outcome of an
// operation that can either succeed with a value of type T or fail with an
// error. This may, for instance, be useful for channels or containers.
type Result[T any] struct {
	value T
	err   error
}

// Ok creates a Result representing a successful outcome with the given value.
func Ok[T any](value T) Result[T] {
	return Result[T]{value: value}
}

// Err creates a Result representing a failed outcome with the given error.
func Err[T any](err error) Result[T] {
	return Result[T]{err: err}
}

// Get returns the value and error contained in the Result. Using this function
// forces the caller to handle potential errors.
func (r Result[T]) Get() (T, error) {
	return r.value, r.err
}
