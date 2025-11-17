// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package future

// Futures provided a simple abstraction for asynchronous computation results.
// A future is a placeholder for a value that may not yet be available, allowing
// code to proceed without blocking until the value is needed. Futures are
// typically used in concurrent programming to represent the result of an
// operation that is executed in a separate goroutine.
//
// This implementation provides a simple Future and Promise abstraction using
// Go channels. A Promise is used to fulfill a Future. Using the provided
// abstractions allows for better readability and error handling in asynchronous
// code and facilitates the transparent integration of future improvements.
//
// The producer side of a Future typically looks as follows:
//  promise, future := future.Create[T]()
//  go func() {
//     // perform some asynchronous operation
//     promise.Fulfill(someOperation())
//  }()
//  return future
//
// Alternatively, if the result is already available, an immediate Future can be
// created using Immediate.

// Promise represents the handle used to fulfill a Future with a Result.
type Promise[T any] struct {
	C chan<- T
}

// Future represents a placeholder for a value that will be available in the
// future. It can be awaited to retrieve the Result once it is fulfilled.
type Future[T any] struct {
	C <-chan T
}

// Create initializes a new Promise and Future pair. The Promise can be used to
// fulfill the Future with a Result, while the Future can be awaited to retrieve
// the Result once it is available.
func Create[T any]() (Promise[T], Future[T]) {
	ch := make(chan T, 1)
	return Promise[T]{C: ch}, Future[T]{C: ch}
}

// Immediate creates a Future that is already fulfilled with the given value.
// This is useful for scenarios where the result is already available and no
// asynchronous computation is needed.
func Immediate[T any](value T) Future[T] {
	ch := make(chan T, 1)
	ch <- value
	close(ch)
	return Future[T]{C: ch}
}

// Fulfill fulfills the Promise with the given Result, making it available to
// any awaiting Future.
func (p Promise[T]) Fulfill(value T) {
	p.C <- value
	close(p.C)
}

// Forward connects the Promise to the given Future, such that when the Future
// is fulfilled, the Promise is also fulfilled with the same Result.
func (p Promise[T]) Forward(f Future[T]) {
	go func() {
		p.C <- <-f.C
		close(p.C)
	}()
}

// Await blocks until the Future is fulfilled and returns the contained value
// and error. This is a convenience method that combines Get and error handling.
// Futures can only be consumed once.
func (f Future[T]) Await() T {
	return <-f.C
}

// Then creates a new Future by applying the given transformation function to
// the result of the original Future once it is fulfilled.
func Then[A, B any](f Future[A], transform func(A) B) Future[B] {
	promise, future := Create[B]()
	go func() {
		result := f.Await()
		value := transform(result)
		promise.Fulfill(value)
	}()
	return future
}
