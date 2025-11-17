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

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCreate_PromiseAndFutureAreLinked(t *testing.T) {
	promise, future := Create[int]()
	promise.Fulfill(12)
	require.Equal(t, 12, future.Await())
}

func TestImmediate_FutureIsFulfilled(t *testing.T) {
	future := Immediate("hello")
	require.Equal(t, "hello", future.Await())
}

func TestForward_CanBeUsedToPipelineFutures(t *testing.T) {
	promise1, future1 := Create[string]()
	promise2, future2 := Create[string]()

	promise2.Forward(future1)

	promise1.Fulfill("hello")
	require.Equal(t, "hello", future2.Await())
}

func TestThen_FutureResultCanBeTransformed(t *testing.T) {
	promise1, future1 := Create[[]int]()
	future2 := Then(future1, func(value []int) int {
		return len(value)
	})

	promise1.Fulfill([]int{1, 2, 3, 4, 5})
	require.Equal(t, 5, future2.Await())
}
