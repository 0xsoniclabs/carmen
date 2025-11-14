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

	"github.com/stretchr/testify/require"
)

func TestNewTask_CreatesTaskWithGivenActionAndNumberOfDependencies(t *testing.T) {
	require := require.New(t)

	counter := 0
	action := func() {
		counter++
	}

	numDependencies := 3
	tk := newTask(action, numDependencies)
	require.EqualValues(numDependencies, tk.numDependencies.Load())
	tk.action()
	require.Equal(1, counter)
}

func TestTask_Run_ExecutesAction(t *testing.T) {
	require := require.New(t)

	counter := 0
	action := func() {
		counter++
	}

	tk := newTask(action, 0)
	tk.run()
	require.Equal(1, counter)
}

func TestTask_Run_DecreasesParentDependencyCountAndReturnsParentWhenReady(t *testing.T) {
	require := require.New(t)

	parent := newTask(func() {}, 2)
	child := newTask(func() {}, 0)
	child.parentTask = parent

	// First run should decrease parent's dependency count but not make it ready.
	result := child.run()
	require.Nil(result)
	require.EqualValues(1, parent.numDependencies.Load())

	// Second run should make parent ready.
	result = child.run()
	require.Equal(parent, result)
	require.EqualValues(0, parent.numDependencies.Load())
}

func TestRunTasks_TaskChain_ProcessesAllTasksInOrder(t *testing.T) {
	require := require.New(t)

	// Different lengths, to cover the sequential and parallel code paths.
	for _, N := range []int{1, 5, 10, 50} {
		t.Run(fmt.Sprintf("%d tasks", N), func(t *testing.T) {
			done := make([]bool, N)
			tasks := make([]*task, N)

			for i := range N {
				tasks[i] = newTask(func() {
					// This task has not been executed yet.
					require.False(done[i])
					if i > 0 {
						// Previous task must have been executed.
						require.True(done[i-1])
					}
					done[i] = true
				}, 1)
				if i > 0 {
					tasks[i-1].parentTask = tasks[i]
				}
			}

			// Mark the first task as ready to run
			tasks[0].numDependencies.Store(0) // remove dependency of first task
			runTasks(tasks)

			for i := range N {
				require.True(done[i], "Task %d was not executed", i)
			}
		})
	}
}

func TestRunTasks_TaskTree_ProcessesAllTasksInOrder(t *testing.T) {
	require := require.New(t)

	// Test a task graph with N child tasks delivering input to a single root
	// task. The root task can only run once all child tasks have completed.
	for _, N := range []int{0, 1, 5, 10, 50} {
		t.Run(fmt.Sprintf("%d leafs", N), func(t *testing.T) {
			done := make([]bool, N+1)
			tasks := make([]*task, 0, N+1)

			// The root task verifying that all child tasks have completed.
			root := newTask(func() {
				for i := range N {
					// All child tasks must have been executed.
					require.True(done[i])
				}
				done[N] = true // mark root as done
			}, N)

			// The child tasks to be executed before the root task.
			for i := range N {
				tasks = append(tasks, newTask(func() {
					// This task has not been executed yet.
					require.False(done[i])
					done[i] = true
				}, 0))
				tasks[i].parentTask = root
			}

			// The root task is the last to list tasks in topological order.
			tasks = append(tasks, root)

			runTasks(tasks)
			require.True(done[N])
		})
	}
}
