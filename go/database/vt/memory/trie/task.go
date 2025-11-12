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
	"sync/atomic"

	"github.com/0xsoniclabs/tracy"
)

// This file provides a simple task execution framework not to be intended for
// use outside of this package. It is customized to the specific needs of the
// trie implementation, by only allowing a tree structure of tasks, where each
// task may have multiple dependencies, but only a single parent task that
// depends on it. However, these properties are not verified.
//
// The intended usage is to
//   1) create a set of tasks, closed under dependencies, topologically sorted
//      such that no task appears before any of its dependencies
//   2) call [runTasks]() with the set of tasks
//
// The framework will execute the tasks in parallel, respecting dependencies,
// and will return only once all tasks have been completed.

// task represents a unit of work to be executed. For this package's purposes, a
// task may represent a small unit of work organized in a tree structure, where
// each task may depend on the result of zero or more child tasks before it can
// be executed.
//
// Each tasks stores the action to be performed, the number of (yet) unfulfilled
// dependencies, and an optional parent task to notify when the task has been
// completed.
type task struct {
	action          func()       // < the action to perform
	numDependencies atomic.Int32 // < number of dependencies before this task can run
	parentTask      *task        // < optional parent task to notify when done
}

// newTask creates a new task with the specified action and number of
// dependencies. The task will only be able to run once all dependencies are
// satisfied.
func newTask(
	action func(),
	numDependencies int,
) *task {
	t := &task{action: action}
	t.numDependencies.Store(int32(numDependencies))
	return t
}

// run executes the task's action and may return a parent task that becomes
// ready to run as a result. If no parent task becomes ready, nil is returned.
func (t *task) run() *task {
	t.action()
	if t.parentTask == nil {
		return nil
	}
	// Decrease the parent's dependency count and check if it is ready. If not,
	// return nil, otherwise return the parent task to be run.
	if t.parentTask.numDependencies.Add(-1) != 0 {
		return nil // not ready yet
	}
	return t.parentTask // ready to run
}

// runTasks executes the given tasks in parallel, respecting their dependencies.
// Note that the provided list of tasks must include all tasks that are needed
// to satisfy dependencies. The function does not validate this. If dependencies
// are missing, the function may deadlock.
func runTasks(tasks []*task) {
	// Cut-off for a small number of tasks, in which case we run sequentially.
	// It is not worth the overhead of parallelism.
	if len(tasks) < 20 {
		for _, task := range tasks {
			task.action()
		}
		return
	}

	const NumWorkers = 7 // + this thread
	completedTasks := atomic.Uint32{}

	// Collect all tasks ready to run (no dependencies).
	workList := make([]*task, 0, len(tasks))
	for _, task := range tasks {
		if task.numDependencies.Load() == 0 {
			workList = append(workList, task)
		}
	}

	// Process tasks until all are done.
	pos := atomic.Int32{}
	processTasks := func() {
		zone := tracy.ZoneBegin("tasks::worker")
		defer zone.End()
		for {
			next := pos.Add(1) - 1
			if int(next) >= len(workList) {
				return
			}

			// Run this task and all tasks that become ready as a result.
			task := workList[next]
			for task != nil {
				task = task.run()
				completedTasks.Add(1)
			}
		}
	}

	// TODO: pre-start and/or re-use workers
	for range NumWorkers {
		go processTasks()
	}

	// This thread also helps with running tasks.
	processTasks()

	// The scheduled tasks are very short and reasonably well balanced, so we
	// can afford to do a busy wait here. This turns out to be faster than to
	// use a typical wait-group, as the processing may have finished before the
	// last worker gets even scheduled.
	// TODO: evaluate the usage of a signaling mechanism here.
	zone := tracy.ZoneBegin("tasks::wait_for_completion")
	for completedTasks.Load() < uint32(len(tasks)) {
	}
	zone.End()
}
