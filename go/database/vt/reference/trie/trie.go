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
	"sync/atomic"

	"github.com/0xsoniclabs/carmen/go/database/vt/commit"
	"github.com/0xsoniclabs/tracy"
)

// Key is a fixed-size byte array used to address values in the trie.
type Key [32]byte

// Value is a fixed-size byte array used to represent data stored in the trie.
type Value [32]byte

// Trie implements an all-in-memory version of a Verkle trie as specified by
// Ethereum. It provides a basic key-value store with fixed-length keys and
// values and the ability to provide a cryptographic commitment of the trie's
// state using Pedersen commitments.
//
// This implementation is not optimized for performance or storage efficiency,
// but serves as a reference for the trie structure and operations. It is
// not intended for production use.
//
// For an overview of the Verkle trie structure, see
// https://blog.ethereum.org/2021/12/02/verkle-tree-structure
type Trie struct {
	config TrieConfig
	root   node
}

type TrieConfig struct {
	ParallelCommit bool
}

func NewTrie(config TrieConfig) *Trie {
	return &Trie{
		config: config,
	}
}

// Get retrieves the value associated with the given key from the trie. All keys
// that have not been set will return the zero value.
func (t *Trie) Get(key Key) Value {
	if t.root == nil {
		return Value{}
	}
	return t.root.get(key, 0)
}

// Set associates the given key with the specified value in the trie. If the key
// already exists, its value will be updated.
func (t *Trie) Set(key Key, value Value) {
	if t.root == nil {
		t.root = &inner{}
	}
	t.root = t.root.set(key, 0, value)
}

// Commit returns the cryptographic commitment of the current state of the trie.
func (t *Trie) Commit() commit.Commitment {
	if t.root == nil {
		return commit.Identity()
	}
	if t.config.ParallelCommit {
		return t.commit_parallel()
	}
	return t.commit_sequential()
}

func (t *Trie) commit_sequential() commit.Commitment {
	return t.root.commit()
}

func (t *Trie) commit_parallel() commit.Commitment {

	//fmt.Printf("\nNewRound\n")

	const NumWorkers = 8
	// Phase 1: collect tasks to be done in parallel
	tasks := make([]*task, 0, 1024)
	zone := tracy.ZoneBegin("trie::commit_parallel::collect_tasks")
	t.root.collectCommitTasks(&tasks)
	zone.End()

	// Phase 2: run tasks in parallel
	if false { // = debug task dependencies
		for i, task := range tasks {
			if task.numDependencies.Load() != 0 {
				//panic(fmt.Sprintf("task %s which is %d of %d has unresolved dependencies", task.name, i, len(tasks)))
				panic(fmt.Sprintf("task %d of %d has unresolved dependencies", i, len(tasks)))
			}
			task.run()
		}
	} else if len(tasks) < 20 {
		// For small number of tasks, run sequentially to avoid overhead.
		for _, task := range tasks {
			task.action()
		}
	} else {

		runTasks := atomic.Uint32{}

		// Collect all tasks ready to run (no dependencies).
		workList := make([]*task, 0, len(tasks))
		for _, task := range tasks {
			if task.numDependencies.Load() == 0 {
				workList = append(workList, task)
			}
		}

		// Process tasks until all are done.
		pos := atomic.Int32{}
		// TODO: re-use workers;
		for range NumWorkers {
			go func() {
				zone := tracy.ZoneBegin("trie::commit_parallel::worker")
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
						runTasks.Add(1)
					}
				}
			}()
		}

		// This thread also helps with running tasks.
		// TODO: clean up the code to avoid duplication.
		zone1 := tracy.ZoneBegin("trie::commit_parallel::main_worker")
		for {
			next := pos.Add(1) - 1
			if int(next) >= len(workList) {
				break
			}

			// Run this task and all tasks that become ready as a result.
			task := workList[next]
			for task != nil {
				task = task.run()
				runTasks.Add(1)
			}
		}
		zone1.End()

		// The scheduled tasks are very short, so we just do a busy wait here.
		// until all tasks are done.
		zone2 := tracy.ZoneBegin("trie::commit_parallel::wait_for_completion")
		for runTasks.Load() < uint32(len(tasks)) {
		}
		zone2.End()

		if want, got := len(tasks), int(runTasks.Load()); want != got {
			panic(fmt.Sprintf("not all tasks were run: want %d, got %d", want, got))
		}
		//fmt.Printf("Committed trie in parallel, ran %d / %d tasks\n", runTasks.Load(), len(tasks))
	}

	// Phase 3: fetch new root commitment
	zone2 := tracy.ZoneBegin("trie::commit_parallel::fetch_root_commitment")
	defer zone2.End()
	return t.root.commit()
}

type task struct {
	//name            string // debug name
	action          func()
	numDependencies atomic.Int32
	parentTask      *task
}

func newTask(
	//name string,
	action func(),
	numDependencies int,
) *task {
	t := &task{ /*name: name,*/ action: action}
	t.numDependencies.Store(int32(numDependencies))
	return t
}

// run executes the task's action and returns an optional parent task that may
// now be ready to run.
func (t *task) run() *task {
	t.action()
	if t.parentTask == nil {
		return nil
	}
	if t.parentTask.numDependencies.Add(-1) != 0 {
		return nil // not ready yet
	}
	return t.parentTask // ready to run
}
