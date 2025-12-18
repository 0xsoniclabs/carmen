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
	"bytes"

	"github.com/0xsoniclabs/carmen/go/database/vt/commit"
	"github.com/0xsoniclabs/tracy"
)

// ---- Nodes ----

// node is an interface for trie nodes, which can be either inner or leaf nodes.
type node interface {
	get(key Key, depth byte) Value
	set(Key Key, depth byte, value Value) node
	commit() commit.Commitment

	// -- parallel commit support --

	// collectCommitTasks requests the node to append tasks required to be
	// performed to update its commitment. The resulting list should list the
	// tasks in order of dependencies, i.e., when processing the list from start
	// to end, tasks producing inputs for other tasks should appear before
	// tasks consuming those inputs. Furthermore, the last task in the list
	// should be the task that updates this node's commitment. When this task is
	// completed, this node's commitment must be up to date.
	collectCommitTasks(tasks *[]*task)
}

// ---- Inner nodes ----

// inner is the type of an inner node in the Verkle trie. It contains an array
// of 256 child nodes, indexed by one byte of the key.
type inner struct {
	children [256]node

	// The cached commitment of this inner node. It is only valid if the
	// no dirtyChildCommitments bit is set.
	commitment commit.Commitment

	// --- Commitment caching ---
	dirtyChildCommitments bitMap                 // < which children have been modified
	oldChildCommitments   [256]commit.Commitment // < previous commitments of children
}

func (i *inner) get(key Key, depth byte) Value {
	next := i.children[key[depth]]
	if next == nil {
		return Value{}
	}
	return next.get(key, depth+1)
}

func (i *inner) set(key Key, depth byte, value Value) node {
	pos := key[depth]
	next := i.children[pos]
	if !i.dirtyChildCommitments.get(pos) {
		i.dirtyChildCommitments.set(pos)
		if next != nil {
			i.oldChildCommitments[pos] = next.commit()
		}
	}
	if next == nil {
		next = newLeaf(key)
	}
	i.children[pos] = next.set(key, depth+1, value)
	return i
}

func (i *inner) commit() commit.Commitment {
	if !i.dirtyChildCommitments.any() {
		return i.commitment
	}
	zone := tracy.ZoneBegin("inner::commit")
	defer zone.End()

	delta := [commit.VectorSize]commit.Value{}
	for j := range i.children {
		if i.dirtyChildCommitments.get(byte(j)) {
			new := i.children[j].commit().ToValue()
			old := i.oldChildCommitments[j].ToValue()
			delta[j] = *new.Sub(old)
		}
	}

	// Update the commitment of this inner node.
	i.commitment.Add(commit.Commit(delta))
	i.dirtyChildCommitments.clear()
	return i.commitment
}

func (i *inner) collectCommitTasks(tasks *[]*task) {
	if !i.dirtyChildCommitments.any() {
		return
	}

	// Produce one task for every dirty child.
	directChildTasks := make([]*task, 0, i.dirtyChildCommitments.popCount())
	delta := [commit.VectorSize]commit.Commitment{}
	for j := range i.children {
		if i.dirtyChildCommitments.get(byte(j)) {
			lengthBefore := len(*tasks)
			i.children[j].collectCommitTasks(tasks)
			childTasks := (*tasks)[lengthBefore:]

			var child *task
			numDependencies := 0
			if len(childTasks) > 0 {
				child = childTasks[len(childTasks)-1]
				numDependencies = 1
			}

			task := newTask(
				func() {
					old := i.oldChildCommitments[j].ToValue()
					new := i.children[j].commit().ToValue()
					poly := [commit.VectorSize]commit.Value{}
					poly[j] = *new.Sub(old)
					delta[j] = commit.Commit(poly)
				},
				numDependencies,
			)
			if child != nil {
				child.parentTask = task
			}

			directChildTasks = append(directChildTasks, task)
		}
	}
	*tasks = append(*tasks, directChildTasks...)

	// Add the aggregation task updating this inner node's commitment.
	aggTask := newTask(
		func() {
			// Sum up the individual deltas from the children.
			deltaCommitment := commit.Commitment{}
			for _, d := range delta {
				if d == (commit.Commitment{}) {
					continue
				}
				deltaCommitment.Add(d)
			}
			i.commitment.Add(deltaCommitment)
			i.dirtyChildCommitments.clear()
		},
		len(directChildTasks),
	)
	for _, childTask := range directChildTasks {
		childTask.parentTask = aggTask
	}

	*tasks = append(*tasks, aggTask)
}

// ---- Leaf nodes ----

// leaf is the type of a leaf node in the Verkle trie. It contains a stem (the
// first 31 bytes of the key) and an array of values indexed by the last byte
// of the key.
type leaf struct {
	stem   [31]byte   // The first 31 bytes of the key leading to this leaf.
	values [256]Value // The values stored in this leaf, indexed by the last byte of the key.
	used   bitMap     // A bitmap indicating which suffixes (last byte of the key) are used.

	// The cached commitment of this inner node. This is only valid if both
	// lowDirty and highDirty are false.
	commitment commit.Commitment

	// --- Commitment caching ---
	c1           commit.Commitment
	c2           commit.Commitment
	lowDirty     bool
	highDirty    bool
	oldUsed      bitMap
	oldValues    [256]Value
	oldValuesSet bitMap // < whether oldValues[i] is set to a meaningful value
}

// newLeaf creates a new leaf node with the given key.
func newLeaf(key Key) *leaf {
	return &leaf{
		stem: [31]byte(key[:31]),
	}
}

var _emptyStemCommitment = commit.Commit([256]commit.Value{
	commit.NewValue(1),
})

func getEmptyStemCommitment() commit.Commitment {
	return _emptyStemCommitment
}

func commitmentForStem(stem []byte) commit.Commitment {
	delta := commit.Commit([256]commit.Value{
		1: commit.NewValueFromLittleEndianBytes(stem),
	})
	res := getEmptyStemCommitment()
	res.Add(delta)
	return res
}

func (l *leaf) get(key Key, _ byte) Value {
	if !bytes.Equal(key[:31], l.stem[:]) {
		return Value{}
	}
	return l.values[key[31]]
}

func (l *leaf) set(key Key, depth byte, value Value) node {
	if bytes.Equal(key[:31], l.stem[:]) {
		suffix := key[31]
		old := l.values[suffix]
		l.values[suffix] = value
		if !l.used.get(suffix) {
			l.used.set(suffix)
		} else if old == value {
			return l
		}
		if !l.oldValuesSet.get(suffix) {
			l.oldValuesSet.set(suffix)
			l.oldValues[suffix] = old
		}
		if suffix < 128 {
			l.lowDirty = true
		} else {
			l.highDirty = true
		}
		return l
	}

	// This leaf needs to be split
	res := &inner{}
	res.children[l.stem[depth]] = l
	res.dirtyChildCommitments.set(l.stem[depth])
	return res.set(key, depth, value)
}

func (l *leaf) commit() commit.Commitment {
	if !l.lowDirty && !l.highDirty {
		return l.commitment
	}
	zone := tracy.ZoneBegin("leaf::commit")
	defer zone.End()

	leafDelta := [commit.VectorSize]commit.Value{}

	// update c1 - lower half + used bit
	if l.lowDirty {
		delta := [commit.VectorSize]commit.Value{}
		for i := range 128 {
			if !l.oldValuesSet.get(byte(i)) {
				continue
			}
			old := commit.NewValueFromLittleEndianBytes(l.oldValues[i][:16])
			if l.oldUsed.get(byte(i)) {
				old.SetBit128()
			}
			new := commit.NewValueFromLittleEndianBytes(l.values[i][:16])
			if l.used.get(byte(i)) {
				new.SetBit128()
			}
			delta[2*i] = *new.Sub(old)

			old = commit.NewValueFromLittleEndianBytes(l.oldValues[i][16:])
			new = commit.NewValueFromLittleEndianBytes(l.values[i][16:])
			delta[2*i+1] = *new.Sub(old)
		}

		newC1 := l.c1
		newC1.Add(commit.Commit(delta))

		deltaC1 := newC1.ToValue()
		deltaC1 = *deltaC1.Sub(l.c1.ToValue())

		leafDelta[2] = deltaC1
		l.c1 = newC1

		l.lowDirty = false
	}

	// update c2 - upper half
	if l.highDirty {
		delta := [commit.VectorSize]commit.Value{}
		for i := range 128 {
			if !l.oldValuesSet.get(byte(i + 128)) {
				continue
			}
			old := commit.NewValueFromLittleEndianBytes(l.oldValues[i+128][:16])
			if l.oldUsed.get(byte(i + 128)) {
				old.SetBit128()
			}
			new := commit.NewValueFromLittleEndianBytes(l.values[i+128][:16])
			if l.used.get(byte(i + 128)) {
				new.SetBit128()
			}
			delta[2*i] = *new.Sub(old)

			old = commit.NewValueFromLittleEndianBytes(l.oldValues[i+128][16:])
			new = commit.NewValueFromLittleEndianBytes(l.values[i+128][16:])
			delta[2*i+1] = *new.Sub(old)
		}

		newC2 := l.c2
		newC2.Add(commit.Commit(delta))

		deltaC2 := newC2.ToValue()
		deltaC2 = *deltaC2.Sub(l.c2.ToValue())

		leafDelta[3] = deltaC2
		l.c2 = newC2

		l.highDirty = false
	}

	// Compute commitment of changes and add to node commitment.
	if l.commitment == (commit.Commitment{}) {
		l.commitment = commitmentForStem(l.stem[:])
	}
	l.commitment.Add(commit.Commit(leafDelta))
	l.oldValuesSet.clear()
	l.oldUsed = l.used
	return l.commitment
}

func (l *leaf) collectCommitTasks(tasks *[]*task) {
	if !l.lowDirty && !l.highDirty {
		return // nothing to do
	}

	leafDelta := [2]commit.Commitment{}

	childTasks := make([]*task, 0, 3)

	// create task for initializing commitment with stem commitment if missing
	missingStemCommit := l.commitment == (commit.Commitment{})
	if missingStemCommit {
		childTasks = append(childTasks, newTask(
			func() {
				l.commitment = commitmentForStem(l.stem[:])
			},
			0,
		))
	}

	// create a task for updating C1
	if l.lowDirty {
		childTasks = append(childTasks, newTask(
			func() {
				delta := [commit.VectorSize]commit.Value{}
				for i := range 128 {
					if !l.oldValuesSet.get(byte(i)) {
						continue
					}
					old := commit.NewValueFromLittleEndianBytes(l.oldValues[i][:16])
					if l.oldUsed.get(byte(i)) {
						old.SetBit128()
					}
					new := commit.NewValueFromLittleEndianBytes(l.values[i][:16])
					if l.used.get(byte(i)) {
						new.SetBit128()
					}
					delta[2*i] = *new.Sub(old)

					old = commit.NewValueFromLittleEndianBytes(l.oldValues[i][16:])
					new = commit.NewValueFromLittleEndianBytes(l.values[i][16:])
					delta[2*i+1] = *new.Sub(old)
				}

				newC1 := l.c1
				newC1.Add(commit.Commit(delta))

				deltaC1 := newC1.ToValue()
				deltaC1 = *deltaC1.Sub(l.c1.ToValue())

				poly := [commit.VectorSize]commit.Value{}
				poly[2] = deltaC1
				leafDelta[0] = commit.Commit(poly)
				l.c1 = newC1

				l.lowDirty = false
			},
			0,
		))
	}

	// create a task for updating C2
	if l.highDirty {
		childTasks = append(childTasks, newTask(
			func() {
				delta := [commit.VectorSize]commit.Value{}
				for i := range 128 {
					if !l.oldValuesSet.get(byte(i + 128)) {
						continue
					}
					old := commit.NewValueFromLittleEndianBytes(l.oldValues[i+128][:16])
					if l.oldUsed.get(byte(i + 128)) {
						old.SetBit128()
					}
					new := commit.NewValueFromLittleEndianBytes(l.values[i+128][:16])
					if l.used.get(byte(i + 128)) {
						new.SetBit128()
					}
					delta[2*i] = *new.Sub(old)

					old = commit.NewValueFromLittleEndianBytes(l.oldValues[i+128][16:])
					new = commit.NewValueFromLittleEndianBytes(l.values[i+128][16:])
					delta[2*i+1] = *new.Sub(old)
				}

				newC2 := l.c2
				newC2.Add(commit.Commit(delta))

				deltaC2 := newC2.ToValue()
				deltaC2 = *deltaC2.Sub(l.c2.ToValue())

				poly := [commit.VectorSize]commit.Value{}
				poly[3] = deltaC2
				leafDelta[1] = commit.Commit(poly)
				l.c2 = newC2

				l.highDirty = false
			},
			0,
		))
	}

	// Create the aggregation task
	aggTask := newTask(
		func() {
			// Compute commitment of changes and add to node commitment.
			if leafDelta[0] != (commit.Commitment{}) {
				l.commitment.Add(leafDelta[0])
			}
			if leafDelta[1] != (commit.Commitment{}) {
				l.commitment.Add(leafDelta[1])
			}
			l.oldValuesSet.clear()
			l.oldUsed = l.used
		},
		len(childTasks),
	)

	// Set up dependencies.
	for _, childTask := range childTasks {
		childTask.parentTask = aggTask
	}
	*tasks = append(*tasks, childTasks...)
	*tasks = append(*tasks, aggTask)
}

func (l *leaf) isUsed(index byte) bool {
	return l.used.get(index)
}
