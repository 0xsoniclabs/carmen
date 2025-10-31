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
	"fmt"

	"github.com/0xsoniclabs/carmen/go/database/vt/commit"
)

// ---- Nodes ----

// node is an interface for trie nodes, which can be either inner or leaf nodes.
type node interface {
	get(key Key, depth byte) Value
	set(Key Key, depth byte, value Value) node
	commit() commit.Commitment
}

// ---- Inner nodes ----

// inner is the type of an inner node in the Verkle trie. It contains an array
// of 256 child nodes, indexed by one byte of the key.
type inner struct {
	children [256]node

	// The cached commitment of this inner node. It is only valid if the
	// commitmentClean flag is true.
	commitment commit.Commitment

	// --- Commitment caching ---
	dirtyChildValues     bitMap
	oldChildrenValues    [256]commit.Value
	oldChildrenValuesSet bitMap
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
	if !i.oldChildrenValuesSet.get(pos) {
		i.oldChildrenValuesSet.set(pos)
		if next != nil {
			i.oldChildrenValues[pos] = next.commit().ToValue()
		}
	}
	if next == nil {
		next = newLeaf(key)
	}
	i.children[pos] = next.set(key, depth+1, value)
	i.dirtyChildValues.set(pos)
	return i
}

func (i *inner) commit() commit.Commitment {
	//return i.commit_naive()
	return i.commit_optimized()
}

func (i *inner) commit_optimized() commit.Commitment {
	if !i.dirtyChildValues.any() {
		return i.commitment
	}

	delta := [commit.VectorSize]commit.Value{}
	for j := range i.children {
		if i.dirtyChildValues.get(byte(j)) {
			old := i.oldChildrenValues[j]
			new := i.children[j].commit().ToValue()
			delta[j] = *new.Sub(old)
		}
	}

	// Update the commitment of this inner node.
	if i.commitment == (commit.Commitment{}) {
		i.commitment = commit.Commit(delta)
	} else {
		i.commitment.Add(commit.Commit(delta))
	}
	i.dirtyChildValues.clear()
	i.oldChildrenValuesSet.clear()
	return i.commitment
}

func (i *inner) commit_naive() commit.Commitment {
	if !i.dirtyChildValues.any() {
		return i.commitment
	}

	// The commitment of an inner node is computed as a Pedersen commitment
	// as follows:
	//
	//   C = Commit([C_i.ToValue() for i in children])
	//
	// For details, see
	// https://blog.ethereum.org/2021/12/02/verkle-tree-structure#commitment-of-internal-nodes

	// Recompute the commitment for this inner node.
	children := [256]commit.Value{}
	for j, child := range i.children {
		if child != nil { // for empty children, the value to commit to is zero
			children[j] = child.commit().ToValue()
		}
	}
	i.commitment = commit.Commit(children)
	i.dirtyChildValues.clear()
	return i.commitment
}

// ---- Leaf nodes ----

// leaf is the type of a leaf node in the Verkle trie. It contains a stem (the
// first 31 bytes of the key) and an array of values indexed by the last byte
// of the key.
type leaf struct {
	stem   [31]byte   // The first 31 bytes of the key leading to this leaf.
	values [256]Value // The values stored in this leaf, indexed by the last byte of the key.
	used   bitMap     // A bitmap indicating which suffixes (last byte of the key) are used.

	// The cached commitment of this inner node. It is only valid if the
	// commitmentClean flag is true.
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
		commitment: commit.Commit([256]commit.Value{
			commit.NewValue(1), // TODO: avoid recomputing this every time
			commit.NewValueFromLittleEndianBytes(key[:31]),
		}),
	}
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
	return res.set(key, depth, value)
}

func (l *leaf) commit() commit.Commitment {
	//return l.commit_naive()
	return l.commit_optimized()
}

func (l *leaf) commit_optimized() commit.Commitment {
	if !l.lowDirty && !l.highDirty {
		return l.commitment
	}

	leafDelta := [commit.VectorSize]commit.Value{}

	// update c1 - lower half + used bit
	if l.lowDirty {
		delta := [commit.VectorSize]commit.Value{}
		for i := range 128 {
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
		if l.c1 == (commit.Commitment{}) {
			newC1 = commit.Commit(delta)
		} else {
			newC1.Add(commit.Commit(delta))
		}

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
		if newC2 == (commit.Commitment{}) {
			newC2 = commit.Commit(delta)
		} else {
			newC2.Add(commit.Commit(delta))
		}

		deltaC2 := newC2.ToValue()
		deltaC2 = *deltaC2.Sub(l.c2.ToValue())

		leafDelta[3] = deltaC2
		l.c2 = newC2

		l.highDirty = false
	}

	// Compute commitment of changes and add to node commitment.
	l.commitment.Add(commit.Commit(leafDelta))
	l.oldValuesSet.clear()
	l.oldUsed = l.used
	return l.commitment
}

func (l *leaf) commit_naive() commit.Commitment {
	if !l.lowDirty && !l.highDirty {
		return l.commitment
	}

	// The commitment of a leaf node is computed as a Pedersen commitment
	// as follows:
	//
	//    C = Commit([1,stem, C1, C2])
	//
	// where C1 and C2 are the Pedersen commitments of the interleaved modified
	// lower and upper halves of the values stored in the leaf node, computed
	// by:
	//
	//   C1 = Commit([v[0][:16]), v[0][16:]), v[1][:16]), v[1][16:]), ...])
	//   C2 = Commit([v[128][:16]), v[128][16:]), v[129][:16]), v[129][16:]), ...])
	//
	// For details on the commitment procedure, see
	// https://blog.ethereum.org/2021/12/02/verkle-tree-structure#commitment-to-the-values-leaf-nodes

	// Compute the commitment for this leaf node.
	values := [2][256]commit.Value{}
	for i, v := range l.values {
		lower := commit.NewValueFromLittleEndianBytes(v[:16])
		upper := commit.NewValueFromLittleEndianBytes(v[16:])

		if l.used.get(byte(i)) {
			lower.SetBit128()
		}

		values[i/128][(2*i)%256] = lower
		values[i/128][(2*i+1)%256] = upper
	}

	c1 := commit.Commit(values[0])
	c2 := commit.Commit(values[1])

	fmt.Printf("\tnaive c1: %x\n", c1.ToValue())
	fmt.Printf("\tnaive c2: %x\n", c2.ToValue())

	l.commitment = commit.Commit([256]commit.Value{
		commit.NewValue(1),
		commit.NewValueFromLittleEndianBytes(l.stem[:]),
		c1.ToValue(),
		c2.ToValue(),
	})
	l.lowDirty = false
	l.highDirty = false
	return l.commitment
}

func (l *leaf) isUsed(index byte) bool {
	return l.used.get(index)
}

type bitMap [256 / 64]uint64

func (b *bitMap) get(index byte) bool {
	return (b[index/64] & (1 << (index % 64))) != 0
}

func (b *bitMap) set(index byte) {
	b[index/64] |= 1 << (index % 64)
}

func (b *bitMap) clear() {
	b[0] = 0
	b[1] = 0
	b[2] = 0
	b[3] = 0
}

func (b *bitMap) any() bool {
	return b[0]|b[1]|b[2]|b[3] != 0
}

func lowEqual(a, b Value) bool {
	for i := 0; i < 16; i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func highEqual(a, b Value) bool {
	for i := 16; i < 32; i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

type halfValue [16]byte

func (v Value) lower() halfValue {
	var hv halfValue
	copy(hv[:], v[:16])
	return hv
}

func (v Value) upper() halfValue {
	var hv halfValue
	copy(hv[:], v[16:])
	return hv
}

func (hv halfValue) equal(other halfValue) bool {
	return bytes.Equal(hv[:], other[:])
}
