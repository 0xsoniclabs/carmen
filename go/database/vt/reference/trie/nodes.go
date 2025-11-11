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
	commitment      commit.Commitment
	commitmentClean bool
}

func (i *inner) get(key Key, depth byte) Value {
	next := i.children[key[depth]]
	if next == nil {
		return Value{}
	}
	return next.get(key, depth+1)
}

func (i *inner) set(key Key, depth byte, value Value) node {
	i.commitmentClean = false
	pos := key[depth]
	next := i.children[pos]
	if next == nil {
		next = newLeaf(key)
	}
	i.children[pos] = next.set(key, depth+1, value)
	return i
}

func (i *inner) commit() commit.Commitment {
	if i.commitmentClean {
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
	i.commitmentClean = true
	return i.commitment
}

// ---- Leaf nodes ----

// leaf is the type of a leaf node in the Verkle trie. It contains a stem (the
// first 31 bytes of the key) and an array of values indexed by the last byte
// of the key.
type leaf struct {
	stem   [31]byte      // The first 31 bytes of the key leading to this leaf.
	values [256]Value    // The values stored in this leaf, indexed by the last byte of the key.
	used   [256 / 8]byte // A bitmap indicating which suffixes (last byte of the key) are used.

	// The cached commitment of this inner node. It is only valid if the
	// commitmentClean flag is true.
	commitment      commit.Commitment
	commitmentClean bool
}

// newLeaf creates a new leaf node with the given key.
func newLeaf(key Key) *leaf {
	return &leaf{
		stem: [31]byte(key[:31]),
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
		l.values[suffix] = value
		l.used[suffix/8] |= 1 << (suffix % 8)
		l.commitmentClean = false
		return l
	}

	// This leaf needs to be split
	res := &inner{}
	res.children[l.stem[depth]] = l
	return res.set(key, depth, value)
}

func (l *leaf) commit() commit.Commitment {
	if l.commitmentClean {
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

		if l.isUsed(byte(i)) {
			lower.SetBit128()
		}

		values[i/128][(2*i)%256] = lower
		values[i/128][(2*i+1)%256] = upper
	}

	c1 := commit.Commit(values[0])
	c2 := commit.Commit(values[1])

	l.commitment = commit.Commit([256]commit.Value{
		commit.NewValue(1),
		commit.NewValueFromLittleEndianBytes(l.stem[:]),
		c1.ToValue(),
		c2.ToValue(),
	})
	l.commitmentClean = true
	return l.commitment
}

func (l *leaf) isUsed(suffix byte) bool {
	return (l.used[suffix/8] & (1 << (suffix % 8))) != 0
}
