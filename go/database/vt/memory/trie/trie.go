package trie

import (
	"bytes"
	"fmt"

	"github.com/0xsoniclabs/carmen/go/database/vt/memory/commit"
)

type Key [32]byte
type Value [32]byte
type stem [31]byte

type Trie struct {
	root node
}

func (t *Trie) Get(key Key) Value {
	if t.root == nil {
		return Value{}
	}
	return t.root.get(key, key[:])
}

func (t *Trie) Set(key Key, value Value) {
	if t.root == nil {
		t.root = newLeaf(key)
	}
	t.root = t.root.set(key, key[:], value)
}

func (t *Trie) Commit() commit.Commitment {
	if t.root == nil {
		return commit.Identity()
	}
	return t.root.commit()
}

func (t *Trie) Dump() {
	if t.root == nil {
		fmt.Printf("Trie is empty\n")
	} else {
		t.root.dump("")
	}

}

// ---- Nodes ----

type node interface {
	get(key Key, path []byte) Value
	set(Key Key, path []byte, value Value) node
	commit() commit.Commitment

	// Debug
	dump(indent string)
}

// ---- Inner nodes ----

type inner struct {
	children        [256]node
	commitment      commit.Commitment
	commitmentClean bool
}

func (i *inner) get(key Key, path []byte) Value {
	next := i.children[path[0]]
	if next == nil {
		return Value{}
	}
	return next.get(key, path[1:])
}

func (i *inner) set(key Key, path []byte, value Value) node {
	i.commitmentClean = false
	pos := path[0]
	next := i.children[pos]
	if next == nil {
		next = newLeaf(key)
	}
	i.children[pos] = next.set(key, path[1:], value)
	return i
}

func (i *inner) commit() commit.Commitment {
	if i.commitmentClean {
		return i.commitment
	}

	// see https://blog.ethereum.org/2021/12/02/verkle-tree-structure#commitment-of-internal-nodes

	// Recompute the commitment for this inner node.
	children := [256]commit.Value{}
	for j, child := range i.children {
		if child != nil {
			children[j] = child.commit().ToValue()
		}
	}
	i.commitment = commit.Commit(children)
	i.commitmentClean = true
	return i.commitment
}

func (i *inner) dump(indent string) {
	fmt.Printf("%sInner node:\n", indent)
	for _, child := range i.children {
		if child == nil {
			continue
		}
		child.dump(indent + "  ")
	}
}

// ---- Leaf nodes ----

type leaf struct {
	stem   [31]byte
	values [256]Value
	used   [256 / 8]byte

	commitment      commit.Commitment
	commitmentClean bool
}

func newLeaf(key Key) *leaf {
	return &leaf{
		stem: stem(key[:31]),
	}
}

func (l *leaf) get(key Key, _ []byte) Value {
	if !bytes.Equal(key[:31], l.stem[:]) {
		return Value{}
	}
	return l.values[key[31]]
}

func (l *leaf) set(key Key, path []byte, value Value) node {
	if bytes.Equal(key[:31], l.stem[:]) {
		suffix := key[31]
		l.values[suffix] = value
		l.used[suffix/8] |= 1 << (suffix % 8)
		l.commitmentClean = false
		return l
	}

	// This leaf needs to be split
	res := &inner{}
	res.children[l.stem[len(key)-len(path)]] = l
	return res.set(key, path, value)
}

func (l *leaf) commit() commit.Commitment {
	if l.commitmentClean {
		return l.commitment
	}

	// see https://blog.ethereum.org/2021/12/02/verkle-tree-structure#commitment-to-the-values-leaf-nodes

	// Compute the commitment for this leaf node.
	values := [2][256]commit.Value{}
	for i, v := range l.values {
		/*
			var lower commit.Value
			if l.isUsed(byte(i)) {
				// set the 128-th bit to 1 in lower + the lower 128 bits from v
				lower = commit.NewValueFromBytes(append([]byte{1}, v[:16]...))
			}
		*/
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

	//fmt.Printf("C1 inputs: %x\n", values[0])

	/*
		fmt.Printf("C1: %x\n", c1)
		fmt.Printf("C2: %x\n", c2)
	*/

	// C = commit(1,stem, c1, c2)

	/*
		fmt.Printf("Commit inputs: %x\n", [256]commit.Value{
			commit.NewValue(1),
			commit.NewValueFromLEBytes(l.stem[:]),
			c1.ToValue(),
			c2.ToValue(),
		})
	*/

	l.commitment = commit.Commit([256]commit.Value{
		commit.NewValue(1),
		commit.NewValueFromLittleEndianBytes(l.stem[:]),
		c1.ToValue(),
		c2.ToValue(),
	})
	l.commitmentClean = true
	return l.commitment
}

func (l *leaf) dump(indent string) {
	fmt.Printf("%sLeaf node: %x\n", indent, l.stem)
	fmt.Printf("%s  Commitment: %x\n", indent, l.commit())
	for i, v := range l.values {
		if l.isUsed(byte(i)) {
			fmt.Printf("%s  %d: %x\n", indent, i, v)
		}
	}
}

func (l *leaf) isUsed(suffix byte) bool {
	return (l.used[suffix/8] & (1 << (suffix % 8))) != 0
}
