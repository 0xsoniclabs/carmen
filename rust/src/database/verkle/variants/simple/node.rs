// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use crate::{
    database::{
        verkle::{
            compute_commitment::compute_leaf_node_commitment,
            crypto::{Commitment, Scalar},
        },
        visitor::NodeVisitor,
    },
    error::{BTResult, Error},
    statistics::trie_count::TrieCountVisitor,
    types::{Key, Value},
};

/// A node in the simple in-memory Verkle trie.
#[derive(Debug)]
pub enum Node {
    Empty,
    Inner(InnerNode),
    Leaf(LeafNode),
}

impl Node {
    /// Returns the value associated with the given key, or the default value if
    /// the key does not exist.
    pub fn lookup(&self, key: &Key, depth: u8) -> Value {
        match self {
            Node::Empty => Value::default(),
            Node::Inner(inner) => inner.lookup(key, depth),
            Node::Leaf(leaf) => leaf.lookup(key),
        }
    }

    /// Stores the value for the given key.
    /// Consumes the node and returns an updated version.
    pub fn store(self, key: &Key, depth: u8, value: &Value) -> Node {
        match self {
            Node::Empty => {
                if depth == 0 {
                    // While conceptually it would suffice to create a leaf node here,
                    // Geth always creates an inner node (and we want to stay compatible).
                    let inner = InnerNode::new();
                    inner.store(key, depth, value)
                } else {
                    let leaf = LeafNode::new(key);
                    leaf.store(key, depth, value)
                }
            }
            Node::Inner(inner) => inner.store(key, depth, value),
            Node::Leaf(leaf) => leaf.store(key, depth, value),
        }
    }

    /// Computes and returns the commitment of this node.
    ///
    /// If the commitment is already up to date, it is returned without recomputation.
    pub fn commit(&mut self) -> Commitment {
        match self {
            Node::Empty => Commitment::default(),
            Node::Inner(inner) => inner.commit(),
            Node::Leaf(leaf) => leaf.commit(),
        }
    }

    /// Returns the current commitment of this node without recomputing it.
    fn get_commitment(&self) -> Commitment {
        match self {
            Node::Empty => Commitment::default(),
            Node::Inner(inner) => inner.commitment,
            Node::Leaf(leaf) => leaf.commitment,
        }
    }

    /// Returns true if the commitment of this node is dirty and needs to be recomputed.
    fn commitment_is_dirty(&self) -> bool {
        match self {
            Node::Empty => false,
            Node::Inner(inner) => inner.commitment_dirty,
            Node::Leaf(leaf) => leaf.commitment_dirty,
        }
    }

    /// Accepts a visitor for recursively traversing the node and its children.
    pub fn accept(&self, visitor: &mut impl NodeVisitor<Self>, level: u64) -> BTResult<(), Error> {
        visitor.visit(self, level)?;
        match self {
            Node::Empty | Node::Leaf(_) => {}
            Node::Inner(inner) => {
                for child in inner.children.iter() {
                    child.accept(visitor, level + 1)?;
                }
            }
        }
        Ok(())
    }
}

/// An inner node in the simple in-memory Verkle trie, containing up to 256 children.
#[derive(Debug)]
pub struct InnerNode {
    children: Box<[Node; 256]>,
    commitment: Commitment,
    commitment_initialized: bool,
    commitment_dirty: bool,
}

impl InnerNode {
    /// Creates a new inner node without any children.
    fn new() -> Self {
        let children = Box::new([const { Node::Empty }; 256]);
        InnerNode {
            children,
            commitment: Commitment::default(),
            commitment_initialized: false,
            commitment_dirty: true,
        }
    }

    /// Creates a new inner node with the given leaf node as child at the given position.
    /// The commitment of the inner node is initialized from the leaf's commitment.
    pub fn new_with_leaf(leaf: LeafNode, position: u8) -> Self {
        let mut inner = Self::new();
        // Take leaf's commitment as initial commitment of the inner node.
        // This is required to ensure that this node's parent can correctly update its commitment
        // by computing the delta between the old and new child's commitments.
        inner.commitment = leaf.commitment;
        inner.children[position as usize] = Node::Leaf(leaf);
        inner
    }

    /// Returns the value associated with the given key, by forwarding the request to
    /// the child at position `key[depth]`.
    pub fn lookup(&self, key: &Key, depth: u8) -> Value {
        self.children[key[depth as usize] as usize].lookup(key, depth + 1)
    }

    /// Stores the value for the given key by forwarding the request to the child at
    /// position `key[depth]`.
    ///
    /// If no child exists at that position, a new leaf node is created.
    ///
    /// Consumes the node and returns an updated version.
    pub fn store(mut self, key: &Key, depth: u8, value: &Value) -> Node {
        self.commitment_dirty = true;

        let pos = key[depth as usize];
        let next = std::mem::replace(&mut self.children[pos as usize], Node::Empty);
        self.children[pos as usize] = next.store(key, depth + 1, value);
        Node::Inner(self)
    }

    /// Computes and returns the commitment of this node, by first updating the commitments
    /// of all children that are dirty and then updating the respective positions within the
    /// commitment of this node.
    ///
    /// If the commitment is already up to date, it is returned without recomputation.
    pub fn commit(&mut self) -> Commitment {
        if !self.commitment_dirty {
            return self.commitment;
        }

        let _span = tracy_client::span!("InnerNode::commit");
        let mut values = [Scalar::zero(); 256];

        for (i, child) in self.children.iter_mut().enumerate() {
            if self.commitment_initialized {
                if child.commitment_is_dirty() {
                    let old_child_commitment = child.get_commitment();
                    let new_child_commitment = child.commit();
                    self.commitment.update(
                        i as u8,
                        old_child_commitment.to_scalar(),
                        new_child_commitment.to_scalar(),
                    );
                }
            } else {
                values[i] = child.commit().to_scalar();
            }
        }

        if !self.commitment_initialized {
            self.commitment = Commitment::new(&values);
        }

        self.commitment_initialized = true;
        self.commitment_dirty = false;
        self.commitment
    }
}

/// A leaf node in the simple in-memory Verkle trie, containing 256 values.
///
/// Alongside the values and commitment, the leaf node also stores:
/// - The 31-byte stem that is common to all keys in this leaf, required to distinguish from keys
///   that share a common prefix, as well as for non-existence proofs.
/// - A bitmap indicating which of the 256 values have been modified at some point, required for
///   future state expiry schemes.
#[derive(Debug)]
pub struct LeafNode {
    stem: [u8; 31],
    values: Box<[Value; 256]>,
    used_bits: [u8; 256 / 8],
    commitment: Commitment,
    commitment_dirty: bool,
}

impl LeafNode {
    /// Creates a new leaf node for the given key, initializing all values to the default [Value].
    pub fn new(key: &Key) -> Self {
        let values = Box::new([Value::default(); 256]);
        LeafNode {
            stem: key[..31].try_into().unwrap(), // safe to unwrap because `Key` is 32 bytes long
            values,
            used_bits: [0; 256 / 8],
            commitment: Commitment::default(),
            commitment_dirty: true,
        }
    }

    /// Returns the value associated with the given key, or the default [Value] if the key does
    /// not match the stem of this leaf.
    pub fn lookup(&self, key: &Key) -> Value {
        if key[..31] != self.stem {
            Value::default()
        } else {
            self.values[key[31] as usize]
        }
    }

    /// Stores the value for the given key.
    ///
    /// If the stem of the key does not match the stem of this leaf, the leaf is split
    /// into an inner node with two children (the existing leaf and a new leaf for the key).
    pub fn store(mut self, key: &Key, depth: u8, value: &Value) -> Node {
        if key[..31] == self.stem {
            let suffix = key[31];
            self.values[suffix as usize] = *value;
            self.used_bits[(suffix / 8) as usize] |= 1 << (suffix % 8);
            self.commitment_dirty = true;
            return Node::Leaf(self);
        }

        // This leaf needs to be split
        let pos = self.stem[depth as usize];
        let inner = InnerNode::new_with_leaf(self, pos);
        inner.store(key, depth, value)
    }

    /// Computes and returns the commitment of this leaf node.
    ///
    /// If the commitment is already up to date, it is returned without recomputation.
    pub fn commit(&mut self) -> Commitment {
        if !self.commitment_dirty {
            return self.commitment;
        }
        let _span = tracy_client::span!("LeafNode::commit");
        self.commitment = compute_leaf_node_commitment(&self.values, &self.used_bits, &self.stem);
        self.commitment_dirty = false;
        self.commitment
    }
}

impl NodeVisitor<Node> for TrieCountVisitor {
    fn visit(&mut self, node: &Node, level: u64) -> BTResult<(), Error> {
        match node {
            Node::Empty => {
                self.count_node(level, "Empty", None);
            }
            Node::Leaf(node) => {
                self.count_node(
                    level,
                    "Leaf",
                    Some(node.used_bits.iter().map(|b| b.count_ones() as u64).sum()),
                );
            }
            Node::Inner(node) => {
                self.count_node(
                    level,
                    "Inner",
                    Some(
                        node.children
                            .iter()
                            .filter(|c| !matches!(c, Node::Empty))
                            .count() as u64,
                    ),
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::verkle::{
        crypto::Scalar,
        test_utils::{make_key, make_leaf_key, make_value},
    };

    // NOTE: Tests for the accept method are in simple/mod.rs

    #[test]
    fn empty_node_store_creates_inner_node() {
        let key = make_key(&[1, 2, 3]);
        let value = make_value(42);
        let node = Node::Empty.store(&key, 0, &value);
        assert!(matches!(node, Node::Inner(_)));
    }

    #[test]
    fn inner_node_new_creates_empty_node() {
        let inner = InnerNode::new();
        for child in inner.children.iter() {
            assert!(matches!(child, Node::Empty));
        }
        assert_eq!(inner.commitment, Commitment::default());
        assert!(inner.commitment_dirty);
    }

    #[test]
    fn inner_node_new_with_leaf_creates_node_with_child() {
        let key = make_key(&[9, 2, 3]);
        let leaf_commitment = Commitment::new(&[Scalar::from(42)]);
        let mut leaf = LeafNode::new(&key);
        leaf.commitment = leaf_commitment;
        let position = key[0];
        let inner = InnerNode::new_with_leaf(leaf, position);
        for (i, child) in inner.children.iter().enumerate() {
            if i == position as usize {
                assert!(matches!(child, Node::Leaf(_)));
            } else {
                assert!(matches!(child, Node::Empty));
            }
        }
        // Newly created inner node has commitment of the leaf.
        assert_eq!(inner.commitment, leaf_commitment);
        assert!(inner.commitment_dirty);
    }

    #[test]
    fn inner_node_lookup_returns_default_value_if_there_is_no_next_node() {
        let inner = InnerNode::new();
        let key = [0; 32];
        let value = inner.lookup(&key, 0);
        assert_eq!(value, Value::default());
    }

    #[test]
    fn inner_node_lookup_returns_value_from_next_node() {
        let key1 = make_key(&[1, 2, 3]);
        let key2 = make_key(&[1, 2, 4]);

        let root = Node::Leaf(LeafNode::new(&key1));
        let root = root.store(&key1, 2, &make_value(42));
        let root = root.store(&key2, 2, &make_value(84));

        assert!(
            matches!(root, Node::Inner(_)),
            "root should be an InnerNode"
        );

        assert_eq!(root.lookup(&key1, 2), make_value(42));
        assert_eq!(root.lookup(&key2, 2), make_value(84));
    }

    #[test]
    fn inner_node_store_creates_new_leaf_if_there_is_no_next_node() {
        let key = make_key(&[1, 2, 3]);
        let inner = InnerNode::new();
        assert!(matches!(inner.children[key[2] as usize], Node::Empty));

        let inner = inner.store(&key, 2, &make_value(42));
        let Node::Inner(inner) = inner else {
            panic!("expected InnerNode after set");
        };
        assert!(matches!(inner.children[key[2] as usize], Node::Leaf(_)));
    }

    #[test]
    fn inner_node_commit_dirty_state_is_tracked() {
        let inner = InnerNode::new();
        assert!(inner.commitment_dirty);

        // Setting a value should mark the commitment as dirty.
        let key = make_key(&[1, 2, 3]);
        let inner = inner.store(&key, 2, &make_value(42));
        let Node::Inner(mut inner) = inner else {
            panic!("expected InnerNode after set");
        };
        assert!(inner.commitment_dirty);

        // Committing should clean the state.
        let fist_commitment = inner.commit();
        assert!(!inner.commitment_dirty);

        // Committing again should return the same commitment.
        let second_commitment = inner.commit();
        assert!(!inner.commitment_dirty);
        assert_eq!(fist_commitment, second_commitment);

        // Setting another value should mark the commitment as dirty again.
        let inner = inner.store(&make_key(&[1, 2, 4]), 2, &make_value(84));
        let Node::Inner(inner) = inner else {
            panic!("expected InnerNode after set");
        };
        assert!(inner.commitment_dirty);
    }

    #[test]
    fn inner_node_commit_computes_commitment_from_children() {
        let inner = InnerNode::new();
        let key1 = make_key(&[1, 2, 3]);

        let inner = inner.store(&key1, 2, &make_value(42));
        let Node::Inner(mut inner) = inner else {
            panic!("expected InnerNode after set");
        };

        assert!(!inner.commitment_initialized);
        let commitment = inner.commit();
        assert!(inner.commitment_initialized);

        let mut child_commitments = vec![Scalar::zero(); 256];
        child_commitments[key1[2] as usize] = inner.children[key1[2] as usize].commit().to_scalar();
        let expected_commitment = Commitment::new(&child_commitments);
        assert_eq!(commitment, expected_commitment);

        let key2 = make_key(&[1, 2, 4]);
        let inner = inner.store(&key2, 2, &make_value(84));
        let Node::Inner(mut inner) = inner else {
            panic!("expected InnerNode after set");
        };

        let commitment = inner.commit();

        child_commitments[key2[2] as usize] = inner.children[key2[2] as usize].commit().to_scalar();
        let expected_commitment = Commitment::new(&child_commitments);
        assert_eq!(commitment, expected_commitment);
    }

    #[test]
    fn leaf_node_new_produces_empty_leaf_with_stem() {
        let key = make_key(&[1, 2, 3, 4, 5]);
        let leaf = LeafNode::new(&key);

        assert_eq!(
            &leaf.stem[..],
            &key[..31],
            "stem should match the first 31 bytes of the key"
        );
        assert_eq!(
            leaf.values,
            Box::new([Value::default(); 256]),
            "all values should be initialized to zero"
        );
        assert_eq!(leaf.used_bits, [0; 256 / 8], "used bitmap should be empty");
    }

    #[test]
    fn leaf_node_lookup_returns_value_for_matching_stem() {
        let key = make_leaf_key(&[1, 2, 3, 4, 5], 1);
        let leaf = LeafNode::new(&key);

        // Initially, the value for the key should be zero.
        assert_eq!(leaf.lookup(&key), Value::default());

        let leaf = leaf.store(&key, 0, &make_value(42));
        let Node::Leaf(leaf) = leaf else {
            panic!("expected LeafNode after set");
        };
        assert_eq!(leaf.lookup(&key), make_value(42),);
    }

    #[test]
    fn leaf_node_lookup_returns_zero_for_non_matching_stem() {
        let key1 = make_key(&[1, 2, 3]);
        let key2 = make_key(&[4, 5, 6]);
        let leaf = LeafNode::new(&key1);
        let leaf = leaf.store(&key1, 0, &make_value(42));

        assert_eq!(
            leaf.lookup(&key2, 0),
            Value::default(),
            "value for non-matching key should be zero"
        );
    }

    #[test]
    fn leaf_node_store_splits_leaf_if_stem_does_not_match() {
        let key1 = make_key(&[1, 2, 3]);
        let key2 = make_key(&[1, 2, 4]);

        let leaf = LeafNode::new(&key1);
        let mut leaf = leaf.store(&key1, 0, &make_value(42));
        let leaf_commitment = leaf.commit();

        let new_node = leaf.store(&key2, 2, &make_value(84));
        let Node::Inner(inner) = new_node else {
            panic!("expected InnerNode after set");
        };

        // Original leaf is now a child of the inner node.
        let value = inner.children[key1[2] as usize].lookup(&key1, 2);
        assert_eq!(value, make_value(42));
        // Inner node has commitment of original leaf.
        assert_eq!(inner.commitment, leaf_commitment);
    }

    #[test]
    fn leaf_node_can_store_and_lookup_values() {
        fn is_used(leaf: &LeafNode, suffix: u8) -> bool {
            leaf.used_bits[(suffix / 8) as usize] & (1 << (suffix % 8)) != 0
        }

        let key1 = make_leaf_key(&[1, 2, 3], 1);
        let key2 = make_leaf_key(&[1, 2, 3], 2);
        let key3 = make_leaf_key(&[1, 2, 3], 3);

        let leaf = LeafNode::new(&key1);

        assert!(!is_used(&leaf, key1[31]));
        assert!(!is_used(&leaf, key2[31]));
        assert!(!is_used(&leaf, key3[31]));

        assert_eq!(leaf.lookup(&key1), Value::default());
        assert_eq!(leaf.lookup(&key2), Value::default());
        assert_eq!(leaf.lookup(&key3), Value::default());

        // Setting a value for key 1 makes the value retrievable and marks the suffix as used.
        let leaf = leaf.store(&key1, 0, &make_value(10));
        let Node::Leaf(leaf) = leaf else {
            panic!("expected LeafNode after set");
        };

        assert!(is_used(&leaf, key1[31]));
        assert!(!is_used(&leaf, key2[31]));
        assert!(!is_used(&leaf, key3[31]));

        assert_eq!(leaf.lookup(&key1), make_value(10));
        assert_eq!(leaf.lookup(&key2), Value::default());
        assert_eq!(leaf.lookup(&key3), Value::default());

        // Setting the value for key 2 to zero does not change the value but marks the suffix as
        // used.
        let leaf = leaf.store(&key2, 0, &Value::default());
        let Node::Leaf(leaf) = leaf else {
            panic!("expected LeafNode after set");
        };

        assert!(is_used(&leaf, key1[31]));
        assert!(is_used(&leaf, key2[31]));
        assert!(!is_used(&leaf, key3[31]));

        assert_eq!(leaf.lookup(&key1), make_value(10));
        assert_eq!(leaf.lookup(&key2), Value::default());
        assert_eq!(leaf.lookup(&key3), Value::default());

        // Resetting the value for key 1 to zero does not change the used bitmap.
        let leaf = leaf.store(&key1, 0, &Value::default());
        let Node::Leaf(leaf) = leaf else {
            panic!("expected LeafNode after set");
        };

        assert!(is_used(&leaf, key1[31]));
        assert!(is_used(&leaf, key2[31]));
        assert!(!is_used(&leaf, key3[31]));

        assert_eq!(leaf.lookup(&key1), Value::default());
        assert_eq!(leaf.lookup(&key2), Value::default());
        assert_eq!(leaf.lookup(&key3), Value::default());
    }

    #[test]
    fn leaf_node_can_compute_commitment() {
        let key1 = make_leaf_key(&[1, 2, 3], 1);
        let key2 = make_leaf_key(&[1, 2, 3], 130);

        let mut val1 = [0; 32];
        val1[8..16].copy_from_slice(&42u64.to_be_bytes());
        let mut val2 = [0; 32];
        val2[8..16].copy_from_slice(&84u64.to_be_bytes());

        let leaf = LeafNode::new(&key1);
        let leaf = leaf.store(&key1, 0, &val1);
        let leaf = leaf.store(&key2, 0, &val2);
        let Node::Leaf(mut leaf) = leaf else {
            panic!("expected LeafNode after set");
        };

        let have = leaf.commit();

        let mut low1 = Scalar::from_le_bytes(&val1[..16]);
        let mut low2 = Scalar::from_le_bytes(&val2[..16]);
        let high1 = Scalar::from_le_bytes(&val1[16..]);
        let high2 = Scalar::from_le_bytes(&val2[16..]);
        low1.set_bit128();
        low2.set_bit128();

        let mut c1_values = vec![Scalar::zero(); 256];
        let mut c2_values = vec![Scalar::zero(); 256];
        c1_values[2] = low1;
        c1_values[3] = high1;
        c2_values[4] = low2;
        c2_values[5] = high2;

        let c1 = Commitment::new(&c1_values);
        let c2 = Commitment::new(&c2_values);
        let mut combined = vec![Scalar::zero(); 256];
        combined[0] = Scalar::from(1);
        combined[1] = Scalar::from_le_bytes(&key1[..31]);
        combined[2] = c1.to_scalar();
        combined[3] = c2.to_scalar();
        let want = Commitment::new(&combined);

        assert_eq!(have, want);
    }

    #[test]
    fn leaf_node_commitment_dirty_state_is_tracked() {
        let key1 = make_leaf_key(&[1, 2, 3], 1);
        let key2 = make_leaf_key(&[1, 2, 3], 130);

        let leaf = LeafNode::new(&key1);
        assert!(leaf.commitment_dirty);

        let leaf = leaf.store(&key1, 0, &make_value(10));
        let Node::Leaf(leaf) = leaf else {
            panic!("expected LeafNode after set");
        };
        assert!(leaf.commitment_dirty);

        let leaf = leaf.store(&key2, 0, &make_value(20));
        let Node::Leaf(mut leaf) = leaf else {
            panic!("expected LeafNode after set");
        };
        assert!(leaf.commitment_dirty);

        let first = leaf.commit();
        assert!(!leaf.commitment_dirty);

        let second = leaf.commit();
        assert!(!leaf.commitment_dirty);
        assert_eq!(first, second);

        let leaf = leaf.store(&key1, 0, &make_value(30));
        let Node::Leaf(mut leaf) = leaf else {
            panic!("expected LeafNode after set");
        };
        assert!(leaf.commitment_dirty);

        let third = leaf.commit();
        assert!(!leaf.commitment_dirty);

        assert_ne!(first, third);
    }
}
