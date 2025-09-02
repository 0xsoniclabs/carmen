// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use zerocopy::{FromBytes, Immutable, IntoBytes, Unaligned};

use crate::types::{Commitment, NodeId, Value};

/// A value of a leaf node in a (file-based) Verkle trie, together with its index.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned,
)]
#[repr(C)]
pub struct ValueWithIndex {
    /// The index of the value in the leaf node.
    pub index: u8,
    /// The value stored in the leaf node.
    pub value: Value,
}

/// A sparsely populated leaf node in a (file-based) Verkle trie.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
#[repr(C)]
pub struct SparseLeafNode<const N: usize> {
    pub commitment: Commitment,
    pub stem: [u8; 31],
    pub values: [ValueWithIndex; N],
}

impl<const N: usize> Default for SparseLeafNode<N> {
    fn default() -> Self {
        let mut values = [ValueWithIndex::default(); N];
        values.iter_mut().enumerate().for_each(|(i, v)| {
            v.index = i as u8;
        });

        SparseLeafNode {
            commitment: Commitment::default(),
            stem: [0; 31],
            values,
        }
    }
}

/// A leaf node with 256 children in a (file-based) Verkle trie.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
#[repr(C)]
pub struct FullLeafNode {
    pub commitment: Commitment,
    pub stem: [u8; 31],
    pub values: [Value; 256],
}

impl Default for FullLeafNode {
    fn default() -> Self {
        FullLeafNode {
            commitment: Commitment::default(),
            stem: [0; 31],
            values: [Value::default(); 256],
        }
    }
}

/// An inner node in a (file-based) Verkle trie.
// NOTE: This type should NOT implement [`Clone`] because there is never be two instances
// corresponding to the same logical node.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned)]
#[repr(C)]
pub struct InnerNode {
    pub commitment: Commitment,
    pub values: [NodeId; 256],
}

impl Default for InnerNode {
    fn default() -> Self {
        InnerNode {
            commitment: Commitment::default(),
            values: [NodeId::from_idx_and_node_type(0, NodeType::Empty); 256],
        }
    }
}

/// A node in a (file-based) Verkle trie.
//
/// Non-empty nodes are stored as boxed to save memory (otherwise the size of [Node] would be
/// dictated by the largest variant).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    Empty,
    Inner(Box<InnerNode>),
    Leaf2(Box<SparseLeafNode<2>>),
    Leaf256(Box<FullLeafNode>),
}

/// A node type of a node in a (file-based) Verkle trie.
/// This type is primarily used for conversion between [`Node`] and indexes in the file storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Empty,
    Inner,
    Leaf2,
    Leaf256,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_leaf_node_default_returns_leaf_node_with_default_values_and_unique_indices() {
        const N: usize = 2;
        let node: SparseLeafNode<N> = SparseLeafNode::default();

        assert_eq!(node.commitment, Commitment::default());
        assert_eq!(node.stem, [0; 31]);

        for (i, value) in node.values.iter().enumerate() {
            assert_eq!(value.index, i as u8);
            assert_eq!(value.value, Value::default());
        }
    }

    #[test]
    fn full_leaf_node_default_returns_leaf_node_with_all_values_set_to_default() {
        let node: FullLeafNode = FullLeafNode::default();
        assert_eq!(node.commitment, Commitment::default());
        assert_eq!(node.stem, [0; 31]);
        assert_eq!(node.values, [Value::default(); 256]);
    }

    #[test]
    fn inner_node_default_returns_inner_node_with_all_values_set_to_empty_node_id() {
        let node: InnerNode = InnerNode::default();
        assert_eq!(node.commitment, Commitment::default());
        assert_eq!(
            node.values,
            [NodeId::from_idx_and_node_type(0, NodeType::Empty); 256]
        );
    }
}
