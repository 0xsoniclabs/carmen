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
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum Node {
    #[default]
    Empty,
    Inner(Box<InnerNode>),
    Leaf2(Box<SparseLeafNode<2>>),
    Leaf256(Box<FullLeafNode>),
}

impl Node {
    pub fn to_node_type(&self) -> NodeType {
        match self {
            Node::Empty => NodeType::Empty,
            Node::Inner(_) => NodeType::Inner,
            Node::Leaf2(_) => NodeType::Leaf2,
            Node::Leaf256(_) => NodeType::Leaf256,
        }
    }
}

impl NodeSize for Node {
    fn node_byte_size(&self) -> usize {
        self.to_node_type().node_byte_size()
    }

    fn min_non_empty_node_size() -> usize {
        NodeType::min_non_empty_node_size()
    }
}

/// A node type of a node in a (file-based) Verkle trie.
/// This type is primarily used for conversion between [`Node`] and indexes in the file storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeType {
    Empty,
    Inner,
    Leaf2,
    Leaf256,
}

impl NodeType {
    pub fn from_node(node: &Node) -> Self {
        match node {
            Node::Empty => NodeType::Empty,
            Node::Inner(_) => NodeType::Inner,
            Node::Leaf2(_) => NodeType::Leaf2,
            Node::Leaf256(_) => NodeType::Leaf256,
        }
    }
}

impl NodeSize for NodeType {
    fn node_byte_size(&self) -> usize {
        let inner_size = match self {
            NodeType::Empty => 0,
            NodeType::Inner => {
                std::mem::size_of::<Box<InnerNode>>() + std::mem::size_of::<InnerNode>()
            }
            NodeType::Leaf2 => {
                std::mem::size_of::<Box<SparseLeafNode<2>>>()
                    + std::mem::size_of::<SparseLeafNode<2>>()
            }
            NodeType::Leaf256 => {
                std::mem::size_of::<Box<FullLeafNode>>() + std::mem::size_of::<FullLeafNode>()
            }
        };
        std::mem::size_of::<Node>() + inner_size
    }

    fn min_non_empty_node_size() -> usize {
        // Because we don't store empty nodes, the minimum size is the smallest non-empty node.
        NodeType::Leaf2.node_byte_size()
    }
}

/// A trait to determine the size of a node.
pub trait NodeSize {
    /// Returns the size of the node in bytes.
    fn node_byte_size(&self) -> usize;

    /// Returns the minimum size of a non-empty node in bytes.
    fn min_non_empty_node_size() -> usize;
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

    #[test]
    fn node_type_byte_size_returns_correct_size() {
        let empty_node = NodeType::Empty;
        let inner_node = NodeType::Inner;
        let leaf2_node = NodeType::Leaf2;
        let leaf256_node = NodeType::Leaf256;

        assert_eq!(empty_node.node_byte_size(), std::mem::size_of::<Node>());
        assert_eq!(
            inner_node.node_byte_size(),
            std::mem::size_of::<Node>()
                + std::mem::size_of::<Box<InnerNode>>()
                + std::mem::size_of::<InnerNode>()
        );
        assert_eq!(
            leaf2_node.node_byte_size(),
            std::mem::size_of::<Node>()
                + std::mem::size_of::<Box<SparseLeafNode<2>>>()
                + std::mem::size_of::<SparseLeafNode<2>>()
        );
        assert_eq!(
            leaf256_node.node_byte_size(),
            std::mem::size_of::<Node>()
                + std::mem::size_of::<Box<FullLeafNode>>()
                + std::mem::size_of::<FullLeafNode>()
        );
    }

    #[test]
    fn node_type_min_non_empty_node_size_returns_size_of_smallest_non_empty_node() {
        assert_eq!(
            NodeType::min_non_empty_node_size(),
            Node::Leaf2(Box::default()).node_byte_size()
        );
    }

    #[test]
    fn node_byte_size_returns_node_type_byte_size() {
        let empty_node = Node::Empty;
        let inner_node = Node::Inner(Box::default());
        let leaf2_node = Node::Leaf2(Box::default());
        let leaf256_node = Node::Leaf256(Box::default());

        assert_eq!(
            NodeType::Empty.node_byte_size(),
            empty_node.node_byte_size()
        );
        assert_eq!(
            NodeType::Inner.node_byte_size(),
            inner_node.node_byte_size()
        );
        assert_eq!(
            NodeType::Leaf2.node_byte_size(),
            leaf2_node.node_byte_size()
        );
        assert_eq!(
            NodeType::Leaf256.node_byte_size(),
            leaf256_node.node_byte_size()
        );
    }

    #[test]
    fn node_min_non_empty_node_size_returns_node_type_min_size() {
        assert_eq!(
            NodeType::min_non_empty_node_size(),
            Node::min_non_empty_node_size()
        );
    }
}
