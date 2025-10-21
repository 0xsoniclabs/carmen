// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::mem;

use crate::{
    database::verkle::variants::managed::nodes::{
        empty::EmptyNode, inner::InnerNode, leaf::FullLeafNode, sparse_leaf::SparseLeafNode,
    },
    types::{AllVariants, DiskRepresentable, DiskRepresentableByType, NodeSize, ToNodeType},
};

pub mod empty;
pub mod id;
pub mod inner;
pub mod leaf;
pub mod sparse_leaf;

/// A node in a managed Verkle trie.
//
/// Non-empty nodes are stored as boxed to save memory (otherwise the size of [Node] would be
/// dictated by the largest variant).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    Empty(EmptyNode),
    Inner(Box<InnerNode>),
    Leaf2(Box<SparseLeafNode<2>>),
    Leaf256(Box<FullLeafNode>),
}

impl ToNodeType for Node {
    type NodeType = NodeType;

    fn to_node_type(&self) -> Self::NodeType {
        match self {
            Node::Empty(_) => NodeType::Empty,
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

impl DiskRepresentableByType for Node {
    type EnumType = NodeType;

    fn from_disk_repr<E>(
        et: &Self::EnumType,
        read_into_buffer: impl FnOnce(&mut [u8]) -> Result<(), E>,
    ) -> Result<Self, E> {
        match et {
            NodeType::Empty => Ok(Node::Empty(EmptyNode)),
            NodeType::Inner => InnerNode::from_disk_repr(read_into_buffer)
                .map(Box::new)
                .map(Node::Inner),
            NodeType::Leaf2 => SparseLeafNode::from_disk_repr(read_into_buffer)
                .map(Box::new)
                .map(Node::Leaf2),
            NodeType::Leaf256 => FullLeafNode::from_disk_repr(read_into_buffer)
                .map(Box::new)
                .map(Node::Leaf256),
        }
    }

    fn to_disk_repr(&self) -> &[u8] {
        match self {
            Node::Empty(_) => &[],
            Node::Inner(n) => n.to_disk_repr(),
            Node::Leaf2(n) => n.to_disk_repr(),
            Node::Leaf256(n) => n.to_disk_repr(),
        }
    }

    fn disk_size(et: &Self::EnumType) -> usize {
        match et {
            NodeType::Empty => 0,
            NodeType::Inner => mem::size_of::<InnerNode>(),
            NodeType::Leaf2 => mem::size_of::<SparseLeafNode<2>>(),
            NodeType::Leaf256 => mem::size_of::<FullLeafNode>(),
        }
    }
}

impl Default for Node {
    fn default() -> Self {
        Node::Empty(EmptyNode)
    }
}

/// A node type of a node in a managed Verkle trie.
/// This type is primarily used for conversion between [`Node`] and indexes in the file storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeType {
    Empty,
    Inner,
    Leaf2,
    Leaf256,
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

impl AllVariants for NodeType {
    fn all_variants() -> &'static [(Self, &'static str)] {
        &[
            (Self::Empty, "empty_node"),
            (Self::Inner, "inner_node"),
            (Self::Leaf2, "leaf_node_2"),
            (Self::Leaf256, "leaf_node_256"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let empty_node = Node::Empty(EmptyNode);
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
