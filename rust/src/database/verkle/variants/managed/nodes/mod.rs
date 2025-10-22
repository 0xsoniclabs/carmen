// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use derive_deftly::Deftly;

use crate::{
    database::verkle::variants::managed::nodes::{
        empty::EmptyNode, id::NodeId, inner::InnerNode, leaf::FullLeafNode,
        sparse_leaf::SparseLeafNode,
    },
    storage::file::derive_deftly_template_FileStorageManager,
    types::NodeSize,
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
#[derive(Debug, Clone, PartialEq, Eq, Deftly)]
#[derive_deftly(FileStorageManager)]
pub enum Node {
    Empty(EmptyNode),
    Inner(Box<InnerNode>),
    Leaf2(Box<Leaf2Node>),
    Leaf256(Box<FullLeafNode>),
}

type Leaf2Node = SparseLeafNode<2>;
type Leaf256Node = FullLeafNode;

impl Node {
    pub fn to_node_type(&self) -> NodeType {
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
