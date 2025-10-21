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
        managed_trie::{
            CachedCommitment, CanStoreResult, LookupResult, ManagedTrieNode, UnionManagedTrieNode,
        },
        verkle::{
            crypto::Commitment,
            variants::managed::{
                NodeId,
                commitment::VerkleCommitmentInput,
                nodes::{
                    empty::EmptyNode, inner::InnerNode, leaf::FullLeafNode,
                    sparse_leaf::SparseLeafNode,
                },
            },
        },
    },
    error::Error,
    types::{Key, NodeSize, Value},
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

impl UnionManagedTrieNode for Node {}

impl ManagedTrieNode for Node {
    type Union = Node;
    type Id = NodeId;
    type Commitment = Commitment;
    type CommitmentInput = VerkleCommitmentInput;

    fn lookup(&self, key: &Key, depth: u8) -> Result<LookupResult<Self::Id>, Error> {
        match self {
            Node::Empty(n) => n.lookup(key, depth),
            Node::Inner(n) => n.lookup(key, depth),
            Node::Leaf2(n) => n.lookup(key, depth),
            Node::Leaf256(n) => n.lookup(key, depth),
        }
    }

    fn can_store(&self, key: &Key, depth: u8) -> Result<CanStoreResult<Self::Id>, Error> {
        match self {
            Node::Empty(n) => n.can_store(key, depth),
            Node::Inner(n) => n.can_store(key, depth),
            Node::Leaf2(n) => n.can_store(key, depth),
            Node::Leaf256(n) => n.can_store(key, depth),
        }
    }

    fn transform(&self, key: &Key, depth: u8) -> Result<Self::Union, Error> {
        match self {
            Node::Empty(n) => n.transform(key, depth),
            Node::Inner(n) => n.transform(key, depth),
            Node::Leaf2(n) => n.transform(key, depth),
            Node::Leaf256(n) => n.transform(key, depth),
        }
    }

    fn reparent(&self, key: &Key, depth: u8, self_id: Self::Id) -> Result<Self::Union, Error> {
        match self {
            Node::Empty(n) => n.reparent(key, depth, self_id),
            Node::Inner(n) => n.reparent(key, depth, self_id),
            Node::Leaf2(n) => n.reparent(key, depth, self_id),
            Node::Leaf256(n) => n.reparent(key, depth, self_id),
        }
    }

    fn replace_child(&mut self, key: &Key, depth: u8, new: NodeId) -> Result<(), Error> {
        match self {
            Node::Empty(n) => n.replace_child(key, depth, new),
            Node::Inner(n) => n.replace_child(key, depth, new),
            Node::Leaf2(n) => n.replace_child(key, depth, new),
            Node::Leaf256(n) => n.replace_child(key, depth, new),
        }
    }

    fn store(&mut self, key: &Key, value: &Value) -> Result<Value, Error> {
        match self {
            Node::Empty(n) => n.store(key, value),
            Node::Inner(n) => n.store(key, value),
            Node::Leaf2(n) => n.store(key, value),
            Node::Leaf256(n) => n.store(key, value),
        }
    }

    fn get_cached_commitment(&self) -> CachedCommitment<Self::Commitment> {
        match self {
            Node::Empty(n) => n.get_cached_commitment(),
            Node::Inner(n) => n.get_cached_commitment(),
            Node::Leaf2(n) => n.get_cached_commitment(),
            Node::Leaf256(n) => n.get_cached_commitment(),
        }
    }

    fn set_cached_commitment(
        &mut self,
        cache: CachedCommitment<Self::Commitment>,
    ) -> Result<(), Error> {
        match self {
            Node::Empty(n) => n.set_cached_commitment(cache),
            Node::Inner(n) => n.set_cached_commitment(cache),
            Node::Leaf2(n) => n.set_cached_commitment(cache),
            Node::Leaf256(n) => n.set_cached_commitment(cache),
        }
    }

    fn get_commitment_input(&self) -> Result<Self::CommitmentInput, Error> {
        match self {
            Node::Empty(n) => n.get_commitment_input(),
            Node::Inner(n) => n.get_commitment_input(),
            Node::Leaf2(n) => n.get_commitment_input(),
            Node::Leaf256(n) => n.get_commitment_input(),
        }
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
