// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::ops::Deref;

use derive_deftly::Deftly;

use crate::{
    database::{
        managed_trie::{LookupResult, ManagedTrieNode, StoreAction, UnionManagedTrieNode},
        verkle::variants::managed::{
            VerkleNodeId,
            commitment::{VerkleCommitment, VerkleCommitmentInput},
            nodes::{
                empty::EmptyNode,
                inner::InnerNode,
                leaf::FullLeafNode,
                sparse_inner::{IdWithIndex, SparseInnerNode},
                sparse_leaf::{SparseLeafNode, ValueWithIndex},
            },
        },
        visitor::NodeVisitor,
    },
    error::{BTResult, Error},
    node_manager::NodeManager,
    statistics::node_count::NodeCountVisitor,
    storage::file::derive_deftly_template_FileStorageManager,
    types::{HasEmptyNode, Key, NodeSize, ToNodeKind, Value},
};

pub mod empty;
pub mod id;
pub mod inner;
pub mod leaf;
pub mod sparse_inner;
pub mod sparse_leaf;

/// A node in a managed Verkle trie.
//
/// Non-empty nodes are stored as boxed to save memory (otherwise the size of the enum would be
/// dictated by the largest variant).
#[derive(Debug, Clone, PartialEq, Eq, Deftly)]
#[derive_deftly(FileStorageManager)]
pub enum VerkleNode {
    Empty(EmptyNode),
    Inner9(Box<Inner9VerkleNode>),
    Inner15(Box<Inner15VerkleNode>),
    Inner21(Box<Inner21VerkleNode>),
    Inner256(Box<Inner256VerkleNode>),
    // Make sure to adjust smallest_inner_type_for when adding new inner types.
    Leaf1(Box<Leaf1VerkleNode>),
    Leaf2(Box<Leaf2VerkleNode>),
    Leaf5(Box<Leaf5VerkleNode>),
    Leaf18(Box<Leaf18VerkleNode>),
    Leaf146(Box<Leaf146VerkleNode>),
    Leaf256(Box<Leaf256VerkleNode>),
    // Make sure to adjust smallest_leaf_type_for when adding new leaf types.
}

type EmptyVerkleNode = EmptyNode;
type Inner9VerkleNode = SparseInnerNode<9>;
type Inner15VerkleNode = SparseInnerNode<15>;
type Inner21VerkleNode = SparseInnerNode<21>;
type Inner256VerkleNode = InnerNode; // TODO Rename to FullInnerNode
type Leaf1VerkleNode = SparseLeafNode<1>;
type Leaf2VerkleNode = SparseLeafNode<2>;
type Leaf5VerkleNode = SparseLeafNode<5>;
type Leaf18VerkleNode = SparseLeafNode<18>;
type Leaf146VerkleNode = SparseLeafNode<146>;
type Leaf256VerkleNode = FullLeafNode;

impl VerkleNode {
    pub fn smallest_inner_type_for(n: usize) -> VerkleNodeKind {
        match n {
            0..=9 => VerkleNodeKind::Inner9,
            10..=15 => VerkleNodeKind::Inner15,
            16..=21 => VerkleNodeKind::Inner21,
            22..=256 => VerkleNodeKind::Inner256,
            _ => panic!("no inner type for more than 256 children"),
        }
    }

    /// Returns the smallest leaf node type capable of storing `n` values.
    pub fn smallest_leaf_type_for(n: usize) -> VerkleNodeKind {
        match n {
            0..=1 => VerkleNodeKind::Leaf1,
            2..=2 => VerkleNodeKind::Leaf2,
            3..=5 => VerkleNodeKind::Leaf5,
            6..=18 => VerkleNodeKind::Leaf18,
            19..=146 => VerkleNodeKind::Leaf146,
            147..=256 => VerkleNodeKind::Leaf256,
            _ => panic!("no leaf type for more than 256 values"),
        }
    }

    /// Returns the commitment input for computing the commitment of this node.
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
        match self {
            VerkleNode::Empty(n) => n.get_commitment_input(),
            VerkleNode::Inner9(n) => n.get_commitment_input(),
            VerkleNode::Inner15(n) => n.get_commitment_input(),
            VerkleNode::Inner21(n) => n.get_commitment_input(),
            VerkleNode::Inner256(n) => n.get_commitment_input(),
            VerkleNode::Leaf1(n) => n.get_commitment_input(),
            VerkleNode::Leaf2(n) => n.get_commitment_input(),
            VerkleNode::Leaf5(n) => n.get_commitment_input(),
            VerkleNode::Leaf18(n) => n.get_commitment_input(),
            VerkleNode::Leaf146(n) => n.get_commitment_input(),
            VerkleNode::Leaf256(n) => n.get_commitment_input(),
        }
    }

    /// Accepts a visitor for recursively traversing the node and its children.
    pub fn accept(
        &self,
        visitor: &mut impl NodeVisitor<Self>,
        manager: &impl NodeManager<Id = VerkleNodeId, Node = VerkleNode>,
        level: u64,
    ) -> BTResult<(), Error> {
        visitor.visit(self, level)?;
        match self {
            VerkleNode::Empty(_) | VerkleNode::Leaf2(_) | VerkleNode::Leaf256(_) => {}
            VerkleNode::Inner256(inner) => {
                for child_id in inner.children.iter() {
                    let child = manager.get_read_access(*child_id)?;
                    child.accept(visitor, manager, level + 1)?;
                }
            }
            _ => unimplemented!(),
        }
        Ok(())
    }
}

impl NodeVisitor<VerkleNode> for NodeCountVisitor {
    fn visit(&mut self, node: &VerkleNode, level: u64) -> BTResult<(), Error> {
        match node {
            VerkleNode::Empty(n) => self.visit(n, level),
            VerkleNode::Inner256(n) => self.visit(n.deref(), level),
            VerkleNode::Leaf2(n) => self.visit(n.deref(), level),
            VerkleNode::Leaf256(n) => self.visit(n.deref(), level),
            _ => unimplemented!(),
        }
    }
}

impl ToNodeKind for VerkleNode {
    type Target = VerkleNodeKind;

    /// Converts the ID to its corresponding node kind. This conversion will always succeed.
    fn to_node_kind(&self) -> Option<Self::Target> {
        match self {
            VerkleNode::Empty(_) => Some(VerkleNodeKind::Empty),
            VerkleNode::Inner9(_) => Some(VerkleNodeKind::Inner9),
            VerkleNode::Inner15(_) => Some(VerkleNodeKind::Inner15),
            VerkleNode::Inner21(_) => Some(VerkleNodeKind::Inner21),
            VerkleNode::Inner256(_) => Some(VerkleNodeKind::Inner256),
            VerkleNode::Leaf1(_) => Some(VerkleNodeKind::Leaf1),
            VerkleNode::Leaf2(_) => Some(VerkleNodeKind::Leaf2),
            VerkleNode::Leaf5(_) => Some(VerkleNodeKind::Leaf5),
            VerkleNode::Leaf18(_) => Some(VerkleNodeKind::Leaf18),
            VerkleNode::Leaf146(_) => Some(VerkleNodeKind::Leaf146),
            VerkleNode::Leaf256(_) => Some(VerkleNodeKind::Leaf256),
        }
    }
}

impl NodeSize for VerkleNode {
    fn node_byte_size(&self) -> usize {
        self.to_node_kind().unwrap().node_byte_size()
    }

    fn min_non_empty_node_size() -> usize {
        VerkleNodeKind::min_non_empty_node_size()
    }
}

impl HasEmptyNode for VerkleNode {
    fn is_empty_node(&self) -> bool {
        matches!(self, VerkleNode::Empty(_))
    }

    fn empty_node() -> Self {
        VerkleNode::Empty(EmptyNode)
    }
}

impl Default for VerkleNode {
    fn default() -> Self {
        VerkleNode::Empty(EmptyNode)
    }
}

impl UnionManagedTrieNode for VerkleNode {}

impl ManagedTrieNode for VerkleNode {
    type Union = VerkleNode;
    type Id = VerkleNodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, key: &Key, depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {
        match self {
            VerkleNode::Empty(n) => n.lookup(key, depth),
            VerkleNode::Inner9(n) => n.lookup(key, depth),
            VerkleNode::Inner15(n) => n.lookup(key, depth),
            VerkleNode::Inner21(n) => n.lookup(key, depth),
            VerkleNode::Inner256(n) => n.lookup(key, depth),
            VerkleNode::Leaf1(n) => n.lookup(key, depth),
            VerkleNode::Leaf2(n) => n.lookup(key, depth),
            VerkleNode::Leaf5(n) => n.lookup(key, depth),
            VerkleNode::Leaf18(n) => n.lookup(key, depth),
            VerkleNode::Leaf146(n) => n.lookup(key, depth),
            VerkleNode::Leaf256(n) => n.lookup(key, depth),
        }
    }

    fn next_store_action(
        &self,
        key: &Key,
        depth: u8,
        self_id: Self::Id,
    ) -> BTResult<StoreAction<Self::Id, Self::Union>, Error> {
        match self {
            VerkleNode::Empty(n) => n.next_store_action(key, depth, self_id),
            VerkleNode::Inner9(n) => n.next_store_action(key, depth, self_id),
            VerkleNode::Inner15(n) => n.next_store_action(key, depth, self_id),
            VerkleNode::Inner21(n) => n.next_store_action(key, depth, self_id),
            VerkleNode::Inner256(n) => n.next_store_action(key, depth, self_id),
            VerkleNode::Leaf1(n) => n.next_store_action(key, depth, self_id),
            VerkleNode::Leaf2(n) => n.next_store_action(key, depth, self_id),
            VerkleNode::Leaf5(n) => n.next_store_action(key, depth, self_id),
            VerkleNode::Leaf18(n) => n.next_store_action(key, depth, self_id),
            VerkleNode::Leaf146(n) => n.next_store_action(key, depth, self_id),
            VerkleNode::Leaf256(n) => n.next_store_action(key, depth, self_id),
        }
    }

    fn replace_child(&mut self, key: &Key, depth: u8, new: VerkleNodeId) -> BTResult<(), Error> {
        match self {
            VerkleNode::Empty(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner9(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner15(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner21(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner256(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf1(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf2(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf5(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf18(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf146(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf256(n) => n.replace_child(key, depth, new),
        }
    }

    fn store(&mut self, key: &Key, value: &Value) -> BTResult<Value, Error> {
        match self {
            VerkleNode::Empty(n) => n.store(key, value),
            VerkleNode::Inner9(n) => n.store(key, value),
            VerkleNode::Inner15(n) => n.store(key, value),
            VerkleNode::Inner21(n) => n.store(key, value),
            VerkleNode::Inner256(n) => n.store(key, value),
            VerkleNode::Leaf1(n) => n.store(key, value),
            VerkleNode::Leaf2(n) => n.store(key, value),
            VerkleNode::Leaf5(n) => n.store(key, value),
            VerkleNode::Leaf18(n) => n.store(key, value),
            VerkleNode::Leaf146(n) => n.store(key, value),
            VerkleNode::Leaf256(n) => n.store(key, value),
        }
    }

    fn get_commitment(&self) -> Self::Commitment {
        match self {
            VerkleNode::Empty(n) => n.get_commitment(),
            VerkleNode::Inner9(n) => n.get_commitment(),
            VerkleNode::Inner15(n) => n.get_commitment(),
            VerkleNode::Inner21(n) => n.get_commitment(),
            VerkleNode::Inner256(n) => n.get_commitment(),
            VerkleNode::Leaf1(n) => n.get_commitment(),
            VerkleNode::Leaf2(n) => n.get_commitment(),
            VerkleNode::Leaf5(n) => n.get_commitment(),
            VerkleNode::Leaf18(n) => n.get_commitment(),
            VerkleNode::Leaf146(n) => n.get_commitment(),
            VerkleNode::Leaf256(n) => n.get_commitment(),
        }
    }

    fn set_commitment(&mut self, cache: Self::Commitment) -> BTResult<(), Error> {
        match self {
            VerkleNode::Empty(n) => n.set_commitment(cache),
            VerkleNode::Inner9(n) => n.set_commitment(cache),
            VerkleNode::Inner15(n) => n.set_commitment(cache),
            VerkleNode::Inner21(n) => n.set_commitment(cache),
            VerkleNode::Inner256(n) => n.set_commitment(cache),
            VerkleNode::Leaf1(n) => n.set_commitment(cache),
            VerkleNode::Leaf2(n) => n.set_commitment(cache),
            VerkleNode::Leaf5(n) => n.set_commitment(cache),
            VerkleNode::Leaf18(n) => n.set_commitment(cache),
            VerkleNode::Leaf146(n) => n.set_commitment(cache),
            VerkleNode::Leaf256(n) => n.set_commitment(cache),
        }
    }
}

/// A node type of a node in a managed Verkle trie.
/// This type is primarily used for conversion between [`VerkleNode`] and indexes in the file
/// storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VerkleNodeKind {
    Empty,
    Inner9,
    Inner15,
    Inner21,
    Inner256,
    Leaf1,
    Leaf2,
    Leaf5,
    Leaf18,
    Leaf146,
    Leaf256,
}

impl NodeSize for VerkleNodeKind {
    fn node_byte_size(&self) -> usize {
        let inner_size = match self {
            VerkleNodeKind::Empty => 0,
            VerkleNodeKind::Inner9 => {
                std::mem::size_of::<Box<SparseInnerNode<9>>>()
                    + std::mem::size_of::<SparseInnerNode<9>>()
            }
            VerkleNodeKind::Inner15 => {
                std::mem::size_of::<Box<SparseInnerNode<15>>>()
                    + std::mem::size_of::<SparseInnerNode<15>>()
            }
            VerkleNodeKind::Inner21 => {
                std::mem::size_of::<Box<SparseInnerNode<21>>>()
                    + std::mem::size_of::<SparseInnerNode<21>>()
            }
            VerkleNodeKind::Inner256 => {
                std::mem::size_of::<Box<InnerNode>>() + std::mem::size_of::<InnerNode>()
            }
            VerkleNodeKind::Leaf1 => {
                std::mem::size_of::<Box<SparseLeafNode<1>>>()
                    + std::mem::size_of::<SparseLeafNode<1>>()
            }
            VerkleNodeKind::Leaf2 => {
                std::mem::size_of::<Box<SparseLeafNode<2>>>()
                    + std::mem::size_of::<SparseLeafNode<2>>()
            }
            VerkleNodeKind::Leaf5 => {
                std::mem::size_of::<Box<SparseLeafNode<5>>>()
                    + std::mem::size_of::<SparseLeafNode<5>>()
            }
            VerkleNodeKind::Leaf18 => {
                std::mem::size_of::<Box<SparseLeafNode<18>>>()
                    + std::mem::size_of::<SparseLeafNode<18>>()
            }
            VerkleNodeKind::Leaf146 => {
                std::mem::size_of::<Box<SparseLeafNode<146>>>()
                    + std::mem::size_of::<SparseLeafNode<146>>()
            }
            VerkleNodeKind::Leaf256 => {
                std::mem::size_of::<Box<FullLeafNode>>() + std::mem::size_of::<FullLeafNode>()
            }
        };
        std::mem::size_of::<VerkleNode>() + inner_size
    }

    fn min_non_empty_node_size() -> usize {
        // Because we don't store empty nodes, the minimum size is the smallest non-empty node.
        VerkleNodeKind::Leaf2.node_byte_size()
    }
}

pub fn make_smallest_inner_node_for(
    n: usize,
    children: &[IdWithIndex],
    commitment: VerkleCommitment,
) -> BTResult<VerkleNode, Error> {
    match VerkleNode::smallest_inner_type_for(n) {
        VerkleNodeKind::Inner9 => Ok(VerkleNode::Inner9(Box::new(
            SparseInnerNode::<9>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner15 => Ok(VerkleNode::Inner15(Box::new(
            SparseInnerNode::<15>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner21 => Ok(VerkleNode::Inner21(Box::new(
            SparseInnerNode::<21>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner256 => Ok(VerkleNode::Inner256(Box::new(InnerNode {
            children: {
                let mut arr = [VerkleNodeId::default(); 256];
                for IdWithIndex { index, id } in children {
                    arr[*index as usize] = *id;
                }
                arr
            },
            commitment,
        }))),
        _ => panic!("received non-inner type in make_smallest_inner_node_for"),
    }
}

/// Creates the smallest leaf node capable of storing `n` values, initialized with the given
/// `stem`, `values` and `commitment`.
#[allow(clippy::large_types_passed_by_value)] // Needs to be copied anyway
pub fn make_smallest_leaf_node_for(
    n: usize,
    stem: [u8; 31],
    values: &[ValueWithIndex],
    commitment: VerkleCommitment,
) -> BTResult<VerkleNode, Error> {
    match VerkleNode::smallest_leaf_type_for(n) {
        VerkleNodeKind::Empty => Ok(VerkleNode::Empty(EmptyNode)),
        VerkleNodeKind::Leaf1 => Ok(VerkleNode::Leaf1(Box::new(
            SparseLeafNode::<1>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf2 => Ok(VerkleNode::Leaf2(Box::new(
            SparseLeafNode::<2>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf5 => Ok(VerkleNode::Leaf5(Box::new(
            SparseLeafNode::<5>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf18 => Ok(VerkleNode::Leaf18(Box::new(
            SparseLeafNode::<18>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf146 => Ok(VerkleNode::Leaf146(Box::new(
            SparseLeafNode::<146>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf256 => {
            let mut new_leaf = FullLeafNode {
                stem,
                commitment,
                ..Default::default()
            };
            for v in values {
                new_leaf.values[v.index as usize] = v.value;
            }
            Ok(VerkleNode::Leaf256(Box::new(new_leaf)))
        }
        VerkleNodeKind::Inner9
        | VerkleNodeKind::Inner15
        | VerkleNodeKind::Inner21
        | VerkleNodeKind::Inner256 => Err(Error::CorruptedState(
            "received non-leaf type in make_smallest_leaf_node_for".to_owned(),
        )
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TreeId;

    // NOTE: Tests for the accept method are in managed/mod.rs

    #[test]
    fn node_type_byte_size_returns_correct_size() {
        let empty_node = VerkleNodeKind::Empty;
        let inner_node = VerkleNodeKind::Inner256;
        let leaf2_node = VerkleNodeKind::Leaf2;
        let leaf256_node = VerkleNodeKind::Leaf256;

        assert_eq!(
            empty_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
        );
        assert_eq!(
            inner_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<InnerNode>>()
                + std::mem::size_of::<InnerNode>()
        );
        assert_eq!(
            leaf2_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseLeafNode<2>>>()
                + std::mem::size_of::<SparseLeafNode<2>>()
        );
        assert_eq!(
            leaf256_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<FullLeafNode>>()
                + std::mem::size_of::<FullLeafNode>()
        );
    }

    #[test]
    fn node_type_min_non_empty_node_size_returns_size_of_smallest_non_empty_node() {
        assert_eq!(
            VerkleNodeKind::min_non_empty_node_size(),
            VerkleNode::Leaf2(Box::default()).node_byte_size()
        );
    }

    #[test]
    fn node_byte_size_returns_node_type_byte_size() {
        let empty_node = VerkleNode::Empty(EmptyNode);
        let inner_node = VerkleNode::Inner256(Box::default());
        let leaf2_node = VerkleNode::Leaf2(Box::default());
        let leaf256_node = VerkleNode::Leaf256(Box::default());

        assert_eq!(
            VerkleNodeKind::Empty.node_byte_size(),
            empty_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Inner256.node_byte_size(),
            inner_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Leaf2.node_byte_size(),
            leaf2_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Leaf256.node_byte_size(),
            leaf256_node.node_byte_size()
        );
    }

    #[test]
    fn node_min_non_empty_node_size_returns_node_type_min_size() {
        assert_eq!(
            VerkleNodeKind::min_non_empty_node_size(),
            VerkleNode::min_non_empty_node_size()
        );
    }

    #[test]
    fn node_count_visitor_visit_visit_nodes() {
        let mut visitor = NodeCountVisitor::default();
        let level = 0;

        let node = VerkleNode::Empty(EmptyNode);
        assert!(visitor.visit(&node, level).is_ok());

        let mut node = Inner256VerkleNode::default();
        for i in 0..256 {
            node.children[i] = VerkleNodeId::from_idx_and_node_kind(1, VerkleNodeKind::Inner256);
        }
        assert!(visitor.visit(&node, level + 1).is_ok());

        let mut node = Leaf2VerkleNode::default();
        for i in 0..2 {
            node.values[i] = ValueWithIndex {
                index: i as u8,
                value: [1; 32],
            };
        }
        let node = VerkleNode::Leaf2(Box::new(node));
        assert!(visitor.visit(&node, level + 2).is_ok());

        let mut node = Leaf256VerkleNode::default();
        for i in 0..256 {
            node.values[i] = [1; 32];
        }
        let node = VerkleNode::Leaf256(Box::new(node));
        assert!(visitor.visit(&node, level + 3).is_ok());

        assert_eq!(visitor.node_count.levels_count.len(), 4);
        assert_eq!(
            visitor.node_count.levels_count[0]
                .get("Empty")
                .unwrap()
                .size_count
                .get(&0),
            Some(&1)
        );
        assert_eq!(
            visitor.node_count.levels_count[1]
                .get("Inner")
                .unwrap()
                .size_count
                .get(&256),
            Some(&1)
        );
        assert_eq!(
            visitor.node_count.levels_count[2]
                .get("Leaf")
                .unwrap()
                .size_count
                .get(&2),
            Some(&1)
        );
        assert_eq!(
            visitor.node_count.levels_count[3]
                .get("Leaf")
                .unwrap()
                .size_count
                .get(&256),
            Some(&1)
        );
    }
}
