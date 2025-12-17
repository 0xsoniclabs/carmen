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
use zerocopy::{FromBytes, Immutable, IntoBytes, Unaligned};

use crate::{
    database::{
        managed_trie::{LookupResult, ManagedTrieNode, StoreAction, UnionManagedTrieNode},
        verkle::{
            KeyedUpdate, KeyedUpdateBatch,
            variants::managed::{
                VerkleNodeId,
                commitment::{VerkleCommitment, VerkleCommitmentInput},
                nodes::{
                    empty::EmptyNode, inner::FullInnerNode, leaf::FullLeafNode,
                    sparse_inner::SparseInnerNode, sparse_leaf::SparseLeafNode,
                },
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

#[cfg(test)]
pub use tests::{NodeAccess, VerkleManagedTrieNode};

/// A node in a managed Verkle trie.
//
/// Non-empty nodes are stored as boxed to save memory (otherwise the size of the enum would be
/// dictated by the largest variant).
#[derive(Debug, Clone, PartialEq, Eq, Deftly)]
#[derive_deftly(FileStorageManager)]
pub enum VerkleNode {
    Empty(EmptyVerkleNode),
    Inner9(Box<Inner9VerkleNode>),
    Inner15(Box<Inner15VerkleNode>),
    Inner21(Box<Inner21VerkleNode>),
    Inner256(Box<Inner256VerkleNode>),
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
type Inner256VerkleNode = FullInnerNode;
type Leaf1VerkleNode = SparseLeafNode<1>;
type Leaf2VerkleNode = SparseLeafNode<2>;
type Leaf5VerkleNode = SparseLeafNode<5>;
type Leaf18VerkleNode = SparseLeafNode<18>;
type Leaf146VerkleNode = SparseLeafNode<146>;
type Leaf256VerkleNode = FullLeafNode;

impl VerkleNode {
    /// Returns the smallest leaf node type capable of storing `n` values.
    pub fn smallest_leaf_type_for(n: usize) -> VerkleNodeKind {
        match n {
            0 => VerkleNodeKind::Empty,
            1..=1 => VerkleNodeKind::Leaf1,
            2..=2 => VerkleNodeKind::Leaf2,
            3..=5 => VerkleNodeKind::Leaf5,
            6..=18 => VerkleNodeKind::Leaf18,
            19..=146 => VerkleNodeKind::Leaf146,
            147..=256 => VerkleNodeKind::Leaf256,
            _ => panic!("no leaf type for more than 256 values"),
        }
    }

    /// Returns the smallest inner node type capable of storing `n` values.
    pub fn smallest_inner_type_for(n: usize) -> VerkleNodeKind {
        match n {
            0 => VerkleNodeKind::Empty,
            1..=9 => VerkleNodeKind::Inner9,
            10..=15 => VerkleNodeKind::Inner15,
            16..=21 => VerkleNodeKind::Inner21,
            22..=256 => VerkleNodeKind::Inner256,
            _ => panic!("no inner type for more than 256 children"),
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

    /// Converts this node to an inner node, if it is one.
    pub fn as_inner_node(&self) -> Option<&dyn VerkleManagedInnerNode> {
        match self {
            VerkleNode::Inner9(n) => Some(n.deref()),
            VerkleNode::Inner15(n) => Some(n.deref()),
            VerkleNode::Inner21(n) => Some(n.deref()),
            VerkleNode::Inner256(n) => Some(n.deref()),
            _ => None,
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
            VerkleNode::Empty(_)
            | VerkleNode::Leaf1(_)
            | VerkleNode::Leaf2(_)
            | VerkleNode::Leaf5(_)
            | VerkleNode::Leaf18(_)
            | VerkleNode::Leaf146(_)
            | VerkleNode::Leaf256(_) => {}
            inner_node => {
                let inner = inner_node.as_inner_node().ok_or(Error::CorruptedState(
                    "expected inner node in accept method. Maybe you added a new leaf variant and forgot to update the accept method".to_owned(),
                ))?;
                for child_id in inner.iter_children() {
                    let child = manager.get_read_access(child_id.item)?;
                    child.accept(visitor, manager, level + 1)?;
                }
            }
        }
        Ok(())
    }
}

impl NodeVisitor<VerkleNode> for NodeCountVisitor {
    fn visit(&mut self, node: &VerkleNode, level: u64) -> BTResult<(), Error> {
        match node {
            VerkleNode::Empty(n) => self.visit(n, level),
            VerkleNode::Inner9(n) => self.visit(n.deref(), level),
            VerkleNode::Inner15(n) => self.visit(n.deref(), level),
            VerkleNode::Inner21(n) => self.visit(n.deref(), level),
            VerkleNode::Inner256(n) => self.visit(n.deref(), level),
            VerkleNode::Leaf1(n) => self.visit(n.deref(), level),
            VerkleNode::Leaf2(n) => self.visit(n.deref(), level),
            VerkleNode::Leaf5(n) => self.visit(n.deref(), level),
            VerkleNode::Leaf18(n) => self.visit(n.deref(), level),
            VerkleNode::Leaf146(n) => self.visit(n.deref(), level),
            VerkleNode::Leaf256(n) => self.visit(n.deref(), level),
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

    fn next_store_action<'a>(
        &self,
        updates: KeyedUpdateBatch<'a>,
        depth: u8,
        self_id: Self::Id,
    ) -> BTResult<StoreAction<'a, Self::Id, Self::Union>, Error> {
        match self {
            VerkleNode::Empty(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner9(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner15(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner21(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner256(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf1(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf2(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf5(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf18(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf146(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf256(n) => n.next_store_action(updates, depth, self_id),
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

    fn store(&mut self, update: &KeyedUpdate) -> BTResult<Value, Error> {
        match self {
            VerkleNode::Empty(n) => n.store(update),
            VerkleNode::Inner9(n) => n.store(update),
            VerkleNode::Inner15(n) => n.store(update),
            VerkleNode::Inner21(n) => n.store(update),
            VerkleNode::Inner256(n) => n.store(update),
            VerkleNode::Leaf1(n) => n.store(update),
            VerkleNode::Leaf2(n) => n.store(update),
            VerkleNode::Leaf5(n) => n.store(update),
            VerkleNode::Leaf18(n) => n.store(update),
            VerkleNode::Leaf146(n) => n.store(update),
            VerkleNode::Leaf256(n) => n.store(update),
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
                std::mem::size_of::<Box<FullInnerNode>>() + std::mem::size_of::<FullInnerNode>()
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
        VerkleNodeKind::Leaf1.node_byte_size()
    }
}

/// An item (value or child ID) stored in a sparse trie node, together with its index.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned,
)]
#[repr(C)]
pub struct ItemWithIndex<T> {
    pub index: u8,
    pub item: T,
}

/// A value of a sparse leaf node in a managed Verkle trie, together with its index.
pub type ValueWithIndex = ItemWithIndex<Value>;
/// An ID in a sparse inner node, together with its index.
pub type VerkleIdWithIndex = ItemWithIndex<VerkleNodeId>;

impl<T> ItemWithIndex<T>
where
    T: Default + PartialEq,
{
    /// Returns a slot in `items` for storing an item with the given index, or `None` if no such
    /// slot exists. A slot is suitable if it either already holds the given index, or if it is
    /// empty (i.e., holds the default item).
    fn get_slot_for<const N: usize>(items: &[ItemWithIndex<T>; N], index: u8) -> Option<usize> {
        let mut empty_slot = None;
        // We always do a linear search over all items to ensure that we never hold the same index
        // twice in different slots. By starting the search at the given index we are very likely
        // to find the matching slot immediately in practice (if index < N).
        for (i, iwi) in items
            .iter()
            .enumerate()
            .cycle()
            .skip(index as usize)
            .take(N)
        {
            if iwi.index == index {
                return Some(i);
            } else if empty_slot.is_none() && iwi.item == T::default() {
                empty_slot = Some(i);
            }
        }
        empty_slot
    }

    /// Returns the number of slots that would be required to store the given items or None if they
    /// already fit.
    fn required_slot_count_for<const N: usize>(
        items: &[ItemWithIndex<T>; N],
        indices: impl Iterator<Item = u8>,
    ) -> Option<usize> {
        let empty_slots = items.iter().filter(|iwi| iwi.item == T::default()).count();
        let mut new_slots = 0;
        for index in indices {
            if items
                .iter()
                .any(|iwi| iwi.index == index && iwi.item != T::default())
            {
                continue;
            }
            new_slots += 1;
        }
        if new_slots <= empty_slots {
            None
        } else {
            Some(N - empty_slots + new_slots)
        }
    }
}

/// Creates the smallest leaf node capable of storing `n` values, initialized with the given
/// `stem`, `values` and `commitment`.
pub fn make_smallest_leaf_node_for(
    n: usize,
    stem: [u8; 31],
    values: &[ValueWithIndex],
    commitment: &VerkleCommitment,
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
                commitment: *commitment,
                ..Default::default()
            };
            for v in values {
                new_leaf.values[v.index as usize] = v.item;
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

/// Creates the smallest inner node capable of storing `n` children, initialized with the given
/// `children` and `commitment`.
pub fn make_smallest_inner_node_for(
    n: usize,
    children: &[VerkleIdWithIndex],
    commitment: &VerkleCommitment,
) -> BTResult<VerkleNode, Error> {
    match VerkleNode::smallest_inner_type_for(n) {
        VerkleNodeKind::Empty => Ok(VerkleNode::Empty(EmptyNode)),
        VerkleNodeKind::Inner9 => Ok(VerkleNode::Inner9(Box::new(
            Inner9VerkleNode::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner15 => Ok(VerkleNode::Inner15(Box::new(
            Inner15VerkleNode::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner21 => Ok(VerkleNode::Inner21(Box::new(
            Inner21VerkleNode::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner256 => {
            let mut new_inner = FullInnerNode {
                commitment: *commitment,
                ..Default::default()
            };
            for c in children {
                new_inner.children[c.index as usize] = c.item;
            }
            Ok(VerkleNode::Inner256(Box::new(new_inner)))
        }
        VerkleNodeKind::Leaf1
        | VerkleNodeKind::Leaf2
        | VerkleNodeKind::Leaf5
        | VerkleNodeKind::Leaf18
        | VerkleNodeKind::Leaf146
        | VerkleNodeKind::Leaf256 => Err(Error::CorruptedState(
            "received non-inner type in make_smallest_inner_node_for".to_owned(),
        )
        .into()),
    }
}

/// A trait to link together full and sparse inner nodes.
/// It provides a set of operations common to all inner node types.
pub trait VerkleManagedInnerNode {
    /// Returns an iterator over all children in the inner node, together with their indexes.
    fn iter_children(&self) -> Box<dyn Iterator<Item = VerkleIdWithIndex> + '_>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TreeId;

    // NOTE: Tests for the accept method are in managed/mod.rs

    #[test]
    fn node_type_byte_size_returns_correct_size() {
        let empty_node = VerkleNodeKind::Empty;
        let inner9_node = VerkleNodeKind::Inner9;
        let inner15_node = VerkleNodeKind::Inner15;
        let inner21_node = VerkleNodeKind::Inner21;
        let inner256_node = VerkleNodeKind::Inner256;
        let leaf1_node = VerkleNodeKind::Leaf1;
        let leaf2_node = VerkleNodeKind::Leaf2;
        let leaf5_node = VerkleNodeKind::Leaf5;
        let leaf18_node = VerkleNodeKind::Leaf18;
        let leaf146_node = VerkleNodeKind::Leaf146;
        let leaf256_node = VerkleNodeKind::Leaf256;

        assert_eq!(
            empty_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
        );
        assert_eq!(
            inner9_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseInnerNode<9>>>()
                + std::mem::size_of::<SparseInnerNode<9>>()
        );
        assert_eq!(
            inner15_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseInnerNode<15>>>()
                + std::mem::size_of::<SparseInnerNode<15>>()
        );
        assert_eq!(
            inner21_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseInnerNode<21>>>()
                + std::mem::size_of::<SparseInnerNode<21>>()
        );
        assert_eq!(
            inner256_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<FullInnerNode>>()
                + std::mem::size_of::<FullInnerNode>()
        );
        assert_eq!(
            leaf1_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseLeafNode<1>>>()
                + std::mem::size_of::<SparseLeafNode<1>>()
        );
        assert_eq!(
            leaf2_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseLeafNode<2>>>()
                + std::mem::size_of::<SparseLeafNode<2>>()
        );
        assert_eq!(
            leaf5_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseLeafNode<5>>>()
                + std::mem::size_of::<SparseLeafNode<5>>()
        );
        assert_eq!(
            leaf18_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseLeafNode<18>>>()
                + std::mem::size_of::<SparseLeafNode<18>>()
        );
        assert_eq!(
            leaf146_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseLeafNode<146>>>()
                + std::mem::size_of::<SparseLeafNode<146>>()
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
            VerkleNode::Leaf1(Box::default()).node_byte_size()
        );
    }

    #[test]
    fn node_byte_size_returns_node_type_byte_size() {
        let empty_node = VerkleNode::Empty(EmptyNode);
        let inner9_node = VerkleNode::Inner9(Box::default());
        let inner15_node = VerkleNode::Inner15(Box::default());
        let inner21_node = VerkleNode::Inner21(Box::default());
        let inner256_node = VerkleNode::Inner256(Box::default());
        let leaf1_node = VerkleNode::Leaf1(Box::default());
        let leaf2_node = VerkleNode::Leaf2(Box::default());
        let leaf5_node = VerkleNode::Leaf5(Box::default());
        let leaf18_node = VerkleNode::Leaf18(Box::default());
        let leaf146_node = VerkleNode::Leaf146(Box::default());
        let leaf256_node = VerkleNode::Leaf256(Box::default());

        assert_eq!(
            VerkleNodeKind::Empty.node_byte_size(),
            empty_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Inner9.node_byte_size(),
            inner9_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Inner15.node_byte_size(),
            inner15_node.node_byte_size()
        );

        assert_eq!(
            VerkleNodeKind::Inner21.node_byte_size(),
            inner21_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Inner256.node_byte_size(),
            inner256_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Leaf1.node_byte_size(),
            leaf1_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Leaf2.node_byte_size(),
            leaf2_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Leaf5.node_byte_size(),
            leaf5_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Leaf18.node_byte_size(),
            leaf18_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Leaf146.node_byte_size(),
            leaf146_node.node_byte_size()
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

        let mut node = FullInnerNode::default();
        for i in 0..256 {
            node.children[i] = VerkleNodeId::from_idx_and_node_kind(1, VerkleNodeKind::Inner256);
        }
        assert!(visitor.visit(&node, level + 1).is_ok());

        let mut node = Leaf2VerkleNode::default();
        for i in 0..2 {
            node.values[i] = ValueWithIndex {
                index: i as u8,
                item: [1; 32],
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

    #[test]
    fn item_with_index_get_slot_returns_slot_with_matching_index_or_empty_slot() {
        type TestItemWithIndex = ItemWithIndex<u8>;
        let mut values = [TestItemWithIndex::default(); 4];
        values[0] = TestItemWithIndex { index: 0, item: 10 };
        values[3] = TestItemWithIndex { index: 5, item: 20 };

        // Matching index
        let slot = TestItemWithIndex::get_slot_for(&values, 0);
        assert_eq!(slot, Some(0));

        // Matching index has precedence over empty slot
        let slot = TestItemWithIndex::get_slot_for(&values, 5);
        assert_eq!(slot, Some(3));

        // No matching index, so we return first empty slot
        let slot = TestItemWithIndex::get_slot_for(&values, 8); // 8 % 4 = 0, so start start search at 0
        assert_eq!(slot, Some(1));

        // No matching index and no empty slot
        values[1] = TestItemWithIndex { index: 1, item: 30 };
        values[2] = TestItemWithIndex { index: 2, item: 40 };
        let slot = TestItemWithIndex::get_slot_for(&values, 250);
        assert_eq!(slot, None);
    }

    #[test]
    fn item_with_index_required_slot_count_for_returns_number_of_required_slots_or_none_if_items_fit()
     {
        let mut items = [ItemWithIndex::default(); 5];
        items[1] = ItemWithIndex {
            index: 1,
            item: [1; 32],
        };
        items[2] = ItemWithIndex {
            index: 10,
            item: [2; 32],
        };
        items[3] = ItemWithIndex {
            index: 100,
            item: Value::default(),
        };
        // `items` now has 2 occupied slots (for indices 1 and 10) and 3 empty slots

        // Enough empty slots for all new indices
        let slots = ItemWithIndex::required_slot_count_for(&items, [100, 101, 102].into_iter());
        assert_eq!(slots, None);

        // Enough empty slots and slots which get overwritten
        let slots =
            ItemWithIndex::required_slot_count_for(&items, [100, 101, 102, 10, 1].into_iter());
        assert_eq!(slots, None);

        // Not enough empty slots
        let slots =
            ItemWithIndex::required_slot_count_for(&items, [100, 101, 102, 103].into_iter());
        assert_eq!(slots, Some(6)); // 2 existing + 1 reused + 3 new
    }

    /// A supertrait combining [`ManagedTrieNode`] and [`NodeHelperTrait`] for use in rstest tests.
    pub trait VerkleManagedTrieNode<T>:
        ManagedTrieNode<Union = VerkleNode, Id = VerkleNodeId, Commitment = VerkleCommitment>
        + NodeAccess<T>
    where
        T: Clone + Copy + Default + PartialEq + Eq + FromBytes + IntoBytes + Immutable + Unaligned,
    {
    }

    impl<const N: usize> VerkleManagedTrieNode<VerkleNodeId> for SparseInnerNode<N> {}

    /// Helper trait to interact with generic node types in rstest tests.
    pub trait NodeAccess<T>
    where
        T: Clone + Copy + Default + PartialEq + Eq + FromBytes + IntoBytes + Immutable + Unaligned,
    {
        fn access_slot(&mut self, slot: usize) -> &mut ItemWithIndex<T>;

        fn get_commitment_input(&self) -> VerkleCommitmentInput;
    }
}
