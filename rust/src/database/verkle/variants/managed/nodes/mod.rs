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
                    empty::EmptyNode, inner::FullInnerNode, inner_delta::InnerDeltaNode,
                    leaf::FullLeafNode, sparse_inner::SparseInnerNode, sparse_leaf::SparseLeafNode,
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
pub mod inner_delta;
pub mod leaf;
pub mod sparse_inner;
pub mod sparse_leaf;

#[cfg(test)]
use tests::VerkleManagedTrieNode;

/// A node in a managed Verkle trie.
//
/// Non-empty nodes are stored as boxed to save memory (otherwise the size of the enum would be
/// dictated by the largest variant).
#[derive(Debug, Clone, PartialEq, Eq, Deftly)]
#[derive_deftly(FileStorageManager)]
pub enum VerkleNode {
    Empty(EmptyVerkleNode),
    Inner3(Box<Inner3VerkleNode>),
    Inner47(Box<Inner47VerkleNode>),
    Inner256(Box<Inner256VerkleNode>),
    InnerDelta(Box<InnerDeltaVerkleNode>),
    Leaf1(Box<Leaf1VerkleNode>),
    Leaf2(Box<Leaf2VerkleNode>),
    Leaf21(Box<Leaf21VerkleNode>),
    Leaf64(Box<Leaf64VerkleNode>),
    Leaf141(Box<Leaf141VerkleNode>),
    Leaf256(Box<Leaf256VerkleNode>),
    // Make sure to adjust smallest_leaf_type_for when adding new leaf types.
}

type EmptyVerkleNode = EmptyNode;
type Inner3VerkleNode = SparseInnerNode<3>;
type Inner47VerkleNode = SparseInnerNode<47>;
type Inner256VerkleNode = FullInnerNode;
type InnerDeltaVerkleNode = InnerDeltaNode;
type Leaf1VerkleNode = SparseLeafNode<1>;
type Leaf2VerkleNode = SparseLeafNode<2>;
type Leaf21VerkleNode = SparseLeafNode<21>;
type Leaf64VerkleNode = SparseLeafNode<64>;
type Leaf141VerkleNode = SparseLeafNode<141>;
type Leaf256VerkleNode = FullLeafNode;

impl VerkleNode {
    /// Returns the smallest leaf node type capable of storing `n` values.
    pub fn smallest_leaf_type_for(n: usize) -> VerkleNodeKind {
        match n {
            0 => VerkleNodeKind::Empty,
            1..=1 => VerkleNodeKind::Leaf1,
            2..=2 => VerkleNodeKind::Leaf2,
            3..=21 => VerkleNodeKind::Leaf21,
            22..=64 => VerkleNodeKind::Leaf64,
            65..=141 => VerkleNodeKind::Leaf141,
            142..=256 => VerkleNodeKind::Leaf256,
            _ => panic!("no leaf type for more than 256 values"),
        }
    }

    /// Returns the smallest inner node type capable of storing `n` values.
    pub fn smallest_inner_type_for(n: usize) -> VerkleNodeKind {
        match n {
            0 => VerkleNodeKind::Empty,
            1..=3 => VerkleNodeKind::Inner3,
            4..=47 => VerkleNodeKind::Inner47,
            48..=256 => VerkleNodeKind::Inner256,
            _ => panic!("no inner type for more than 256 children"),
        }
    }

    /// Returns the commitment input for computing the commitment of this node.
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
        match self {
            VerkleNode::Empty(n) => n.get_commitment_input(),
            VerkleNode::Inner3(n) => n.get_commitment_input(),
            VerkleNode::Inner47(n) => n.get_commitment_input(),
            VerkleNode::Inner256(n) => n.get_commitment_input(),
            VerkleNode::InnerDelta(n) => n.get_commitment_input(),
            VerkleNode::Leaf1(n) => n.get_commitment_input(),
            VerkleNode::Leaf2(n) => n.get_commitment_input(),
            VerkleNode::Leaf21(n) => n.get_commitment_input(),
            VerkleNode::Leaf64(n) => n.get_commitment_input(),
            VerkleNode::Leaf141(n) => n.get_commitment_input(),
            VerkleNode::Leaf256(n) => n.get_commitment_input(),
        }
    }

    /// Converts this node to an inner node, if it is one.
    pub fn as_inner_node(&self) -> Option<&dyn ManagedInnerNode> {
        match self {
            VerkleNode::Inner3(n) => Some(n.deref()),
            VerkleNode::Inner47(n) => Some(n.deref()),
            VerkleNode::Inner256(n) => Some(n.deref()),
            VerkleNode::InnerDelta(n) => Some(n.deref()),
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
            | VerkleNode::Leaf21(_)
            | VerkleNode::Leaf64(_)
            | VerkleNode::Leaf141(_)
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
            _ => unimplemented!(),
        }
        Ok(())
    }
}

impl NodeVisitor<VerkleNode> for NodeCountVisitor {
    fn visit(&mut self, node: &VerkleNode, level: u64) -> BTResult<(), Error> {
        match node {
            VerkleNode::Empty(n) => self.visit(n, level),
            VerkleNode::Inner3(n) => self.visit(n.deref(), level),
            VerkleNode::Inner47(n) => self.visit(n.deref(), level),
            VerkleNode::Inner256(n) => self.visit(n.deref(), level),
            VerkleNode::InnerDelta(n) => self.visit(n.deref(), level),
            VerkleNode::Leaf1(n) => self.visit(n.deref(), level),
            VerkleNode::Leaf2(n) => self.visit(n.deref(), level),
            VerkleNode::Leaf21(n) => self.visit(n.deref(), level),
            VerkleNode::Leaf64(n) => self.visit(n.deref(), level),
            VerkleNode::Leaf141(n) => self.visit(n.deref(), level),
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
            VerkleNode::Inner3(_) => Some(VerkleNodeKind::Inner3),
            VerkleNode::Inner47(_) => Some(VerkleNodeKind::Inner47),
            VerkleNode::Inner256(_) => Some(VerkleNodeKind::Inner256),
            VerkleNode::InnerDelta(_) => Some(VerkleNodeKind::InnerDelta),
            VerkleNode::Leaf1(_) => Some(VerkleNodeKind::Leaf1),
            VerkleNode::Leaf2(_) => Some(VerkleNodeKind::Leaf2),
            VerkleNode::Leaf21(_) => Some(VerkleNodeKind::Leaf21),
            VerkleNode::Leaf64(_) => Some(VerkleNodeKind::Leaf64),
            VerkleNode::Leaf141(_) => Some(VerkleNodeKind::Leaf141),
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

impl UnionManagedTrieNode for VerkleNode {
    fn copy_on_write(&self, id: Self::Id, changed_children: Vec<u8>) -> Self {
        match self {
            VerkleNode::Empty(n) => VerkleNode::Empty(*n),
            VerkleNode::Inner3(n) => VerkleNode::Inner3(n.clone()),
            VerkleNode::Inner47(n) => VerkleNode::Inner47(n.clone()),
            VerkleNode::Inner256(n) => {
                if changed_children.len() <= InnerDeltaNode::DELTA_SIZE {
                    println!("full to delta: {}", changed_children.len());
                    VerkleNode::InnerDelta(Box::new(InnerDeltaNode::from_full_inner(n, id)))
                } else {
                    println!("full to full: {}", changed_children.len());
                    VerkleNode::Inner256(n.clone())
                }
            }
            VerkleNode::InnerDelta(n) => {
                let s = ItemWithIndex::get_slots_for(
                    &n.children_delta,
                    changed_children.clone().into_iter(),
                );
                if ItemWithIndex::get_slots_for(
                    &n.children_delta,
                    changed_children.clone().into_iter(),
                )
                .is_some()
                {
                    println!("delta to full: {}, {:?}", changed_children.len(), s);
                    VerkleNode::Inner256(Box::new(FullInnerNode::from(&**n)))
                } else {
                    println!(
                        "delta to delta but inner: {} {:?}",
                        changed_children.len(),
                        s
                    );
                    VerkleNode::Inner256(Box::new(FullInnerNode::from(&**n)))
                    //VerkleNode::InnerDelta(n.clone())
                }
            }
            VerkleNode::Leaf1(n) => VerkleNode::Leaf1(n.clone()),
            VerkleNode::Leaf2(n) => VerkleNode::Leaf2(n.clone()),
            VerkleNode::Leaf21(n) => VerkleNode::Leaf21(n.clone()),
            VerkleNode::Leaf64(n) => VerkleNode::Leaf64(n.clone()),
            VerkleNode::Leaf141(n) => VerkleNode::Leaf141(n.clone()),
            VerkleNode::Leaf256(n) => VerkleNode::Leaf256(n.clone()),
        }
    }
}

impl ManagedTrieNode for VerkleNode {
    type Union = VerkleNode;
    type Id = VerkleNodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, key: &Key, depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {
        match self {
            VerkleNode::Empty(n) => n.lookup(key, depth),
            VerkleNode::Inner3(n) => n.lookup(key, depth),
            VerkleNode::Inner47(n) => n.lookup(key, depth),
            VerkleNode::Inner256(n) => n.lookup(key, depth),
            VerkleNode::InnerDelta(n) => n.lookup(key, depth),
            VerkleNode::Leaf1(n) => n.lookup(key, depth),
            VerkleNode::Leaf2(n) => n.lookup(key, depth),
            VerkleNode::Leaf21(n) => n.lookup(key, depth),
            VerkleNode::Leaf64(n) => n.lookup(key, depth),
            VerkleNode::Leaf141(n) => n.lookup(key, depth),
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
            VerkleNode::Inner3(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner47(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner256(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::InnerDelta(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf1(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf2(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf21(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf64(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf141(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf256(n) => n.next_store_action(updates, depth, self_id),
        }
    }

    fn replace_child(&mut self, key: &Key, depth: u8, new: VerkleNodeId) -> BTResult<(), Error> {
        match self {
            VerkleNode::Empty(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner3(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner47(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner256(n) => n.replace_child(key, depth, new),
            VerkleNode::InnerDelta(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf1(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf2(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf21(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf64(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf141(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf256(n) => n.replace_child(key, depth, new),
        }
    }

    fn store(&mut self, update: &KeyedUpdate) -> BTResult<Value, Error> {
        match self {
            VerkleNode::Empty(n) => n.store(update),
            VerkleNode::Inner3(n) => n.store(update),
            VerkleNode::Inner47(n) => n.store(update),
            VerkleNode::Inner256(n) => n.store(update),
            VerkleNode::InnerDelta(n) => n.store(update),
            VerkleNode::Leaf1(n) => n.store(update),
            VerkleNode::Leaf2(n) => n.store(update),
            VerkleNode::Leaf21(n) => n.store(update),
            VerkleNode::Leaf64(n) => n.store(update),
            VerkleNode::Leaf141(n) => n.store(update),
            VerkleNode::Leaf256(n) => n.store(update),
        }
    }

    fn get_commitment(&self) -> Self::Commitment {
        match self {
            VerkleNode::Empty(n) => n.get_commitment(),
            VerkleNode::Inner3(n) => n.get_commitment(),
            VerkleNode::Inner47(n) => n.get_commitment(),
            VerkleNode::Inner256(n) => n.get_commitment(),
            VerkleNode::InnerDelta(n) => n.get_commitment(),
            VerkleNode::Leaf1(n) => n.get_commitment(),
            VerkleNode::Leaf2(n) => n.get_commitment(),
            VerkleNode::Leaf21(n) => n.get_commitment(),
            VerkleNode::Leaf64(n) => n.get_commitment(),
            VerkleNode::Leaf141(n) => n.get_commitment(),
            VerkleNode::Leaf256(n) => n.get_commitment(),
        }
    }

    fn set_commitment(&mut self, cache: Self::Commitment) -> BTResult<(), Error> {
        match self {
            VerkleNode::Empty(n) => n.set_commitment(cache),
            VerkleNode::Inner3(n) => n.set_commitment(cache),
            VerkleNode::Inner47(n) => n.set_commitment(cache),
            VerkleNode::Inner256(n) => n.set_commitment(cache),
            VerkleNode::InnerDelta(n) => n.set_commitment(cache),
            VerkleNode::Leaf1(n) => n.set_commitment(cache),
            VerkleNode::Leaf2(n) => n.set_commitment(cache),
            VerkleNode::Leaf21(n) => n.set_commitment(cache),
            VerkleNode::Leaf64(n) => n.set_commitment(cache),
            VerkleNode::Leaf141(n) => n.set_commitment(cache),
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
    Inner3,
    Inner47,
    Inner256,
    InnerDelta,
    Leaf1,
    Leaf2,
    Leaf21,
    Leaf64,
    Leaf141,
    Leaf256,
}

impl NodeSize for VerkleNodeKind {
    fn node_byte_size(&self) -> usize {
        let inner_size = match self {
            VerkleNodeKind::Empty => 0,
            VerkleNodeKind::Inner3 => {
                std::mem::size_of::<Box<SparseInnerNode<3>>>()
                    + std::mem::size_of::<SparseInnerNode<3>>()
            }
            VerkleNodeKind::Inner47 => {
                std::mem::size_of::<Box<SparseInnerNode<47>>>()
                    + std::mem::size_of::<SparseInnerNode<47>>()
            }
            VerkleNodeKind::Inner256 => {
                std::mem::size_of::<Box<FullInnerNode>>() + std::mem::size_of::<FullInnerNode>()
            }
            VerkleNodeKind::InnerDelta => {
                std::mem::size_of::<Box<InnerDeltaNode>>() + std::mem::size_of::<InnerDeltaNode>()
            }
            VerkleNodeKind::Leaf1 => {
                std::mem::size_of::<Box<SparseLeafNode<1>>>()
                    + std::mem::size_of::<SparseLeafNode<1>>()
            }
            VerkleNodeKind::Leaf2 => {
                std::mem::size_of::<Box<SparseLeafNode<2>>>()
                    + std::mem::size_of::<SparseLeafNode<2>>()
            }
            VerkleNodeKind::Leaf21 => {
                std::mem::size_of::<Box<SparseLeafNode<21>>>()
                    + std::mem::size_of::<SparseLeafNode<21>>()
            }
            VerkleNodeKind::Leaf64 => {
                std::mem::size_of::<Box<SparseLeafNode<64>>>()
                    + std::mem::size_of::<SparseLeafNode<64>>()
            }
            VerkleNodeKind::Leaf141 => {
                std::mem::size_of::<Box<SparseLeafNode<141>>>()
                    + std::mem::size_of::<SparseLeafNode<141>>()
            }
            VerkleNodeKind::Leaf256 => {
                std::mem::size_of::<Box<FullLeafNode>>() + std::mem::size_of::<FullLeafNode>()
            }
        };
        std::mem::size_of::<VerkleNode>() + inner_size
    }

    fn min_non_empty_node_size() -> usize {
        // Because we don't store empty nodes, the minimum size is the smallest non-empty node.
        VerkleNodeKind::Inner3.node_byte_size()
    }
}

/// An item in a trie node, together with its index.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned,
)]
#[repr(C)]
pub struct ItemWithIndex<T>
where
    T: Clone + Copy + PartialEq + Eq + FromBytes + IntoBytes + Immutable + Unaligned,
{
    /// The index of the item in the node.
    pub index: u8,
    /// The item stored in the node.
    pub item: T,
}

/// A value of a leaf node in a managed Verkle trie, together with its index.
pub type ValueWithIndex = ItemWithIndex<Value>;
/// An ID in a sparse inner node, together with its index.
pub type VerkleIdWithIndex = ItemWithIndex<VerkleNodeId>;

impl<T> ItemWithIndex<T>
where
    T: Clone + Copy + PartialEq + Eq + FromBytes + IntoBytes + Immutable + Unaligned + Default,
{
    /// Returns a slot for storing an item with the given index, or `None` if no such slot exists.
    /// A slot is suitable if it either already holds the given index, or if it is empty
    /// (i.e., holds the default item).
    fn get_slot_for<const N: usize>(values: &[ItemWithIndex<T>; N], index: u8) -> Option<usize> {
        let mut empty_slot = None;
        // We always do a linear search over all item to ensure that we never hold the same index
        // twice in different slots. By starting the search at the given index we are very likely
        // to find the matching slot immediately in practice (if index < N).
        for (i, vwi) in values
            .iter()
            .enumerate()
            .cycle()
            .skip(index as usize)
            .take(N)
        {
            if vwi.index == index {
                return Some(i);
            } else if empty_slot.is_none() && vwi.item == T::default() {
                empty_slot = Some(i);
            }
        }
        empty_slot
    }

    /// Returns the number of slots that would be required to store the given values or None if they
    /// already fit.
    fn get_slots_for<const N: usize>(
        values: &[ItemWithIndex<T>; N],
        indices: impl Iterator<Item = u8>,
    ) -> Option<usize> {
        let empty_slots = values.iter().filter(|vwi| vwi.item == T::default()).count();
        let mut new_slots = 0;
        for index in indices {
            if values
                .iter()
                .any(|vwi| vwi.index == index && vwi.item != T::default())
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
        VerkleNodeKind::Leaf21 => Ok(VerkleNode::Leaf21(Box::new(
            SparseLeafNode::<21>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf64 => Ok(VerkleNode::Leaf64(Box::new(
            SparseLeafNode::<64>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf141 => Ok(VerkleNode::Leaf141(Box::new(
            SparseLeafNode::<141>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf256 => {
            let mut new_leaf = FullLeafNode {
                stem,
                commitment,
                ..Default::default()
            };
            for v in values {
                new_leaf.values[v.index as usize] = v.item;
            }
            Ok(VerkleNode::Leaf256(Box::new(new_leaf)))
        }
        VerkleNodeKind::Inner3
        | VerkleNodeKind::Inner47
        | VerkleNodeKind::Inner256
        | VerkleNodeKind::InnerDelta => Err(Error::CorruptedState(
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
    commitment: VerkleCommitment,
) -> BTResult<VerkleNode, Error> {
    match VerkleNode::smallest_inner_type_for(n) {
        VerkleNodeKind::Empty => Ok(VerkleNode::Empty(EmptyNode)),
        VerkleNodeKind::Inner3 => Ok(VerkleNode::Inner3(Box::new(
            Inner3VerkleNode::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner47 => Ok(VerkleNode::Inner47(Box::new(
            Inner47VerkleNode::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner256 => {
            let mut new_inner = FullInnerNode {
                commitment,
                ..Default::default()
            };
            for c in children {
                new_inner.children[c.index as usize] = c.item;
            }
            Ok(VerkleNode::Inner256(Box::new(new_inner)))
        }
        VerkleNodeKind::InnerDelta => Err(Error::CorruptedState(
            "InnerDelta is not a valid choice for make_smallest_inner_node_for".to_owned(),
        )
        .into()),
        VerkleNodeKind::Leaf1
        | VerkleNodeKind::Leaf2
        | VerkleNodeKind::Leaf21
        | VerkleNodeKind::Leaf64
        | VerkleNodeKind::Leaf141
        | VerkleNodeKind::Leaf256 => Err(Error::CorruptedState(
            "received non-inner type in make_smallest_inner_node_for".to_owned(),
        )
        .into()),
    }
}

/// A trait to link together full and sparse inner nodes.
/// It provides a set of operations common to all inner node types.
pub trait ManagedInnerNode {
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
        let inner3_node = VerkleNodeKind::Inner3;
        let inner47_node = VerkleNodeKind::Inner47;
        let inner256_node = VerkleNodeKind::Inner256;
        let leaf1_node = VerkleNodeKind::Leaf1;
        let leaf2_node = VerkleNodeKind::Leaf2;
        let leaf21_node = VerkleNodeKind::Leaf21;
        let leaf64_node = VerkleNodeKind::Leaf64;
        let leaf141_node = VerkleNodeKind::Leaf141;
        let leaf256_node = VerkleNodeKind::Leaf256;

        assert_eq!(
            empty_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
        );
        assert_eq!(
            inner3_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseInnerNode<3>>>()
                + std::mem::size_of::<SparseInnerNode<3>>()
        );
        assert_eq!(
            inner47_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseInnerNode<47>>>()
                + std::mem::size_of::<SparseInnerNode<47>>()
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
            leaf21_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseLeafNode<21>>>()
                + std::mem::size_of::<SparseLeafNode<21>>()
        );
        assert_eq!(
            leaf64_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseLeafNode<64>>>()
                + std::mem::size_of::<SparseLeafNode<64>>()
        );
        assert_eq!(
            leaf141_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseLeafNode<141>>>()
                + std::mem::size_of::<SparseLeafNode<141>>()
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
            VerkleNode::Inner3(Box::default()).node_byte_size()
        );
    }

    #[test]
    fn node_byte_size_returns_node_type_byte_size() {
        let empty_node = VerkleNode::Empty(EmptyNode);
        let inner3_node = VerkleNode::Inner3(Box::default());
        let inner47_node = VerkleNode::Inner47(Box::default());
        let inner256_node = VerkleNode::Inner256(Box::default());
        let leaf1_node = VerkleNode::Leaf1(Box::default());
        let leaf2_node = VerkleNode::Leaf2(Box::default());
        let leaf21_node = VerkleNode::Leaf21(Box::default());
        let leaf64_node = VerkleNode::Leaf64(Box::default());
        let leaf141_node = VerkleNode::Leaf141(Box::default());
        let leaf256_node = VerkleNode::Leaf256(Box::default());

        assert_eq!(
            VerkleNodeKind::Empty.node_byte_size(),
            empty_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Inner3.node_byte_size(),
            inner3_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Inner47.node_byte_size(),
            inner47_node.node_byte_size()
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
            VerkleNodeKind::Leaf21.node_byte_size(),
            leaf21_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Leaf64.node_byte_size(),
            leaf64_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Leaf141.node_byte_size(),
            leaf141_node.node_byte_size()
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
    fn get_slots_for_returns_number_of_required_slots_or_none_if_values_fit() {
        let mut node = SparseLeafNode::<5>::default();
        node.values[1] = ValueWithIndex {
            index: 1,
            item: [1; 32],
        };
        node.values[2] = ValueWithIndex {
            index: 10,
            item: [1; 32],
        };
        node.values[3] = ValueWithIndex {
            index: 100,
            item: Value::default(),
        };
        // node now has 2 occupied slots (for indices 1 and 10) and 3 empty slots

        // Enough empty slots for all new indices
        let slots = ValueWithIndex::get_slots_for(&node.values, [100, 101, 102].into_iter());
        assert_eq!(slots, None);

        // Enough empty slots and slots which get overwritten
        let slots = ValueWithIndex::get_slots_for(&node.values, [100, 101, 102, 10, 1].into_iter());
        assert_eq!(slots, None);

        // Not enough empty slots
        let slots = ValueWithIndex::get_slots_for(&node.values, [100, 101, 102, 103].into_iter());
        assert_eq!(slots, Some(6)); // 2 existing + 4 new
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
        let slot = ItemWithIndex::get_slot_for(&values, 0);
        assert_eq!(slot, Some(0));

        // Matching index has precedence over empty slot
        let slot = ItemWithIndex::get_slot_for(&values, 5);
        assert_eq!(slot, Some(3));

        // No matching index, so we return first empty slot
        let slot = ItemWithIndex::get_slot_for(&values, 8); // 8 % 2 = 0, so start start search at 0
        assert_eq!(slot, Some(1));

        // No matching index and no empty slot
        values[1] = TestItemWithIndex { index: 1, item: 30 };
        values[2] = TestItemWithIndex { index: 2, item: 40 };
        let slot = ItemWithIndex::get_slot_for(&values, 250);
        assert_eq!(slot, None);
    }

    /// A trait to link together multiple managed trie node variants in rstest tests.
    pub trait VerkleManagedTrieNode<T>:
        ManagedTrieNode<Union = VerkleNode, Id = VerkleNodeId, Commitment = VerkleCommitment>
        + NodeHelperTrait<T>
    where
        T: Clone + Copy + PartialEq + Eq + FromBytes + IntoBytes + Immutable + Unaligned + Default,
    {
    }

    impl<const N: usize> VerkleManagedTrieNode<Value> for SparseLeafNode<N> {}
    impl<const N: usize> VerkleManagedTrieNode<VerkleNodeId> for SparseInnerNode<N> {}

    /// Helper trait to interact with nodes in rstest tests.
    pub trait NodeHelperTrait<T>
    where
        T: Clone + Copy + PartialEq + Eq + FromBytes + IntoBytes + Immutable + Unaligned + Default,
    {
        fn access_slot(&mut self, slot: usize) -> &mut ItemWithIndex<T>;
        fn get_commitment_input(&self) -> VerkleCommitmentInput;
    }

    impl<const N: usize> NodeHelperTrait<Value> for SparseLeafNode<N> {
        /// Returns a reference to the specified slot (modulo N).
        fn access_slot(&mut self, slot: usize) -> &mut ValueWithIndex {
            &mut self.values[slot % N]
        }

        fn get_commitment_input(&self) -> VerkleCommitmentInput {
            self.get_commitment_input().unwrap()
        }
    }

    impl<const N: usize> NodeHelperTrait<VerkleNodeId> for SparseInnerNode<N> {
        /// Returns a reference to the specified slot (modulo N).
        fn access_slot(&mut self, slot: usize) -> &mut VerkleIdWithIndex {
            &mut self.children[slot % N]
        }

        fn get_commitment_input(&self) -> VerkleCommitmentInput {
            self.get_commitment_input().unwrap()
        }
    }
}
