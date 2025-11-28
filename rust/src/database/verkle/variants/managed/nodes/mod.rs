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
use zerocopy::{FromBytes, Immutable, IntoBytes, Unaligned};

use crate::{
    database::{
        managed_trie::{LookupResult, ManagedTrieNode, StoreAction, UnionManagedTrieNode},
        verkle::variants::managed::{
            VerkleNodeId,
            commitment::{VerkleCommitment, VerkleCommitmentInput},
            nodes::{
                empty::EmptyNode, inner::FullInnerNode, leaf::FullLeafNode,
                sparse_inner::SparseInnerNode, sparse_leaf::SparseLeafNode,
            },
        },
    },
    error::{BTResult, Error},
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
    Empty(EmptyVerkleNode),
    Inner2(Box<Inner2VerkleNode>),
    Inner256(Box<Inner256VerkleNode>),
    Leaf2(Box<Leaf2VerkleNode>),
    Leaf256(Box<Leaf256VerkleNode>),
    // Make sure to adjust smallest_leaf_type_for when adding new leaf types.
}

type EmptyVerkleNode = EmptyNode;
type Inner2VerkleNode = SparseInnerNode<2>;
type Inner256VerkleNode = FullInnerNode;
type Leaf2VerkleNode = SparseLeafNode<2>;
type Leaf256VerkleNode = FullLeafNode;

impl VerkleNode {
    /// Returns the smallest leaf node type capable of storing `n` values.
    pub fn smallest_leaf_type_for(n: usize) -> VerkleNodeKind {
        match n {
            0 => VerkleNodeKind::Empty,
            1..=2 => VerkleNodeKind::Leaf2,
            3..=256 => VerkleNodeKind::Leaf256,
            _ => panic!("no leaf type for more than 256 values"),
        }
    }

    /// Returns the smallest inner node type capable of storing `n` values.
    pub fn smallest_inner_type_for(n: usize) -> VerkleNodeKind {
        match n {
            0 => VerkleNodeKind::Empty,
            1..=2 => VerkleNodeKind::Inner2,
            3..=256 => VerkleNodeKind::Inner256,
            _ => panic!("no inner type for more than 256 children"),
        }
    }

    /// Returns the commitment input for computing the commitment of this node.
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
        match self {
            VerkleNode::Empty(n) => n.get_commitment_input(),
            VerkleNode::Inner2(n) => n.get_commitment_input(),
            VerkleNode::Inner256(n) => n.get_commitment_input(),
            VerkleNode::Leaf2(n) => n.get_commitment_input(),
            VerkleNode::Leaf256(n) => n.get_commitment_input(),
        }
    }
}

impl ToNodeKind for VerkleNode {
    type Target = VerkleNodeKind;

    /// Converts the ID to its corresponding node kind. This conversion will always succeed.
    fn to_node_kind(&self) -> Option<Self::Target> {
        match self {
            VerkleNode::Empty(_) => Some(VerkleNodeKind::Empty),
            VerkleNode::Inner2(_) => Some(VerkleNodeKind::Inner2),
            VerkleNode::Inner256(_) => Some(VerkleNodeKind::Inner256),
            VerkleNode::Leaf2(_) => Some(VerkleNodeKind::Leaf2),
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
            VerkleNode::Inner2(n) => n.lookup(key, depth),
            VerkleNode::Inner256(n) => n.lookup(key, depth),
            VerkleNode::Leaf2(n) => n.lookup(key, depth),
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
            VerkleNode::Inner2(n) => n.next_store_action(key, depth, self_id),
            VerkleNode::Inner256(n) => n.next_store_action(key, depth, self_id),
            VerkleNode::Leaf2(n) => n.next_store_action(key, depth, self_id),
            VerkleNode::Leaf256(n) => n.next_store_action(key, depth, self_id),
        }
    }

    fn replace_child(&mut self, key: &Key, depth: u8, new: VerkleNodeId) -> BTResult<(), Error> {
        match self {
            VerkleNode::Empty(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner2(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner256(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf2(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf256(n) => n.replace_child(key, depth, new),
        }
    }

    fn store(&mut self, key: &Key, value: &Value) -> BTResult<Value, Error> {
        match self {
            VerkleNode::Empty(n) => n.store(key, value),
            VerkleNode::Inner2(n) => n.store(key, value),
            VerkleNode::Inner256(n) => n.store(key, value),
            VerkleNode::Leaf2(n) => n.store(key, value),
            VerkleNode::Leaf256(n) => n.store(key, value),
        }
    }

    fn get_commitment(&self) -> Self::Commitment {
        match self {
            VerkleNode::Empty(n) => n.get_commitment(),
            VerkleNode::Inner2(n) => n.get_commitment(),
            VerkleNode::Inner256(n) => n.get_commitment(),
            VerkleNode::Leaf2(n) => n.get_commitment(),
            VerkleNode::Leaf256(n) => n.get_commitment(),
        }
    }

    fn set_commitment(&mut self, cache: Self::Commitment) -> BTResult<(), Error> {
        match self {
            VerkleNode::Empty(n) => n.set_commitment(cache),
            VerkleNode::Inner2(n) => n.set_commitment(cache),
            VerkleNode::Inner256(n) => n.set_commitment(cache),
            VerkleNode::Leaf2(n) => n.set_commitment(cache),
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
    Inner2,
    Inner256,
    Leaf2,
    Leaf256,
}

impl NodeSize for VerkleNodeKind {
    fn node_byte_size(&self) -> usize {
        let inner_size = match self {
            VerkleNodeKind::Empty => 0,
            VerkleNodeKind::Inner2 => {
                std::mem::size_of::<Box<SparseInnerNode<2>>>()
                    + std::mem::size_of::<SparseInnerNode<2>>()
            }
            VerkleNodeKind::Inner256 => {
                std::mem::size_of::<Box<FullInnerNode>>() + std::mem::size_of::<FullInnerNode>()
            }
            VerkleNodeKind::Leaf2 => {
                std::mem::size_of::<Box<SparseLeafNode<2>>>()
                    + std::mem::size_of::<SparseLeafNode<2>>()
            }
            VerkleNodeKind::Leaf256 => {
                std::mem::size_of::<Box<FullLeafNode>>() + std::mem::size_of::<FullLeafNode>()
            }
        };
        std::mem::size_of::<VerkleNode>() + inner_size
    }

    fn min_non_empty_node_size() -> usize {
        // TODO: Maybe now is the innernode 2?
        // Because we don't store empty nodes, the minimum size is the smallest non-empty node.
        VerkleNodeKind::Leaf2.node_byte_size()
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
type ValueWithIndex = ItemWithIndex<Value>;
/// An ID in a sparse inner node, together with its index.
type VerkleIdWithIndex = ItemWithIndex<VerkleNodeId>;

impl<T> ItemWithIndex<T>
where
    T: Clone + Copy + PartialEq + Eq + FromBytes + IntoBytes + Immutable + Unaligned + Default,
{
    /// Returns a slot for storing a value with the given index, or `None` if no such slot exists.
    /// A slot is suitable if it either already holds the given index, or if it is empty
    /// (i.e., holds the default value).
    fn get_slot_for<const N: usize>(values: &[ItemWithIndex<T>; N], index: u8) -> Option<usize>
    where
        T: Clone + Copy + PartialEq + Eq + FromBytes + IntoBytes + Immutable + Unaligned + Default,
    {
        let mut empty_slot = None;
        // We always do a linear search over all values to ensure that we never hold the same index
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
}

/// Creates the smallest leaf node capable of storing `n` values, initialized with the given
/// `stem`, `values` and `commitment`.
pub fn make_smallest_leaf_node_for(
    n: usize,
    stem: [u8; 31],
    values: &[ValueWithIndex],
    commitment: VerkleCommitment,
) -> BTResult<VerkleNode, Error> {
    match VerkleNode::smallest_leaf_type_for(n) {
        VerkleNodeKind::Empty => Ok(VerkleNode::Empty(EmptyNode)),
        VerkleNodeKind::Leaf2 => Ok(VerkleNode::Leaf2(Box::new(
            SparseLeafNode::<2>::from_existing(stem, values, commitment)?,
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
        VerkleNodeKind::Inner2 | VerkleNodeKind::Inner256 => Err(Error::CorruptedState(
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
        VerkleNodeKind::Inner2 => Ok(VerkleNode::Inner2(Box::new(
            SparseInnerNode::<2>::from_existing(children, commitment)?,
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
        VerkleNodeKind::Leaf2 | VerkleNodeKind::Leaf256 => Err(Error::CorruptedState(
            "received non-inner type in make_smallest_inner_node_for".to_owned(),
        )
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_type_byte_size_returns_correct_size() {
        let empty_node = VerkleNodeKind::Empty;
        let inner2_node = VerkleNodeKind::Inner2;
        let inner256_node = VerkleNodeKind::Inner256;
        let leaf2_node = VerkleNodeKind::Leaf2;
        let leaf256_node = VerkleNodeKind::Leaf256;

        assert_eq!(
            empty_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
        );
        assert_eq!(
            inner2_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseInnerNode<2>>>()
                + std::mem::size_of::<SparseInnerNode<2>>()
        );
        assert_eq!(
            inner256_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<FullInnerNode>>()
                + std::mem::size_of::<FullInnerNode>()
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
        let inner2_node = VerkleNode::Inner2(Box::default());
        let inner256_node = VerkleNode::Inner256(Box::default());
        let leaf2_node = VerkleNode::Leaf2(Box::default());
        let leaf256_node = VerkleNode::Leaf256(Box::default());

        assert_eq!(
            VerkleNodeKind::Empty.node_byte_size(),
            empty_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Inner2.node_byte_size(),
            inner2_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Inner256.node_byte_size(),
            inner256_node.node_byte_size()
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
        let slot = TestItemWithIndex::get_slot_for(&values, 8); // 8 % 2 = 0, so start start search at 0
        assert_eq!(slot, Some(1));

        // No matching index and no empty slot
        values[1] = TestItemWithIndex { index: 1, item: 30 };
        values[2] = TestItemWithIndex { index: 2, item: 40 };
        let slot = TestItemWithIndex::get_slot_for(&values, 250);
        assert_eq!(slot, None);
    }
}
