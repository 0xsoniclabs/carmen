// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::borrow::Cow;

use zerocopy::{FromBytes, Immutable, IntoBytes, Unaligned};

use crate::{
    database::{
        managed_trie::{LookupResult, ManagedTrieNode, StoreAction},
        verkle::variants::managed::{
            VerkleNode, VerkleNodeId,
            commitment::{OnDiskVerkleCommitment, VerkleCommitment, VerkleCommitmentInput},
            nodes::make_smallest_inner_node_for,
        },
    },
    error::{BTResult, Error},
    types::{DiskRepresentable, Key},
};

/// A value of a leaf node in a managed Verkle trie, together with its index.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned,
)]
#[repr(C)]
pub struct IdWithIndex {
    pub index: u8,
    pub id: VerkleNodeId,
}

/// A sparsely populated leaf node in a managed Verkle trie.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
pub struct SparseInnerNode<const N: usize> {
    pub children: [IdWithIndex; N],
    pub commitment: VerkleCommitment,
}

impl<const N: usize> SparseInnerNode<N> {
    /// Creates a sparse leaf node from existing stem, values, and commitment.
    /// Returns an error if there are more than N non-zero values.
    pub fn from_existing(
        children: &[IdWithIndex],
        commitment: VerkleCommitment,
    ) -> BTResult<Self, Error> {
        let mut inner = SparseInnerNode {
            commitment,
            ..Default::default()
        };

        // Insert values from previous leaf using get_slot_for to ensure no duplicate indices.
        for vwi in children {
            if vwi.id == VerkleNodeId::default() {
                continue;
            }
            let slot =
                Self::get_slot_for(&inner.children, vwi.index).ok_or(Error::CorruptedState(
                    "too many non-zero IDs to fit into sparse inner node".to_owned(),
                ))?;
            inner.children[slot] = *vwi;
        }

        Ok(inner)
    }

    /// Returns the values and stem of this leaf node as commitment input.
    // TODO: This should not have to pass 256 values: https://github.com/0xsoniclabs/sonic-admin/issues/384
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
        let mut ids = [VerkleNodeId::default(); 256];
        for IdWithIndex { index, id } in &self.children {
            ids[*index as usize] = *id;
        }
        Ok(VerkleCommitmentInput::Inner(ids))
    }

    /// Returns a slot for storing a value with the given index, or `None` if no such slot exists.
    /// A slot is suitable if it either already holds the given index, or if it is empty
    /// (i.e., holds the default value).
    fn get_slot_for(values: &[IdWithIndex], index: u8) -> Option<usize> {
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
            } else if empty_slot.is_none() && vwi.id == VerkleNodeId::default() {
                empty_slot = Some(i);
            }
        }
        empty_slot
    }
}

impl<const N: usize> Default for SparseInnerNode<N> {
    fn default() -> Self {
        let mut children = [IdWithIndex::default(); N];
        children.iter_mut().enumerate().for_each(|(i, v)| {
            v.index = i as u8;
        });

        SparseInnerNode {
            children,
            commitment: VerkleCommitment::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned)]
#[repr(C)]
pub struct OnDiskSparseInnerNode<const N: usize> {
    pub children: [IdWithIndex; N],
    pub commitment: OnDiskVerkleCommitment,
}

impl<const N: usize> From<OnDiskSparseInnerNode<N>> for SparseInnerNode<N> {
    fn from(on_disk: OnDiskSparseInnerNode<N>) -> Self {
        SparseInnerNode {
            children: on_disk.children,
            commitment: VerkleCommitment::from(on_disk.commitment),
        }
    }
}

impl<const N: usize> From<&SparseInnerNode<N>> for OnDiskSparseInnerNode<N> {
    fn from(node: &SparseInnerNode<N>) -> Self {
        OnDiskSparseInnerNode {
            children: node.children,
            commitment: OnDiskVerkleCommitment::from(&node.commitment),
        }
    }
}

impl<const N: usize> DiskRepresentable for SparseInnerNode<N> {
    fn from_disk_repr<E>(
        read_into_buffer: impl FnOnce(&mut [u8]) -> Result<(), E>,
    ) -> Result<Self, E> {
        OnDiskSparseInnerNode::<N>::from_disk_repr(read_into_buffer).map(Into::into)
    }

    fn to_disk_repr(&'_ self) -> Cow<'_, [u8]> {
        Cow::Owned(
            OnDiskSparseInnerNode::from(self)
                .to_disk_repr()
                .into_owned(),
        )
    }

    fn size() -> usize {
        std::mem::size_of::<OnDiskSparseInnerNode<N>>()
    }
}

impl<const N: usize> ManagedTrieNode for SparseInnerNode<N> {
    type Union = VerkleNode;
    type Id = VerkleNodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, key: &Key, depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {
        for IdWithIndex { index, id } in &self.children {
            if *index == key[depth as usize] {
                return Ok(LookupResult::Node(*id));
            }
        }
        Ok(LookupResult::Node(VerkleNodeId::default()))
    }

    fn next_store_action(
        &self,
        key: &Key,
        depth: u8,
        self_id: Self::Id,
    ) -> BTResult<StoreAction<Self::Id, Self::Union>, Error> {
        // If we have a free/matching slot, we can store the value directly.
        if let Some(slot) = Self::get_slot_for(&self.children, key[depth as usize]) {
            return Ok(StoreAction::Descend {
                index: key[depth as usize] as usize,
                id: self.children[slot].id,
            });
        }

        // If the stems match but we don't have a free/matching slot, convert to a bigger leaf.
        Ok(StoreAction::HandleTransform(make_smallest_inner_node_for(
            N + 1,
            &self.children,
            self.commitment,
        )?))
    }

    fn replace_child(&mut self, key: &Key, depth: u8, new: VerkleNodeId) -> BTResult<(), Error> {
        if let Some(slot) = Self::get_slot_for(&self.children, key[depth as usize]) {
            self.children[slot] = IdWithIndex {
                index: key[depth as usize],
                id: new,
            };
            return Ok(());
        }
        Err(Error::CorruptedState(
            "tried to replace child that does not exist in sparse inner node".to_owned(),
        )
        .into())
    }

    fn get_commitment(&self) -> Self::Commitment {
        self.commitment
    }

    fn set_commitment(&mut self, cache: Self::Commitment) -> BTResult<(), Error> {
        self.commitment = cache;
        Ok(())
    }
}
