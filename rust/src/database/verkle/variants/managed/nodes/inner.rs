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
            Node,
            commitment::{OnDiskVerkleCommitment, VerkleCommitment, VerkleCommitmentInput},
            nodes::{NodeType, id::NodeId},
        },
    },
    error::Error,
    types::{DiskRepresentable, Key, TreeId},
};

/// An inner node in a managed Verkle trie.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InnerNode {
    pub children: [NodeId; 256],
    pub commitment: VerkleCommitment,
}

#[derive(Debug, Clone, PartialEq, Eq, Immutable, FromBytes, IntoBytes, Unaligned)]
#[repr(C)]
pub struct OnDiskInnerNode {
    pub children: [NodeId; 256],
    pub commitment: OnDiskVerkleCommitment,
}

impl From<&InnerNode> for OnDiskInnerNode {
    fn from(node: &InnerNode) -> Self {
        OnDiskInnerNode {
            children: node.children,
            commitment: OnDiskVerkleCommitment::from(&node.commitment),
        }
    }
}

impl From<OnDiskInnerNode> for InnerNode {
    fn from(node: OnDiskInnerNode) -> Self {
        InnerNode {
            children: node.children,
            commitment: VerkleCommitment::from(node.commitment),
        }
    }
}

impl DiskRepresentable for InnerNode {
    fn from_disk_repr<E>(
        read_into_buffer: impl FnOnce(&mut [u8]) -> Result<(), E>,
    ) -> Result<Self, E>
    where
        Self: Sized,
    {
        OnDiskInnerNode::from_disk_repr(read_into_buffer).map(Into::into)
    }

    fn to_disk_repr(&'_ self) -> Cow<'_, [u8]> {
        Cow::Owned(OnDiskInnerNode::from(self).to_disk_repr().into_owned())
    }

    fn size() -> usize {
        std::mem::size_of::<OnDiskInnerNode>()
    }
}

impl InnerNode {
    pub fn get_commitment_input(&self) -> Result<VerkleCommitmentInput, Error> {
        Ok(VerkleCommitmentInput::Inner(self.children))
    }
}

impl Default for InnerNode {
    fn default() -> Self {
        InnerNode {
            children: [NodeId::from_idx_and_node_type(0, NodeType::Empty); 256],
            commitment: VerkleCommitment::default(),
        }
    }
}

impl ManagedTrieNode for InnerNode {
    type Union = Node;
    type Id = NodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, key: &Key, depth: u8) -> Result<LookupResult<Self::Id>, Error> {
        Ok(LookupResult::Node(
            self.children[key[depth as usize] as usize],
        ))
    }

    fn next_store_action(
        &self,
        key: &Key,
        depth: u8,
        _self_id: Self::Id,
    ) -> Result<StoreAction<Self::Id, Self::Union>, Error> {
        let pos = key[depth as usize] as usize;
        Ok(StoreAction::Descend(pos, self.children[pos]))
    }

    fn replace_child(&mut self, key: &Key, depth: u8, new: NodeId) -> Result<(), Error> {
        self.children[key[depth as usize] as usize] = new;
        Ok(())
    }

    fn get_commitment(&self) -> Self::Commitment {
        self.commitment
    }

    fn set_commitment(&mut self, cache: Self::Commitment) -> Result<(), Error> {
        self.commitment = cache;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inner_node_default_returns_inner_node_with_all_children_set_to_empty_node_id() {
        let node: InnerNode = InnerNode::default();
        assert_eq!(node.commitment, VerkleCommitment::default());
        assert_eq!(
            node.children,
            [NodeId::from_idx_and_node_type(0, NodeType::Empty); 256]
        );
    }
}
