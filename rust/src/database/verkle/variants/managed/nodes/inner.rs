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

use crate::{
    database::{
        managed_trie::{LookupResult, ManagedTrieNode, StoreAction},
        verkle::variants::managed::{
            VerkleNode,
            commitment::{VerkleCommitment, VerkleCommitmentInput},
            nodes::{VerkleNodeKind, id::VerkleNodeId},
        },
    },
    error::{BTResult, Error},
    types::{Key, TreeId},
};

/// An inner node in a managed Verkle trie.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned)]
#[repr(C)]
pub struct InnerNode {
    pub children: [VerkleNodeId; 256],
    pub commitment: VerkleCommitment,
}

impl InnerNode {
    /// Returns the children of this inner node as commitment input.
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
        Ok(VerkleCommitmentInput::Inner(self.children))
    }
}

impl Default for InnerNode {
    fn default() -> Self {
        InnerNode {
            children: [VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty); 256],
            commitment: VerkleCommitment::default(),
        }
    }
}

impl ManagedTrieNode for InnerNode {
    type Union = VerkleNode;
    type Id = VerkleNodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, key: &Key, depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {
        Ok(LookupResult::Node(
            self.children[key[depth as usize] as usize],
        ))
    }

    fn next_store_action(
        &self,
        key: &Key,
        depth: u8,
        _self_id: Self::Id,
    ) -> BTResult<StoreAction<Self::Id, Self::Union>, Error> {
        let index = key[depth as usize] as usize;
        Ok(StoreAction::Descend {
            index,
            id: self.children[index],
        })
    }

    fn replace_child(&mut self, key: &Key, depth: u8, new: VerkleNodeId) -> BTResult<(), Error> {
        self.children[key[depth as usize] as usize] = new;
        Ok(())
    }

    fn get_commitment(&self) -> Self::Commitment {
        self.commitment
    }

    fn set_commitment(&mut self, commitment: Self::Commitment) -> BTResult<(), Error> {
        self.commitment = commitment;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use super::*;
    use crate::{
        database::{
            managed_trie::TrieCommitment,
            verkle::{test_utils::FromIndexValues, variants::managed::nodes::VerkleNodeKind},
        },
        types::TreeId,
    };

    #[test]
    fn inner_node_default_returns_inner_node_with_all_children_set_to_empty_node_id() {
        let node: InnerNode = InnerNode::default();
        assert_eq!(node.commitment, VerkleCommitment::default());
        assert_eq!(
            node.children,
            [VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty); 256]
        );
    }

    #[test]
    fn get_commitment_input_returns_children() {
        let node = InnerNode {
            children: array::from_fn(|i| {
                VerkleNodeId::from_idx_and_node_kind(i as u64, VerkleNodeKind::Inner)
            }),
            ..Default::default()
        };
        let result = node.get_commitment_input().unwrap();
        assert_eq!(result, VerkleCommitmentInput::Inner(node.children));
    }

    #[test]
    fn lookup_returns_id_of_child_at_key_index() {
        let mut node = InnerNode::default();
        let depth = 10;
        let index = 78;
        let key = Key::from_index_values(1, &[(depth, index)]);
        let child_id = VerkleNodeId::from_idx_and_node_kind(42, VerkleNodeKind::Inner);
        node.children[index as usize] = child_id;

        let result = node.lookup(&key, depth as u8).unwrap();
        assert_eq!(result, LookupResult::Node(child_id));
    }

    #[test]
    fn next_store_action_is_descend_into_child_at_key_index() {
        let mut node = InnerNode::default();
        let depth = 10;
        let index = 78;
        let key = Key::from_index_values(1, &[(depth, index)]);
        let child_id = VerkleNodeId::from_idx_and_node_kind(42, VerkleNodeKind::Inner);
        node.children[index as usize] = child_id;

        let result = node
            .next_store_action(
                &key,
                depth as u8,
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Inner),
            )
            .unwrap();
        assert_eq!(
            result,
            StoreAction::Descend {
                index: index as usize,
                id: child_id
            }
        );
    }

    #[test]
    fn replace_child_sets_child_id_at_key_index() {
        let mut node = InnerNode::default();
        let depth = 5;
        let index = 200;
        let key = Key::from_index_values(1, &[(depth, index)]);
        let new_child_id = VerkleNodeId::from_idx_and_node_kind(99, VerkleNodeKind::Leaf256);

        node.replace_child(&key, depth as u8, new_child_id).unwrap();
        assert_eq!(node.children[index as usize], new_child_id);
    }

    #[test]
    fn commitment_can_be_set_and_retrieved() {
        let mut node = InnerNode::default();
        assert_eq!(node.get_commitment(), VerkleCommitment::default());

        let mut new_commitment = VerkleCommitment::default();
        new_commitment.modify_child(5);

        node.set_commitment(new_commitment).unwrap();
        assert_eq!(node.get_commitment(), new_commitment);
    }
}
