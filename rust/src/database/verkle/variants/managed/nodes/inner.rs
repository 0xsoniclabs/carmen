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
            Node,
            commitment::{VerkleCommitment, VerkleCommitmentInput},
            nodes::{NodeType, id::NodeId},
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
    pub children: [NodeId; 256],
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
            children: [NodeId::from_idx_and_node_type(0, NodeType::Empty); 256],
            commitment: VerkleCommitment::default(),
        }
    }
}

impl ManagedTrieNode for InnerNode {
    type Union = Node;
    type Id = NodeId;
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
        let pos = key[depth as usize] as usize;
        Ok(StoreAction::Descend {
            index: pos,
            id: self.children[pos],
        })
    }

    fn replace_child(&mut self, key: &Key, depth: u8, new: NodeId) -> BTResult<(), Error> {
        self.children[key[depth as usize] as usize] = new;
        Ok(())
    }

    fn get_commitment(&self) -> Self::Commitment {
        self.commitment
    }

    fn set_commitment(&mut self, cache: Self::Commitment) -> BTResult<(), Error> {
        self.commitment = cache;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        database::{
            managed_trie::TrieCommitment,
            verkle::{test_utils::FromIndexValues, variants::managed::nodes::NodeType},
        },
        types::TreeId,
    };

    #[test]
    fn inner_node_default_returns_inner_node_with_all_children_set_to_node_id() {
        let node: InnerNode = InnerNode::default();
        assert_eq!(node.commitment, VerkleCommitment::default());
        assert_eq!(
            node.children,
            [NodeId::from_idx_and_node_type(0, NodeType::Empty); 256]
        );
    }

    #[test]
    fn get_commitment_input_returns_children() {
        let node = InnerNode {
            children: (0..=255)
                .map(|i| NodeId::from_idx_and_node_type(i as u64, NodeType::Inner))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            ..Default::default()
        };
        let result = node.get_commitment_input().unwrap();
        assert_eq!(result, VerkleCommitmentInput::Inner(node.children));
    }

    #[test]
    fn lookup_returns_id_of_child_at_key_position() {
        let mut node = InnerNode::default();
        let depth = 10;
        let position = 78;
        let key = Key::from_index_values(1, &[(depth, position)]);
        let child_id = NodeId::from_idx_and_node_type(42, NodeType::Inner);
        node.children[position as usize] = child_id;

        let result = node.lookup(&key, depth as u8).unwrap();
        assert_eq!(result, LookupResult::Node(child_id));
    }

    #[test]
    fn next_store_action_is_descend_into_child_at_key_position() {
        let mut node = InnerNode::default();
        let depth = 10;
        let position = 78;
        let key = Key::from_index_values(1, &[(depth, position)]);
        let child_id = NodeId::from_idx_and_node_type(42, NodeType::Inner);
        node.children[position as usize] = child_id;

        let result = node
            .next_store_action(
                &key,
                depth as u8,
                NodeId::from_idx_and_node_type(0, NodeType::Inner),
            )
            .unwrap();
        assert_eq!(
            result,
            StoreAction::Descend {
                index: position as usize,
                id: child_id
            }
        );
    }

    #[test]
    fn replace_child_sets_child_id_at_key_position() {
        let mut node = InnerNode::default();
        let depth = 5;
        let position = 200;
        let key = Key::from_index_values(1, &[(depth, position)]);
        let new_child_id = NodeId::from_idx_and_node_type(99, NodeType::Leaf256);

        node.replace_child(&key, depth as u8, new_child_id).unwrap();
        assert_eq!(node.children[position as usize], new_child_id);
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
