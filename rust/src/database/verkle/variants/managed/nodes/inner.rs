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
        managed_trie::{DescendAction, LookupResult, ManagedTrieNode, StoreAction},
        verkle::{
            KeyedUpdateBatch,
            variants::managed::{
                VerkleNode,
                commitment::{VerkleCommitment, VerkleCommitmentInput},
                nodes::{
                    VerkleIdWithIndex, VerkleManagedInnerNode, VerkleNodeKind, id::VerkleNodeId,
                },
            },
        },
        visitor::NodeVisitor,
    },
    error::{BTResult, Error},
    statistics::node_count::NodeCountVisitor,
    types::{Key, ToNodeKind, TreeId},
};

/// An inner node in a managed Verkle trie.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned)]
#[repr(C)]
pub struct FullInnerNode {
    pub children: [VerkleNodeId; 256],
    pub commitment: VerkleCommitment,
}

impl FullInnerNode {
    /// Returns the children of this inner node as commitment input.
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
        Ok(VerkleCommitmentInput::Inner(self.children))
    }
}

impl Default for FullInnerNode {
    fn default() -> Self {
        FullInnerNode {
            children: [VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty); 256],
            commitment: VerkleCommitment::default(),
        }
    }
}

impl ManagedTrieNode for FullInnerNode {
    type Union = VerkleNode;
    type Id = VerkleNodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, key: &Key, depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {
        Ok(LookupResult::Node(
            self.children[key[depth as usize] as usize],
        ))
    }

    fn next_store_action<'a>(
        &self,
        updates: KeyedUpdateBatch<'a>,
        depth: u8,
        _self_id: Self::Id,
    ) -> BTResult<StoreAction<'a, Self::Id, Self::Union>, Error> {
        let mut descent_actions = Vec::new();
        for sub_updates in updates.split(depth) {
            let index = sub_updates.first_key()[depth as usize] as usize;
            descent_actions.push(DescendAction {
                id: self.children[index],
                updates: sub_updates,
            });
        }
        Ok(StoreAction::Descend(descent_actions))
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

impl NodeVisitor<FullInnerNode> for NodeCountVisitor {
    fn visit(&mut self, node: &FullInnerNode, level: u64) -> BTResult<(), Error> {
        self.count_node(
            level,
            "Inner",
            node.children
                .iter()
                .filter(|child| child.to_node_kind().unwrap() != VerkleNodeKind::Empty)
                .count() as u64,
        );
        Ok(())
    }
}

impl VerkleManagedInnerNode for FullInnerNode {
    fn iter_children(&self) -> Box<dyn Iterator<Item = VerkleIdWithIndex> + '_> {
        Box::new(
            self.children
                .iter()
                .enumerate()
                .map(|(index, child_id)| VerkleIdWithIndex {
                    index: index as u8,
                    item: *child_id,
                }),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use super::*;
    use crate::{
        database::{
            managed_trie::TrieCommitment,
            verkle::{
                KeyedUpdateBatch, test_utils::FromIndexValues,
                variants::managed::nodes::VerkleNodeKind,
            },
        },
        types::{TreeId, Value},
    };

    #[test]
    fn inner_node_default_returns_inner_node_with_all_children_set_to_empty_node_id() {
        let node: FullInnerNode = FullInnerNode::default();
        assert_eq!(node.commitment, VerkleCommitment::default());
        assert_eq!(
            node.children,
            [VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty); 256]
        );
    }

    #[test]
    fn get_commitment_input_returns_children() {
        let node = FullInnerNode {
            children: array::from_fn(|i| {
                VerkleNodeId::from_idx_and_node_kind(i as u64, VerkleNodeKind::Inner256)
            }),
            ..Default::default()
        };
        let result = node.get_commitment_input().unwrap();
        assert_eq!(result, VerkleCommitmentInput::Inner(node.children));
    }

    #[test]
    fn lookup_returns_id_of_child_at_key_index() {
        let mut node = FullInnerNode::default();
        let depth = 10;
        let index = 78;
        let key = Key::from_index_values(1, &[(depth, index)]);
        let child_id = VerkleNodeId::from_idx_and_node_kind(42, VerkleNodeKind::Inner256);
        node.children[index as usize] = child_id;

        let result = node.lookup(&key, depth as u8).unwrap();
        assert_eq!(result, LookupResult::Node(child_id));
    }

    #[test]
    fn next_store_action_is_descend_with_one_descent_action_for_each_unique_index_at_depth() {
        let mut node = FullInnerNode::default();
        let depth = 10;
        let key1 = Key::from_index_values(1, &[(depth, 1)]);
        let key2 = Key::from_index_values(1, &[(depth, 2)]);
        let key3_1 = Key::from_index_values(1, &[(depth, 3), (depth + 1, 1)]);
        let key3_2 = Key::from_index_values(1, &[(depth, 3), (depth + 1, 2)]);
        let child_id1 = VerkleNodeId::from_idx_and_node_kind(100, VerkleNodeKind::Inner256);
        let child_id2 = VerkleNodeId::from_idx_and_node_kind(101, VerkleNodeKind::Inner256);
        let child_id3 = VerkleNodeId::from_idx_and_node_kind(102, VerkleNodeKind::Inner256);
        node.children[1] = child_id1;
        node.children[2] = child_id2;
        node.children[3] = child_id3;
        let updates = KeyedUpdateBatch::from_key_value_pairs(&[
            (key1, Value::default()),
            (key2, Value::default()),
            (key3_1, Value::default()),
            (key3_2, Value::default()),
        ]);

        let result = node
            .next_store_action(
                updates.clone(),
                depth as u8,
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Inner256),
            )
            .unwrap();
        assert_eq!(
            result,
            StoreAction::Descend(vec![
                DescendAction {
                    id: child_id1,
                    updates: KeyedUpdateBatch::from_key_value_pairs(&[(key1, Value::default())])
                },
                DescendAction {
                    id: child_id2,
                    updates: KeyedUpdateBatch::from_key_value_pairs(&[(key2, Value::default())])
                },
                DescendAction {
                    id: child_id3,
                    updates: KeyedUpdateBatch::from_key_value_pairs(&[
                        (key3_1, Value::default()),
                        (key3_2, Value::default())
                    ])
                },
            ])
        );
    }

    #[test]
    fn replace_child_sets_child_id_at_key_index() {
        let mut node = FullInnerNode::default();
        let depth = 5;
        let index = 200;
        let key = Key::from_index_values(1, &[(depth, index)]);
        let new_child_id = VerkleNodeId::from_idx_and_node_kind(99, VerkleNodeKind::Leaf256);

        node.replace_child(&key, depth as u8, new_child_id).unwrap();
        assert_eq!(node.children[index as usize], new_child_id);
    }

    #[test]
    fn commitment_can_be_set_and_retrieved() {
        let mut node = FullInnerNode::default();
        assert_eq!(node.get_commitment(), VerkleCommitment::default());

        let mut new_commitment = VerkleCommitment::default();
        new_commitment.modify_child(5);

        node.set_commitment(new_commitment).unwrap();
        assert_eq!(node.get_commitment(), new_commitment);
    }
}
