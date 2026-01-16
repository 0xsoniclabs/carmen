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
        managed_trie::{LookupResult, ManagedTrieNode, StoreAction},
        verkle::{
            KeyedUpdateBatch,
            variants::managed::{
                FullInnerNode, VerkleNode, VerkleNodeId,
                commitment::{
                    VerkleCommitment, VerkleCommitmentInput, VerkleInnerCommitment,
                    VerkleLeafCommitment,
                },
                nodes::{make_smallest_inner_node_for, make_smallest_leaf_node_for},
            },
        },
        visitor::NodeVisitor,
    },
    error::{BTResult, Error},
    statistics::node_count::NodeCountVisitor,
    types::{Key, Value},
};

/// An empty node in a managed Verkle trie.
/// This is a zero-sized type that only exists for implementing traits on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyNode;

impl EmptyNode {
    /// Returns an error as EmptyNode does not support commitment inputs.
    /// This method only exists for implementation symmetry in [`VerkleNode::get_commitment_input`].
    #[cfg_attr(not(test), allow(unused))]
    #[allow(clippy::unused_self, clippy::trivially_copy_pass_by_ref)]
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
        Err(Error::UnsupportedOperation(
            "EmptyNode does not support get_commitment_input".to_owned(),
        )
        .into())
    }
}

impl ManagedTrieNode for EmptyNode {
    type Union = VerkleNode;
    type Id = VerkleNodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, _key: &Key, _depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {
        Ok(LookupResult::Value(Value::default()))
    }

    fn next_store_action<'a>(
        &self,
        updates: KeyedUpdateBatch<'a>,
        depth: u8,
        _self_id: Self::Id,
    ) -> BTResult<StoreAction<'a, Self::Id, Self::Union>, Error> {
        if depth == 0 {
            // While conceptually it would suffice to create a leaf node here,
            // Geth always creates an inner node (and we want to stay compatible).
            let inner = FullInnerNode::default();
            Ok(StoreAction::HandleTransform(VerkleNode::Inner256(
                Box::new(inner),
            )))
        } else {
            // Safe to unwrap: Slice is always 31 bytes
            let stem = updates.first_key()[..31].try_into().unwrap();
            if updates.all_stems_match(&stem) {
                let new_leaf = make_smallest_leaf_node_for(
                    updates.split(31).count(), // this counts the number of distinct keys
                    stem,
                    &[],
                    &VerkleLeafCommitment::default(),
                )?;
                Ok(StoreAction::HandleTransform(new_leaf))
            } else {
                // Because we have non-matching stems, we need an inner node
                let new_inner = make_smallest_inner_node_for(
                    updates.split(depth).count(),
                    &[],
                    &VerkleInnerCommitment::default(),
                )?;
                Ok(StoreAction::HandleTransform(new_inner))
            }
        }
    }

    fn get_commitment(&self) -> Self::Commitment {
        // This could also be a leaf commitment - it doesn't matter
        VerkleCommitment::Inner(VerkleInnerCommitment::default())
    }
}

impl NodeVisitor<EmptyNode> for NodeCountVisitor {
    fn visit(&mut self, _node: &EmptyNode, level: u64) -> BTResult<(), Error> {
        self.count_node(level, "Empty", 0); // num children doesn't matter
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        database::verkle::{
            KeyedUpdateBatch,
            test_utils::FromIndexValues,
            variants::managed::{commitment::VerkleLeafCommitment, nodes::VerkleNodeKind},
        },
        error::BTError,
        types::TreeId,
    };

    #[test]
    fn get_commitment_input_returns_error() {
        let node = EmptyNode;
        let result = node.get_commitment_input();
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::UnsupportedOperation(e)) if e.contains(
                "EmptyNode does not support get_commitment_input",
            )
        ));
    }

    #[test]
    fn lookup_returns_default_value() {
        let node = EmptyNode;
        let key = Key::default();
        let result = node.lookup(&key, 0).unwrap();
        assert_eq!(result, LookupResult::Value(Value::default()));
    }

    #[test]
    fn next_store_action_at_depth_zero_transforms_to_inner_node() {
        let node = EmptyNode;
        let update = KeyedUpdateBatch::from_key_value_pairs(&[(Key::default(), Value::default())]);
        let action = node
            .next_store_action(
                update,
                0,
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty),
            )
            .unwrap();
        assert!(matches!(action, StoreAction::HandleTransform(_)));
    }

    #[test]
    fn next_store_action_at_non_zero_depth_with_non_matching_stems_transforms_to_inner_node() {
        let node = EmptyNode;
        let updates = KeyedUpdateBatch::from_key_value_pairs(&[
            ([1; 32], Value::default()),
            ([2; 32], Value::default()),
        ]);
        let action = node
            .next_store_action(
                updates.clone(),
                1,
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty),
            )
            .unwrap();
        match action {
            StoreAction::HandleTransform(inner) => {
                assert!(matches!(inner, VerkleNode::Inner9(_)));
            }
            _ => panic!("expected HandleTransform to inner node"),
        }
    }

    #[test]
    fn next_store_action_at_non_zero_depth_with_matching_stems_transforms_to_leaf_with_capacity_for_len_update_slots()
     {
        let node = EmptyNode;
        let updates = KeyedUpdateBatch::from_key_value_pairs(&[
            ([1; 32], Value::default()),
            ([1; 32], Value::default()),
        ]);
        let action = node
            .next_store_action(
                updates.clone(),
                1,
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty),
            )
            .unwrap();
        match action {
            StoreAction::HandleTransform(leaf) => {
                // This new leaf is able to store the value
                assert_eq!(
                    leaf.next_store_action(
                        updates.clone(),
                        1,
                        VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty) // type does not matter
                    ),
                    Ok(StoreAction::Store (updates))
                );
            }
            _ => panic!("expected HandleTransform to leaf node"),
        }
    }

    #[test]
    fn next_store_action_converts_to_leaf_of_minimal_necessary_size() {
        let mut updates = Vec::new();
        for i in 0..smallest_leaf_size() {
            updates.push((
                Key::from_index_values(0, &[(31, i as u8)]),
                Value::default(),
            ));
        }
        updates.push((Key::from_index_values(0, &[(31, 0)]), Value::default()));

        let update = KeyedUpdateBatch::from_key_value_pairs(&updates);
        let empty = EmptyNode;
        let action = empty
            .next_store_action(
                update,
                1,
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty),
            )
            .unwrap();
        match action {
            StoreAction::HandleTransform(leaf) => {
                assert_eq!(
                    leaf,
                    make_smallest_leaf_node_for(
                        smallest_leaf_size(),
                        [0; 31],
                        &[],
                        &VerkleLeafCommitment::default(),
                    )
                    .unwrap(),
                );
            }
            _ => panic!("expected HandleTransform to leaf node"),
        }
    }

    #[test]
    fn next_store_action_splits_update_at_byte_31_to_determine_leaf_size() {
        let mut updates = Vec::new();
        for i in 0..256 {
            updates.push((
                Key::from_index_values(0, &[(31, i as u8)]),
                Value::default(),
            ));
        }

        let update = KeyedUpdateBatch::from_key_value_pairs(&updates);
        let empty = EmptyNode;
        let action = empty
            .next_store_action(
                update,
                1,
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty),
            )
            .unwrap();
        match action {
            StoreAction::HandleTransform(leaf) => {
                assert_eq!(
                    leaf,
                    make_smallest_leaf_node_for(
                        256,
                        [0; 31],
                        &[],
                        &VerkleLeafCommitment::default()
                    )
                    .unwrap(),
                );
            }
            _ => panic!("expected HandleTransform to leaf node"),
        }
    }

    #[test]
    fn get_commitment_returns_default_commitment() {
        let node = EmptyNode;
        let commitment = node.get_commitment();
        assert_eq!(
            commitment,
            VerkleCommitment::Inner(VerkleInnerCommitment::default())
        );
    }

    fn smallest_leaf_size() -> usize {
        let prev =
            make_smallest_leaf_node_for(1, [0; 31], &[], &VerkleLeafCommitment::default()).unwrap();
        for i in 2..=256 {
            let leaf =
                make_smallest_leaf_node_for(i, [0; 31], &[], &VerkleLeafCommitment::default())
                    .unwrap();
            if prev != leaf {
                return i - 1;
            }
        }
        unreachable!();
    }
}
