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
        verkle::{
            KeyedUpdate,
            variants::managed::{
                KeyedUpdateBatch, VerkleNode, VerkleNodeId,
                commitment::{OnDiskVerkleCommitment, VerkleCommitment, VerkleCommitmentInput},
                nodes::{VerkleIdWithIndex, make_smallest_inner_node_for},
            },
        },
        visitor::NodeVisitor,
    },
    error::{BTResult, Error},
    statistics::node_count::NodeCountVisitor,
    types::{DiskRepresentable, Key, Value},
};

/// A leaf node with 256 children in a managed Verkle trie.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullLeafNode {
    pub stem: [u8; 31],
    pub values: [Value; 256],
    pub commitment: VerkleCommitment,
}

impl FullLeafNode {
    /// Returns the values and stem of this leaf node as commitment input.
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
        Ok(VerkleCommitmentInput::Leaf(self.values, self.stem))
    }
}

impl Default for FullLeafNode {
    fn default() -> Self {
        FullLeafNode {
            stem: [0; 31],
            values: [Value::default(); 256],
            commitment: VerkleCommitment::default(),
        }
    }
}

// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, Immutable, FromBytes, IntoBytes, Unaligned)]
#[repr(C)]
pub struct OnDiskFullLeafNode {
    pub stem: [u8; 31],
    pub values: [Value; 256],
    pub commitment: OnDiskVerkleCommitment,
}

impl From<&FullLeafNode> for OnDiskFullLeafNode {
    fn from(node: &FullLeafNode) -> Self {
        OnDiskFullLeafNode {
            stem: node.stem,
            values: node.values,
            commitment: OnDiskVerkleCommitment::from(&node.commitment),
        }
    }
}

impl From<OnDiskFullLeafNode> for FullLeafNode {
    fn from(node: OnDiskFullLeafNode) -> Self {
        FullLeafNode {
            stem: node.stem,
            values: node.values,
            commitment: VerkleCommitment::from(node.commitment),
        }
    }
}

impl DiskRepresentable for FullLeafNode {
    fn from_disk_repr<E>(
        read_into_buffer: impl FnOnce(&mut [u8]) -> Result<(), E>,
    ) -> Result<Self, E> {
        OnDiskFullLeafNode::from_disk_repr(read_into_buffer).map(Into::into)
    }

    fn to_disk_repr(&'_ self) -> Cow<'_, [u8]> {
        Cow::Owned(OnDiskFullLeafNode::from(self).to_disk_repr().into_owned())
    }

    fn size() -> usize {
        std::mem::size_of::<OnDiskFullLeafNode>()
    }
}

impl ManagedTrieNode for FullLeafNode {
    type Union = VerkleNode;
    type Id = VerkleNodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, key: &Key, _depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {
        if key[..31] != self.stem[..] {
            Ok(LookupResult::Value(Value::default()))
        } else {
            Ok(LookupResult::Value(self.values[key[31] as usize]))
        }
    }

    fn next_store_action<'a>(
        &self,
        updates: KeyedUpdateBatch<'a>,
        depth: u8,
        self_id: Self::Id,
    ) -> BTResult<StoreAction<'a, Self::Id, Self::Union>, Error> {
        // If not all keys match the stem, we have to introduce a new inner node.
        if !updates.all_stems_match(&self.stem) {
            let index = self.stem[depth as usize];
            let self_child = VerkleIdWithIndex {
                index,
                item: self_id,
            };
            let slots = updates
                .split(depth)
                .map(|batch| {
                    (self.stem[depth as usize] != batch.first_key()[depth as usize]) as usize
                })
                .sum::<usize>()
                + 1;
            let dirty_index = self.commitment.is_dirty().then_some(index);
            let inner = make_smallest_inner_node_for(
                slots,
                &[self_child],
                &VerkleCommitment::from_existing(&self.commitment, dirty_index),
            )?;
            return Ok(StoreAction::HandleReparent(inner));
        }

        // All updates fit into this leaf.
        Ok(StoreAction::Store(updates))
    }

    fn store(&mut self, update: &KeyedUpdate) -> BTResult<Value, Error> {
        let key = update.key();
        if self.stem[..] != key[..31] {
            return Err(Error::CorruptedState(
                "called store on a leaf with non-matching stem".to_owned(),
            )
            .into());
        }

        let suffix = key[31];
        let prev_value = self.values[suffix as usize];
        update.apply_to_value(&mut self.values[suffix as usize]);
        Ok(prev_value)
    }

    fn get_commitment(&self) -> Self::Commitment {
        self.commitment
    }

    fn set_commitment(&mut self, cache: Self::Commitment) -> BTResult<(), Error> {
        self.commitment = cache;
        Ok(())
    }
}

impl NodeVisitor<FullLeafNode> for NodeCountVisitor {
    fn visit(&mut self, node: &FullLeafNode, level: u64) -> BTResult<(), Error> {
        self.count_node(
            level,
            "Leaf",
            node.values
                .iter()
                .filter(|value| **value != Value::default())
                .count() as u64,
        );
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
            verkle::{
                KeyedUpdateBatch, test_utils::FromIndexValues,
                variants::managed::nodes::VerkleNodeKind,
            },
        },
        error::BTError,
        types::{TreeId, Value},
    };

    #[test]
    fn full_leaf_node_default_returns_leaf_node_with_all_values_set_to_default() {
        let node: FullLeafNode = FullLeafNode::default();
        assert_eq!(node.stem, [0; 31]);
        assert_eq!(node.values, [Value::default(); 256]);
        assert_eq!(node.commitment, VerkleCommitment::default());
    }

    #[test]
    fn can_be_converted_to_and_from_on_disk_representation() {
        let original_node = FullLeafNode {
            values: array::from_fn(|i| Value::from_index_values(i as u8, &[])),
            stem: <[u8; 31]>::from_index_values(5, &[]),
            commitment: {
                // We deliberately only create a default commitment, since this type does
                // not preserve all of its fields when converting to/from on-disk representation.
                let mut commitment = VerkleCommitment::default();
                commitment.test_only_mark_as_clean();
                commitment
            },
        };
        let disk_repr = original_node.to_disk_repr();
        let deserialized_node = FullLeafNode::from_disk_repr::<()>(|buf| {
            buf.copy_from_slice(&disk_repr);
            Ok(())
        })
        .unwrap();
        assert_eq!(original_node, deserialized_node);
    }

    #[test]
    fn get_commitment_input_returns_values_and_stem() {
        let node = FullLeafNode {
            stem: <[u8; 31]>::from_index_values(3, &[]),
            values: array::from_fn(|i| Value::from_index_values(i as u8, &[])),
            ..Default::default()
        };
        let result = node.get_commitment_input().unwrap();
        assert_eq!(result, VerkleCommitmentInput::Leaf(node.values, node.stem));
    }

    #[test]
    fn lookup_with_matching_stem_returns_value_at_final_key_index() {
        let index = 78;
        let key = Key::from_index_values(1, &[(31, index)]);
        let mut node = FullLeafNode {
            stem: key[..31].try_into().unwrap(),
            ..Default::default()
        };
        let value = Value::from_index_values(42, &[]);
        node.values[index as usize] = value;

        let result = node.lookup(&key, 0).unwrap();
        assert_eq!(result, LookupResult::Value(value));

        // Depth is irrelevant
        let result = node.lookup(&key, 42).unwrap();
        assert_eq!(result, LookupResult::Value(value));

        // Mismatching stem returns default value
        let other_key = Key::from_index_values(7, &[]);
        let other_result = node.lookup(&other_key, 0).unwrap();
        assert_eq!(other_result, LookupResult::Value(Value::default()));

        // Other index has default value
        let other_key = Key::from_index_values(1, &[(31, index + 1)]);
        let other_result = node.lookup(&other_key, 0).unwrap();
        assert_eq!(other_result, LookupResult::Value(Value::default()));
    }

    #[test]
    fn next_store_action_with_non_matching_stem_is_reparent() {
        let divergence_at = 5;
        let mut commitment = VerkleCommitment::default();
        commitment.store(123, Value::from_index_values(99, &[]));
        let node = FullLeafNode {
            stem: <[u8; 31]>::from_index_values(1, &[(divergence_at, 56)]),
            commitment,
            ..Default::default()
        };
        let key = Key::from_index_values(1, &[(divergence_at, 97)]);
        let self_id = VerkleNodeId::from_idx_and_node_kind(33, VerkleNodeKind::Leaf256);
        let updates = KeyedUpdateBatch::from_key_value_pairs(&[(key, Value::default())]);

        let result = node
            .next_store_action(updates, divergence_at as u8, self_id)
            .unwrap();
        match result {
            StoreAction::HandleReparent(VerkleNode::Inner9(inner)) => {
                let slot = VerkleIdWithIndex::get_slot_for(&inner.children, 56).unwrap();
                assert_eq!(inner.children[slot].item, self_id);
                // Newly created inner node has commitment of the leaf.
                assert_ne!(inner.get_commitment(), VerkleCommitment::default());
                assert_eq!(inner.get_commitment().commitment(), commitment.commitment());
            }
            _ => panic!("expected HandleReparent with inner node"),
        }
    }

    #[rstest::rstest]
    fn leaf_with_dirty_commitment_is_marked_as_changed_in_new_parent(
        #[values(true, false)] leaf_is_dirty: bool,
    ) {
        let mut commitment = VerkleCommitment::default();
        if !leaf_is_dirty {
            commitment.test_only_mark_as_clean();
        }
        let stem = [42; 31];
        let node = FullLeafNode {
            stem,
            commitment,
            ..Default::default()
        };
        let updates = KeyedUpdateBatch::from_key_value_pairs(&[([99; 32], Value::default())]);
        match node
            .next_store_action(updates, 0, VerkleNodeId::default())
            .unwrap()
        {
            StoreAction::HandleReparent(VerkleNode::Inner9(inner)) => {
                assert_eq!(
                    inner.get_commitment().index_changed(stem[0] as usize),
                    leaf_is_dirty
                );
            }
            _ => panic!("expected HandleReparent with inner node"),
        }
    }

    #[test]
    fn next_store_action_with_non_matching_stem_returns_parent_large_enough_for_all_updates() {
        let depth = 1;

        // The update has 9 batches that diverge at depth 1, but one of them has the same key as
        // the leaf's stem, so an inner node with at least 9 slots is needed
        let mut node = FullLeafNode {
            stem: [0; 31],
            ..Default::default()
        };
        let updates = KeyedUpdateBatch::from_key_value_pairs(
            &(0..9)
                .map(|i| (Key::from_index_values(0, &[(depth, i)]), Value::default()))
                .collect::<Vec<_>>(),
        );

        let result = node
            .next_store_action(updates.clone(), depth as u8, VerkleNodeId::default())
            .unwrap();
        match result {
            StoreAction::HandleReparent(inner) => {
                assert!(matches!(inner, VerkleNode::Inner9(_)));
                // This new inner node is big enough to hold all leaves that will be created
                assert!(matches!(
                    inner.next_store_action(updates.clone(), 1, VerkleNodeId::default()),
                    Ok(StoreAction::Descend(_))
                ));
            }
            _ => panic!("expected HandleReparent"),
        }

        // The update has 9 batches that diverge at depth 1, but all of them have a different key
        // than the leaf's stem, so an inner node with at least 10 slots is needed
        node.stem[depth] = 10;

        let result = node
            .next_store_action(updates.clone(), depth as u8, VerkleNodeId::default())
            .unwrap();
        match result {
            StoreAction::HandleReparent(inner) => {
                assert!(matches!(inner, VerkleNode::Inner15(_)));
                // This new inner node is big enough to hold all leaves that will be created
                assert!(matches!(
                    inner.next_store_action(updates, 1, VerkleNodeId::default()),
                    Ok(StoreAction::Descend(_))
                ));
            }
            _ => panic!("expected HandleReparent"),
        }
    }

    #[test]
    fn next_store_action_with_matching_stem_is_store() {
        let index = 78;
        let key = Key::from_index_values(1, &[(31, index)]);
        let node = FullLeafNode {
            stem: key[..31].try_into().unwrap(),
            ..Default::default()
        };
        let updates = KeyedUpdateBatch::from_key_value_pairs(&[(key, Value::default())]);

        let result = node
            .next_store_action(
                updates.clone(),
                0,
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Leaf256),
            )
            .unwrap();
        assert_eq!(result, StoreAction::Store(updates));
    }

    #[test]
    fn store_sets_value_at_final_key_index() {
        let index = 78;
        let key = Key::from_index_values(1, &[(31, index)]);
        let mut node = FullLeafNode {
            stem: key[..31].try_into().unwrap(),
            ..Default::default()
        };
        let value = Value::from_index_values(42, &[]);
        let update = KeyedUpdate::FullSlot { key, value };

        node.store(&update).unwrap();
        assert_eq!(node.values[index as usize], value);
    }

    #[test]
    fn store_with_non_matching_stem_returns_error() {
        let key = Key::from_index_values(1, &[(31, 78)]);
        let mut node = FullLeafNode::default();
        let value = Value::from_index_values(42, &[]);
        let update = KeyedUpdate::FullSlot { key, value };

        let result = node.store(&update);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::CorruptedState(_))
        ));
    }

    #[test]
    fn commitment_can_be_set_and_retrieved() {
        let mut node = FullLeafNode::default();
        assert_eq!(node.get_commitment(), VerkleCommitment::default());

        let mut new_commitment = VerkleCommitment::default();
        new_commitment.store(5, Value::from_index_values(4, &[]));

        node.set_commitment(new_commitment).unwrap();
        assert_eq!(node.get_commitment(), new_commitment);
    }
}
