// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{array, borrow::Cow};

use zerocopy::{FromBytes, Immutable, IntoBytes, Unaligned};

use crate::{
    database::{
        managed_trie::{LookupResult, ManagedTrieNode, StoreAction},
        verkle::{
            KeyedUpdate, KeyedUpdateBatch,
            variants::managed::{
                FullLeafNode, SparseLeafNode, VerkleNode, VerkleNodeId,
                commitment::{
                    OnDiskVerkleLeafCommitment, VerkleCommitment, VerkleCommitmentInput,
                    VerkleInnerCommitment, VerkleLeafCommitment,
                },
                nodes::{
                    ItemWithIndex, ValueWithIndex, VerkleIdWithIndex, make_smallest_inner_node_for,
                    make_smallest_leaf_node_for,
                },
            },
        },
        visitor::NodeVisitor,
    },
    error::{BTError, BTResult, Error},
    statistics::node_count::NodeCountVisitor,
    storage,
    types::{DiskRepresentable, Key, Value},
};

/// The leaf delta node is a space-saving optimization for managed Verkle trie archives:
/// Archives update tries non-destructively using copy-on-write.
/// Instead of copying all children each time, delta nodes instead store a set of differences
/// for specific indices, compared to some earlier base leaf node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeafDeltaNode {
    pub stem: [u8; 31],
    pub values: [Value; 256],
    // We can not use `Value::default()` to indicate an unused slot because a value could be
    // non-default in the base and then be updated to default in the delta. Therefore we need an
    // explicit `Option`.
    pub values_delta: [ItemWithIndex<Option<Value>>; Self::DELTA_SIZE],
    pub base_node_id: VerkleNodeId,
    pub commitment: VerkleLeafCommitment,
}

impl LeafDeltaNode {
    /// The number of values for which a delta can be stored.
    /// This represents a tradeoff between how large the delta node itself is on disk,
    /// versus how often a base leaf node has to be introduced.
    pub const DELTA_SIZE: usize = 10;

    pub fn from_full_leaf(base_node: &FullLeafNode, base_node_id: VerkleNodeId) -> Self {
        LeafDeltaNode {
            stem: base_node.stem,
            values: base_node.values,
            values_delta: array::from_fn(|i| ItemWithIndex {
                index: i as u8,
                item: None,
            }),
            base_node_id,
            commitment: base_node.commitment,
        }
    }

    pub fn from_sparse_leaf<const N: usize>(
        base_node: &SparseLeafNode<N>,
        base_node_id: VerkleNodeId,
    ) -> Self {
        LeafDeltaNode {
            stem: base_node.stem,
            values: {
                let mut values = [Value::default(); 256];
                for ItemWithIndex { index, item } in &base_node.values {
                    values[*index as usize] = *item;
                }
                values
            },
            values_delta: array::from_fn(|i| ItemWithIndex {
                index: i as u8,
                item: None,
            }),
            base_node_id,
            commitment: base_node.commitment,
        }
    }

    /// Returns the values and stem of this leaf node as commitment input.
    // TODO: This should not have to pass 256 values: https://github.com/0xsoniclabs/sonic-admin/issues/384
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
        let mut values = self.values;
        for ItemWithIndex { index, item: value } in &self.values_delta {
            if let Some(value) = value {
                values[*index as usize] = *value;
            }
        }
        Ok(VerkleCommitmentInput::Leaf(values, self.stem))
    }
}

// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned)]
#[repr(C)]
pub struct OnDiskLeafDeltaNode {
    pub stem: [u8; 31],
    pub values_delta: [ItemWithIndex<[u8; 33]>; LeafDeltaNode::DELTA_SIZE],
    pub base_node_id: VerkleNodeId,
    pub commitment: OnDiskVerkleLeafCommitment,
}

// This still needs to load the base leaf node to reconstruct the `values`.
impl TryFrom<OnDiskLeafDeltaNode> for LeafDeltaNode {
    type Error = BTError<Error>;

    fn try_from(on_disk: OnDiskLeafDeltaNode) -> Result<Self, Self::Error> {
        Ok(LeafDeltaNode {
            stem: on_disk.stem,
            values: [Value::default(); 256],
            values_delta: array::from_fn(|i| ItemWithIndex {
                index: on_disk.values_delta[i].index,
                item: if on_disk.values_delta[i].item[0] == 0 {
                    None
                } else {
                    Some(on_disk.values_delta[i].item[1..].try_into().unwrap())
                },
            }),
            base_node_id: on_disk.base_node_id,
            commitment: VerkleLeafCommitment::try_from(on_disk.commitment)?,
        })
    }
}

impl From<&LeafDeltaNode> for OnDiskLeafDeltaNode {
    fn from(node: &LeafDeltaNode) -> Self {
        OnDiskLeafDeltaNode {
            stem: node.stem,
            values_delta: array::from_fn(|i| ItemWithIndex {
                index: node.values_delta[i].index,
                item: if let Some(value) = node.values_delta[i].item {
                    let mut item = [0u8; 33];
                    item[0] = 1;
                    item[1..].copy_from_slice(&value);
                    item
                } else {
                    [0u8; 33]
                },
            }),
            base_node_id: node.base_node_id,
            commitment: OnDiskVerkleLeafCommitment::from(&node.commitment),
        }
    }
}

impl DiskRepresentable for LeafDeltaNode {
    const DISK_REPR_SIZE: usize = std::mem::size_of::<OnDiskLeafDeltaNode>();

    fn from_disk_repr(
        read_into_buffer: impl FnOnce(&mut [u8]) -> BTResult<(), storage::Error>,
    ) -> BTResult<Self, storage::Error> {
        OnDiskLeafDeltaNode::from_disk_repr(read_into_buffer).and_then(|on_disk| {
            LeafDeltaNode::try_from(on_disk)
                .map_err(|e| storage::Error::DatabaseCorruption(e.to_string()).into())
        })
    }

    fn to_disk_repr(&'_ self) -> Cow<'_, [u8]> {
        Cow::Owned(OnDiskLeafDeltaNode::from(self).to_disk_repr().into_owned())
    }
}

impl ManagedTrieNode for LeafDeltaNode {
    type Union = VerkleNode;
    type Id = VerkleNodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, key: &Key, _depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {
        if key[..31] != self.stem[..] {
            return Ok(LookupResult::Value(Value::default()));
        }
        let slot = ItemWithIndex::get_slot_for(&self.values_delta, key[31]);
        if let Some(slot) = slot
            && let slot_item = self.values_delta[slot]
            && slot_item.index == key[31]
            && let Some(item) = slot_item.item
        {
            Ok(LookupResult::Value(item))
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
            let dirty_index = (!self.commitment.is_clean()).then_some(index);
            let inner = make_smallest_inner_node_for(
                slots,
                &[self_child],
                &VerkleInnerCommitment::from_leaf(&self.commitment, dirty_index),
            )?;
            return Ok(StoreAction::HandleReparent(inner));
        }

        if ItemWithIndex::required_slot_count_for(
            &self.values_delta,
            updates.clone().split(31).map(|u| u.first_key()[31]),
        ) > Self::DELTA_SIZE
        {
            // If the stems match but we don't have a free/matching slot, convert to a bigger leaf.
            let mut values: [_; 256] = array::from_fn(|i| ValueWithIndex {
                index: i as u8,
                item: self.values[i],
            });
            for ItemWithIndex { index, item } in &self.values_delta {
                if let Some(item) = item {
                    values[*index as usize].item = *item;
                }
            }
            let slots = ItemWithIndex::required_slot_count_for(
                &values,
                updates.clone().split(31).map(|u| u.first_key()[31]),
            );
            return Ok(StoreAction::HandleTransform(make_smallest_leaf_node_for(
                slots,
                self.stem,
                &values,
                &self.commitment,
            )?));
        }

        // All updates fit into this leaf delta node.
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

        let slot = ItemWithIndex::get_slot_for(&self.values_delta, key[31]).ok_or(
            Error::CorruptedState(
                "no available slot for storing value in leaf delta node".to_owned(),
            ),
        )?;
        let prev_value = self.values_delta[slot]
            .item
            .unwrap_or(self.values[key[31] as usize]);

        self.values_delta[slot].index = key[31];
        let value = &mut self.values_delta[slot].item;
        if value.is_none() {
            *value = Some(self.values[key[31] as usize]);
        }
        update.apply_to_value(value.as_mut().unwrap());

        Ok(prev_value)
    }

    fn get_commitment(&self) -> Self::Commitment {
        VerkleCommitment::Leaf(self.commitment)
    }

    fn set_commitment(&mut self, commitment: Self::Commitment) -> BTResult<(), Error> {
        self.commitment = commitment.into_leaf()?;
        Ok(())
    }
}

impl NodeVisitor<LeafDeltaNode> for NodeCountVisitor {
    fn visit(&mut self, node: &LeafDeltaNode, level: u64) -> BTResult<(), Error> {
        self.count_node(
            level,
            "Leaf",
            node.values
                .iter()
                .filter(|value| **value != Value::default())
                .count() as u64
                + node
                    .values_delta
                    .iter()
                    .filter(|value| value.item.is_some())
                    .count() as u64,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
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

    /// A random stem used by nodes created through [`make_empty_leaf_delta`].
    const STEM: [u8; 31] = [
        199, 138, 41, 113, 63, 133, 10, 244, 221, 149, 172, 110, 253, 27, 18, 76, 151, 202, 22, 80,
        37, 162, 130, 217, 143, 28, 241, 137, 212, 77, 126,
    ];

    /// Creates an empty leaf delta node with stem [`STEM`].
    fn make_empty_leaf_delta() -> LeafDeltaNode {
        LeafDeltaNode {
            stem: STEM,
            values: [Value::default(); 256],
            values_delta: array::from_fn(|i| ItemWithIndex {
                index: i as u8,
                item: None,
            }),
            base_node_id: VerkleNodeId::default(),
            commitment: VerkleLeafCommitment::default(),
        }
    }

    fn make_node_id() -> VerkleNodeId {
        VerkleNodeId::from_idx_and_node_kind(123, VerkleNodeKind::Leaf2) // The actual node kind is irrelevant for tests
    }

    #[test]
    fn can_be_converted_to_and_from_on_disk_representation() {
        let mut original_node = make_empty_leaf_delta();
        original_node.values_delta[3] = ItemWithIndex {
            index: 200,
            item: Some([1; 32]),
        };
        original_node.base_node_id = make_node_id();
        // We deliberately only create a default commitment, since this type does
        // not preserve all of its fields when converting to/from on-disk representation.
        original_node.commitment = VerkleLeafCommitment::default();

        let disk_repr = original_node.to_disk_repr();
        let deserialized_node = LeafDeltaNode::from_disk_repr(|buf| {
            buf.copy_from_slice(&disk_repr);
            Ok(())
        })
        .unwrap();
        assert_eq!(original_node, deserialized_node);
    }

    #[test]
    fn from_full_leaf_copies_stem_and_values_and_commitment_and_sets_id_of_base_node() {
        let mut base_leaf = FullLeafNode {
            stem: STEM,
            values: [Value::default(); 256],
            commitment: VerkleLeafCommitment::default(),
        };
        base_leaf.values[2] = [1; 32];
        base_leaf.commitment.store(1, [1; 32]);

        let full_leaf_id = VerkleNodeId::from_idx_and_node_kind(1, VerkleNodeKind::Leaf256);

        let node = LeafDeltaNode::from_full_leaf(&base_leaf, full_leaf_id);
        assert_eq!(node.stem, base_leaf.stem);
        assert_eq!(node.commitment, base_leaf.commitment);
        assert_eq!(node.values, base_leaf.values);
        assert_eq!(node.base_node_id, full_leaf_id);
        assert_eq!(
            node.values_delta,
            array::from_fn(|i| ItemWithIndex {
                index: i as u8,
                item: None,
            })
        );
    }

    #[test]
    fn from_sparse_leaf_copies_stem_and_values_and_commitment_and_sets_id_of_base_node() {
        let mut base_leaf = SparseLeafNode::<9> {
            stem: STEM,
            values: array::from_fn(|i| ItemWithIndex {
                index: i as u8,
                item: Value::default(),
            }),
            commitment: VerkleLeafCommitment::default(),
        };
        base_leaf.commitment.store(255, [1; 32]);
        base_leaf.values[0] = ItemWithIndex {
            index: 255,
            item: [1; 32],
        };

        let full_leaf_id = VerkleNodeId::from_idx_and_node_kind(1, VerkleNodeKind::Leaf256);

        let node = LeafDeltaNode::from_sparse_leaf(&base_leaf, full_leaf_id);
        assert_eq!(node.stem, base_leaf.stem);
        assert_eq!(node.commitment, base_leaf.commitment);
        assert_eq!(node.values[255], [1; 32]);
        assert_eq!(node.values[..255], [Value::default(); 255]);
        assert_eq!(node.base_node_id, full_leaf_id);
        assert_eq!(
            node.values_delta,
            array::from_fn(|i| ItemWithIndex {
                index: i as u8,
                item: None,
            })
        );
    }

    #[test]
    fn get_commitment_input_returns_values_and_stem() {
        let mut node = make_empty_leaf_delta();
        node.values[10] = [10; 32];
        node.values[11] = [10; 32];
        node.values_delta[2] = ItemWithIndex {
            index: 11,
            item: Some([11; 32]),
        };

        let mut expected_values = [Value::default(); 256];
        expected_values[10] = [10; 32];
        expected_values[11] = [11; 32];
        let result = node.get_commitment_input().unwrap();
        assert_eq!(
            result,
            VerkleCommitmentInput::Leaf(expected_values, node.stem)
        );
    }

    #[test]
    fn lookup_with_matching_stem_returns_value_at_final_key_index() {
        let idx1 = 3;
        let value1 = [3; 32];
        let idx2 = 11;
        let value2 = [11; 32];

        // Create leaf delta node with idx1 set to value1 and idx2 set to value1 in base values and
        // idx2 set to value2 in delta.
        let mut node = make_empty_leaf_delta();
        node.values[idx1] = value1;
        node.values[idx2] = value1;
        node.values_delta[8] = ItemWithIndex {
            index: idx2 as u8,
            item: Some(value2),
        };

        // Lookup idx from base values which is not in delta.
        let key = [&STEM[..], &[idx1 as u8]].concat().try_into().unwrap();
        let result = node.lookup(&key, 0).unwrap();
        assert_eq!(result, LookupResult::Value(value1));

        // Lookup idx from delta values which overwrites value in base values.
        let key = [&STEM[..], &[idx2 as u8]].concat().try_into().unwrap();
        let result = node.lookup(&key, 0).unwrap();
        assert_eq!(result, LookupResult::Value(value2));

        // Depth is irrelevant.
        let result = node.lookup(&key, 42).unwrap();
        assert_eq!(result, LookupResult::Value(value2));

        // Mismatching stem returns default value.
        let other_key = Key::from_index_values(7, &[]);
        let other_result = node.lookup(&other_key, 0).unwrap();
        assert_eq!(other_result, LookupResult::Value(Value::default()));

        // Other index has default value.
        let other_key = [&STEM[..], &[0]].concat().try_into().unwrap();
        let other_result = node.lookup(&other_key, 0).unwrap();
        assert_eq!(other_result, LookupResult::Value(Value::default()));
    }

    #[test]
    fn next_store_action_with_non_matching_stems_is_reparent() {
        let mut node = make_empty_leaf_delta();

        // Set a non-default commitment to verify it is copied over.
        let mut commitment = VerkleCommitment::Leaf(VerkleLeafCommitment::default());
        commitment.store(123, [1; 32]);
        node.set_commitment(commitment).unwrap();

        let divergence_at = 5;
        let key1: Key = [&STEM[..], &[0]].concat().try_into().unwrap();
        let mut key2: Key = key1;
        key2[divergence_at] = 57;
        let self_id = make_node_id();
        let update = KeyedUpdateBatch::from_key_value_pairs(&[(key1, [1; 32]), (key2, [2; 32])]);
        let result = node
            .next_store_action(update, divergence_at as u8, self_id)
            .unwrap();
        match result {
            StoreAction::HandleReparent(VerkleNode::Inner9(inner)) => {
                let slot =
                    VerkleIdWithIndex::get_slot_for(&inner.children, STEM[divergence_at]).unwrap();
                assert_eq!(inner.children[slot].item, self_id);
                // Newly created inner node has commitment of the leaf.
                assert_ne!(
                    inner.get_commitment(),
                    VerkleCommitment::Leaf(VerkleLeafCommitment::default())
                );
                assert_eq!(inner.get_commitment().commitment(), commitment.commitment());
            }
            _ => panic!("expected HandleReparent with inner node"),
        }
    }

    #[rstest::rstest]
    fn sparse_leaf_with_dirty_commitment_is_marked_as_changed_in_new_parent(
        #[values(true, false)] leaf_is_dirty: bool,
    ) {
        let mut node = make_empty_leaf_delta();

        let mut commitment = VerkleLeafCommitment::default();
        if leaf_is_dirty {
            commitment.store(5, [0u8; 32]); // Arbitrary
        }
        node.set_commitment(VerkleCommitment::Leaf(commitment))
            .unwrap();
        let updates = KeyedUpdateBatch::from_key_value_pairs(&[([99; 32], Value::default())]);
        match node
            .next_store_action(updates, 0, VerkleNodeId::default())
            .unwrap()
        {
            StoreAction::HandleReparent(VerkleNode::Inner9(inner)) => {
                assert_eq!(
                    inner.get_commitment().index_changed(STEM[0] as usize),
                    leaf_is_dirty
                );
            }
            _ => panic!("expected HandleReparent with inner node"),
        }
    }

    #[test]
    fn next_store_action_with_non_matching_stem_returns_parent_large_enough_for_all_updates() {
        let mut node = make_empty_leaf_delta();

        let depth = 1;

        // The update has 9 batches that diverge at depth 1, but one of them has the same key as
        // the leaf's stem, so an inner node with at least 9 slots is needed
        node.stem = [0; _];
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
    fn next_store_action_with_matching_stems_is_store_if_enough_usable_slots_exists() {
        let index = 142;

        // Create node with one slot that can be overwritten and one free slot.
        let mut node = make_empty_leaf_delta();
        node.values_delta = array::from_fn(|i| ItemWithIndex {
            index: i as u8,
            item: Some(Value::default()),
        });
        node.values_delta[4].index = index; // one slot can be overwritten
        node.values_delta[5].item = None; // one free slot
        let key1: Key = [&STEM[..], &[index]].concat().try_into().unwrap();
        let key2: Key = [&STEM[..], &[index + 1]].concat().try_into().unwrap();
        let update = KeyedUpdateBatch::from_key_value_pairs(&[(key1, [1; 32]), (key2, [2; 32])]);
        let result = node
            .next_store_action(update.clone(), 0, make_node_id())
            .unwrap();
        assert_eq!(result, StoreAction::Store(update));
    }

    #[test]
    fn next_store_action_with_matching_stems_is_transform_to_bigger_leaf_if_not_enough_usable_slots()
     {
        // Create node with only one free slot.
        let mut node = make_empty_leaf_delta();
        node.values_delta = array::from_fn(|i| ItemWithIndex {
            index: i as u8,
            item: Some(Value::default()),
        });
        node.values_delta[0].item = None;

        // Set a non-default commitment to verify it is copied over.
        let mut commitment = VerkleCommitment::Leaf(VerkleLeafCommitment::default());
        commitment.store(0, [1; 32]);
        node.set_commitment(commitment).unwrap();

        // Try to store two values, which requires transforming to a bigger leaf.
        let index = 250;
        let key1: Key = [&STEM[..], &[index]].concat().try_into().unwrap();
        let key2: Key = [&STEM[..], &[index + 1]].concat().try_into().unwrap();
        let update = KeyedUpdateBatch::from_key_value_pairs(&[(key1, [1; 32]), (key2, [2; 32])]);
        let result = node
            .next_store_action(update.clone(), 0, make_node_id())
            .unwrap();
        match result {
            StoreAction::HandleTransform(bigger_leaf) => {
                // This new leaf is big enough to store the value
                assert_eq!(
                    bigger_leaf.next_store_action(update.clone(), 0, make_node_id()),
                    Ok(StoreAction::Store(update))
                );
                // It contains all previous values
                assert_eq!(
                    bigger_leaf.get_commitment_input(),
                    node.get_commitment_input()
                );
                // The commitment is copied over
                assert_eq!(bigger_leaf.get_commitment(), node.get_commitment());
            }
            _ => panic!("expected HandleTransform"),
        }
    }

    #[test]
    fn store_sets_value_at_final_key_index() {
        let mut node = make_empty_leaf_delta();
        let index = 78;
        node.values_delta[3].index = index;
        let key = [&STEM[..], &[index]].concat().try_into().unwrap();
        let value = Value::from_index_values(42, &[]);
        let update = KeyedUpdate::FullSlot { key, value };
        node.store(&update).unwrap();
        let commitment_input = node.get_commitment_input().unwrap();
        match commitment_input {
            VerkleCommitmentInput::Leaf(values, _) => {
                assert_eq!(values[index as usize], value);
            }
            _ => panic!("expected Leaf commitment input"),
        }
    }

    #[test]
    fn store_with_non_matching_stem_returns_error() {
        let mut node = make_empty_leaf_delta();
        let update = KeyedUpdate::FullSlot {
            key: Key::from_index_values(1, &[(31, 78)]),
            value: [1; 32],
        };
        let result = node.store(&update);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::CorruptedState(e)) if e.contains("called store on a leaf with non-matching stem")
        ));
    }

    #[test]
    fn store_returns_error_if_no_free_slot() {
        // Create node with no free slots.
        let mut node = make_empty_leaf_delta();
        node.values_delta = array::from_fn(|i| ItemWithIndex {
            index: i as u8,
            item: Some(Value::default()),
        });

        let update = KeyedUpdate::FullSlot {
            key: [&STEM[..], &[255]].concat().try_into().unwrap(),
            value: [1; 32],
        };
        let result = node.store(&update);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::CorruptedState(e)) if e.contains("no available slot for storing value in leaf delta node")
        ));
    }

    #[test]
    fn commitment_can_be_set_and_retrieved() {
        let mut node = make_empty_leaf_delta();
        assert_eq!(
            node.get_commitment(),
            VerkleCommitment::Leaf(VerkleLeafCommitment::default())
        );

        let mut new_commitment = VerkleCommitment::Leaf(VerkleLeafCommitment::default());
        new_commitment.store(5, Value::from_index_values(4, &[]));

        node.set_commitment(new_commitment).unwrap();
        assert_eq!(node.get_commitment(), new_commitment);
    }
}
