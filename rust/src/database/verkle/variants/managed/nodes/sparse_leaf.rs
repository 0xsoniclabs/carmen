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
            KeyedUpdate, KeyedUpdateBatch,
            variants::managed::{
                VerkleNode, VerkleNodeId,
                commitment::{
                    OnDiskVerkleLeafCommitment, VerkleCommitment, VerkleCommitmentInput,
                    VerkleInnerCommitment, VerkleLeafCommitment,
                },
                nodes::{
                    ValueWithIndex, VerkleIdWithIndex, make_smallest_inner_node_for,
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

/// A sparsely populated leaf node in a managed Verkle trie.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseLeafNode<const N: usize> {
    pub stem: [u8; 31],
    pub values: [ValueWithIndex; N],
    pub commitment: VerkleLeafCommitment,
}

impl<const N: usize> SparseLeafNode<N> {
    /// Creates a sparse leaf node from existing stem, values, and commitment.
    /// Returns an error if there are more than N non-zero values.
    pub fn from_existing(
        stem: [u8; 31],
        values: &[ValueWithIndex],
        commitment: &VerkleLeafCommitment,
    ) -> BTResult<Self, Error> {
        let mut leaf = SparseLeafNode {
            stem,
            commitment: *commitment,
            ..Default::default()
        };

        // Insert values from previous leaf using get_slot_for to ensure no duplicate indices.
        for vwi in values {
            if vwi.item == Value::default() {
                continue;
            }
            let slot = ValueWithIndex::get_slot_for(&leaf.values, vwi.index).ok_or(
                Error::CorruptedState(format!(
                    "too many non-zero values to fit into sparse leaf of size {N}"
                )),
            )?;
            leaf.values[slot] = *vwi;
        }

        Ok(leaf)
    }

    /// Returns the values and stem of this leaf node as commitment input.
    // TODO: This should not have to pass 256 values: https://github.com/0xsoniclabs/sonic-admin/issues/384
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
        let mut values = [Value::default(); 256];
        for ValueWithIndex { index, item: value } in &self.values {
            values[*index as usize] = *value;
        }
        Ok(VerkleCommitmentInput::Leaf(values, self.stem))
    }
}

impl<const N: usize> Default for SparseLeafNode<N> {
    fn default() -> Self {
        let mut values = [ValueWithIndex::default(); N];
        values.iter_mut().enumerate().for_each(|(i, v)| {
            v.index = i as u8;
        });

        SparseLeafNode {
            stem: [0; 31],
            values,
            commitment: VerkleLeafCommitment::default(),
        }
    }
}

// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned)]
#[repr(C)]
pub struct OnDiskSparseLeafNode<const N: usize> {
    pub stem: [u8; 31],
    pub values: [ValueWithIndex; N],
    pub commitment: OnDiskVerkleLeafCommitment,
}

impl<const N: usize> TryFrom<OnDiskSparseLeafNode<N>> for SparseLeafNode<N> {
    type Error = BTError<Error>;

    fn try_from(on_disk: OnDiskSparseLeafNode<N>) -> Result<Self, Self::Error> {
        Ok(SparseLeafNode {
            stem: on_disk.stem,
            values: on_disk.values,
            commitment: VerkleLeafCommitment::try_from(on_disk.commitment)?,
        })
    }
}

impl<const N: usize> From<&SparseLeafNode<N>> for OnDiskSparseLeafNode<N> {
    fn from(node: &SparseLeafNode<N>) -> Self {
        OnDiskSparseLeafNode {
            stem: node.stem,
            values: node.values,
            commitment: OnDiskVerkleLeafCommitment::from(&node.commitment),
        }
    }
}

impl<const N: usize> DiskRepresentable for SparseLeafNode<N> {
    const DISK_REPR_SIZE: usize = std::mem::size_of::<OnDiskSparseLeafNode<N>>();

    fn from_disk_repr(
        read_into_buffer: impl FnOnce(&mut [u8]) -> BTResult<(), storage::Error>,
    ) -> BTResult<Self, storage::Error> {
        OnDiskSparseLeafNode::<N>::from_disk_repr(read_into_buffer).and_then(|on_disk| {
            SparseLeafNode::<N>::try_from(on_disk)
                .map_err(|e| storage::Error::DatabaseCorruption(e.to_string()).into())
        })
    }

    fn to_disk_repr(&'_ self) -> Cow<'_, [u8]> {
        Cow::Owned(OnDiskSparseLeafNode::from(self).to_disk_repr().into_owned())
    }
}

impl<const N: usize> ManagedTrieNode for SparseLeafNode<N> {
    type Union = VerkleNode;
    type Id = VerkleNodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, key: &Key, _depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {
        if key[..31] != self.stem[..] {
            return Ok(LookupResult::Value(Value::default()));
        }
        for ValueWithIndex { index, item: value } in &self.values {
            if *index == key[31] {
                return Ok(LookupResult::Value(*value));
            }
        }
        Ok(LookupResult::Value(Value::default()))
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

        if let Some(slots) = ValueWithIndex::required_slot_count_for(
            &self.values,
            updates.clone().split(31).map(|u| u.first_key()[31]),
        ) {
            // If the stems match but we don't have a free/matching slot, convert to a bigger leaf.
            return Ok(StoreAction::HandleTransform(make_smallest_leaf_node_for(
                slots,
                self.stem,
                &self.values,
                &self.commitment,
            )?));
        }

        // All updates fit into this sparse leaf.
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

        let slot = ValueWithIndex::get_slot_for(&self.values, key[31]).ok_or(
            Error::CorruptedState("no available slot for storing value in sparse leaf".to_owned()),
        )?;
        let prev_value = self.values[slot].item;
        self.values[slot].index = key[31];
        update.apply_to_value(&mut self.values[slot].item);

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

impl<const N: usize> NodeVisitor<SparseLeafNode<N>> for NodeCountVisitor {
    fn visit(&mut self, node: &SparseLeafNode<N>, level: u64) -> BTResult<(), Error> {
        self.count_node(
            level,
            "Leaf",
            node.values
                .iter()
                .filter(|value| value.item != Value::default())
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
                KeyedUpdateBatch,
                test_utils::FromIndexValues,
                variants::managed::nodes::{NodeAccess, VerkleManagedTrieNode, VerkleNodeKind},
            },
        },
        error::BTError,
        types::{TreeId, Value},
    };

    /// A random stem used by nodes created through [`make_leaf`].
    const STEM: [u8; 31] = [
        199, 138, 41, 113, 63, 133, 10, 244, 221, 149, 172, 110, 253, 27, 18, 76, 151, 202, 22, 80,
        37, 162, 130, 217, 143, 28, 241, 137, 212, 77, 126,
    ];

    /// The default value used by nodes created through [`make_leaf`].
    const LEAF_DEFAULT_VALUE: Value = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1,
    ];

    /// The index at which [`VALUE_1`] is stored in nodes created through [`make_leaf`].
    const INDEX_1: u8 = 99;

    /// A random value stored at [`INDEX_1`] in nodes created through [`make_leaf`].
    const VALUE_1: Value = [
        166, 44, 74, 233, 251, 79, 182, 249, 35, 197, 45, 50, 195, 162, 212, 116, 96, 23, 91, 167,
        136, 247, 205, 100, 142, 115, 103, 29, 77, 105, 53, 21,
    ];

    impl<const N: usize> VerkleManagedTrieNode<Value> for SparseLeafNode<N> {}

    impl<const N: usize> NodeAccess<Value> for SparseLeafNode<N> {
        /// Returns a reference to the specified slot (modulo N).
        fn access_slot(&mut self, slot: usize) -> &mut ValueWithIndex {
            &mut self.values[slot % N]
        }

        fn access_stem(&mut self) -> Option<&mut [u8; 31]> {
            Some(&mut self.stem)
        }

        fn get_commitment_input(&self) -> VerkleCommitmentInput {
            self.get_commitment_input().unwrap()
        }
    }

    /// Creates a leaf of size N with stem [`STEM`], the first slot set to [`INDEX_1`] and
    /// [`VALUE_1`], and all other slots set to [`LEAF_DEFAULT_VALUE`] and a unique index in
    /// ([`INDEX_1`]..[`INDEX_1`] + N).
    fn make_leaf<const N: usize>() -> SparseLeafNode<N> {
        let mut values = [ValueWithIndex::default(); N];
        values[0] = ValueWithIndex {
            index: INDEX_1,
            item: VALUE_1,
        };
        for (i, value) in values.iter_mut().enumerate().skip(1) {
            *value = ValueWithIndex {
                index: INDEX_1 + i as u8,
                item: LEAF_DEFAULT_VALUE,
            }
        }
        SparseLeafNode {
            stem: STEM,
            values,
            ..Default::default()
        }
    }

    #[rstest_reuse::template]
    #[rstest::rstest]
    #[case::leaf2(Box::new(make_leaf::<2>()) as Box<dyn VerkleManagedTrieNode<Value>>)]
    #[case::leaf7(Box::new(make_leaf::<7>()) as Box<dyn VerkleManagedTrieNode<Value>>)]
    #[case::leaf99(Box::new(make_leaf::<99>()) as Box<dyn VerkleManagedTrieNode<Value>>)]
    fn different_leaf_sizes(#[case] node: Box<dyn VerkleManagedTrieNode<Value>>) {}

    fn make_node_id() -> VerkleNodeId {
        VerkleNodeId::from_idx_and_node_kind(123, VerkleNodeKind::Leaf2) // The actual node kind is irrelevant for tests
    }

    #[test]
    fn sparse_leaf_node_default_returns_leaf_node_with_default_values_and_unique_indices() {
        const N: usize = 2;
        let node: SparseLeafNode<N> = SparseLeafNode::default();

        assert_eq!(node.stem, [0; 31]);
        assert_eq!(node.commitment, VerkleLeafCommitment::default());

        for (i, value) in node.values.iter().enumerate() {
            assert_eq!(value.index, i as u8);
            assert_eq!(value.item, Value::default());
        }
    }

    #[test]
    fn can_be_converted_to_and_from_on_disk_representation() {
        let mut original_node = make_leaf::<99>();
        // We deliberately only create a default commitment, since this type does
        // not preserve all of its fields when converting to/from on-disk representation.
        original_node.commitment = VerkleLeafCommitment::default();
        let disk_repr = original_node.to_disk_repr();
        let deserialized_node = SparseLeafNode::<99>::from_disk_repr(|buf| {
            buf.copy_from_slice(&disk_repr);
            Ok(())
        })
        .unwrap();
        assert_eq!(original_node, deserialized_node);
    }

    #[test]
    fn from_existing_copies_stem_and_values_and_commitment_correctly() {
        let mut commitment = VerkleLeafCommitment::default();
        commitment.store(2, VALUE_1);

        // Case 1: Contains an index that fits at the corresponding slot in a SparseLeaf<3>.
        {
            let values = [ValueWithIndex {
                index: 2,
                item: VALUE_1,
            }];
            let node = SparseLeafNode::<3>::from_existing(STEM, &values, &commitment).unwrap();
            assert_eq!(node.stem, STEM);
            assert_eq!(node.commitment, commitment);
            // Index is put into the correct slot
            assert_eq!(node.values[0].index, 0);
            assert_eq!(node.values[0].item, Value::default());
            assert_eq!(node.values[1].index, 1);
            assert_eq!(node.values[1].item, Value::default());
            assert_eq!(node.values[2], values[0]);
        }

        // Case 2: Index does not have a corresponding slot in a SparseLeaf<3>.
        {
            let values = [ValueWithIndex {
                index: 18,
                item: VALUE_1,
            }];
            let node = SparseLeafNode::<3>::from_existing(STEM, &values, &commitment).unwrap();
            // The value is put into the first available slot.
            // Note that the search begins at slot 18 % 3, which happens to be 0.
            assert_eq!(node.values[0], values[0]);
        }

        // Case 3: The first index does not fit, but the second one would have.
        {
            let values = [
                ValueWithIndex {
                    index: 18,
                    item: VALUE_1,
                },
                ValueWithIndex {
                    index: 0,
                    item: VALUE_1,
                },
                ValueWithIndex {
                    index: 1,
                    item: VALUE_1,
                },
            ];
            let node = SparseLeafNode::<3>::from_existing(STEM, &values, &commitment).unwrap();
            // Since the first slot is taken by index 18, index 0 and 1 get shifted back by one.
            assert_eq!(node.values[0], values[0]);
            assert_eq!(node.values[1], values[1]);
            assert_eq!(node.values[2], values[2]);
        }

        // Case 4: There are more values that can fit into a SparseLeaf<2>, but some of them are
        // zero and can be skipped.
        {
            let values = [
                ValueWithIndex {
                    index: 20,
                    item: VALUE_1,
                },
                ValueWithIndex {
                    index: 0,
                    item: Value::default(),
                },
                ValueWithIndex {
                    index: 1,
                    item: VALUE_1,
                },
            ];
            let node = SparseLeafNode::<2>::from_existing(STEM, &values, &commitment).unwrap();
            assert_eq!(node.values[0], values[0]);
            assert_eq!(node.values[1], values[2]);
        }
    }

    #[test]
    fn from_existing_returns_error_if_too_many_non_zero_values_are_provided() {
        let values = [
            ValueWithIndex {
                index: 0,
                item: VALUE_1,
            },
            ValueWithIndex {
                index: 1,
                item: VALUE_1,
            },
            ValueWithIndex {
                index: 2,
                item: VALUE_1,
            },
        ];
        let commitment = VerkleLeafCommitment::default();
        let result = SparseLeafNode::<2>::from_existing(STEM, &values, &commitment);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::CorruptedState(e)) if e.contains("too many non-zero values to fit into sparse leaf of size 2")
        ));
    }

    #[test]
    fn get_commitment_input_returns_values_and_stem() {
        let node = make_leaf::<2>();
        let mut expected_values = [Value::default(); 256];
        for ValueWithIndex { index, item: value } in &node.values {
            expected_values[*index as usize] = *value;
        }
        let result = node.get_commitment_input().unwrap();
        assert_eq!(
            result,
            VerkleCommitmentInput::Leaf(expected_values, node.stem)
        );
    }

    #[rstest_reuse::apply(different_leaf_sizes)]
    fn lookup_with_matching_stem_returns_value_at_final_key_index(
        #[case] node: Box<dyn VerkleManagedTrieNode<Value>>,
    ) {
        let key = [&STEM[..], &[INDEX_1]].concat().try_into().unwrap();
        let result = node.lookup(&key, 0).unwrap();
        assert_eq!(result, LookupResult::Value(VALUE_1));

        // Depth is irrelevant
        let result = node.lookup(&key, 42).unwrap();
        assert_eq!(result, LookupResult::Value(VALUE_1));

        // Mismatching stem returns default value
        let other_key = Key::from_index_values(7, &[]);
        let other_result = node.lookup(&other_key, 0).unwrap();
        assert_eq!(other_result, LookupResult::Value(Value::default()));

        // Other index has default value
        let other_key = [&STEM[..], &[INDEX_1 - 1]].concat().try_into().unwrap();
        let other_result = node.lookup(&other_key, 0).unwrap();
        assert_eq!(other_result, LookupResult::Value(Value::default()));
    }

    #[rstest_reuse::apply(different_leaf_sizes)]
    fn next_store_action_with_non_matching_stems_is_reparent(
        #[case] mut node: Box<dyn VerkleManagedTrieNode<Value>>,
    ) {
        let mut commitment = VerkleCommitment::Leaf(VerkleLeafCommitment::default());
        commitment.store(123, VALUE_1);
        node.set_commitment(commitment).unwrap();

        let divergence_at = 5;
        let key1: Key = [&STEM[..], &[0u8]].concat().try_into().unwrap();
        let mut key2: Key = [&STEM[..], &[0u8]].concat().try_into().unwrap();
        key2[divergence_at] = 57;
        let self_id = make_node_id();
        let update = KeyedUpdateBatch::from_key_value_pairs(&[(key1, VALUE_1), (key2, VALUE_1)]);
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

    #[rstest_reuse::apply(different_leaf_sizes)]
    fn sparse_leaf_with_dirty_commitment_is_marked_as_changed_in_new_parent(
        #[case] mut node: Box<dyn VerkleManagedTrieNode<Value>>,
        #[values(true, false)] leaf_is_dirty: bool,
    ) {
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

    #[rstest_reuse::apply(different_leaf_sizes)]
    fn next_store_action_with_non_matching_stem_returns_parent_large_enough_for_all_updates(
        #[case] mut node: Box<dyn VerkleManagedTrieNode<Value>>,
    ) {
        let depth = 1;

        // The update has 9 batches that diverge at depth 1, but one of them has the same key as
        // the leaf's stem, so an inner node with at least 9 slots is needed
        *node.access_stem().unwrap() = [0; _];
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
        node.access_stem().unwrap()[depth] = 10;

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

    #[rstest_reuse::apply(different_leaf_sizes)]
    fn next_store_action_with_matching_stems_is_store_if_enough_usable_slots_exists(
        #[case] node: Box<dyn VerkleManagedTrieNode<Value>>,
    ) {
        let mut node = node;
        let index = 142;
        node.access_slot(4).index = index; // one slot can be overwritten
        node.access_slot(5).item = Value::default(); // one free slot
        let key1: Key = [&STEM[..], &[index]].concat().try_into().unwrap();
        let key2: Key = [&STEM[..], &[index + 1]].concat().try_into().unwrap();
        let update = KeyedUpdateBatch::from_key_value_pairs(&[(key1, VALUE_1), (key2, VALUE_1)]);
        let result = node
            .next_store_action(update.clone(), 0, make_node_id())
            .unwrap();
        assert_eq!(result, StoreAction::Store(update));
    }

    #[rstest_reuse::apply(different_leaf_sizes)]
    fn next_store_action_with_matching_stems_is_transform_to_bigger_leaf_if_not_enough_usable_slots(
        #[case] node: Box<dyn VerkleManagedTrieNode<Value>>,
    ) {
        let mut node = node;
        node.access_slot(5).item = Value::default(); // one free slot
        let mut commitment = VerkleCommitment::Leaf(VerkleLeafCommitment::default());
        commitment.store(7, VALUE_1);
        node.set_commitment(commitment).unwrap();

        let index = 250;
        let key1: Key = [&STEM[..], &[index]].concat().try_into().unwrap();
        let key2: Key = [&STEM[..], &[index + 1]].concat().try_into().unwrap();
        let update = KeyedUpdateBatch::from_key_value_pairs(&[(key1, VALUE_1), (key2, VALUE_1)]);
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
                    bigger_leaf.get_commitment_input().unwrap(),
                    node.get_commitment_input()
                );
                // The commitment is copied over
                assert_eq!(bigger_leaf.get_commitment(), node.get_commitment());
            }
            _ => panic!("expected HandleTransform"),
        }
    }

    #[rstest_reuse::apply(different_leaf_sizes)]
    fn store_sets_value_at_final_key_index(#[case] node: Box<dyn VerkleManagedTrieNode<Value>>) {
        let mut node = node;
        let index = 78;
        node.access_slot(3).index = index;
        let key = [&STEM[..], &[index]].concat().try_into().unwrap();
        let value = Value::from_index_values(42, &[]);
        let update = KeyedUpdate::FullSlot { key, value };
        node.store(&update).unwrap();
        let commitment_input = node.get_commitment_input();
        match commitment_input {
            VerkleCommitmentInput::Leaf(values, _) => {
                assert_eq!(values[index as usize], value);
            }
            _ => panic!("expected Leaf commitment input"),
        }
    }

    #[rstest_reuse::apply(different_leaf_sizes)]
    fn store_with_non_matching_stem_returns_error(
        #[case] node: Box<dyn VerkleManagedTrieNode<Value>>,
    ) {
        let mut node = node;
        let key = Key::from_index_values(1, &[(31, 78)]);
        let update = KeyedUpdate::FullSlot {
            key,
            value: VALUE_1,
        };
        let result = node.store(&update);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::CorruptedState(e)) if e.contains("called store on a leaf with non-matching stem")
        ));
    }

    #[rstest_reuse::apply(different_leaf_sizes)]
    fn store_returns_error_if_no_free_slot(#[case] node: Box<dyn VerkleManagedTrieNode<Value>>) {
        let mut node = node;
        let key = [&STEM[..], &[INDEX_1 - 1]].concat().try_into().unwrap();
        let update = KeyedUpdate::FullSlot {
            key,
            value: VALUE_1,
        };
        let result = node.store(&update);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::CorruptedState(e)) if e.contains("no available slot for storing value in sparse leaf")
        ));
    }

    #[test]
    fn commitment_can_be_set_and_retrieved() {
        let mut node = SparseLeafNode::<7>::default();
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
