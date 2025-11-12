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
            FullLeafNode, InnerNode, VerkleNode, VerkleNodeId,
            commitment::{VerkleCommitment, VerkleCommitmentInput},
        },
    },
    error::{BTResult, Error},
    types::{Key, Value},
};

/// A value of a leaf node in a managed Verkle trie, together with its index.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned,
)]
#[repr(C)]
pub struct ValueWithIndex {
    /// The index of the value in the leaf node.
    pub index: u8,
    /// The value stored in the leaf node.
    pub value: Value,
}

/// A sparsely populated leaf node in a managed Verkle trie.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
#[repr(C)]
pub struct SparseLeafNode<const N: usize> {
    pub stem: [u8; 31],
    pub values: [ValueWithIndex; N],
    pub commitment: VerkleCommitment,
}

impl<const N: usize> SparseLeafNode<N> {
    /// Returns the values and stem of this leaf node as commitment input.
    // TODO: This should not have to pass 256 values: https://github.com/0xsoniclabs/sonic-admin/issues/384
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
        let mut values = [Value::default(); 256];
        for ValueWithIndex { index, value } in &self.values {
            values[*index as usize] = *value;
        }
        Ok(VerkleCommitmentInput::Leaf(values, self.stem))
    }

    /// Returns a slot for storing a value with the given index, or `None` if no such slot exists.
    /// A slot is suitable if it either already holds the given index, or if it is empty
    /// (i.e., holds the default value).
    fn get_slot_for(&self, index: u8) -> Option<usize> {
        let mut empty_slot = None;
        // We always do a linear search over all values to ensure that we never hold the same index
        // twice in different slots. By starting the search at the given index we are very likely
        // to find the matching slot immediately in practice (if index < N).
        for (i, vwi) in self
            .values
            .iter()
            .enumerate()
            .cycle()
            .skip(index as usize)
            .take(N)
        {
            if vwi.index == index {
                return Some(i);
            } else if empty_slot.is_none() && vwi.value == Value::default() {
                empty_slot = Some(i);
            }
        }
        empty_slot
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
            commitment: VerkleCommitment::default(),
        }
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
        for ValueWithIndex { index, value } in &self.values {
            if *index == key[31] {
                return Ok(LookupResult::Value(*value));
            }
        }
        Ok(LookupResult::Value(Value::default()))
    }

    fn next_store_action(
        &self,
        key: &Key,
        depth: u8,
        self_id: Self::Id,
    ) -> BTResult<StoreAction<Self::Id, Self::Union>, Error> {
        // If key does not match the stem, we have to introduce a new inner node.
        if key[..31] != self.stem[..] {
            let index = key[depth as usize];
            let mut inner = InnerNode::default();
            inner.children[index as usize] = self_id;
            return Ok(StoreAction::HandleReparent(VerkleNode::Inner(Box::new(
                inner,
            ))));
        }

        // If we have a free/matching slot, we can store the value directly.
        if self.get_slot_for(key[31]).is_some() {
            return Ok(StoreAction::Store {
                index: key[31] as usize,
            });
        }

        // If the stems match but we don't have a free/matching slot, convert to a full leaf.
        let new_leaf = FullLeafNode {
            stem: self.stem,
            values: {
                let mut values = [Value::default(); 256];
                for ValueWithIndex { index, value } in &self.values {
                    values[*index as usize] = *value;
                }
                values
            },
            commitment: self.commitment,
        };
        Ok(StoreAction::HandleTransform(VerkleNode::Leaf256(Box::new(
            new_leaf,
        ))))
    }

    fn store(&mut self, key: &Key, value: &Value) -> BTResult<Value, Error> {
        if self.stem[..] != key[..31] {
            return Err(Error::CorruptedState(
                "called store on a leaf with non-matching stem".to_owned(),
            )
            .into());
        }

        let slot = self.get_slot_for(key[31]).ok_or(Error::CorruptedState(
            "no available slot for storing value in sparse leaf".to_owned(),
        ))?;
        let prev_value = self.values[slot].value;
        self.values[slot] = ValueWithIndex {
            index: key[31],
            value: *value,
        };

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        database::{
            managed_trie::TrieCommitment,
            verkle::{test_utils::FromIndexValues, variants::managed::nodes::VerkleNodeKind},
        },
        error::BTError,
        types::{TreeId, Value},
    };

    const STEM: [u8; 31] = [
        4u8, 32, 15, 6, 7, 8, 9, 10, 11, 12, 18, 14, 19, 29, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 32, 29, 30, 31, 80, 72,
    ];

    const DEFAULT_VALUE: Value = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1,
    ];

    const INDEX1: u8 = 99;
    const VALUE1: Value = [
        0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31,
    ];

    /// Creates a leaf of size N with stem [`STEM`], the first slot set to [`INDEX1`] and
    /// [`VALUE1`], and all other slots set to [`DEFAULT_VALUE`] and a unique index in
    /// ([`INDEX1`]..[`INDEX1`] + N).
    fn make_leaf<const N: usize>() -> SparseLeafNode<N> {
        let mut values = [ValueWithIndex::default(); N];
        values[0] = ValueWithIndex {
            index: INDEX1,
            value: VALUE1,
        };
        for (i, value) in values.iter_mut().enumerate().skip(1) {
            *value = ValueWithIndex {
                index: INDEX1 + i as u8,
                value: DEFAULT_VALUE,
            }
        }
        SparseLeafNode {
            stem: STEM[..31].try_into().unwrap(),
            values,
            ..Default::default()
        }
    }

    trait VerkleManagedTrieNode:
        ManagedTrieNode<Union = VerkleNode, Id = VerkleNodeId, Commitment = VerkleCommitment>
        + GenericHelperTrait
    {
    }

    impl<const N: usize> VerkleManagedTrieNode for SparseLeafNode<N> {}

    /// Helper trait to interact with generic leaf nodes in rstest tests.
    trait GenericHelperTrait {
        fn access_slot(&mut self, slot: usize) -> &mut ValueWithIndex;
        fn get_commitment_input(&self) -> VerkleCommitmentInput;
    }

    impl<const N: usize> GenericHelperTrait for SparseLeafNode<N> {
        /// Returns a reference to the specified slot (modulo N).
        fn access_slot(&mut self, slot: usize) -> &mut ValueWithIndex {
            &mut self.values[slot % N]
        }

        fn get_commitment_input(&self) -> VerkleCommitmentInput {
            self.get_commitment_input().unwrap()
        }
    }

    #[rstest_reuse::template]
    #[rstest::rstest]
    #[case::leaf2(Box::new(make_leaf::<2>()) as Box<dyn VerkleManagedTrieNode>)]
    #[case::leaf7(Box::new(make_leaf::<7>()) as Box<dyn VerkleManagedTrieNode>)]
    #[case::leaf99(Box::new(make_leaf::<99>()) as Box<dyn VerkleManagedTrieNode>)]
    fn different_leaf_sizes(#[case] node: Box<dyn VerkleManagedTrieNode>) {}

    fn make_node_id() -> VerkleNodeId {
        VerkleNodeId::from_idx_and_node_kind(123, VerkleNodeKind::Leaf2) // The actual node kind is irrelevant for tests
    }

    #[test]
    fn sparse_leaf_node_default_returns_leaf_node_with_default_values_and_unique_indices() {
        const N: usize = 2;
        let node: SparseLeafNode<N> = SparseLeafNode::default();

        assert_eq!(node.stem, [0; 31]);
        assert_eq!(node.commitment, VerkleCommitment::default());

        for (i, value) in node.values.iter().enumerate() {
            assert_eq!(value.index, i as u8);
            assert_eq!(value.value, Value::default());
        }
    }

    #[test]
    fn get_commitment_input_returns_values_and_stem() {
        let node = make_leaf::<2>();
        let mut expected_values = [Value::default(); 256];
        for ValueWithIndex { index, value } in &node.values {
            expected_values[*index as usize] = *value;
        }
        let result = node.get_commitment_input().unwrap();
        assert_eq!(
            result,
            VerkleCommitmentInput::Leaf(expected_values, node.stem)
        );
    }

    #[test]
    fn get_slot_returns_slot_with_matching_index_or_empty_slot() {
        let mut node = make_leaf::<7>();

        // Matching index
        let slot = node.get_slot_for(INDEX1);
        assert_eq!(slot, Some(0));

        // Matching index has precedence over empty slot
        node.values[1].value = Value::default();
        let slot = node.get_slot_for(INDEX1 + 3);
        assert_eq!(slot, Some(3));

        // No matching index, so we return first empty slot
        let slot = node.get_slot_for(250);
        assert_eq!(slot, Some(1));

        // No matching index and no empty slot
        node.values[1].value = VALUE1;
        let slot = node.get_slot_for(250);
        assert_eq!(slot, None);
    }

    #[rstest_reuse::apply(different_leaf_sizes)]
    fn lookup_with_matching_stem_returns_value_at_final_key_index(
        #[case] node: Box<dyn VerkleManagedTrieNode>,
    ) {
        let key = [&STEM[..31], &[INDEX1]].concat().try_into().unwrap();
        let result = node.lookup(&key, 0).unwrap();
        assert_eq!(result, LookupResult::Value(VALUE1));

        // Depth is irrelevant
        let result = node.lookup(&key, 42).unwrap();
        assert_eq!(result, LookupResult::Value(VALUE1));

        // Mismatching stem returns default value
        let other_key = Key::from_index_values(7, &[]);
        let other_result = node.lookup(&other_key, 0).unwrap();
        assert_eq!(other_result, LookupResult::Value(Value::default()));

        // Other index has default value
        let other_key = Key::from_index_values(1, &[(31, INDEX1 + 1)]);
        let other_result = node.lookup(&other_key, 0).unwrap();
        assert_eq!(other_result, LookupResult::Value(Value::default()));
    }

    #[rstest_reuse::apply(different_leaf_sizes)]
    fn next_store_action_with_non_matching_stem_is_reparent(
        #[case] node: Box<dyn VerkleManagedTrieNode>,
    ) {
        let depth = 10;
        let index = 78;
        let key = Key::from_index_values(1, &[(depth, index)]);
        let self_id = make_node_id();

        let result = node.next_store_action(&key, depth as u8, self_id).unwrap();
        match result {
            StoreAction::HandleReparent(VerkleNode::Inner(inner)) => {
                assert_eq!(inner.children[index as usize], self_id);
            }
            _ => panic!("expected HandleReparent with inner node"),
        }
    }

    #[rstest_reuse::apply(different_leaf_sizes)]
    fn next_store_action_with_matching_stem_is_store_if_matching_slot_exists(
        #[case] node: Box<dyn VerkleManagedTrieNode>,
    ) {
        let mut node = node;
        let index = 142;
        node.access_slot(4).index = index;
        let key: Key = [&STEM[..31], &[index]].concat().try_into().unwrap();
        let result = node.next_store_action(&key, 0, make_node_id()).unwrap();
        assert_eq!(
            result,
            StoreAction::Store {
                index: index as usize
            }
        );
    }

    #[rstest_reuse::apply(different_leaf_sizes)]
    fn next_store_action_with_matching_stem_is_store_if_slot_with_default_value_exists(
        #[case] node: Box<dyn VerkleManagedTrieNode>,
    ) {
        let mut node = node;
        let index = 200;
        node.access_slot(5).value = Value::default();
        let key: Key = [&STEM[..31], &[index]].concat().try_into().unwrap();
        let result = node.next_store_action(&key, 0, make_node_id()).unwrap();
        assert_eq!(
            result,
            StoreAction::Store {
                index: index as usize
            }
        );
    }

    #[rstest_reuse::apply(different_leaf_sizes)]
    fn next_store_action_with_matching_stem_is_transform_to_bigger_leaf_if_no_free_slots(
        #[case] node: Box<dyn VerkleManagedTrieNode>,
    ) {
        let mut node = node;
        let mut commitment = VerkleCommitment::default();
        commitment.store(7, VALUE1);
        node.set_commitment(commitment).unwrap();

        let index = 250;
        let key: Key = [&STEM[..31], &[index]].concat().try_into().unwrap();
        let result = node.next_store_action(&key, 0, make_node_id()).unwrap();
        match result {
            StoreAction::HandleTransform(bigger_leaf) => {
                // This new leaf is big enough to store the value
                assert_eq!(
                    bigger_leaf.next_store_action(&key, 0, make_node_id()),
                    Ok(StoreAction::Store {
                        index: index as usize
                    })
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
    fn store_sets_value_at_final_key_index(#[case] node: Box<dyn VerkleManagedTrieNode>) {
        let mut node = node;
        let index = 78;
        node.access_slot(3).index = index;
        let key = [&STEM[..31], &[index]].concat().try_into().unwrap();
        let value = Value::from_index_values(42, &[]);

        node.store(&key, &value).unwrap();
        let commitment_input = node.get_commitment_input();
        match commitment_input {
            VerkleCommitmentInput::Leaf(values, _) => {
                assert_eq!(values[index as usize], value);
            }
            _ => panic!("expected Leaf commitment input"),
        }
    }

    #[rstest_reuse::apply(different_leaf_sizes)]
    fn store_with_non_matching_stem_returns_error(#[case] node: Box<dyn VerkleManagedTrieNode>) {
        let mut node = node;
        let key = Key::from_index_values(1, &[(31, 78)]);
        let result = node.store(&key, &VALUE1);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::CorruptedState(e)) if e.contains("called store on a leaf with non-matching stem")
        ));
    }

    #[rstest_reuse::apply(different_leaf_sizes)]
    fn store_returns_error_if_no_free_slot(#[case] node: Box<dyn VerkleManagedTrieNode>) {
        let mut node = node;
        let key = [&STEM[..31], &[INDEX1 - 1]].concat().try_into().unwrap();
        let result = node.store(&key, &VALUE1);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::CorruptedState(e)) if e.contains("no available slot for storing value in sparse leaf")
        ));
    }

    #[test]
    fn commitment_can_be_set_and_retrieved() {
        let mut node = SparseLeafNode::<7>::default();
        assert_eq!(node.get_commitment(), VerkleCommitment::default());

        let mut new_commitment = VerkleCommitment::default();
        new_commitment.store(5, Value::from_index_values(4, &[]));

        node.set_commitment(new_commitment).unwrap();
        assert_eq!(node.get_commitment(), new_commitment);
    }
}
