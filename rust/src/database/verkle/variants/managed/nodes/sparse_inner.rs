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
        managed_trie::{DescendAction, LookupResult, ManagedTrieNode, StoreAction},
        verkle::{
            KeyedUpdateBatch,
            variants::managed::{
                VerkleNode,
                commitment::{OnDiskVerkleCommitment, VerkleCommitment, VerkleCommitmentInput},
                nodes::{
                    VerkleIdWithIndex, VerkleManagedInnerNode, VerkleNodeKind, id::VerkleNodeId,
                    make_smallest_inner_node_for,
                },
            },
        },
        visitor::NodeVisitor,
    },
    error::{BTResult, Error},
    statistics::node_count::NodeCountVisitor,
    types::{DiskRepresentable, Key, ToNodeKind},
};

/// An inner node in a managed Verkle trie.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseInnerNode<const N: usize> {
    pub children: [VerkleIdWithIndex; N],
    pub commitment: VerkleCommitment,
}

impl<const N: usize> SparseInnerNode<N> {
    /// Creates a sparse inner node from existing children and commitment.
    /// Returns an error if there are more than N non-default children.
    pub fn from_existing(
        children: &[VerkleIdWithIndex],
        commitment: &VerkleCommitment,
    ) -> BTResult<Self, Error> {
        let mut inner = SparseInnerNode {
            commitment: *commitment,
            ..Default::default()
        };

        // Insert values from previous node using get_slot_for to ensure no duplicate indices.
        for iwi in children {
            if iwi.item == VerkleNodeId::default() {
                continue;
            }
            let slot = VerkleIdWithIndex::get_slot_for(&inner.children, iwi.index).ok_or(
                Error::CorruptedState(format!(
                    "too many non-default IDs to fit into sparse inner of size {N}"
                )),
            )?;
            inner.children[slot] = *iwi;
        }

        Ok(inner)
    }

    /// Returns the children of this inner node as commitment input.
    // TODO: This should not have to pass 256 IDs: https://github.com/0xsoniclabs/sonic-admin/issues/384
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
        let mut children = [VerkleNodeId::default(); 256];
        for VerkleIdWithIndex { index, item: value } in &self.children {
            children[*index as usize] = *value;
        }
        Ok(VerkleCommitmentInput::Inner(children))
    }
}

impl<const N: usize> Default for SparseInnerNode<N> {
    fn default() -> Self {
        let mut children = [VerkleIdWithIndex::default(); N];
        children.iter_mut().enumerate().for_each(|(i, v)| {
            v.index = i as u8;
        });

        SparseInnerNode {
            children,
            commitment: VerkleCommitment::default(),
        }
    }
}

// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned)]
#[repr(C)]
pub struct OnDiskSparseInnerNode<const N: usize> {
    pub children: [VerkleIdWithIndex; N],
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
        let slot = VerkleIdWithIndex::get_slot_for(&self.children, key[depth as usize]);
        match slot {
            Some(slot) => Ok(LookupResult::Node(self.children[slot].item)),
            _ => Ok(LookupResult::Node(VerkleNodeId::default())),
        }
    }

    fn next_store_action<'a>(
        &self,
        updates: KeyedUpdateBatch<'a>,
        depth: u8,
        _self_id: Self::Id,
    ) -> BTResult<StoreAction<'a, Self::Id, Self::Union>, Error> {
        let slots = VerkleIdWithIndex::required_slot_count_for(
            &self.children,
            updates
                .borrowed()
                .split(depth)
                .map(|u| u.first_key()[depth as usize]),
        );

        if let Some(slots) = slots {
            Ok(StoreAction::HandleTransform(make_smallest_inner_node_for(
                slots,
                &self.children,
                &self.commitment,
            )?))
        } else {
            let mut descent_actions = Vec::new();
            for sub_updates in updates.split(depth) {
                let index = sub_updates.first_key()[depth as usize];
                let slot = VerkleIdWithIndex::get_slot_for(&self.children, index).ok_or(
                    Error::CorruptedState(
                        "no available slot for storing value in sparse inner node".to_owned(),
                    ),
                )?;
                descent_actions.push(DescendAction {
                    id: self.children[slot].item,
                    updates: sub_updates,
                });
            }
            Ok(StoreAction::Descend(descent_actions))
        }
    }

    fn replace_child(&mut self, key: &Key, depth: u8, new: VerkleNodeId) -> BTResult<(), Error> {
        let index = key[depth as usize];
        match VerkleIdWithIndex::get_slot_for(&self.children, index) {
            Some(slot) => {
                self.children[slot] = VerkleIdWithIndex { index, item: new };
                Ok(())
            }
            _ => Err(Error::CorruptedState(
                "no slot found for replacing child in sparse inner".to_owned(),
            )
            .into()),
        }
    }

    fn get_commitment(&self) -> Self::Commitment {
        self.commitment
    }

    fn set_commitment(&mut self, commitment: Self::Commitment) -> BTResult<(), Error> {
        self.commitment = commitment;
        Ok(())
    }
}

impl<const N: usize> NodeVisitor<SparseInnerNode<N>> for NodeCountVisitor {
    fn visit(&mut self, node: &SparseInnerNode<N>, level: u64) -> BTResult<(), Error> {
        self.count_node(
            level,
            "Inner",
            node.children
                .iter()
                .filter(|child| child.item.to_node_kind().unwrap() != VerkleNodeKind::Empty)
                .count() as u64,
        );
        Ok(())
    }
}

impl<const N: usize> VerkleManagedInnerNode for SparseInnerNode<N> {
    fn iter_children(&self) -> Box<dyn Iterator<Item = VerkleIdWithIndex> + '_> {
        Box::new(self.children.iter().copied())
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
                test_utils::FromIndexValues,
                variants::managed::nodes::{NodeAccess, VerkleManagedTrieNode, VerkleNodeKind},
            },
        },
        error::BTError,
        types::{TreeId, Value},
    };

    const TEST_CHILD_NODE_KIND: VerkleNodeKind = VerkleNodeKind::Inner9;

    fn make_inner<const N: usize>() -> SparseInnerNode<N> {
        SparseInnerNode::<N> {
            children: array::from_fn(|i| VerkleIdWithIndex {
                index: i as u8,
                item: VerkleNodeId::from_idx_and_node_kind(i as u64, TEST_CHILD_NODE_KIND),
            }),
            commitment: VerkleCommitment::default(),
        }
    }

    #[rstest_reuse::template]
    #[rstest::rstest]
    #[case::inner3(Box::new(make_inner::<3>()) as Box<dyn VerkleManagedTrieNode<VerkleNodeId>>)]
    #[case::inner7(Box::new(make_inner::<7>()) as Box<dyn VerkleManagedTrieNode<VerkleNodeId>>)]
    #[case::inner99(Box::new(make_inner::<99>()) as Box<dyn VerkleManagedTrieNode<VerkleNodeId>>)]
    fn different_inner_sizes(#[case] node: Box<dyn VerkleManagedTrieNode<VerkleNodeId>>) {}

    #[test]
    fn sparse_inner_node_default_returns_inner_node_with_default_ids_and_unique_indices() {
        const N: usize = 2;
        let node: SparseInnerNode<N> = SparseInnerNode::default();

        assert_eq!(node.commitment, VerkleCommitment::default());

        for (i, value) in node.children.iter().enumerate() {
            assert_eq!(value.index, i as u8);
            assert_eq!(value.item, VerkleNodeId::default());
        }
    }

    #[test]
    fn can_be_converted_to_and_from_on_disk_representation() {
        let mut original_node = make_inner::<99>();
        original_node.commitment = {
            // We deliberately only create a default commitment, since this type does
            // not preserve all of its fields when converting to/from on-disk representation.
            let mut commitment = VerkleCommitment::default();
            commitment.test_only_mark_as_clean();
            commitment
        };
        let disk_repr = original_node.to_disk_repr();
        let deserialized_node = SparseInnerNode::<99>::from_disk_repr::<()>(|buf| {
            buf.copy_from_slice(&disk_repr);
            Ok(())
        })
        .unwrap();
        assert_eq!(original_node, deserialized_node);
    }

    #[test]
    fn from_existing_copies_children_and_commitment_correctly() {
        let ID = VerkleNodeId::from_idx_and_node_kind(42, TEST_CHILD_NODE_KIND);
        let mut commitment = VerkleCommitment::default();
        commitment.modify_child(2);

        // Case 1: Contains an index that fits at the corresponding slot in a SparseInnerNode<3>.
        {
            let children = [VerkleIdWithIndex { index: 2, item: ID }];
            let node = SparseInnerNode::<3>::from_existing(&children, &commitment).unwrap();
            assert_eq!(node.commitment, commitment);
            // The child is put into the correct slot.
            assert_eq!(node.children[0].index, 0);
            assert_eq!(node.children[0].item, VerkleNodeId::default());
            assert_eq!(node.children[1].index, 1);
            assert_eq!(node.children[1].item, VerkleNodeId::default());
            assert_eq!(node.children[2], children[0]);
        }

        // Case 2: Index does not have a corresponding slot in a SparseInnerNode<3>.
        {
            let children = [VerkleIdWithIndex {
                index: 18,
                item: ID,
            }];
            let node = SparseInnerNode::<3>::from_existing(&children, &commitment).unwrap();
            // The child is put into the first available slot.
            // Note that the search begins at slot 18 % 3, which happens to be 0.
            assert_eq!(node.children[0], children[0]);
        }

        // Case 3: The first index does not fit, but the second one would have.
        {
            let children = [
                VerkleIdWithIndex {
                    index: 18,
                    item: ID,
                },
                VerkleIdWithIndex { index: 0, item: ID },
                VerkleIdWithIndex { index: 1, item: ID },
            ];
            let node = SparseInnerNode::<3>::from_existing(&children, &commitment).unwrap();
            // Since the first slot is taken by index 18, index 0 and 1 get shifted back by one.
            assert_eq!(node.children[0], children[0]);
            assert_eq!(node.children[1], children[1]);
            assert_eq!(node.children[2], children[2]);
        }

        // Case 4: There are more children that can fit into a SparseInnerNode<2>, but some of them
        // are the default ID and can be skipped.
        {
            let children = [
                VerkleIdWithIndex {
                    index: 20,
                    item: ID,
                },
                VerkleIdWithIndex {
                    index: 0,
                    item: VerkleNodeId::default(),
                },
                VerkleIdWithIndex { index: 1, item: ID },
            ];
            let node = SparseInnerNode::<2>::from_existing(&children, &commitment).unwrap();
            assert_eq!(node.children[0], children[0]);
            assert_eq!(node.children[1], children[2]);
        }
    }

    #[test]
    fn from_existing_returns_error_if_too_many_non_default_children_are_provided() {
        let ID = VerkleNodeId::from_idx_and_node_kind(42, TEST_CHILD_NODE_KIND);
        let children = [
            VerkleIdWithIndex { index: 0, item: ID },
            VerkleIdWithIndex { index: 1, item: ID },
            VerkleIdWithIndex { index: 2, item: ID },
        ];
        let commitment = VerkleCommitment::default();
        let result = SparseInnerNode::<2>::from_existing(&children, &commitment);

        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::CorruptedState(e)) if e.contains("too many non-default IDs to fit into sparse inner of size 2")));
    }

    #[test]
    fn get_commitment_input_returns_children() {
        let node = make_inner::<16>();
        let mut expected_children = [VerkleNodeId::default(); 256];
        for VerkleIdWithIndex { index, item } in &node.children {
            expected_children[*index as usize] = *item;
        }
        let result = node.get_commitment_input().unwrap();
        assert_eq!(result, VerkleCommitmentInput::Inner(expected_children));
    }

    #[rstest_reuse::apply(different_inner_sizes)]
    fn lookup_returns_id_of_child_at_key_index(
        #[case] mut node: Box<dyn VerkleManagedTrieNode<VerkleNodeId>>,
    ) {
        // Lookup an index that exists
        let key = Key::from_index_values(1, &[(1, 2)]);
        let result = node.lookup(&key, 1).unwrap();
        assert_eq!(
            result,
            LookupResult::Node(VerkleNodeId::from_idx_and_node_kind(
                2,
                TEST_CHILD_NODE_KIND
            ))
        );

        // Lookup an index that exists but is empty
        node.access_slot(2).item = VerkleNodeId::default();
        let result = node.lookup(&key, 1).unwrap();
        assert_eq!(result, LookupResult::Node(VerkleNodeId::default()));

        // Lookup an index that does not exist
        let key = Key::from_index_values(1, &[(1, 250)]);
        let result = node.lookup(&key, 1).unwrap();
        assert_eq!(result, LookupResult::Node(VerkleNodeId::default()));
    }

    #[rstest_reuse::apply(different_inner_sizes)]
    fn next_store_action_with_available_slot_is_descend(
        #[case] node: Box<dyn VerkleManagedTrieNode<VerkleNodeId>>,
    ) {
        let key = Key::from_index_values(1, &[(1, 2)]);
        let updates = KeyedUpdateBatch::from_key_value_pairs(&[(key, Value::default())]);
        let result = node
            .next_store_action(
                updates.clone(),
                1,
                VerkleNodeId::default(), // Irrelevant
            )
            .unwrap();
        assert_eq!(
            result,
            StoreAction::Descend(vec![DescendAction {
                id: VerkleNodeId::from_idx_and_node_kind(2, TEST_CHILD_NODE_KIND),
                updates
            }])
        );
    }

    #[rstest_reuse::apply(different_inner_sizes)]
    fn next_store_action_with_no_available_slot_is_handle_transform(
        #[case] node: Box<dyn VerkleManagedTrieNode<VerkleNodeId>>,
    ) {
        let key = Key::from_index_values(1, &[(1, 250)]);
        let updates = KeyedUpdateBatch::from_key_value_pairs(&[(key, Value::default())]);
        let result = node
            .next_store_action(
                updates.clone(),
                1,
                VerkleNodeId::default(), // Irrelevant
            )
            .unwrap();
        match result {
            StoreAction::HandleTransform(bigger_inner) => {
                assert_eq!(
                    bigger_inner
                        .next_store_action(updates.clone(), 1, VerkleNodeId::default())
                        .unwrap(),
                    StoreAction::Descend(vec![DescendAction {
                        id: VerkleNodeId::default(),
                        updates
                    }])
                );
                // It contains all previous values
                assert_eq!(
                    bigger_inner.get_commitment_input().unwrap(),
                    node.get_commitment_input()
                );
                // The commitment is copied over
                assert_eq!(bigger_inner.get_commitment(), node.get_commitment());
            }
            _ => panic!("expected HandleTransform action"),
        }
    }

    #[rstest_reuse::apply(different_inner_sizes)]
    fn replace_child_sets_child_id_at_key_index(
        #[case] mut node: Box<dyn VerkleManagedTrieNode<VerkleNodeId>>,
    ) {
        // Existing index
        let key = Key::from_index_values(1, &[(1, 2)]);
        let new_id = VerkleNodeId::from_idx_and_node_kind(999, TEST_CHILD_NODE_KIND);
        node.replace_child(&key, 1, new_id).unwrap();
        let result = node.lookup(&key, 1).unwrap();
        assert_eq!(result, LookupResult::Node(new_id));

        // Non-existing index but with available slot
        node.access_slot(1).item = VerkleNodeId::default(); // Free up slot 1
        let key = Key::from_index_values(1, &[(1, 250)]);
        let new_id = VerkleNodeId::from_idx_and_node_kind(1000, TEST_CHILD_NODE_KIND);
        node.replace_child(&key, 1, new_id).unwrap();
        let result = node.lookup(&key, 1).unwrap();
        assert_eq!(result, LookupResult::Node(new_id));
    }

    #[rstest_reuse::apply(different_inner_sizes)]
    fn replace_child_returns_error_if_no_slot_available(
        #[case] mut node: Box<dyn VerkleManagedTrieNode<VerkleNodeId>>,
    ) {
        let key = Key::from_index_values(1, &[(1, 250)]);
        let new_id = VerkleNodeId::from_idx_and_node_kind(1000, TEST_CHILD_NODE_KIND);
        let result = node.replace_child(&key, 1, new_id);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::CorruptedState(e)) if e.contains("no slot found for replacing child in sparse inner")
        ));
    }

    #[test]
    fn commitment_can_be_set_and_retrieved() {
        let mut node = SparseInnerNode::<3>::default();
        assert_eq!(node.get_commitment(), VerkleCommitment::default());

        let mut new_commitment = VerkleCommitment::default();
        new_commitment.modify_child(5);

        node.set_commitment(new_commitment).unwrap();
        assert_eq!(node.get_commitment(), new_commitment);
    }

    impl<const N: usize> NodeAccess<VerkleNodeId> for SparseInnerNode<N> {
        /// Returns a reference to the specified slot (modulo N).
        fn access_slot(&mut self, slot: usize) -> &mut VerkleIdWithIndex {
            &mut self.children[slot % N]
        }

        fn access_stem(&mut self) -> Option<&mut [u8; 31]> {
            None
        }

        fn get_commitment_input(&self) -> VerkleCommitmentInput {
            self.get_commitment_input().unwrap()
        }
    }
}
