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
        verkle::{
            KeyedUpdate, KeyedUpdateBatch,
            variants::managed::{
                VerkleNode,
                commitment::{VerkleCommitment, VerkleCommitmentInput},
                nodes::{
                    ManagedInnerNode, VerkleIdWithIndex, VerkleNodeKind, id::VerkleNodeId,
                    make_smallest_inner_node_for,
                },
            },
        },
        visitor::NodeVisitor,
    },
    error::{BTResult, Error},
    statistics::node_count::NodeCountVisitor,
    types::{Key, ToNodeKind},
};

/// An inner node in a managed Verkle trie.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned)]
#[repr(C)]
pub struct SparseInnerNode<const N: usize> {
    pub children: [VerkleIdWithIndex; N],
    pub commitment: VerkleCommitment,
}

impl<const N: usize> SparseInnerNode<N> {
    /// Creates a sparse inner node from existing children and commitment.
    /// Returns an error if there are more than N non-zero children.
    pub fn from_existing(
        children: &[VerkleIdWithIndex],
        commitment: VerkleCommitment,
    ) -> BTResult<Self, Error> {
        let mut inner = SparseInnerNode {
            commitment,
            ..Default::default()
        };

        // Insert values from previous leaf using get_slot_for to ensure no duplicate indices.
        for vwi in children {
            if vwi.item == VerkleNodeId::default() {
                continue;
            }
            let slot = VerkleIdWithIndex::get_slot_for(&inner.children, vwi.index).ok_or(
                Error::CorruptedState(format!(
                    "too many non-zero IDs to fit into sparse inner of size {N}"
                )),
            )?;
            inner.children[slot] = *vwi;
        }

        Ok(inner)
    }

    /// Returns the children of this inner node as commitment input.
    // TODO: This should not have to pass 256 values: https://github.com/0xsoniclabs/sonic-admin/issues/384
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
        let mut values = [VerkleNodeId::default(); 256];
        for VerkleIdWithIndex { index, item: value } in &self.children {
            values[*index as usize] = *value;
        }
        Ok(VerkleCommitmentInput::Inner(values))
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

impl<const N: usize> ManagedTrieNode for SparseInnerNode<N> {
    type Union = VerkleNode;
    type Id = VerkleNodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, key: &Key, depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {
        let slot = VerkleIdWithIndex::get_slot_for(&self.children, key[depth as usize]);
        match slot {
            Some(slot) if self.children[slot].index == key[depth as usize] => {
                Ok(LookupResult::Node(self.children[slot].item))
            }
            _ => Ok(LookupResult::Node(VerkleNodeId::default())),
        }
    }

    fn next_store_action(
        &self,
        key: KeyedUpdate,
        depth: u8,
        _self_id: Self::Id,
    ) -> BTResult<StoreAction<Self::Id, Self::Union>, Error> {
        let index = key[depth as usize];
        let slot = VerkleIdWithIndex::get_slot_for(&self.children, index);

        match slot {
            None => Ok(StoreAction::HandleTransform(make_smallest_inner_node_for(
                N + 1,
                &self.children,
                self.commitment,
            )?)),
            Some(slot) => Ok(StoreAction::Descend {
                index: index as usize,
                id: self.children[slot].item,
            }),
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

impl<const N: usize> ManagedInnerNode for SparseInnerNode<N> {
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
        types::TreeId,
    };

    fn make_inner<const N: usize>() -> SparseInnerNode<N> {
        SparseInnerNode::<N> {
            children: array::from_fn(|i| VerkleIdWithIndex {
                index: i as u8,
                item: VerkleNodeId::from_idx_and_node_kind(i as u64, VerkleNodeKind::Inner9),
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
    fn from_existing_copies_children_and_commitment_correctly() {
        let ID = VerkleNodeId::from_idx_and_node_kind(42, VerkleNodeKind::Inner9);
        let mut commitment = VerkleCommitment::default();
        commitment.modify_child(2);

        // Case 1: Contains an index that fits at the corresponding slot in a SparseInnerNode<3>.
        {
            let children = [VerkleIdWithIndex { index: 2, item: ID }];
            let node = SparseInnerNode::<3>::from_existing(&children, commitment).unwrap();
            assert_eq!(node.commitment, commitment);
            // Index is put into the correct slot
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
            let node = SparseInnerNode::<3>::from_existing(&children, commitment).unwrap();
            // The value is put into the first available slot.
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
            let node = SparseInnerNode::<3>::from_existing(&children, commitment).unwrap();
            // Since the first slot is taken by index 18, index 0 and 1 get shifted back by one.
            assert_eq!(node.children[0], children[0]);
            assert_eq!(node.children[1], children[1]);
            assert_eq!(node.children[2], children[2]);
        }

        // Case 4: There are more values that can fit into a SparseInnerNode<2>, but some of them
        // are zero and can be skipped.
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
            let node = SparseInnerNode::<2>::from_existing(&children, commitment).unwrap();
            assert_eq!(node.children[0], children[0]);
            assert_eq!(node.children[1], children[2]);
        }
    }

    #[test]
    fn from_existing_returns_error_if_too_many_non_zero_values_are_provided() {
        let ID = VerkleNodeId::from_idx_and_node_kind(42, VerkleNodeKind::Inner9);
        let children = [
            VerkleIdWithIndex { index: 0, item: ID },
            VerkleIdWithIndex { index: 1, item: ID },
            VerkleIdWithIndex { index: 2, item: ID },
        ];
        let commitment = VerkleCommitment::default();
        let result = SparseInnerNode::<2>::from_existing(&children, commitment);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::CorruptedState(e)) if e.contains("too many non-zero IDs to fit into sparse inner of size 2")));
    }

    #[test]
    fn sparse_inner_node_default_returns_inner_node_with_all_children_set_to_empty_node_id() {
        let node: SparseInnerNode<3> = SparseInnerNode::default();
        assert_eq!(node.commitment, VerkleCommitment::default());
        assert_eq!(
            node.children,
            [
                VerkleIdWithIndex {
                    index: 0,
                    item: VerkleNodeId::default(),
                },
                VerkleIdWithIndex {
                    index: 1,
                    item: VerkleNodeId::default(),
                },
                VerkleIdWithIndex {
                    index: 2,
                    item: VerkleNodeId::default(),
                },
            ]
        );
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
                VerkleNodeKind::Inner9
            ))
        );

        // Lookup an index that exists but it's empty
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
        let result = node
            .next_store_action(
                &key,
                1,
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Inner9), // Irrelevant
            )
            .unwrap();
        assert_eq!(
            result,
            StoreAction::Descend {
                index: 2,
                id: VerkleNodeId::from_idx_and_node_kind(2, VerkleNodeKind::Inner9)
            }
        );
    }

    #[rstest_reuse::apply(different_inner_sizes)]
    fn next_store_action_with_no_available_slot_is_handle_transform(
        #[case] node: Box<dyn VerkleManagedTrieNode<VerkleNodeId>>,
    ) {
        let key = Key::from_index_values(1, &[(1, 250)]);
        let result = node
            .next_store_action(
                &key,
                1,
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Inner9), // Irrelevant
            )
            .unwrap();
        match result {
            StoreAction::HandleTransform(bigger_inner) => {
                assert_eq!(
                    bigger_inner
                        .next_store_action(&key, 1, VerkleNodeId::default())
                        .unwrap(),
                    StoreAction::Descend {
                        index: 250,
                        id: VerkleNodeId::default()
                    }
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
        let new_id = VerkleNodeId::from_idx_and_node_kind(999, VerkleNodeKind::Inner9);
        node.replace_child(&key, 1, new_id).unwrap();
        let result = node.lookup(&key, 1).unwrap();
        assert_eq!(result, LookupResult::Node(new_id));

        // Non-existing index but with available slot
        node.access_slot(1).item = VerkleNodeId::default(); // Free up slot at index 1
        let key = Key::from_index_values(1, &[(1, 250)]);
        let new_id = VerkleNodeId::from_idx_and_node_kind(1000, VerkleNodeKind::Inner9);
        node.replace_child(&key, 1, new_id).unwrap();
        let result = node.lookup(&key, 1).unwrap();
        assert_eq!(result, LookupResult::Node(new_id));
    }

    #[rstest_reuse::apply(different_inner_sizes)]
    fn replace_child_returns_error_if_no_slot_available(
        #[case] mut node: Box<dyn VerkleManagedTrieNode<VerkleNodeId>>,
    ) {
        let key = Key::from_index_values(1, &[(1, 250)]);
        let new_id = VerkleNodeId::from_idx_and_node_kind(1000, VerkleNodeKind::Inner9);
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

        fn get_commitment_input(&self) -> VerkleCommitmentInput {
            self.get_commitment_input().unwrap()
        }
    }
}
