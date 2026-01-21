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
        managed_trie::{DescendAction, LookupResult, ManagedTrieNode, StoreAction},
        verkle::{
            KeyedUpdateBatch,
            variants::managed::{
                FullInnerNode, VerkleNode,
                commitment::{
                    OnDiskVerkleInnerCommitment, VerkleCommitment, VerkleCommitmentInput,
                    VerkleInnerCommitment,
                },
                nodes::{
                    VerkleIdWithIndex, VerkleManagedInnerNode, VerkleNodeKind, id::VerkleNodeId,
                },
            },
        },
        visitor::NodeVisitor,
    },
    error::{BTError, BTResult, Error},
    statistics::node_count::NodeCountVisitor,
    storage,
    types::{DiskRepresentable, Key, ToNodeKind},
};

/// The inner delta node is a space-saving optimization for managed Verkle trie archives:
/// Archives update tries non-destructively using copy-on-write. This results in the upper parts
/// of a trie, which mostly consist of full inner nodes, to be copied frequently.
/// Instead of copying all 256 children each time, delta nodes instead store a set of differences
/// for specific indices, compared to some earlier full inner node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InnerDeltaNode {
    pub children: [VerkleNodeId; 256],
    pub children_delta: [VerkleIdWithIndex; Self::DELTA_SIZE],
    pub full_inner_node_id: VerkleNodeId,
    pub commitment: VerkleInnerCommitment,
}

impl InnerDeltaNode {
    /// The number of child indices for which a delta can be stored.
    /// This represents a tradeoff between how large the delta node itself is on disk,
    /// versus how often a full inner node has to be introduced.
    /// Preliminary benchmarking has shown 10 to be a good number.
    pub const DELTA_SIZE: usize = 10;

    /// Creates a [`InnerDeltaNode`] from a [`FullInnerNode`] and its id.
    pub fn from_full_inner(inner_node: &FullInnerNode, inner_node_id: VerkleNodeId) -> Self {
        InnerDeltaNode {
            children: inner_node.children,
            children_delta: array::from_fn(|i| VerkleIdWithIndex {
                index: i as u8,
                item: VerkleNodeId::default(),
            }),
            full_inner_node_id: inner_node_id,
            commitment: inner_node.commitment,
        }
    }

    /// Returns the children of this inner node as commitment input.
    // TODO: This should not have to pass 256 IDs: https://github.com/0xsoniclabs/sonic-admin/issues/384
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
        let mut children = self.children;
        for VerkleIdWithIndex { index, item } in self.children_delta {
            if item != VerkleNodeId::default() {
                children[index as usize] = item;
            }
        }
        Ok(VerkleCommitmentInput::Inner(children))
    }
}

// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned)]
#[repr(C)]
struct OnDiskInnerDeltaNode {
    pub children_delta: [VerkleIdWithIndex; InnerDeltaNode::DELTA_SIZE],
    pub full_inner_node_id: VerkleNodeId,
    pub commitment: OnDiskVerkleInnerCommitment,
}

// This still needs to load the full inner node to reconstruct the `children`.
impl TryFrom<OnDiskInnerDeltaNode> for InnerDeltaNode {
    type Error = BTError<Error>;

    fn try_from(delta_node: OnDiskInnerDeltaNode) -> Result<Self, Self::Error> {
        Ok(InnerDeltaNode {
            children: [VerkleNodeId::default(); 256],
            children_delta: delta_node.children_delta,
            full_inner_node_id: delta_node.full_inner_node_id,
            commitment: VerkleInnerCommitment::try_from(delta_node.commitment)?,
        })
    }
}

impl From<&InnerDeltaNode> for OnDiskInnerDeltaNode {
    fn from(node: &InnerDeltaNode) -> Self {
        OnDiskInnerDeltaNode {
            children_delta: node.children_delta,
            full_inner_node_id: node.full_inner_node_id,
            commitment: OnDiskVerkleInnerCommitment::from(&node.commitment),
        }
    }
}

impl DiskRepresentable for InnerDeltaNode {
    const DISK_REPR_SIZE: usize = std::mem::size_of::<OnDiskInnerDeltaNode>();

    fn from_disk_repr(
        read_into_buffer: impl FnOnce(&mut [u8]) -> BTResult<(), storage::Error>,
    ) -> BTResult<Self, storage::Error> {
        OnDiskInnerDeltaNode::from_disk_repr(read_into_buffer).and_then(|on_disk| {
            InnerDeltaNode::try_from(on_disk)
                .map_err(|e| storage::Error::DatabaseCorruption(e.to_string()).into())
        })
    }

    fn to_disk_repr(&'_ self) -> Cow<'_, [u8]> {
        Cow::Owned(OnDiskInnerDeltaNode::from(self).to_disk_repr().into_owned())
    }
}

impl ManagedTrieNode for InnerDeltaNode {
    type Union = VerkleNode;
    type Id = VerkleNodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, key: &Key, depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {
        let slot = VerkleIdWithIndex::get_slot_for(&self.children_delta, key[depth as usize]);
        if let Some(slot) = slot
            && self.children_delta[slot].index == key[depth as usize]
            && self.children_delta[slot].item != VerkleNodeId::default()
        {
            Ok(LookupResult::Node(self.children_delta[slot].item))
        } else {
            Ok(LookupResult::Node(
                self.children[key[depth as usize] as usize],
            ))
        }
    }

    fn next_store_action<'a>(
        &self,
        updates: KeyedUpdateBatch<'a>,
        depth: u8,
        _self_id: Self::Id,
    ) -> BTResult<StoreAction<'a, Self::Id, Self::Union>, Error> {
        let slots = VerkleIdWithIndex::required_slot_count_for(
            &self.children_delta,
            updates
                .borrowed()
                .split(depth)
                .map(|u| u.first_key()[depth as usize]),
        );

        if slots.is_some() {
            Ok(StoreAction::HandleTransform(VerkleNode::Inner256(
                Box::new(FullInnerNode::from(self.clone())),
            )))
        } else {
            let mut descent_actions = Vec::new();
            for sub_updates in updates.split(depth) {
                let index = sub_updates.first_key()[depth as usize];
                let slot = VerkleIdWithIndex::get_slot_for(&self.children_delta, index).ok_or(
                    Error::CorruptedState(
                        "no available slot for storing value in inner delta node".to_owned(),
                    ),
                )?;
                let id = if self.children_delta[slot].item != VerkleNodeId::default() {
                    self.children_delta[slot].item
                } else {
                    self.children[index as usize]
                };
                descent_actions.push(DescendAction {
                    id,
                    updates: sub_updates,
                });
            }
            Ok(StoreAction::Descend(descent_actions))
        }
    }

    fn replace_child(&mut self, key: &Key, depth: u8, new: VerkleNodeId) -> BTResult<(), Error> {
        let index = key[depth as usize];
        match VerkleIdWithIndex::get_slot_for(&self.children_delta, index) {
            Some(slot) => {
                self.children_delta[slot] = VerkleIdWithIndex { index, item: new };
                Ok(())
            }
            _ => Err(Error::CorruptedState(
                "no slot found for replacing child in inner delta node".to_owned(),
            )
            .into()),
        }
    }

    fn get_commitment(&self) -> Self::Commitment {
        VerkleCommitment::Inner(self.commitment)
    }

    fn set_commitment(&mut self, commitment: Self::Commitment) -> BTResult<(), Error> {
        self.commitment = commitment.into_inner()?;
        Ok(())
    }
}

impl NodeVisitor<InnerDeltaNode> for NodeCountVisitor {
    fn visit(&mut self, node: &InnerDeltaNode, level: u64) -> BTResult<(), Error> {
        self.count_node(
            level,
            "Inner",
            node.children
                .iter()
                .filter(|child| child.to_node_kind().unwrap() != VerkleNodeKind::Empty)
                .count() as u64
                + node
                    .children_delta
                    .iter()
                    .filter(|child| {
                        child.item.to_node_kind().unwrap() != VerkleNodeKind::Empty
                            && node.children[child.index as usize].to_node_kind().unwrap()
                                == VerkleNodeKind::Empty
                    })
                    .count() as u64,
        );
        Ok(())
    }
}

impl VerkleManagedInnerNode for InnerDeltaNode {
    fn iter_children(&self) -> Box<dyn Iterator<Item = VerkleIdWithIndex> + '_> {
        Box::new(self.children_delta.iter().copied())
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
        error::BTError,
        types::{HasEmptyId, TreeId, Value},
    };

    const TEST_CHILD_NODE_KIND: VerkleNodeKind = VerkleNodeKind::Inner9;

    fn make_inner_with_empty_delta() -> InnerDeltaNode {
        InnerDeltaNode {
            children: array::from_fn(|i| {
                VerkleNodeId::from_idx_and_node_kind(i as u64, TEST_CHILD_NODE_KIND)
            }),
            children_delta: array::from_fn(|i| VerkleIdWithIndex {
                index: i as u8,
                item: VerkleNodeId::default(),
            }),
            full_inner_node_id: VerkleNodeId::default(),
            commitment: VerkleInnerCommitment::default(),
        }
    }

    #[test]
    fn can_be_converted_to_and_from_on_disk_representation() {
        let original_node = InnerDeltaNode {
            // Set children to empty since they are not preserved in on-disk representation
            children: [VerkleNodeId::empty_id(); 256],
            children_delta: array::from_fn(|i| VerkleIdWithIndex {
                index: i as u8,
                item: VerkleNodeId::from_idx_and_node_kind(
                    i as u64 + 1000,
                    VerkleNodeKind::Inner15,
                ),
            }),
            full_inner_node_id: VerkleNodeId::from_idx_and_node_kind(42, VerkleNodeKind::Inner256),
            // We deliberately only create a default commitment, since this type does
            // not preserve all of its fields when converting to/from on-disk representation.
            commitment: VerkleInnerCommitment::default(),
        };
        let disk_repr = original_node.to_disk_repr();
        let deserialized_node = InnerDeltaNode::from_disk_repr(|buf| {
            buf.copy_from_slice(&disk_repr);
            Ok(())
        })
        .unwrap();
        assert_eq!(original_node, deserialized_node);
    }

    #[test]
    fn from_full_inner_copies_children_and_commitment_and_sets_id_of_full_inner_node() {
        let mut full_inner = FullInnerNode {
            children: array::from_fn(|i| {
                VerkleNodeId::from_idx_and_node_kind(i as u64, TEST_CHILD_NODE_KIND)
            }),
            commitment: VerkleInnerCommitment::default(),
        };
        full_inner.commitment.modify_child(2);

        let full_inner_node_id =
            VerkleNodeId::from_idx_and_node_kind(100, VerkleNodeKind::Inner256);

        let node = InnerDeltaNode::from_full_inner(&full_inner, full_inner_node_id);
        assert_eq!(node.commitment, full_inner.commitment);
        assert_eq!(node.full_inner_node_id, full_inner_node_id);
        assert_eq!(node.children, full_inner.children);
        assert_eq!(
            node.children_delta,
            array::from_fn(|i| {
                VerkleIdWithIndex {
                    index: i as u8,
                    item: VerkleNodeId::default(),
                }
            })
        );
    }

    #[test]
    fn get_commitment_input_returns_children() {
        let mut full_inner_node = FullInnerNode::default();
        full_inner_node.children[77] =
            VerkleNodeId::from_idx_and_node_kind(888, TEST_CHILD_NODE_KIND);
        full_inner_node.children[99] =
            VerkleNodeId::from_idx_and_node_kind(999, TEST_CHILD_NODE_KIND);

        let full_inner_node_id = VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Inner256);

        let mut node = InnerDeltaNode::from_full_inner(&full_inner_node, full_inner_node_id);
        // Override one previously 0 child
        node.children_delta[3] = VerkleIdWithIndex {
            index: 33,
            item: VerkleNodeId::from_idx_and_node_kind(333, TEST_CHILD_NODE_KIND),
        };
        // Override one previously non-0 child
        node.children_delta[7] = VerkleIdWithIndex {
            index: 77,
            item: VerkleNodeId::from_idx_and_node_kind(777, TEST_CHILD_NODE_KIND),
        };

        let mut expected_children = [VerkleNodeId::default(); 256];
        expected_children[33] = VerkleNodeId::from_idx_and_node_kind(333, TEST_CHILD_NODE_KIND);
        expected_children[77] = VerkleNodeId::from_idx_and_node_kind(777, TEST_CHILD_NODE_KIND);
        expected_children[99] = VerkleNodeId::from_idx_and_node_kind(999, TEST_CHILD_NODE_KIND);

        let result = node.get_commitment_input().unwrap();
        assert_eq!(result, VerkleCommitmentInput::Inner(expected_children));
    }

    #[test]
    fn lookup_returns_id_of_child_at_key_index() {
        let mut node = make_inner_with_empty_delta();
        // Lookup an index that is not in the delta
        let key = Key::from_index_values(1, &[(1, 2)]);
        let result = node.lookup(&key, 1).unwrap();
        assert_eq!(
            result,
            LookupResult::Node(VerkleNodeId::from_idx_and_node_kind(
                2,
                TEST_CHILD_NODE_KIND
            ))
        );

        // Lookup an index that is in the delta
        let id = VerkleNodeId::from_idx_and_node_kind(2, VerkleNodeKind::Inner15);
        node.children_delta[2].item = id;
        let result = node.lookup(&key, 1).unwrap();
        assert_eq!(result, LookupResult::Node(id));
    }

    #[test]
    fn next_store_action_with_available_slot_is_descend() {
        let node = make_inner_with_empty_delta();
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
                updates,
                id: VerkleNodeId::from_idx_and_node_kind(2, TEST_CHILD_NODE_KIND),
            }])
        );
    }

    #[test]
    fn next_store_action_with_no_available_slot_is_handle_transform() {
        let mut node = make_inner_with_empty_delta();
        node.children_delta = array::from_fn(|i| VerkleIdWithIndex {
            index: i as u8,
            item: VerkleNodeId::from_idx_and_node_kind(i as u64, TEST_CHILD_NODE_KIND),
        });

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
                        id: VerkleNodeId::from_idx_and_node_kind(250, TEST_CHILD_NODE_KIND),
                        updates,
                    }])
                );
                // It contains all previous values
                assert_eq!(
                    bigger_inner.get_commitment_input().unwrap(),
                    node.get_commitment_input().unwrap()
                );
                // The commitment is copied over
                assert_eq!(bigger_inner.get_commitment(), node.get_commitment());
            }
            _ => panic!("expected HandleTransform action"),
        }
    }

    #[test]
    fn replace_child_sets_child_id_at_key_index() {
        let mut node = make_inner_with_empty_delta();
        // Existing index
        let key = Key::from_index_values(1, &[(1, 2)]);
        let new_id = VerkleNodeId::from_idx_and_node_kind(999, TEST_CHILD_NODE_KIND);
        node.replace_child(&key, 1, new_id).unwrap();
        let result = node.lookup(&key, 1).unwrap();
        assert_eq!(result, LookupResult::Node(new_id));

        // Non-existing index but with available slot
        node.children_delta[1].item = VerkleNodeId::default(); // Free up slot at index 1
        let key = Key::from_index_values(1, &[(1, 250)]);
        let new_id = VerkleNodeId::from_idx_and_node_kind(1000, TEST_CHILD_NODE_KIND);
        node.replace_child(&key, 1, new_id).unwrap();
        let result = node.lookup(&key, 1).unwrap();
        assert_eq!(result, LookupResult::Node(new_id));
    }

    #[test]
    fn replace_child_returns_error_if_no_slot_available() {
        let mut node = make_inner_with_empty_delta();
        node.children_delta = array::from_fn(|i| VerkleIdWithIndex {
            index: i as u8,
            item: VerkleNodeId::from_idx_and_node_kind(i as u64, TEST_CHILD_NODE_KIND),
        });
        let key = Key::from_index_values(1, &[(1, 250)]);
        let new_id = VerkleNodeId::from_idx_and_node_kind(1000, TEST_CHILD_NODE_KIND);
        let result = node.replace_child(&key, 1, new_id);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::CorruptedState(e)) if e.contains("no slot found for replacing child in inner delta node")
        ));
    }

    #[test]
    fn commitment_can_be_set_and_retrieved() {
        let mut node = make_inner_with_empty_delta();
        assert_eq!(
            node.get_commitment(),
            VerkleCommitment::Inner(VerkleInnerCommitment::default())
        );

        let mut new_commitment = VerkleCommitment::Inner(VerkleInnerCommitment::default());
        new_commitment.modify_child(5);

        node.set_commitment(new_commitment).unwrap();
        assert_eq!(node.get_commitment(), new_commitment);
    }
}
