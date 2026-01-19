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
                InnerDeltaNode, VerkleNode,
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
    types::{DiskRepresentable, Key, ToNodeKind, TreeId},
};

/// An inner node in a managed Verkle trie.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullInnerNode {
    pub children: [VerkleNodeId; 256],
    pub commitment: VerkleInnerCommitment,
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
            commitment: VerkleInnerCommitment::default(),
        }
    }
}

// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, Immutable, FromBytes, IntoBytes, Unaligned)]
#[repr(C)]
pub struct OnDiskFullInnerNode {
    pub children: [VerkleNodeId; 256],
    pub commitment: OnDiskVerkleInnerCommitment,
}

impl TryFrom<OnDiskFullInnerNode> for FullInnerNode {
    type Error = BTError<Error>;

    fn try_from(node: OnDiskFullInnerNode) -> Result<Self, Self::Error> {
        Ok(FullInnerNode {
            children: node.children,
            commitment: VerkleInnerCommitment::try_from(node.commitment)?,
        })
    }
}

impl From<&FullInnerNode> for OnDiskFullInnerNode {
    fn from(node: &FullInnerNode) -> Self {
        OnDiskFullInnerNode {
            children: node.children,
            commitment: OnDiskVerkleInnerCommitment::from(&node.commitment),
        }
    }
}

impl DiskRepresentable for FullInnerNode {
    const DISK_REPR_SIZE: usize = std::mem::size_of::<OnDiskFullInnerNode>();

    fn from_disk_repr(
        read_into_buffer: impl FnOnce(&mut [u8]) -> BTResult<(), storage::Error>,
    ) -> BTResult<Self, storage::Error> {
        OnDiskFullInnerNode::from_disk_repr(read_into_buffer).and_then(|on_disk| {
            FullInnerNode::try_from(on_disk)
                .map_err(|e| storage::Error::DatabaseCorruption(e.to_string()).into())
        })
    }

    fn to_disk_repr(&'_ self) -> Cow<'_, [u8]> {
        Cow::Owned(OnDiskFullInnerNode::from(self).to_disk_repr().into_owned())
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
        VerkleCommitment::Inner(self.commitment)
    }

    fn set_commitment(&mut self, commitment: Self::Commitment) -> BTResult<(), Error> {
        self.commitment = commitment.into_inner()?;
        Ok(())
    }
}

impl From<InnerDeltaNode> for FullInnerNode {
    fn from(delta_node: InnerDeltaNode) -> Self {
        FullInnerNode {
            children: {
                let mut children = delta_node.children;
                for VerkleIdWithIndex { index, item } in delta_node.children_delta {
                    if item != VerkleNodeId::default() {
                        children[index as usize] = item;
                    }
                }
                children
            },
            commitment: delta_node.commitment,
        }
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
                KeyedUpdateBatch,
                test_utils::FromIndexValues,
                variants::managed::nodes::{ItemWithIndex, VerkleNodeKind},
            },
        },
        types::{TreeId, Value},
    };

    #[test]
    fn inner_node_default_returns_inner_node_with_all_children_set_to_empty_node_id() {
        let node: FullInnerNode = FullInnerNode::default();
        assert_eq!(node.commitment, VerkleInnerCommitment::default());
        assert_eq!(
            node.children,
            [VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty); 256]
        );
    }

    #[test]
    fn can_be_converted_to_and_from_on_disk_representation() {
        let original_node = FullInnerNode {
            children: array::from_fn(|i| {
                VerkleNodeId::from_idx_and_node_kind(i as u64, VerkleNodeKind::Inner256)
            }),
            commitment: {
                // We deliberately only create a default commitment, since this type does
                // not preserve all of its fields when converting to/from on-disk representation.
                let mut commitment = VerkleInnerCommitment::default();
                commitment.mark_clean();
                commitment
            },
        };
        let disk_repr = original_node.to_disk_repr();
        let deserialized_node = FullInnerNode::from_disk_repr(|buf| {
            buf.copy_from_slice(&disk_repr);
            Ok(())
        })
        .unwrap();
        assert_eq!(original_node, deserialized_node);
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
        assert_eq!(
            node.get_commitment(),
            VerkleCommitment::Inner(VerkleInnerCommitment::default())
        );

        let mut new_commitment = VerkleCommitment::Inner(VerkleInnerCommitment::default());
        new_commitment.modify_child(5);

        node.set_commitment(new_commitment).unwrap();
        assert_eq!(node.get_commitment(), new_commitment);
    }

    #[test]
    fn from_inner_delta_node_applies_delta_on_top_of_old_children_and_copies_commitment() {
        let old_children = array::from_fn(|i| {
            VerkleNodeId::from_idx_and_node_kind(i as u64, VerkleNodeKind::Inner256)
        });
        let mut children_delta = array::from_fn(|i| ItemWithIndex {
            index: i as u8,
            item: VerkleNodeId::default(),
        });
        children_delta[1] = ItemWithIndex {
            index: 2,
            item: VerkleNodeId::from_idx_and_node_kind(500, VerkleNodeKind::Leaf1),
        };
        children_delta[3] = ItemWithIndex {
            index: 4,
            item: VerkleNodeId::from_idx_and_node_kind(600, VerkleNodeKind::Leaf1),
        };

        let mut commitment = VerkleInnerCommitment::default();
        commitment.modify_child(10);
        commitment.modify_child(20);

        let delta_node = InnerDeltaNode {
            children: old_children,
            children_delta,
            commitment,
            full_inner_node_id: VerkleNodeId::default(),
        };

        let full_inner_node = FullInnerNode::from(delta_node);

        for i in 0..256 {
            let expected_child = match i {
                2 => VerkleNodeId::from_idx_and_node_kind(500, VerkleNodeKind::Leaf1),
                4 => VerkleNodeId::from_idx_and_node_kind(600, VerkleNodeKind::Leaf1),
                _ => VerkleNodeId::from_idx_and_node_kind(i as u64, VerkleNodeKind::Inner256),
            };
            assert_eq!(full_inner_node.children[i], expected_child);
        }
        assert_eq!(full_inner_node.commitment, commitment);
    }
}
