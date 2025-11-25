// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::collections::HashMap;

use zerocopy::{FromBytes, Immutable, IntoBytes, Unaligned};

use crate::{
    database::{
        managed_trie::{ManagedTrieNode, TrieCommitment, TrieUpdateLog},
        verkle::{
            compute_commitment::compute_leaf_node_commitment,
            crypto::{Commitment, Scalar},
            variants::managed::{VerkleNode, VerkleNodeId},
        },
    },
    error::{BTResult, Error},
    node_manager::NodeManager,
    types::Value,
};

/// The commitment of a managed verkle trie node, together with metadata required to recompute
/// it after the node has been modified.
///
/// NOTE: While this type is meant to be part of trie nodes, a dirty commitment should never
/// be persisted to disk. The dirty flag and changed bits are nevertheless part of the on-disk
/// representation, so that the entire node can be transmuted to/from bytes using zerocopy.
/// Related issue: <https://github.com/0xsoniclabs/sonic-admin/issues/373>
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned)]
#[repr(C)]
pub struct VerkleCommitment {
    /// The commitment of the node, or of the node previously at this position in the trie.
    commitment: Commitment,
    /// A bitfield indicating which slots in a leaf node have been used before.
    /// This allows to distinguish between empty slots and slots that have been set to zero.
    used_slots: [u8; 256 / 8],
    /// Whether this commitment has been computed at least once. This dictates whether
    /// point-wise updates over dirty children can be used, or a full computation is required.
    /// Not being initialized does not imply the commitment being zero, as it may have been
    /// created from an existing commitment using [`VerkleCommitment::from_existing`].
    /// TODO: Consider merging this with `dirty` flag into an enum that is not stored on disk.
    initialized: u8,
    /// Whether the commitment is dirty and needs to be recomputed.
    // bool does not implement FromBytes, so we use u8 instead
    dirty: u8,
    /// A bitfield indicating which children or slots have been changed since
    /// the last commitment computation.
    changed: [u8; 256 / 8],
}

impl VerkleCommitment {
    /// Creates a new commitment that is meant to replace an existing commitment at a certain
    /// position within the trie. The new commitment is considered to be clean and uninitialized,
    /// however copies the existing commitment value. This allows to compute the delta between
    /// the commitment that used to be stored at this position, and the new commitment after
    /// it has been initialized.
    pub fn from_existing(existing: &VerkleCommitment) -> Self {
        VerkleCommitment {
            commitment: existing.commitment,
            used_slots: [0u8; 256 / 8],
            initialized: 0,
            dirty: 0,
            changed: [0u8; 256 / 8],
        }
    }

    pub fn commitment(&self) -> Commitment {
        self.commitment
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty != 0
    }
}

impl Default for VerkleCommitment {
    fn default() -> Self {
        Self {
            commitment: Commitment::default(),
            used_slots: [0u8; 256 / 8],
            initialized: 0,
            dirty: 0,
            changed: [0u8; 256 / 8],
        }
    }
}

impl TrieCommitment for VerkleCommitment {
    fn modify_child(&mut self, index: usize) {
        self.dirty = 1;
        self.changed[index / 8] |= 1 << (index % 8);
    }

    fn store(&mut self, index: usize, _prev: Value) {
        self.used_slots[index / 8] |= 1 << (index % 8);
        self.changed[index / 8] |= 1 << (index % 8);
        self.dirty = 1;
    }
}

// TODO: Avoid copying all 256 values / children: https://github.com/0xsoniclabs/sonic-admin/issues/384
#[allow(clippy::large_enum_variant)]
#[derive(Debug, PartialEq, Eq)]
pub enum VerkleCommitmentInput {
    Leaf([Value; 256], [u8; 31]),
    Inner([VerkleNodeId; 256]),
}

/// Recomputes the commitments of all dirty nodes recorded in the given update log.
///
/// This function assumes the update log and the node manager to be in a consistent state:
/// - All nodes marked as dirty in the update log must exist in the node manager.
/// - All nodes marked as dirty in the update log must have a dirty [`VerkleCommitment`].
/// - For a dirty inner node on level `L`, all of its children in [`VerkleCommitment::changed`] must
///   be contained in the update log on level `L+1`.
///
/// After successful completion, the update log is cleared.
pub fn update_commitments(
    log: &TrieUpdateLog<VerkleNodeId>,
    manager: &impl NodeManager<Id = VerkleNodeId, Node = VerkleNode>,
) -> BTResult<(), Error> {
    if log.count() == 0 {
        return Ok(());
    }

    let mut previous_commitments = HashMap::new();
    for level in (0..log.levels()).rev() {
        let dirty_nodes_ids = log.dirty_nodes(level);
        for id in dirty_nodes_ids {
            let mut lock = manager.get_write_access(id)?;
            let mut vc = lock.get_commitment();
            assert_eq!(vc.dirty, 1);

            previous_commitments.insert(id, vc.commitment);

            match lock.get_commitment_input()? {
                VerkleCommitmentInput::Leaf(values, stem) => {
                    // TODO: Consider caching leaf node commitments https://github.com/0xsoniclabs/sonic-admin/issues/386
                    vc.commitment = compute_leaf_node_commitment(&values, &vc.used_slots, &stem);
                }
                VerkleCommitmentInput::Inner(children) => {
                    let mut scalars = [Scalar::zero(); 256];
                    for (i, child_id) in children.iter().enumerate() {
                        if vc.initialized == 0 {
                            scalars[i] = manager
                                .get_read_access(*child_id)?
                                .get_commitment()
                                .commitment
                                .to_scalar();
                            continue;
                        }

                        if vc.changed[i / 8] & (1 << (i % 8)) == 0 {
                            continue;
                        }

                        let child_commitment = manager.get_read_access(*child_id)?.get_commitment();
                        assert_eq!(child_commitment.dirty, 0);
                        vc.commitment.update(
                            i as u8,
                            previous_commitments[child_id].to_scalar(),
                            child_commitment.commitment.to_scalar(),
                        );
                    }

                    if vc.initialized == 0 {
                        vc.commitment = Commitment::new(&scalars);
                        vc.initialized = 1;
                    }
                }
            }

            vc.dirty = 0;
            vc.changed.fill(0);
            lock.set_commitment(vc)?;
        }
    }

    log.clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        database::verkle::{
            crypto::Scalar,
            test_utils::FromIndexValues,
            variants::managed::{InnerNode, nodes::leaf::FullLeafNode},
        },
        node_manager::in_memory_node_manager::InMemoryNodeManager,
        types::{HasEmptyId, Key},
    };

    #[test]
    fn verkle_commitment_from_existing_copies_commitment() {
        let original = VerkleCommitment {
            commitment: Commitment::new(&[Scalar::from(42), Scalar::from(33)]),
            used_slots: [1u8; 256 / 8],
            initialized: 1,
            dirty: 1,
            changed: [1u8; 256 / 8],
        };
        let new = VerkleCommitment::from_existing(&original);
        assert_eq!(new.commitment, original.commitment);
        assert_eq!(new.used_slots, [0u8; 256 / 8]);
        assert_eq!(new.initialized, 0);
        assert_eq!(new.dirty, 0);
        assert_eq!(new.changed, [0u8; 256 / 8]);
    }

    #[test]
    fn verkle_commitment__commitment_returns_stored_commitment() {
        let commitment = Commitment::new(&[Scalar::from(42), Scalar::from(33)]);
        let vc = VerkleCommitment {
            commitment,
            ..Default::default()
        };
        assert_eq!(vc.commitment(), commitment);
    }

    #[test]
    fn verkle_commitment_is_dirty_returns_correct_value() {
        let vc = VerkleCommitment {
            dirty: 0,
            ..Default::default()
        };
        assert!(!vc.is_dirty());

        let vc = VerkleCommitment {
            dirty: 1,
            ..Default::default()
        };
        assert!(vc.is_dirty());
    }

    #[test]
    fn verkle_commitment_default_returns_clean_cache_with_default_commitment() {
        let vc: VerkleCommitment = VerkleCommitment::default();
        assert_eq!(vc.commitment, Commitment::default());
        assert_eq!(vc.used_slots, [0u8; 256 / 8]);
        assert_eq!(vc.dirty, 0);
    }

    #[test]
    fn verkle_commitment_modify_child_marks_dirty_and_changed() {
        let mut vc = VerkleCommitment::default();
        assert!(!vc.is_dirty());
        vc.modify_child(42);
        assert!(vc.is_dirty());
        for i in 0..256 {
            assert_eq!(vc.changed[i / 8] & (1 << (i % 8)) != 0, i == 42);
        }
    }

    #[test]
    fn verkle_commitment_store_marks_used_slots_and_changed_and_dirty() {
        let mut vc = VerkleCommitment::default();
        assert!(!vc.is_dirty());
        vc.store(42, [0u8; 32]);
        assert!(vc.is_dirty());
        for i in 0..256 {
            assert_eq!(vc.used_slots[i / 8] & (1 << (i % 8)) != 0, i == 42);
            assert_eq!(vc.changed[i / 8] & (1 << (i % 8)) != 0, i == 42);
        }
    }

    #[test]
    fn update_commitments_processes_dirty_nodes_from_leaves_to_root() {
        let manager = InMemoryNodeManager::<VerkleNodeId, VerkleNode>::new(10);
        let log = TrieUpdateLog::<VerkleNodeId>::new();

        let key = Key::from_index_values(33, &[(0, 7), (1, 4), (31, 255)]);

        // Set up simple chain: root -> inner -> leaf

        let mut leaf = FullLeafNode {
            stem: key[..31].try_into().unwrap(),
            ..Default::default()
        };
        leaf.store(&key, &[42u8; 32]).unwrap();
        leaf.commitment.store(key[31] as usize, [0u8; 32]);
        let expected_leaf_commitment =
            compute_leaf_node_commitment(&leaf.values, &leaf.commitment.used_slots, &leaf.stem);
        let leaf_id = manager.add(VerkleNode::Leaf256(Box::new(leaf))).unwrap();
        log.mark_dirty(2, leaf_id);

        let mut inner = InnerNode {
            children: {
                let mut children = [VerkleNodeId::empty_id(); 256];
                children[key[1] as usize] = leaf_id;
                children
            },
            ..Default::default()
        };
        inner.commitment.modify_child(key[1] as usize);
        let expected_inner_commitment = {
            let mut scalars = [Scalar::zero(); 256];
            scalars[key[1] as usize] = expected_leaf_commitment.to_scalar();
            Commitment::new(&scalars)
        };
        let inner_id = manager.add(VerkleNode::Inner(Box::new(inner))).unwrap();
        log.mark_dirty(1, inner_id);

        let mut root = InnerNode {
            children: {
                let mut children = [VerkleNodeId::empty_id(); 256];
                children[key[0] as usize] = inner_id;
                children
            },
            ..Default::default()
        };
        root.commitment.modify_child(key[0] as usize);
        let expected_root_commitment = {
            let mut scalars = [Scalar::zero(); 256];
            scalars[key[0] as usize] = expected_inner_commitment.to_scalar();
            Commitment::new(&scalars)
        };
        let root_id = manager.add(VerkleNode::Inner(Box::new(root))).unwrap();
        log.mark_dirty(0, root_id);

        // Run commitment update
        update_commitments(&log, &manager).unwrap();

        {
            let leaf_node_commitment = manager.get_read_access(leaf_id).unwrap().get_commitment();
            assert_eq!(leaf_node_commitment.commitment, expected_leaf_commitment);
            assert!(!leaf_node_commitment.is_dirty());
            assert!(leaf_node_commitment.changed.iter().all(|&b| b == 0));
        }

        {
            let inner_node_commitment = manager.get_read_access(inner_id).unwrap().get_commitment();
            assert_eq!(inner_node_commitment.commitment, expected_inner_commitment);
            assert!(!inner_node_commitment.is_dirty());
            assert!(inner_node_commitment.changed.iter().all(|&b| b == 0));
        }

        {
            let root_node_commitment = manager.get_read_access(root_id).unwrap().get_commitment();
            assert_eq!(root_node_commitment.commitment, expected_root_commitment);
            assert!(!root_node_commitment.is_dirty());
            assert!(root_node_commitment.changed.iter().all(|&b| b == 0));
        }

        assert_eq!(log.count(), 0);
    }
}
