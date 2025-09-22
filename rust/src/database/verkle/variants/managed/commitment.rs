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
            crypto::Commitment,
            variants::managed::{Node, NodeId},
        },
    },
    error::Error,
    node_manager::NodeManager,
    types::Value,
};

/// A commitment of managed verkle trie node, together with metadata required to recompute
/// it after modifications to children or slots.
///
/// NOTE: While this type is meant to be part of trie nodes, a dirty commitment should never
/// be persisted to disk. The dirty flag is nevertheless part of the on-disk representation,
/// so that the entire node can be transmuted to/from bytes using zerocopy.
/// Related issue: https://github.com/0xsoniclabs/sonic-admin/issues/373
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned)]
#[repr(C)]
pub struct VerkleCommitment {
    commitment: Commitment,
    /// A bitfield indicating which slots in a leaf node have been used before.
    /// This allows to distinguish between empty slots and slots that have been set to zero.
    used_slots: [u8; 256 / 8],
    /// Whether the commitment is dirty and needs to be recomputed.
    // bool does not implement FromBytes, so we use u8 instead
    dirty: u8,
    /// A bitfield indicating which children or slots have been changed since
    /// the last commitment computation.
    changed: [u8; 256 / 8],
}

#[cfg_attr(not(test), expect(unused))]
impl VerkleCommitment {
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

    fn store(&mut self, index: usize, _prev_value: Value) {
        self.used_slots[index / 8] |= 1 << (index % 8);
        self.changed[index / 8] |= 1 << (index % 8);
        self.dirty = 1;
    }
}

// TODO: Avoid copying all 256 values / children: https://github.com/0xsoniclabs/sonic-admin/issues/384
#[allow(clippy::large_enum_variant)]
pub enum VerkleCommitmentInput {
    Leaf([Value; 256], [u8; 31]),
    Inner([NodeId; 256]),
}

pub fn update_commitments(
    log: &TrieUpdateLog<NodeId>,
    manager: &impl NodeManager<Id = NodeId, NodeType = Node>,
) -> Result<(), Error> {
    let mut previous_commitments = HashMap::new();
    for dirty_nodes in log.dirty_nodes_by_level.read().unwrap().iter().rev() {
        for id in dirty_nodes.iter() {
            let mut lock = manager.get_write_access(*id)?;
            let mut com = lock.get_commitment();
            assert_eq!(com.dirty, 1);

            previous_commitments.insert(*id, com.commitment);

            match lock.get_commitment_input()? {
                VerkleCommitmentInput::Leaf(values, stem) => {
                    // TODO: Consider caching leaf node commitments https://github.com/0xsoniclabs/sonic-admin/issues/386
                    com.commitment = compute_leaf_node_commitment(&values, &com.used_slots, &stem);
                }
                VerkleCommitmentInput::Inner(children) => {
                    for (i, child_id) in children.iter().enumerate() {
                        if com.changed[i / 8] & (1 << (i % 8)) == 0 {
                            continue;
                        }

                        let child_commitment = manager.get_read_access(*child_id)?.get_commitment();
                        assert_eq!(child_commitment.dirty, 0);
                        com.commitment.update(
                            i as u8,
                            previous_commitments[child_id].to_scalar(),
                            child_commitment.commitment.to_scalar(),
                        );
                    }
                }
            }

            com.dirty = 0;
            // TODO: Test this (currently not caught by any test!)
            com.changed.fill(0);
            lock.set_commitment(com)?;
        }
    }
    // TODO: Test
    log.clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::verkle::crypto::Scalar;

    #[test]
    fn verkle_commitment__commitment_returns_stored_commitment() {
        let commitment = Commitment::new(&[Scalar::from(42), Scalar::from(33)]);
        let cache = VerkleCommitment {
            commitment,
            ..Default::default()
        };
        assert_eq!(cache.commitment(), commitment);
    }

    #[test]
    fn verkle_commitment_is_dirty_returns_correct_value() {
        let clean_cache = VerkleCommitment {
            dirty: 0,
            ..Default::default()
        };
        assert!(!clean_cache.is_dirty());

        let dirty_cache = VerkleCommitment {
            dirty: 1,
            ..Default::default()
        };
        assert!(dirty_cache.is_dirty());
    }

    #[test]
    fn verkle_commitment_default_returns_clean_cache_with_default_commitment() {
        let cache: VerkleCommitment = VerkleCommitment::default();
        assert_eq!(cache.commitment, Commitment::default());
        assert_eq!(cache.used_slots, [0u8; 256 / 8]);
        assert_eq!(cache.dirty, 0);
    }

    #[test]
    fn verkle_commitment_modify_child_marks_dirty_and_changed() {
        let mut cache = VerkleCommitment::default();
        assert!(!cache.is_dirty());
        cache.modify_child(42);
        assert!(cache.is_dirty());
        for i in 0..256 {
            assert_eq!(cache.changed[i / 8] & (1 << (i % 8)) != 0, i == 42);
        }
    }

    #[test]
    fn verkle_commitment_store_marks_used_slots_and_changed_and_dirty() {
        let mut cache = VerkleCommitment::default();
        assert!(!cache.is_dirty());
        cache.store(42, [0u8; 32]);
        assert!(cache.is_dirty());
        for i in 0..256 {
            assert_eq!(cache.used_slots[i / 8] & (1 << (i % 8)) != 0, i == 42);
            assert_eq!(cache.changed[i / 8] & (1 << (i % 8)) != 0, i == 42);
        }
    }
}
