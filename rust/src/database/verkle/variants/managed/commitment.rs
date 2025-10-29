// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use dashmap::DashMap;
use zerocopy::{FromBytes, Immutable, IntoBytes, Unaligned};

use crate::{
    database::{
        managed_trie::{ManagedTrieNode, TrieCommitment, TrieUpdateLog},
        verkle::{
            crypto::{Commitment, Scalar},
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
    committed_used_slots: [u8; 256 / 8],
    /// Whether the commitment is dirty and needs to be recomputed.
    // bool does not implement FromBytes, so we use u8 instead
    dirty: u8,

    // FIXME Just hacking - these are only needed for Verkle leaf nodes
    // TODO: Also store scalars?
    c1: Commitment,
    c2: Commitment,
    // TODO Naming
    committed_values: [Value; 256],
    /// A bitfield indicating which children or slots have been changed since
    /// the last commitment computation.
    changed: [u8; 256 / 8],
}

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
            committed_used_slots: [0u8; 256 / 8],
            dirty: 0,
            c1: Commitment::default(),
            c2: Commitment::default(),
            committed_values: [Value::default(); 256],
            changed: [0u8; 256 / 8],
        }
    }
}

impl TrieCommitment for VerkleCommitment {
    fn modify_child(&mut self, index: usize) {
        self.dirty = 1;
        self.changed[index / 8] |= 1 << (index % 8);
    }

    fn store(&mut self, index: usize, prev_value: Value) {
        if self.changed[index / 8] & (1 << (index % 8)) == 0 {
            self.changed[index / 8] |= 1 << (index % 8);
            self.committed_values[index] = prev_value;
            self.dirty = 1;
        }
    }
}

// TODO: Avoid copying all 256 values / children: https://github.com/0xsoniclabs/sonic-admin/issues/384
#[allow(clippy::large_enum_variant)]
pub enum VerkleCommitmentInput {
    Leaf([Value; 256], [u8; 31]),
    Inner([NodeId; 256]),
}

pub fn process_update(
    manager: &(impl NodeManager<Id = NodeId, NodeType = Node> + Send + Sync),
    id: NodeId,
    previous_commitments: &DashMap<NodeId, Commitment>,
) {
    let mut lock = manager.get_write_access(id).unwrap();
    let mut vc = lock.get_commitment();
    assert_eq!(vc.dirty, 1);
    previous_commitments.insert(id, vc.commitment);
    match lock.get_commitment_input().unwrap() {
        VerkleCommitmentInput::Leaf(values, stem) => {
            let _span = tracy_client::span!("update leaf");
            for (i, value) in vc.committed_values.iter().enumerate() {
                if vc.changed[i / 8] & (1 << (i % 8)) == 0 {
                    continue;
                }
                let mut prev_lower = Scalar::from_le_bytes(&value[..16]);
                let prev_upper = Scalar::from_le_bytes(&value[16..]);
                if vc.committed_used_slots[i / 8] & (1 << (i % 8)) != 0 {
                    prev_lower.set_bit128();
                }
                let mut lower = Scalar::from_le_bytes(&values[i][..16]);
                let upper = Scalar::from_le_bytes(&values[i][16..]);
                lower.set_bit128();
                let c = if i < 128 { &mut vc.c1 } else { &mut vc.c2 };
                c.update(((i * 2) % 256) as u8, prev_lower, lower);
                c.update(((i * 2 + 1) % 256) as u8, prev_upper, upper);
                vc.committed_used_slots[i / 8] |= 1 << (i % 8);
            }
            vc.committed_values.fill(Value::default());
            let combined = [
                Scalar::from(1),
                Scalar::from_le_bytes(&stem),
                vc.c1.to_scalar(),
                vc.c2.to_scalar(),
            ];
            vc.commitment = Commitment::new(&combined);
        }
        VerkleCommitmentInput::Inner(children) => {
            let _span = tracy_client::span!("update inner");

            let changed_count = vc
                .changed
                .iter()
                .map(|b| b.count_ones() as usize)
                .sum::<usize>();
            let use_batch_update = changed_count > 32;

            let mut deltas = [Scalar::zero(); 256];
            for (i, child_id) in children.iter().enumerate() {
                if vc.changed[i / 8] & (1 << (i % 8)) == 0 {
                    continue;
                }
                let child_commitment = manager.get_read_access(*child_id).unwrap().get_commitment();
                assert_eq!(child_commitment.dirty, 0);
                let prev_commitment = previous_commitments
                    .get(child_id)
                    .expect("previous commitment should have been set in lower level");
                let prev_commitment = prev_commitment.to_scalar();

                if use_batch_update {
                    deltas[i] = child_commitment.commitment().to_scalar() - prev_commitment;
                } else {
                    vc.commitment.update(
                        i as u8,
                        prev_commitment,
                        child_commitment.commitment().to_scalar(),
                    );
                }
            }
            if use_batch_update {
                vc.commitment = vc.commitment + Commitment::new(&deltas);
            }
        }
    }
    vc.dirty = 0;
    // TODO: Test this (currently not caught by any test!)
    vc.changed.fill(0);
    lock.set_commitment(vc).unwrap();
}

pub fn update_commitments(
    log: &TrieUpdateLog<NodeId>,
    manager: &(impl NodeManager<Id = NodeId, NodeType = Node> + Send + Sync),
) -> Result<(), Error> {
    const MIN_BATCH_SIZE: usize = 4;

    if log.count() == 0 {
        return Ok(());
    }

    let hardware_parallelism = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);

    let previous_commitments = DashMap::new();

    for level in (0..log.levels()).rev() {
        let dirty_nodes = log.dirty_nodes(level);

        // TODO: How to set this? Depending on hardware + available parallelism?
        let num_threads = if dirty_nodes.len() < MIN_BATCH_SIZE {
            1
        } else {
            let batch_size = dirty_nodes.len().div_ceil(hardware_parallelism);
            dirty_nodes.len().div_ceil(batch_size)
        };

        if num_threads == 1 {
            for id in dirty_nodes {
                process_update(manager, id, &previous_commitments);
            }
            continue;
        }

        let next_idx = AtomicUsize::new(0);
        rayon::scope(|s| {
            for _ in 0..num_threads {
                let previous_commitments = &previous_commitments;
                let dirty_nodes = &dirty_nodes;
                let next_idx = &next_idx;

                s.spawn(move |_| {
                    let _span = tracy_client::span!("looping over dirty nodes");
                    let total_nodes = dirty_nodes.len();
                    loop {
                        let idx = next_idx.fetch_add(1, Ordering::SeqCst);
                        if idx >= total_nodes {
                            break;
                        }
                        let id = dirty_nodes[idx];
                        process_update(manager, id, previous_commitments);
                    }
                });
            }
        });
    }

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
        assert_eq!(cache.committed_used_slots, [0u8; 256 / 8]);
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
    fn verkle_commitment_store_marks_dirty_and_changed() {
        let mut cache = VerkleCommitment::default();
        assert!(!cache.is_dirty());
        cache.store(42, [0u8; 32]);
        assert!(cache.is_dirty());
        for i in 0..256 {
            assert_eq!(cache.changed[i / 8] & (1 << (i % 8)) != 0, i == 42);
        }
    }

    #[test]
    fn verkle_commitment_store_remembers_committed_value() {
        let mut cache = VerkleCommitment::default();
        cache.store(42, [1u8; 32]);
        assert_eq!(cache.committed_values[42], [1u8; 32]);
        // Only the first previous value (= the committed one) is remembered.
        cache.store(42, [7u8; 32]);
        assert_eq!(cache.committed_values[42], [1u8; 32]);
    }
}
