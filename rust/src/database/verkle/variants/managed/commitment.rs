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
    ops::DerefMut,
    sync::{Arc, atomic::Ordering},
};

use dashmap::DashMap;
use rayon::{Scope, prelude::*};
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
    sync::{RwLockWriteGuard, atomic::AtomicUsize},
    types::{HasEmptyId, Value},
};

/// The commitment of a managed verkle trie node, together with metadata required to recompute
/// it after the node has been modified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct VerkleCommitment {
    /// The commitment of the node, or of the node previously at this position in the trie.
    commitment: Commitment,
    /// A bitfield indicating which indices in a leaf node have been used before.
    /// This allows to distinguish between empty indices and indices that have been set to zero.
    committed_used_indices: [u8; 256 / 8],
    /// Whether this commitment has been computed at least once. This dictates whether
    /// point-wise updates over dirty children can be used, or a full computation is required.
    /// Not being initialized does not imply the commitment being zero, as it may have been
    /// created from an existing commitment using [`VerkleCommitment::from_existing`].
    /// TODO: Consider merging this with `dirty` flag into an enum that is not stored on disk.
    initialized: bool,
    /// Whether the commitment is dirty and needs to be recomputed.
    dirty: bool,
    /// A bitfield indicating which children or values have been changed since
    /// the last commitment computation.
    changed_indices: [u8; 256 / 8],
    /// The two partial commitments used as part of the leaf commitment computation.
    c1: Commitment,
    c2: Commitment,
    /// The values that were committed at the time of the last commitment computation.
    /// This is only needed for leaf nodes.
    // TODO: This could be avoided by recomputing leaf commitments directly after storing values.
    // https://github.com/0xsoniclabs/sonic-admin/issues/542
    committed_values: [Value; 256],
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
            committed_used_indices: [0u8; 256 / 8],
            initialized: false,
            dirty: false,
            changed_indices: [0u8; 256 / 8],
            c1: Commitment::default(),
            c2: Commitment::default(),
            committed_values: [Value::default(); 256],
        }
    }

    pub fn commitment(&self) -> Commitment {
        self.commitment
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }
}

impl Default for VerkleCommitment {
    fn default() -> Self {
        Self {
            commitment: Commitment::default(),
            committed_used_indices: [0u8; 256 / 8],
            initialized: false,
            dirty: false,
            c1: Commitment::default(),
            c2: Commitment::default(),
            committed_values: [Value::default(); 256],
            changed_indices: [0u8; 256 / 8],
        }
    }
}

impl TrieCommitment for VerkleCommitment {
    fn modify_child(&mut self, index: usize) {
        self.dirty = true;
        self.changed_indices[index / 8] |= 1 << (index % 8);
    }

    fn store(&mut self, index: usize, prev: Value) {
        if self.changed_indices[index / 8] & (1 << (index % 8)) == 0 {
            self.changed_indices[index / 8] |= 1 << (index % 8);
            self.committed_values[index] = prev;
            self.dirty = true;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, Unaligned, Immutable)]
#[repr(C)]
pub struct OnDiskVerkleCommitment {
    commitment: Commitment,
    pub committed_used_indices: [u8; 256 / 8],
    c1: Commitment,
    c2: Commitment,
}

impl From<OnDiskVerkleCommitment> for VerkleCommitment {
    fn from(odvc: OnDiskVerkleCommitment) -> Self {
        VerkleCommitment {
            commitment: odvc.commitment,
            committed_used_indices: odvc.committed_used_indices,
            c1: odvc.c1,
            c2: odvc.c2,
            initialized: true,
            dirty: false,
            committed_values: [Value::default(); 256],
            changed_indices: [0u8; 256 / 8],
        }
    }
}

impl From<&VerkleCommitment> for OnDiskVerkleCommitment {
    fn from(value: &VerkleCommitment) -> Self {
        assert!(value.initialized);
        assert!(!value.dirty);

        OnDiskVerkleCommitment {
            commitment: value.commitment,
            committed_used_indices: value.committed_used_indices,
            c1: value.c1,
            c2: value.c2,
        }
    }
}

// TODO: Avoid copying all 256 values / children: https://github.com/0xsoniclabs/sonic-admin/issues/384
#[allow(clippy::large_enum_variant)]
#[derive(Debug, PartialEq, Eq)]
pub enum VerkleCommitmentInput {
    Leaf([Value; 256], [u8; 31]),
    Inner([VerkleNodeId; 256]),
}

pub fn update_commitments_sequential(
    log: &TrieUpdateLog<VerkleNodeId>,
    manager: &(impl NodeManager<Id = VerkleNodeId, Node = VerkleNode> + Send + Sync),
) -> BTResult<(), Error> {
    if log.count() == 0 {
        return Ok(());
    }

    let previous_commitments = DashMap::new();
    for level in (0..log.levels()).rev() {
        let dirty_nodes = log.dirty_nodes(level);
        for id in dirty_nodes.iter() {
            process_update(manager, *id, &previous_commitments);
        }
    }
    // TODO: Test
    log.clear();
    Ok(())
}

pub fn process_update(
    manager: &(impl NodeManager<Id = VerkleNodeId, Node = VerkleNode> + Send + Sync),
    id: VerkleNodeId,
    previous_commitments: &DashMap<VerkleNodeId, Commitment>,
) {
    let mut lock = manager.get_write_access(id).unwrap();
    let mut vc = lock.get_commitment();
    assert!(vc.dirty);
    previous_commitments.insert(id, vc.commitment);
    match lock.get_commitment_input().unwrap() {
        VerkleCommitmentInput::Leaf(values, stem) => {
            let _span = tracy_client::span!("update leaf");

            compute_leaf_node_commitment(
                vc.changed_indices,
                &vc.committed_values,
                &values,
                &stem,
                &mut vc.committed_used_indices,
                &mut vc.c1,
                &mut vc.c2,
                &mut vc.commitment,
            );
            vc.committed_values.fill(Value::default()); // TODO: Is this needed? Also naming of field...
        }
        VerkleCommitmentInput::Inner(children) => {
            let _span = tracy_client::span!("update inner");

            let changed_count = vc
                .changed_indices
                .iter()
                .map(|b| b.count_ones() as usize)
                .sum::<usize>();
            let use_batch_update = changed_count > 32;

            let mut scalars = [Scalar::zero(); 256];
            for (i, child_id) in children.iter().enumerate() {
                if !vc.initialized {
                    scalars[i] = manager
                        .get_read_access(*child_id)
                        .unwrap()
                        .get_commitment()
                        .commitment
                        .to_scalar();
                    continue;
                }

                if vc.changed_indices[i / 8] & (1 << (i % 8)) == 0 {
                    continue;
                }

                let child_commitment = manager.get_read_access(*child_id).unwrap().get_commitment();
                assert!(!child_commitment.dirty);
                let prev_commitment = previous_commitments
                    .get(child_id)
                    .expect("previous commitment should have been set in lower level");
                let prev_commitment = prev_commitment.to_scalar();

                if use_batch_update {
                    scalars[i] = child_commitment.commitment().to_scalar() - prev_commitment;
                } else {
                    vc.commitment.update(
                        i as u8,
                        prev_commitment,
                        child_commitment.commitment().to_scalar(),
                    );
                }
            }

            if vc.initialized && use_batch_update {
                vc.commitment = vc.commitment + Commitment::new(&scalars);
            }

            if !vc.initialized {
                vc.commitment = Commitment::new(&scalars);
            }
        }
    }

    // TODO: Include initialized in VerkleCommitment::is_dirty to prevent eviction?
    vc.initialized = true;
    vc.dirty = false;
    // TODO: Test this (currently not caught by any test!)
    vc.changed_indices.fill(0);
    lock.set_commitment(vc).unwrap();
}

pub fn update_commitments_concurrent(
    log: &TrieUpdateLog<VerkleNodeId>,
    manager: &(impl NodeManager<Id = VerkleNodeId, Node = VerkleNode> + Send + Sync),
) -> BTResult<(), Error> {
    const MIN_BATCH_SIZE: usize = 4;

    if log.count() == 0 {
        return Ok(());
    }

    let _span = tracy_client::span!("update_commitments");

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

// TODO: Instead of returning a delta, return commitment (needs self position)
fn update_commitment_thread(
    dbg_recursion_level: usize,
    mut node: RwLockWriteGuard<'_, impl DerefMut<Target = VerkleNode>>,
    index_in_parent: usize,
    delta_commitment: &mut Commitment,
    manager: &(impl NodeManager<Id = VerkleNodeId, Node = VerkleNode> + Send + Sync),
) {
    // eprintln!(
    //     "{}updating commitment in thread",
    //     "  ".repeat(dbg_recursion_level)
    // );
    // eprintln!(
    //     "{}address of delta: {:p}",
    //     "  ".repeat(dbg_recursion_level),
    //     delta
    // );

    let mut vc = node.get_commitment();
    assert!(vc.dirty); // FIXME: Why is this assertion failing with bertha?

    match node.get_commitment_input().unwrap() {
        VerkleCommitmentInput::Leaf(values, stem) => {
            // eprintln!("{}its a leaf", "  ".repeat(dbg_recursion_level));
            let _span = tracy_client::span!("update leaf");

            compute_leaf_node_commitment(
                vc.changed_indices,
                &vc.committed_values,
                &values,
                &stem,
                &mut vc.committed_used_indices,
                &mut vc.c1,
                &mut vc.c2,
                &mut vc.commitment,
            );
        }
        VerkleCommitmentInput::Inner(children) => {
            // eprintln!("{}its an inner", "  ".repeat(dbg_recursion_level));
            let _span = tracy_client::span!("update inner");

            let mut child_delta_commitments = [Commitment::default(); 256];

            child_delta_commitments
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, delta_i)| {
                    if !vc.initialized && vc.changed_indices[i / 8] & (1 << (i % 8)) == 0 {
                        // FIXME HACK: New nodes that are reparented are not marked as dirty in
                        // parent - need to recurse here anyway. For other
                        // implementations this was not an issue since we
                        // always worked through the log. => We should
                        // probably mark as dirty in parent?
                        if manager
                            .get_read_access(children[i])
                            .unwrap()
                            .get_commitment()
                            .is_dirty()
                        {
                            let child_node = manager.get_write_access(children[i]).unwrap();
                            update_commitment_thread(
                                dbg_recursion_level + 1,
                                child_node,
                                i,
                                delta_i,
                                manager,
                            );
                            return;
                        }

                        delta_i.update(
                            i as u8,
                            Scalar::zero(),
                            manager
                                .get_read_access(children[i])
                                .unwrap()
                                .get_commitment()
                                .commitment()
                                .to_scalar(),
                        );

                        return;
                    }

                    if vc.changed_indices[i / 8] & (1 << (i % 8)) == 0 {
                        return;
                    }
                    // eprintln!("{}processing child {}", "  ".repeat(dbg_recursion_level), i);
                    let child_node = manager.get_write_access(children[i]).unwrap();
                    update_commitment_thread(
                        dbg_recursion_level + 1,
                        child_node,
                        i,
                        delta_i,
                        manager,
                    );
                });

            let _span = tracy_client::span!("sum up child delta commitments");
            let delta_commitment = child_delta_commitments
                .into_iter()
                .enumerate()
                .filter_map(|(i, c)| {
                    if vc.changed_indices[i / 8] & (1 << (i % 8)) == 0 {
                        None
                    } else {
                        Some(c)
                    }
                })
                .collect::<Vec<_>>()
                .into_par_iter()
                .chunks(4)
                .fold(Commitment::default, |acc, chunk| {
                    let mut res = acc;
                    for c in chunk {
                        res = res + c;
                    }
                    res
                })
                .reduce(Commitment::default, |acc, delta_commitment| {
                    acc + delta_commitment
                });

            if !vc.initialized {
                vc.commitment = delta_commitment;
                vc.initialized = true;
            } else {
                vc.commitment = vc.commitment + delta_commitment;
            }
        }
    }

    // let delta = vc.commitment.to_scalar() - node.get_commitment().commitment().to_scalar();
    delta_commitment.update(
        index_in_parent as u8,
        node.get_commitment().commitment().to_scalar(),
        vc.commitment.to_scalar(),
    );
    // eprintln!(
    //     "{}setting delta to {:?}",
    //     "  ".repeat(dbg_recursion_level),
    //     delta
    // );
    vc.dirty = false;
    // TODO: Test this (currently not caught by any test!)
    vc.changed_indices.fill(0);
    node.set_commitment(vc).unwrap();
}

pub fn update_commitments_concurrent_recursive(
    root_id: VerkleNodeId,
    log: &TrieUpdateLog<VerkleNodeId>,
    manager: &(impl NodeManager<Id = VerkleNodeId, Node = VerkleNode> + Send + Sync),
) -> BTResult<(), Error> {
    // TODO: This is still useful to have, maybe not get rid of log entirely?
    if log.count() == 0 {
        return Ok(());
    }

    if log.count() <= 8 {
        return update_commitments_concurrent(log, manager);
    }

    let _span = tracy_client::span!("update_commitments_concurrent_recursive");

    let mut delta_commitment = Commitment::default();
    let root = manager.get_write_access(root_id).unwrap();
    update_commitment_thread(0, root, 0, &mut delta_commitment, manager);

    log.clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        database::verkle::{
            compute_commitment::compute_leaf_node_commitment,
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
            committed_used_indices: [1u8; 256 / 8],
            initialized: true,
            dirty: true,
            changed_indices: [7u8; 256 / 8],
            c1: Commitment::new(&[Scalar::from(7)]),
            c2: Commitment::new(&[Scalar::from(11)]),
            committed_values: [[7u8; 32]; 256],
        };
        let new = VerkleCommitment::from_existing(&original);
        assert_eq!(new.commitment, original.commitment);
        assert_eq!(new.committed_used_indices, [0u8; 256 / 8]);
        assert!(!new.initialized);
        assert!(!new.dirty);
        assert_eq!(new.changed_indices, [0u8; 256 / 8]);
        assert_eq!(new.c1, Commitment::default());
        assert_eq!(new.c2, Commitment::default());
        assert_eq!(new.committed_values, [Value::default(); 256]);
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
            dirty: false,
            ..Default::default()
        };
        assert!(!vc.is_dirty());

        let vc = VerkleCommitment {
            dirty: true,
            ..Default::default()
        };
        assert!(vc.is_dirty());
    }

    #[test]
    fn verkle_commitment_default_returns_clean_cache_with_default_commitment() {
        let vc: VerkleCommitment = VerkleCommitment::default();
        assert_eq!(vc.commitment, Commitment::default());
        assert_eq!(vc.committed_used_indices, [0u8; 256 / 8]);
        assert!(!vc.dirty);
    }

    #[test]
    fn verkle_commitment_modify_child_marks_dirty_and_changed() {
        let mut vc = VerkleCommitment::default();
        assert!(!vc.is_dirty());
        vc.modify_child(42);
        assert!(vc.is_dirty());
        for i in 0..256 {
            assert_eq!(vc.changed_indices[i / 8] & (1 << (i % 8)) != 0, i == 42);
        }
    }

    #[test]
    fn verkle_commitment_store_marks_dirty_and_changed() {
        let mut vc = VerkleCommitment::default();
        assert!(!vc.is_dirty());
        vc.store(42, [0u8; 32]);
        assert!(vc.is_dirty());
        for i in 0..256 {
            assert_eq!(vc.changed_indices[i / 8] & (1 << (i % 8)) != 0, i == 42);
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

        let expected_leaf_commitment = {
            let mut vc = leaf.commitment;
            compute_leaf_node_commitment(
                vc.changed_indices,
                &[Value::default(); 256],
                &leaf.values,
                &leaf.stem,
                &mut vc.committed_used_indices,
                &mut vc.c1,
                &mut vc.c2,
                &mut vc.commitment,
            );
            vc.commitment
        };
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
        update_commitments_concurrent(&log, &manager).unwrap();

        {
            let leaf_node_commitment = manager.get_read_access(leaf_id).unwrap().get_commitment();
            assert_eq!(leaf_node_commitment.commitment, expected_leaf_commitment);
            assert!(!leaf_node_commitment.is_dirty());
            assert!(leaf_node_commitment.changed_indices.iter().all(|&b| b == 0));
        }

        {
            let inner_node_commitment = manager.get_read_access(inner_id).unwrap().get_commitment();
            assert_eq!(inner_node_commitment.commitment, expected_inner_commitment);
            assert!(!inner_node_commitment.is_dirty());
            assert!(
                inner_node_commitment
                    .changed_indices
                    .iter()
                    .all(|&b| b == 0)
            );
        }

        {
            let root_node_commitment = manager.get_read_access(root_id).unwrap().get_commitment();
            assert_eq!(root_node_commitment.commitment, expected_root_commitment);
            assert!(!root_node_commitment.is_dirty());
            assert!(root_node_commitment.changed_indices.iter().all(|&b| b == 0));
        }

        assert_eq!(log.count(), 0);
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
