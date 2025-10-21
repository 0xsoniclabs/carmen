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

use crate::{
    database::{
        managed_trie::{ManagedTrieNode, TrieUpdateLog},
        verkle::{
            crypto::{Commitment, Scalar},
            variants::managed::{Node, NodeId},
        },
    },
    error::Error,
    node_manager::NodeManager,
    types::Value,
};

// TODO: Avoid copying all 256 values / children: https://github.com/0xsoniclabs/sonic-admin/issues/384
#[allow(clippy::large_enum_variant)]
pub enum VerkleCommitmentInput {
    Leaf([Value; 256], [u8; 256 / 8], [u8; 31]),
    Inner([NodeId; 256]),
}

pub fn update_commitments<T>(
    log: &mut TrieUpdateLog<NodeId>,
    manager: &impl NodeManager<Id = NodeId, NodeType = T>,
) -> Result<(), Error>
where
    T: ManagedTrieNode<
            Union = Node,
            Id = NodeId,
            Commitment = Commitment,
            CommitmentInput = VerkleCommitmentInput,
        >,
{
    let mut previous_commitments = HashMap::new();
    for dirty_nodes in log.dirty_nodes_by_level.iter().rev() {
        for (id, child_mask) in dirty_nodes.iter() {
            let mut lock = manager.get_write_access(*id)?;
            let mut cache = lock.get_cached_commitment();
            // assert_eq!(cache.dirty, 1);

            previous_commitments.insert(*id, cache.commitment());

            match lock.get_commitment_input()? {
                VerkleCommitmentInput::Leaf(values, _used_bits, stem) => {
                    // TODO: Consider caching leaf node commitments https://github.com/0xsoniclabs/sonic-admin/issues/386
                    // cache.commitment = compute_leaf_node_commitment(&values, &used_bits, &stem);

                    // let mut c1_changed = false;
                    // let mut c2_changed = false;
                    for (i, value) in cache.committed_values.iter().enumerate() {
                        if cache.changed_slots[i] == 0 {
                            continue;
                        }

                        let mut prev_lower = Scalar::from_le_bytes(&value[..16]);
                        let prev_upper = Scalar::from_le_bytes(&value[16..]);
                        if cache.committed_used_bits[i / 8] & (1 << (i % 8)) != 0 {
                            prev_lower.set_bit128();
                        }

                        let mut lower = Scalar::from_le_bytes(&values[i][..16]);
                        let upper = Scalar::from_le_bytes(&values[i][16..]);
                        lower.set_bit128();

                        let c = if i < 128 {
                            // c1_changed = true;
                            &mut cache.c1
                        } else {
                            // c2_changed = true;
                            &mut cache.c2
                        };
                        c.update(((i * 2) % 256) as u8, prev_lower, lower);
                        c.update(((i * 2 + 1) % 256) as u8, prev_upper, upper);
                        cache.committed_used_bits[i / 8] |= 1 << (i % 8);
                    }
                    cache.committed_values.fill(Value::default());
                    cache.changed_slots.fill(0);

                    // if c1_changed {
                    //     self.c1_scalar = self.c1.to_scalar();
                    // }
                    // if c2_changed {
                    //     self.c2_scalar = self.c2.to_scalar();
                    // }

                    let combined = [
                        Scalar::from(1),
                        Scalar::from_le_bytes(&stem),
                        cache.c1.to_scalar(),
                        cache.c2.to_scalar(),
                    ];
                    cache.commitment = Commitment::new(&combined);
                }
                VerkleCommitmentInput::Inner(children) => {
                    for (i, child_id) in children.iter().enumerate() {
                        if child_mask[i / 8] & 1 << (i as u8 % 8) == 0 {
                            continue;
                        }

                        let child_commitment =
                            manager.get_read_access(*child_id)?.get_cached_commitment();
                        assert_eq!(child_commitment.dirty, 0);
                        cache.commitment.update(
                            i as u8,
                            previous_commitments[child_id].to_scalar(),
                            child_commitment.commitment().to_scalar(),
                        );
                    }
                }
            }

            cache.dirty = 0;
            lock.set_cached_commitment(cache)?;
        }
    }
    // TODO: Test
    log.clear();
    Ok(())
}
