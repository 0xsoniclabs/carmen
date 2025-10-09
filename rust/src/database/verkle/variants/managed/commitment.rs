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
            compute_commitment::compute_leaf_node_commitment,
            crypto::Commitment,
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
                VerkleCommitmentInput::Leaf(values, used_bits, stem) => {
                    // TODO: Consider caching leaf node commitments https://github.com/0xsoniclabs/sonic-admin/issues/386
                    cache.commitment = compute_leaf_node_commitment(&values, &used_bits, &stem);
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
