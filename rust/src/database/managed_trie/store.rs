// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::sync::RwLockWriteGuard;

use crate::{
    database::managed_trie::{
        TrieCommitment, TrieUpdateLog,
        managed_trie_node::{CanStoreResult, UnionManagedTrieNode},
    },
    error::Error,
    node_manager::NodeManager,
    types::{Key, Value},
};

pub fn store<T: UnionManagedTrieNode>(
    mut root_id: RwLockWriteGuard<T::Id>,
    key: &Key,
    value: &Value,
    manager: &impl NodeManager<Id = T::Id, NodeType = T>,
    update_log: &TrieUpdateLog<T::Id>,
) -> Result<(), Error>
where
    T::Id: Copy + Eq + std::hash::Hash + std::fmt::Debug,
{
    let mut parent_lock = None;
    let mut current_id = *root_id;
    // TODO Test: Traversing tree with set sets dirty flag (except on leaf in case of split)
    let mut current_lock = manager.get_write_access(current_id)?;
    let mut depth = 0;

    loop {
        match current_lock.can_store(key, depth)? {
            CanStoreResult::Yes(slot_idx) => {
                let prev_value = current_lock.store(key, value)?;
                let mut trie_commitment = current_lock.get_commitment();
                trie_commitment.store(slot_idx, prev_value);
                current_lock.set_commitment(trie_commitment)?;
                update_log.add(depth as usize, current_id);
                return Ok(());
            }
            CanStoreResult::Descend(child_idx, new_id) => {
                let mut trie_commitment = current_lock.get_commitment();
                trie_commitment.modify_child(child_idx);
                current_lock.set_commitment(trie_commitment)?;
                update_log.add(depth as usize, current_id);

                parent_lock = Some(current_lock);
                current_lock = manager.get_write_access(new_id)?;
                current_id = new_id;
                depth += 1;
            }
            // TODO TEST: Parent lock is held for entire duration of transform / replace child
            CanStoreResult::Transform => {
                let new_node = current_lock.transform(key, depth)?;
                let new_id = manager.add(new_node).unwrap();
                if let Some(lock) = &mut parent_lock {
                    lock.replace_child(key, depth - 1, new_id)?;
                } else {
                    *root_id = new_id;
                    // TODO: drop lock on root_id here?
                }

                // TODO: Fetching the node again here may interfere with cache eviction (https://github.com/0xsoniclabs/sonic-admin/issues/380)
                current_lock = manager.get_write_access(new_id)?;
                // TODO TEST: Transform releases lock on current id before calling delete
                manager.delete(current_id)?;
                update_log.delete(depth as usize, current_id);
                current_id = new_id;

                // No need to log the update here, we are visiting the node again next iteration.
            }
            CanStoreResult::Reparent => {
                let new_node = current_lock.reparent(key, depth, current_id)?;
                let new_id = manager.add(new_node).unwrap();
                if let Some(lock) = &mut parent_lock {
                    lock.replace_child(key, depth - 1, new_id)?;
                } else {
                    *root_id = new_id;
                    // TODO: drop lock on root_id here?
                }
                // TODO TEST: We need to update a leaf and then reparent it, before recomputing
                //            commitments
                update_log.move_down(current_id, depth as usize);
                current_lock = manager.get_write_access(new_id)?;
                current_id = new_id;

                // No need to log the update here, we are visiting the node again next iteration.
            }
        }
    }
}

#[cfg(test)]
mod tests {

    // TODO TEST: Reparenting does not mark child as dirty
    // TODO TEST: Root lock is released after traversing far enough into tree

    // TODO TEST: Concurrent read/write access (instrument an implementation with channels)

    // TODO: Consider having a faux MPT-style node here as well (using hash commitments)
}
