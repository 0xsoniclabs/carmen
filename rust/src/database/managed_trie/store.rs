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
        TrieUpdateLog,
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
    update_log: &mut TrieUpdateLog<T::Id>,
) -> Result<(), Error>
where
    T::Id: Copy + Eq + std::hash::Hash,
{
    let mut parent_lock = None;
    let mut current_id = *root_id;
    // TODO Test: Traversing tree with set sets dirty flag (except on leaf in case of split)
    let mut current_lock = manager.get_write_access(current_id)?;
    let mut depth = 0;

    loop {
        match current_lock.can_store(key, depth)? {
            CanStoreResult::Yes => {
                let prev_value = current_lock.store(key, value)?;
                let mut cache = current_lock.get_cached_commitment();
                cache.store(key[31] as usize, prev_value);
                current_lock.set_cached_commitment(cache)?;
                update_log.add(depth, current_id, key[31]);
                return Ok(());
            }
            CanStoreResult::Descend(new_id) => {
                let mut cache = current_lock.get_cached_commitment();
                cache.dirty = 1;
                current_lock.set_cached_commitment(cache)?;
                update_log.add(depth, current_id, key[depth as usize]);

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
                update_log.delete(current_id);
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
                current_lock = manager.get_write_access(new_id)?;
                current_id = new_id;

                // No need to log the update here, we are visiting the node again next iteration.
            }
        }
    }
}
