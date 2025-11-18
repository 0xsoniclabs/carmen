// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use crate::{
    database::managed_trie::{
        TrieCommitment, TrieUpdateLog,
        managed_trie_node::{StoreAction, UnionManagedTrieNode},
    },
    error::{BTResult, Error},
    node_manager::NodeManager,
    sync::RwLockWriteGuard,
    types::{Key, Value},
};

/// Stores the given key-value pair into the managed trie rooted at `root_id`.
///
/// In case the root node of the trie changes, the `root_id` guard is updated accordingly.
///
/// At most two nodes are write-locked at any given time during tree traversal, allowing for
/// concurrent lookup/store operations on different parts of the tree. The lock on the root ID is
/// held until the algorithm has descended two levels into the tree.
///
/// The `update_log` is updated to reflect which nodes need to have their commitments recomputed
/// after the store operation.
pub fn store<T>(
    root_id: RwLockWriteGuard<T::Id>,
    key: &Key,
    value: &Value,
    manager: &impl NodeManager<Id = T::Id, Node = T>,
    update_log: &TrieUpdateLog<T::Id>,
) -> BTResult<(), Error>
where
    T: UnionManagedTrieNode,
    T::Id: Copy + Eq + std::hash::Hash + std::fmt::Debug,
{
    let mut parent_lock = None;
    // Wrap the root ID lock into an Option so we can release it once we are deep enough in the tree
    let mut root_id = Some(root_id);
    let mut current_id = **root_id.as_ref().unwrap();
    let mut current_lock = manager.get_write_access(current_id)?;
    let mut depth = 0;

    loop {
        match current_lock.next_store_action(key, depth, current_id)? {
            StoreAction::Store { index } => {
                let prev_value = current_lock.store(key, value)?;

                let mut trie_commitment = current_lock.get_commitment();
                trie_commitment.store(index, prev_value);
                current_lock.set_commitment(trie_commitment)?;
                update_log.mark_dirty(depth as usize, current_id);

                return Ok(());
            }
            StoreAction::Descend { index, id } => {
                let mut trie_commitment = current_lock.get_commitment();
                trie_commitment.modify_child(index);
                current_lock.set_commitment(trie_commitment)?;
                update_log.mark_dirty(depth as usize, current_id);

                let had_parent_lock = parent_lock.is_some();
                parent_lock = Some(current_lock);
                current_lock = manager.get_write_access(id)?;
                current_id = id;
                depth += 1;

                if had_parent_lock {
                    root_id = None;
                }
            }
            StoreAction::HandleTransform(new_node) => {
                let new_id = manager.add(new_node).unwrap();
                if let Some(lock) = &mut parent_lock {
                    lock.replace_child(key, depth - 1, new_id)?;
                } else {
                    **root_id.as_mut().unwrap() = new_id;
                }

                // TODO: Fetching the node again here may interfere with cache eviction (https://github.com/0xsoniclabs/sonic-admin/issues/380)
                current_lock = manager.get_write_access(new_id)?;
                manager.delete(current_id)?;
                update_log.delete(depth as usize, current_id);
                current_id = new_id;

                // No need to log the update here, we are visiting the node again next iteration.
            }
            StoreAction::HandleReparent(new_node) => {
                let new_id = manager.add(new_node).unwrap();
                if let Some(lock) = &mut parent_lock {
                    lock.replace_child(key, depth - 1, new_id)?;
                } else {
                    **root_id.as_mut().unwrap() = new_id;
                }

                // TODO: Fetching the node again here may interfere with cache eviction (https://github.com/0xsoniclabs/sonic-admin/issues/380)
                current_lock = manager.get_write_access(new_id)?;
                update_log.move_down(depth as usize, current_id);
                current_id = new_id;

                // No need to log the update here, we are visiting the node again next iteration.
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        database::managed_trie::test_utils::{
            Id, RcNodeExpectation, RcNodeManager, TestNodeCommitment,
        },
        sync::{Arc, RwLock, thread},
    };

    const KEY: Key = [7u8; 32];
    const VALUE: Value = [42u8; 32];

    /// Sets up common boilerplate for store tests.
    fn boilerplate() -> (Arc<RcNodeManager>, TrieUpdateLog<Id>, Id, RwLock<Id>) {
        let manager = Arc::new(RcNodeManager::new());
        let log = TrieUpdateLog::<Id>::new();
        let root_id = manager.insert(manager.make());
        let root_id_lock = RwLock::new(root_id);
        (manager, log, root_id, root_id_lock)
    }

    /// Helper function for setting up expectations for a commitment update.
    fn expect_commitment_update(
        manager: &RcNodeManager,
        node_id: Id,
        slot_idx: usize,
        prev_value: Value,
    ) {
        manager.expect(
            node_id,
            RcNodeExpectation::GetCommitment {
                result: TestNodeCommitment::default(),
            },
        );
        manager.expect(
            node_id,
            RcNodeExpectation::SetCommitment {
                commitment: TestNodeCommitment::expected(slot_idx, prev_value),
            },
        );
    }

    /// Helper function for descending into a child node.
    fn descend_into(manager: &RcNodeManager, parent_id: Id, child_id: Id, key: Key, depth: u8) {
        let child_idx = 55;
        manager.expect(
            parent_id,
            RcNodeExpectation::NextStoreAction {
                key,
                depth,
                self_id: parent_id,
                result: StoreAction::Descend {
                    index: child_idx,
                    id: child_id,
                },
            },
        );
        expect_commitment_update(manager, parent_id, child_idx, Value::default());
        manager.expect_write_access(child_id, vec![parent_id]);
    }

    /// Helper function for completing a store operation on the given node.
    fn complete_store(
        manager: &RcNodeManager,
        node_id: <RcNodeManager as NodeManager>::Id,
        key: Key,
        value: Value,
        depth: u8,
    ) {
        let slot_idx = 77;
        manager.expect(
            node_id,
            RcNodeExpectation::NextStoreAction {
                key,
                depth,
                self_id: node_id,
                result: StoreAction::Store { index: slot_idx },
            },
        );
        let prev_value = Value::from([77u8; 32]);
        manager.expect(
            node_id,
            RcNodeExpectation::Store {
                key,
                value,
                result: prev_value,
            },
        );
        expect_commitment_update(manager, node_id, slot_idx, prev_value);
    }

    #[test]
    fn store_sets_value_and_marks_node_and_commitment_and_log_as_dirty() {
        let (manager, log, root_id, root_id_lock) = boilerplate();

        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &KEY, &VALUE, &*manager, &log).unwrap();
            });

            let slot_idx = 99;
            manager.expect_write_access(root_id, vec![]);
            manager.expect(
                root_id,
                RcNodeExpectation::NextStoreAction {
                    key: KEY,
                    depth: 0,
                    self_id: root_id,
                    result: StoreAction::Store { index: slot_idx },
                },
            );
            let prev_value = Value::from([77u8; 32]);
            manager.expect(
                root_id,
                RcNodeExpectation::Store {
                    key: KEY,
                    value: VALUE,
                    result: prev_value,
                },
            );
            expect_commitment_update(&manager, root_id, slot_idx, prev_value);
            manager.wait_for_unlock(root_id);
            assert!(manager.is_dirty(root_id));
            assert_eq!(log.count(), 1);
            assert_eq!(log.dirty_nodes(0), [root_id]);
        });
    }

    #[test]
    fn descending_marks_node_and_commitment_and_log_as_dirty() {
        let (manager, log, root_id, root_id_lock) = boilerplate();
        let child_id = manager.insert(manager.make());

        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &KEY, &VALUE, &*manager, &log).unwrap();
            });

            let child_idx = 17;
            manager.expect_write_access(root_id, vec![]);
            manager.expect(
                root_id,
                RcNodeExpectation::NextStoreAction {
                    key: KEY,
                    depth: 0,
                    self_id: root_id,
                    result: StoreAction::Descend {
                        index: child_idx,
                        id: child_id,
                    },
                },
            );
            expect_commitment_update(&manager, root_id, child_idx, Value::default());

            // Root should be locked while we acquire lock on the child
            manager.expect_write_access(child_id, vec![root_id]);
            complete_store(&manager, child_id, KEY, VALUE, 1);
            manager.wait_for_unlock(child_id);

            // While we did not store anything in the root directly, it should be marked dirty.
            assert!(manager.is_dirty(root_id));
            assert_eq!(log.count(), 2);
            assert_eq!(log.dirty_nodes(0), [root_id]);
        });
    }

    #[test]
    fn descending_two_levels_deep_releases_lock_on_root_id() {
        let (manager, log, root_id, root_id_lock) = boilerplate();
        let child_id = manager.insert(manager.make());
        let grandchild_id = manager.insert(manager.make());

        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &KEY, &VALUE, &*manager, &log).unwrap();
            });

            manager.expect_write_access(root_id, vec![]);
            assert!(root_id_lock.try_read().is_err());
            descend_into(&manager, root_id, child_id, KEY, 0);
            assert!(root_id_lock.try_read().is_err());
            descend_into(&manager, child_id, grandchild_id, KEY, 1);
            assert!(root_id_lock.try_read().is_ok());
            complete_store(&manager, grandchild_id, KEY, VALUE, 2);
        });
    }

    #[test]
    fn transform_adds_new_node_and_deletes_old_one_and_updates_parent() {
        let (manager, log, root_id, root_id_lock) = boilerplate();
        let child_id = manager.insert(manager.make());

        // We insert the child node into the log to simulate a case where we first modify the node
        // and then transform it in a subsequent store operation (e.g. because a sparse leaf is
        // becoming too large).
        // This way we can verify that the old id is removed from the log after the transform.
        log.mark_dirty(1, child_id);

        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &KEY, &VALUE, &*manager, &log).unwrap();
            });

            manager.expect_write_access(root_id, vec![]);
            descend_into(&manager, root_id, child_id, KEY, 0);

            let new_child = manager.make();
            let new_child_id = new_child.id();
            manager.expect(
                child_id,
                RcNodeExpectation::NextStoreAction {
                    key: KEY,
                    depth: 1,
                    self_id: child_id,
                    result: StoreAction::HandleTransform(new_child.clone()),
                },
            );

            manager.expect_add(new_child);
            manager.expect(
                root_id,
                RcNodeExpectation::ReplaceChild {
                    key: KEY,
                    depth: 0,
                    new: new_child_id,
                },
            );

            manager.expect_write_access(new_child_id, vec![root_id, child_id]);
            // At this point the lock on the old child should be released
            manager.expect_locked(&[root_id, new_child_id]);
            manager.expect_delete(child_id);

            complete_store(&manager, new_child_id, KEY, VALUE, 1);
            manager.wait_for_unlock(new_child_id);

            // The old child should be deleted from the log
            assert_eq!(log.count(), 2);
            assert_eq!(log.dirty_nodes(0), [root_id]);
            assert_eq!(log.dirty_nodes(1), [new_child_id]);
        });
    }

    #[test]
    fn transform_on_root_updates_root_id() {
        let (manager, log, root_id, root_id_lock) = boilerplate();

        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &KEY, &VALUE, &*manager, &log).unwrap();
            });

            manager.expect_write_access(root_id, vec![]);
            let new_root = manager.make();
            let new_root_id = new_root.id();
            manager.expect(
                root_id,
                RcNodeExpectation::NextStoreAction {
                    key: KEY,
                    depth: 0,
                    self_id: root_id,
                    result: StoreAction::HandleTransform(new_root.clone()),
                },
            );

            manager.expect_add(new_root);
            manager.expect_write_access(new_root_id, vec![root_id]);
            manager.expect_locked(&[new_root_id]);
            manager.expect_delete(root_id);

            complete_store(&manager, new_root_id, KEY, VALUE, 0);
            manager.wait_for_unlock(new_root_id);

            // The root id should be updated to the new id
            let updated_root_id = *root_id_lock.read().unwrap();
            assert_eq!(updated_root_id, new_root_id);
        });
    }

    #[test]
    fn reparent_adds_new_node_and_updates_parent_without_marking_original_child_as_dirty() {
        let (manager, log, root_id, root_id_lock) = boilerplate();
        let child_id = manager.insert(manager.make());

        // We insert the child node into the log to simulate a case where we first modify the node
        // and then reparent it in a subsequent store operation.
        // This way we can verify that the id is moved down a level in the log after reparenting.
        // Note that this is different from the dirty flag in the node manager, which is not set.
        log.mark_dirty(1, child_id);

        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &KEY, &VALUE, &*manager, &log).unwrap();
            });

            manager.expect_write_access(root_id, vec![]);
            assert!(!manager.is_dirty(child_id));
            descend_into(&manager, root_id, child_id, KEY, 0);

            let new_parent_node = manager.make();
            let new_parent_id = new_parent_node.id();
            manager.expect(
                child_id,
                RcNodeExpectation::NextStoreAction {
                    key: KEY,
                    depth: 1,
                    self_id: child_id,
                    result: StoreAction::HandleReparent(new_parent_node.clone()),
                },
            );

            manager.expect_add(new_parent_node);
            manager.expect(
                root_id,
                RcNodeExpectation::ReplaceChild {
                    key: KEY,
                    depth: 0,
                    new: new_parent_id,
                },
            );

            manager.expect_write_access(new_parent_id, vec![root_id, child_id]);
            // At this point the lock on the original child should be released
            manager.expect_locked(&[root_id, new_parent_id]);

            complete_store(&manager, new_parent_id, KEY, VALUE, 1);
            manager.wait_for_unlock(new_parent_id);

            // The original child should not be marked as dirty
            assert!(!manager.is_dirty(child_id));

            // All three nodes should be present in the log, but the original child should have
            // moved down.
            assert_eq!(log.count(), 3);
            assert_eq!(log.dirty_nodes(0), [root_id]);
            assert_eq!(log.dirty_nodes(1), [new_parent_id]);
            assert_eq!(log.dirty_nodes(2), [child_id]);
        });
    }

    #[test]
    fn reparenting_root_updates_root_id() {
        let (manager, log, root_id, root_id_lock) = boilerplate();

        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &KEY, &VALUE, &*manager, &log).unwrap();
            });

            manager.expect_write_access(root_id, vec![]);
            let new_root = manager.make();
            let new_root_id = new_root.id();
            manager.expect(
                root_id,
                RcNodeExpectation::NextStoreAction {
                    key: KEY,
                    depth: 0,
                    self_id: root_id,
                    result: StoreAction::HandleReparent(new_root.clone()),
                },
            );

            manager.expect_add(new_root);
            manager.expect_write_access(new_root_id, vec![root_id]);
            manager.expect_locked(&[new_root_id]);

            complete_store(&manager, new_root_id, KEY, VALUE, 0);
            manager.wait_for_unlock(new_root_id);

            // The root id should be updated to the new id
            let updated_root_id = *root_id_lock.read().unwrap();
            assert_eq!(updated_root_id, new_root_id);
        });
    }
}
