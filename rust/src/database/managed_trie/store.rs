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
    database::{
        managed_trie::{
            DescendAction, TrieCommitment, TrieUpdateLog,
            managed_trie_node::{StoreAction, UnionManagedTrieNode},
        },
        verkle::KeyedUpdateBatch,
    },
    error::{BTResult, Error},
    node_manager::NodeManager,
    sync::RwLockWriteGuard,
    types::{HasEmptyId, HasEmptyNode},
};

/// Data for descending with an update batch into a child node.
struct DescendUpdates<'a, T, ID> {
    /// Index of this node in the parent's children, if this node has a parent.
    parent_index: Option<usize>,
    /// Write guard for the current node, or None if the node is the empty node.
    node: Option<RwLockWriteGuard<'a, T>>,
    /// ID of the current node.
    node_id: ID,
    /// Whether this node was created during the current store operation.
    is_new: bool,
    /// Updates to apply to this node.
    updates: KeyedUpdateBatch<'a>,
}

/// Stores the given key-value pair into the managed trie rooted at `root_id`.
///
/// In case the root node of the trie changes, the `root_id` guard is updated accordingly.
///
/// The updates are pushed through the tree level by level, acquiring write locks on nodes as
/// needed. During traversal, write locks on nodes in at most 3 adjacent levels will be held.
/// The lock on the root ID is held until the algorithm has descended one level into the tree.
///
/// The `update_log` is updated to reflect which nodes need to have their commitments recomputed
/// after the store operation.
pub fn store<T>(
    root_id: RwLockWriteGuard<T::Id>,
    updates: &KeyedUpdateBatch,
    manager: &impl NodeManager<Id = T::Id, Node = T>,
    update_log: &TrieUpdateLog<T::Id>,
    is_archive: bool,
) -> BTResult<(), Error>
where
    T: UnionManagedTrieNode + HasEmptyNode + Clone,
    T::Id: Copy + Eq + std::hash::Hash + std::fmt::Debug + HasEmptyId,
{
    let _span = tracy_client::span!("push updates through all levels");
    let updates = updates.borrowed(); // Ensure we have a Cow::Borrowed.
    // Wrap the root ID lock into an Option so we can release it once we are deep enough in the
    // tree.
    let mut root_id = Some(root_id);
    let mut current_node_updates = vec![DescendUpdates {
        parent_index: None,
        node: Some(manager.get_write_access(**root_id.as_ref().unwrap())?),
        node_id: **root_id.as_ref().unwrap(),
        is_new: false,
        updates,
    }];

    let mut next_node_updates = Vec::new();
    let mut depth = 0;
    // This is a no-op and only used for type inference.
    let mut parent_node_updates: Vec<_> = current_node_updates.drain(0..0).collect(); // = Vec::new();

    let mut empty_node = T::empty_node();

    while !current_node_updates.is_empty() {
        let span = tracy_client::span!("push updates through level");
        span.emit_value(depth as u64);
        let mut i = 0;
        while let Some(current_node_update) = current_node_updates.get_mut(i) {
            let next_store_action = current_node_update
                .node
                .as_ref()
                .map(|guard| &***guard)
                .unwrap_or(&empty_node)
                .next_store_action(
                    // The `updates` passed into store were converted to a Cow::Borrowed so all
                    // split updates are also borrowed which means the clone is
                    // cheap.
                    current_node_update.updates.clone(),
                    depth,
                    current_node_update.node_id,
                )?;
            // Clones the node if we are in archive mode and the node is not new and not the
            // empty node.
            let mut clone_if_archive = || -> BTResult<(), Error> {
                if is_archive
                    && !current_node_update.is_new
                    && !current_node_update.node_id.is_empty_id()
                {
                    current_node_update.node_id =
                        manager.add((*current_node_update.node.as_ref().unwrap()).clone())?;
                    current_node_update.node =
                        Some(manager.get_write_access(current_node_update.node_id)?);
                    if let Some(index) = current_node_update.parent_index {
                        parent_node_updates[index]
                            .node
                            .as_mut()
                            .unwrap()
                            .replace_child(
                                current_node_update.updates.first_key(),
                                depth - 1,
                                current_node_update.node_id,
                            )?;
                    } else {
                        **root_id.as_mut().unwrap() = current_node_update.node_id;
                    }
                }
                Ok(())
            };
            match next_store_action {
                StoreAction::Store(stores) => {
                    clone_if_archive()?;
                    let current_node_mut: &mut T = current_node_update
                        .node
                        .as_mut()
                        .map(|guard| &mut ***guard)
                        .unwrap_or(&mut empty_node);
                    let mut trie_commitment = current_node_mut.get_commitment();

                    for update in stores.iter() {
                        let prev_value = current_node_mut.store(update)?;

                        trie_commitment.store(update.key()[31] as usize, prev_value);
                    }

                    current_node_mut.set_commitment(trie_commitment)?;
                    update_log.mark_dirty(depth as usize, current_node_update.node_id);

                    i += 1;
                }
                StoreAction::Descend(descent_actions) => {
                    clone_if_archive()?;
                    let current_node_mut: &mut T = current_node_update
                        .node
                        .as_mut()
                        .map(|guard| &mut ***guard)
                        .unwrap_or(&mut empty_node);
                    let mut trie_commitment = current_node_mut.get_commitment();
                    for DescendAction { id, updates } in descent_actions {
                        let index = updates.first_key()[depth as usize] as usize;
                        trie_commitment.modify_child(index);

                        next_node_updates.push(DescendUpdates {
                            parent_index: Some(i),
                            node: if id.is_empty_id() {
                                None
                            } else {
                                Some(manager.get_write_access(id)?)
                            },
                            is_new: false,
                            node_id: id,
                            updates,
                        });
                    }

                    current_node_mut.set_commitment(trie_commitment)?;
                    update_log.mark_dirty(depth as usize, current_node_update.node_id);
                    i += 1;
                }
                StoreAction::HandleTransform(new_node) => {
                    let new_id = manager.add(new_node).unwrap();
                    if let Some(index) = current_node_update.parent_index {
                        parent_node_updates[index]
                            .node
                            .as_mut()
                            .unwrap()
                            .replace_child(
                                current_node_update.updates.first_key(),
                                depth - 1,
                                new_id,
                            )?;
                    } else {
                        **root_id.as_mut().unwrap() = new_id;
                    }

                    current_node_update.node = if new_id.is_empty_id() {
                        None
                    } else {
                        // TODO: Fetching the node again here may interfere with cache eviction (https://github.com/0xsoniclabs/sonic-admin/issues/380)
                        Some(manager.get_write_access(new_id)?)
                    };
                    let old_id = current_node_update.node_id;
                    current_node_update.node_id = new_id;
                    current_node_update.is_new = true;

                    if !is_archive {
                        manager.delete(old_id)?;
                        update_log.delete(depth as usize, old_id);
                    }

                    // No need to log the update here, we are visiting the node again next
                    // iteration.

                    // `i` stays the same, because the current node was replaced and the new node at
                    // the same index needs to be processed next.
                }
                StoreAction::HandleReparent(new_node) => {
                    let new_id = manager.add(new_node).unwrap();
                    if let Some(index) = current_node_update.parent_index {
                        parent_node_updates[index]
                            .node
                            .as_mut()
                            .unwrap()
                            .replace_child(
                                current_node_update.updates.first_key(),
                                depth - 1,
                                new_id,
                            )?;
                    } else {
                        **root_id.as_mut().unwrap() = new_id;
                    }

                    current_node_update.node = if new_id.is_empty_id() {
                        None
                    } else {
                        // TODO: Fetching the node again here may interfere with cache eviction (https://github.com/0xsoniclabs/sonic-admin/issues/380)
                        Some(manager.get_write_access(new_id)?)
                    };
                    let old_id = current_node_update.node_id;
                    current_node_update.node_id = new_id;
                    current_node_update.is_new = true;

                    update_log.move_down(depth as usize, old_id);

                    // No need to log the update here, we are visiting the node again next
                    // iteration.

                    // `i` stays the same, because the current node was replaced and the new node at
                    // the same index needs to be processed next.
                }
            }
        }

        (parent_node_updates, current_node_updates, next_node_updates) =
            (current_node_updates, next_node_updates, parent_node_updates);
        next_node_updates.clear();

        depth += 1;
        if depth == 1 {
            root_id = None;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        database::{
            managed_trie::test_utils::{
                Id, RcNodeExpectation, RcNodeManager, TestNodeCommitment, spin_until_some,
            },
            verkle::KeyedUpdate,
        },
        sync::{Arc, RwLock, thread},
        types::{Key, Value},
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

    /// Helper function which sets up expectations for a descent as next store action and the update
    /// of the commitment for all children that are descended into.
    fn descend_into(
        manager: &Arc<RcNodeManager>,
        mut node_id: Id,
        parent_id: Option<Id>,
        descent_actions: Vec<DescendAction<'static, Id>>,
        updates: &KeyedUpdateBatch<'static>,
        depth: u8,
        is_archive: bool,
    ) -> Id {
        manager.expect(
            node_id,
            RcNodeExpectation::NextStoreAction {
                updates: updates.clone(),
                depth,
                self_id: node_id,
                result: StoreAction::Descend(descent_actions.clone()),
            },
        );
        if is_archive {
            let new_node = manager.make();
            let new_node_id = new_node.id();
            manager.expect(
                node_id,
                RcNodeExpectation::Clone {
                    new_id: new_node_id,
                },
            );
            manager.expect_add(new_node);
            manager.expect_write_access(new_node_id, vec![node_id]);
            if let Some(parent_id) = parent_id {
                manager.expect(
                    parent_id,
                    RcNodeExpectation::ReplaceChild {
                        key: *updates.first_key(),
                        depth: depth - 1,
                        new: new_node_id,
                    },
                );
            }
            node_id = new_node_id;
        }
        manager.expect(
            node_id,
            RcNodeExpectation::GetCommitment {
                result: TestNodeCommitment::default(),
            },
        );
        let mut locked = if let Some(parent_id) = parent_id {
            vec![node_id, parent_id]
        } else {
            vec![node_id]
        };
        for DescendAction { id, .. } in descent_actions {
            manager.expect_write_access(id, locked.clone());
            locked.push(id);
        }
        manager.expect(
            node_id,
            RcNodeExpectation::SetCommitment {
                commitment: TestNodeCommitment::expected(updates.iter().map(|keyed_update| {
                    (
                        keyed_update.key()[depth as usize] as usize,
                        Value::default(),
                    )
                })),
            },
        );
        node_id
    }

    /// Helper function which sets up expectations for a store as next store action and the update
    /// of the commitment for all stored keys.
    #[allow(clippy::too_many_arguments)]
    fn complete_store(
        manager: &Arc<RcNodeManager>,
        parent_id: Option<Id>,
        mut node_id: <RcNodeManager as NodeManager>::Id,
        is_new: bool,
        locked_ids: &[Id],
        updates: &KeyedUpdateBatch<'static>,
        depth: u8,
        is_archive: bool,
    ) -> Id {
        manager.expect(
            node_id,
            RcNodeExpectation::NextStoreAction {
                updates: updates.clone(),
                depth,
                self_id: node_id,
                result: StoreAction::Store(updates.clone()),
            },
        );
        if is_archive && !is_new {
            let new_node = manager.make();
            let new_node_id = new_node.id();
            manager.expect(
                node_id,
                RcNodeExpectation::Clone {
                    new_id: new_node_id,
                },
            );
            manager.expect_add(new_node);
            let mut locked_ids = locked_ids.to_vec();
            locked_ids.push(node_id);
            manager.expect_write_access(new_node_id, locked_ids);
            if let Some(parent_id) = parent_id {
                manager.expect(
                    parent_id,
                    RcNodeExpectation::ReplaceChild {
                        key: *updates.first_key(),
                        depth: depth - 1,
                        new: new_node_id,
                    },
                );
            }
            node_id = new_node_id;
        }
        manager.expect(
            node_id,
            RcNodeExpectation::GetCommitment {
                result: TestNodeCommitment::default(),
            },
        );
        let prev_value = Value::from([77u8; 32]);
        for update in updates.iter() {
            manager.expect(
                node_id,
                RcNodeExpectation::Store {
                    update: update.clone(),
                    result: prev_value,
                },
            );
        }
        manager.expect(
            node_id,
            RcNodeExpectation::SetCommitment {
                commitment: TestNodeCommitment::expected(
                    updates.iter().map(|keyed_update| {
                        (keyed_update.key()[depth as usize] as usize, prev_value)
                    }),
                ),
            },
        );
        node_id
    }

    #[rstest_reuse::template]
    #[rstest::rstest]
    #[case::live(false)]
    #[case::archive(true)]
    fn is_archive(#[case] is_archive: bool) {}

    #[rstest_reuse::apply(is_archive)]
    fn store_sets_value_and_marks_node_and_commitment_and_log_as_dirty(#[case] is_archive: bool) {
        let (manager, log, mut root_id, root_id_lock) = boilerplate();

        let updates = KeyedUpdateBatch::from_key_value_pairs(&[(KEY, VALUE)]);
        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &updates, &*manager, &log, is_archive).unwrap();
            });

            manager.expect_write_access(root_id, vec![]);
            manager.expect(
                root_id,
                RcNodeExpectation::NextStoreAction {
                    updates: updates.clone(),
                    depth: 0,
                    self_id: root_id,
                    result: StoreAction::Store(updates.clone()),
                },
            );
            if is_archive {
                let new_root = manager.make();
                let new_root_id = new_root.id();
                manager.expect(
                    root_id,
                    RcNodeExpectation::Clone {
                        new_id: new_root_id,
                    },
                );
                manager.expect_add(new_root);
                manager.expect_write_access(new_root_id, vec![root_id]);
                root_id = new_root_id;
            }
            manager.expect(
                root_id,
                RcNodeExpectation::GetCommitment {
                    result: TestNodeCommitment::default(),
                },
            );
            let prev_value = Value::from([77u8; 32]);
            manager.expect(
                root_id,
                RcNodeExpectation::Store {
                    update: KeyedUpdate::FullSlot {
                        key: KEY,
                        value: VALUE,
                    },
                    result: prev_value,
                },
            );
            manager.expect(
                root_id,
                RcNodeExpectation::SetCommitment {
                    commitment: TestNodeCommitment::expected_single(KEY[31] as usize, prev_value),
                },
            );
            manager.wait_for_unlock(root_id);
            assert!(manager.is_dirty(root_id));
            assert_eq!(log.count(), 1);
            assert_eq!(log.dirty_nodes(0), [root_id]);
        });
    }

    #[rstest_reuse::apply(is_archive)]
    fn descending_marks_node_and_commitment_and_log_as_dirty(#[case] is_archive: bool) {
        let (manager, log, mut root_id, root_id_lock) = boilerplate();
        let mut child_id = manager.insert(manager.make());

        let updates = KeyedUpdateBatch::from_key_value_pairs(&[(KEY, VALUE)]);
        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &updates, &*manager, &log, is_archive).unwrap();
            });

            manager.expect_write_access(root_id, vec![]);
            root_id = descend_into(
                &manager,
                root_id,
                None,
                vec![DescendAction {
                    updates: updates.clone(),
                    id: child_id,
                }],
                &updates,
                0,
                is_archive,
            );

            child_id = complete_store(
                &manager,
                Some(root_id),
                child_id,
                false,
                &[root_id],
                &updates,
                1,
                is_archive,
            );
            manager.wait_for_unlock(root_id);

            // While we did not store anything in the root directly, it should be marked dirty.
            assert!(manager.is_dirty(root_id));
            assert_eq!(log.count(), 2);
            assert_eq!(log.dirty_nodes(0), [root_id]);
        });
    }

    #[rstest_reuse::apply(is_archive)]
    fn descending_one_level_deep_releases_lock_on_root_id(#[case] is_archive: bool) {
        let (manager, log, mut root_id, root_id_lock) = boilerplate();
        let mut child_id = manager.insert(manager.make());

        let updates = KeyedUpdateBatch::from_key_value_pairs(&[(KEY, VALUE)]);
        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &updates, &*manager, &log, is_archive).unwrap();
            });

            manager.expect_write_access(root_id, vec![]);
            assert!(root_id_lock.try_read().is_err());
            root_id = descend_into(
                &manager,
                root_id,
                None,
                vec![DescendAction {
                    id: child_id,
                    updates: updates.clone(),
                }],
                &updates,
                0,
                is_archive,
            );
            let _guard = spin_until_some(
                || root_id_lock.try_read().ok(),
                "timed out waiting for root_id to be unlocked",
            );
            child_id = complete_store(
                &manager,
                Some(root_id),
                child_id,
                false,
                &[root_id],
                &updates,
                1,
                is_archive,
            );
        });
    }

    #[rstest_reuse::apply(is_archive)]
    fn transform_adds_new_node_and_deletes_old_one_and_updates_parent(#[case] is_archive: bool) {
        let (manager, log, mut root_id, root_id_lock) = boilerplate();
        let child_id = manager.insert(manager.make());

        let updates = KeyedUpdateBatch::from_key_value_pairs(&[(KEY, VALUE)]);
        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &updates, &*manager, &log, is_archive).unwrap();
            });

            manager.expect_write_access(root_id, vec![]);
            root_id = descend_into(
                &manager,
                root_id,
                None,
                vec![DescendAction {
                    id: child_id,
                    updates: updates.clone(),
                }],
                &updates,
                0,
                is_archive,
            );

            let new_child = manager.make();
            let mut new_child_id = new_child.id();
            manager.expect(
                child_id,
                RcNodeExpectation::NextStoreAction {
                    updates: updates.clone(),
                    depth: 1,
                    self_id: child_id,
                    result: StoreAction::HandleTransform(new_child.clone_non_rc()),
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
            manager.wait_for_unlock(child_id);
            if !is_archive {
                manager.expect_delete(child_id);
            }

            new_child_id = complete_store(
                &manager,
                Some(root_id),
                new_child_id,
                true,
                &[root_id],
                &updates,
                1,
                is_archive,
            );
            manager.wait_for_unlock(new_child_id);

            // The old child should be deleted from the log
            assert_eq!(log.count(), 2);
            assert_eq!(log.dirty_nodes(0), [root_id]);
            assert_eq!(log.dirty_nodes(1), [new_child_id]);
        });
    }

    #[rstest_reuse::apply(is_archive)]
    fn transform_on_root_updates_root_id(#[case] is_archive: bool) {
        let (manager, log, root_id, root_id_lock) = boilerplate();

        let updates = KeyedUpdateBatch::from_key_value_pairs(&[(KEY, VALUE)]);
        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &updates, &*manager, &log, is_archive).unwrap();
            });

            manager.expect_write_access(root_id, vec![]);
            let new_root = manager.make();
            let mut new_root_id = new_root.id();
            manager.expect(
                root_id,
                RcNodeExpectation::NextStoreAction {
                    updates: updates.clone(),
                    depth: 0,
                    self_id: root_id,
                    result: StoreAction::HandleTransform(new_root.clone_non_rc()),
                },
            );

            manager.expect_add(new_root);
            manager.expect_write_access(new_root_id, vec![root_id]);
            manager.wait_for_unlock(root_id);
            if !is_archive {
                manager.expect_delete(root_id);
            }

            new_root_id = complete_store(
                &manager,
                None,
                new_root_id,
                true,
                &[root_id],
                &updates,
                0,
                is_archive,
            );
            manager.wait_for_unlock(new_root_id);

            // The root id should be updated to the new id
            let updated_root_id = *root_id_lock.read().unwrap();
            assert_eq!(updated_root_id, new_root_id);
        });
    }

    #[rstest_reuse::apply(is_archive)]
    fn reparent_adds_new_node_and_updates_parent_without_marking_original_child_as_dirty(
        #[case] is_archive: bool,
    ) {
        let (manager, log, mut root_id, root_id_lock) = boilerplate();
        let child_id = manager.insert(manager.make());

        let updates = KeyedUpdateBatch::from_key_value_pairs(&[(KEY, VALUE)]);
        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &updates, &*manager, &log, is_archive).unwrap();
            });

            manager.expect_write_access(root_id, vec![]);
            assert!(!manager.is_dirty(child_id));
            root_id = descend_into(
                &manager,
                root_id,
                None,
                vec![DescendAction {
                    id: child_id,
                    updates: updates.clone(),
                }],
                &updates,
                0,
                is_archive,
            );

            let new_parent_node = manager.make();
            let mut new_parent_id = new_parent_node.id();
            manager.expect(
                child_id,
                RcNodeExpectation::NextStoreAction {
                    updates: updates.clone(),
                    depth: 1,
                    self_id: child_id,
                    result: StoreAction::HandleReparent(new_parent_node.clone_non_rc()),
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
            manager.wait_for_unlock(child_id);

            new_parent_id = complete_store(
                &manager,
                Some(root_id),
                new_parent_id,
                true,
                &[root_id],
                &updates,
                1,
                is_archive,
            );
            manager.wait_for_unlock(new_parent_id);

            // The original child should not be marked as dirty
            assert!(!manager.is_dirty(child_id));

            assert_eq!(log.count(), 2);
            assert_eq!(log.dirty_nodes(0), [root_id]);
            assert_eq!(log.dirty_nodes(1), [new_parent_id]);
        });
    }

    #[rstest_reuse::apply(is_archive)]
    fn reparenting_root_updates_root_id(#[case] is_archive: bool) {
        let (manager, log, root_id, root_id_lock) = boilerplate();

        let updates = KeyedUpdateBatch::from_key_value_pairs(&[(KEY, VALUE)]);
        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &updates, &*manager, &log, is_archive).unwrap();
            });

            manager.expect_write_access(root_id, vec![]);
            let new_root = manager.make();
            let mut new_root_id = new_root.id();
            manager.expect(
                root_id,
                RcNodeExpectation::NextStoreAction {
                    updates: updates.clone(),
                    depth: 0,
                    self_id: root_id,
                    result: StoreAction::HandleReparent(new_root.clone_non_rc()),
                },
            );

            manager.expect_add(new_root);
            manager.expect_write_access(new_root_id, vec![root_id]);
            manager.wait_for_unlock(root_id);

            new_root_id = complete_store(
                &manager,
                Some(root_id),
                new_root_id,
                true,
                &[root_id],
                &updates,
                0,
                is_archive,
            );
            manager.wait_for_unlock(new_root_id);

            // The root id should be updated to the new id
            let updated_root_id = *root_id_lock.read().unwrap();
            assert_eq!(updated_root_id, new_root_id);
        });
    }

    #[rstest_reuse::apply(is_archive)]
    fn store_applies_storeaction_store_immediately(#[case] is_archive: bool) {
        let (manager, log, mut root_id, root_id_lock) = boilerplate();

        let key1 = [1; 32];
        let key2 = [2; 32];
        let updates = KeyedUpdateBatch::from_key_value_pairs(&[(key1, VALUE), (key2, VALUE)]);
        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &updates, &*manager, &log, is_archive).unwrap();
            });

            manager.expect_write_access(root_id, vec![]);
            manager.expect(
                root_id,
                RcNodeExpectation::NextStoreAction {
                    updates: updates.clone(),
                    depth: 0,
                    self_id: root_id,
                    result: StoreAction::Store(updates.clone()),
                },
            );
            if is_archive {
                let new_root = manager.make();
                let new_root_id = new_root.id();
                manager.expect(
                    root_id,
                    RcNodeExpectation::Clone {
                        new_id: new_root_id,
                    },
                );
                manager.expect_add(new_root);
                manager.expect_write_access(new_root_id, vec![root_id]);
                root_id = new_root_id;
            }
            manager.expect(
                root_id,
                RcNodeExpectation::GetCommitment {
                    result: TestNodeCommitment::default(),
                },
            );
            let prev_value = Value::from([77u8; 32]);
            manager.expect(
                root_id,
                RcNodeExpectation::Store {
                    update: KeyedUpdate::FullSlot {
                        key: key1,
                        value: VALUE,
                    },
                    result: prev_value,
                },
            );
            manager.expect(
                root_id,
                RcNodeExpectation::Store {
                    update: KeyedUpdate::FullSlot {
                        key: key2,
                        value: VALUE,
                    },
                    result: prev_value,
                },
            );
            manager.expect(
                root_id,
                RcNodeExpectation::SetCommitment {
                    commitment: TestNodeCommitment::expected(
                        [
                            (key1[31] as usize, prev_value),
                            (key2[31] as usize, prev_value),
                        ]
                        .into_iter(),
                    ),
                },
            );
            manager.wait_for_unlock(root_id);
        });
    }

    #[rstest_reuse::apply(is_archive)]
    fn store_handles_storeaction_descend_by_processing_it_in_the_next_level(
        #[case] is_archive: bool,
    ) {
        let (manager, log, mut root_id, root_id_lock) = boilerplate();
        let mut child1_id = manager.insert(manager.make());
        let mut child2_id = manager.insert(manager.make());

        let key1 = [1; 32];
        let key2 = [2; 32];
        let updates = KeyedUpdateBatch::from_key_value_pairs(&[(key1, VALUE), (key2, VALUE)]);
        let sub_update1 = KeyedUpdateBatch::from_key_value_pairs(&[(key1, VALUE)]);
        let sub_update2 = KeyedUpdateBatch::from_key_value_pairs(&[(key2, VALUE)]);
        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &updates, &*manager, &log, is_archive).unwrap();
            });

            manager.expect_write_access(root_id, vec![]);
            root_id = descend_into(
                &manager,
                root_id,
                None,
                vec![
                    DescendAction {
                        id: child1_id,
                        updates: sub_update1.clone(),
                    },
                    DescendAction {
                        id: child2_id,
                        updates: sub_update2.clone(),
                    },
                ],
                &updates,
                0,
                is_archive,
            );

            child1_id = complete_store(
                &manager,
                Some(root_id),
                child1_id,
                false,
                &[root_id, child2_id],
                &sub_update1,
                1,
                is_archive,
            );
            child2_id = complete_store(
                &manager,
                Some(root_id),
                child2_id,
                false,
                &[root_id, child1_id],
                &sub_update2,
                1,
                is_archive,
            );
            manager.wait_for_unlock(root_id);
        });
    }

    #[rstest_reuse::apply(is_archive)]
    fn store_applies_storeaction_transform_immediately_and_then_processes_transformed_node(
        #[case] is_archive: bool,
    ) {
        let (manager, log, root_id, root_id_lock) = boilerplate();

        let key1 = [1; 32];
        let key2 = [2; 32];
        let updates = KeyedUpdateBatch::from_key_value_pairs(&[(key1, VALUE), (key2, VALUE)]);
        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &updates, &*manager, &log, is_archive).unwrap();
            });

            manager.expect_write_access(root_id, vec![]);
            let transformed_node = manager.make();
            let mut transformed_node_id = transformed_node.id();
            manager.expect(
                root_id,
                RcNodeExpectation::NextStoreAction {
                    updates: updates.clone(),
                    depth: 0,
                    self_id: root_id,
                    result: StoreAction::HandleTransform(transformed_node.clone_non_rc()),
                },
            );
            manager.expect_add(transformed_node);
            manager.expect_write_access(transformed_node_id, vec![root_id]);
            manager.wait_for_unlock(root_id);
            if !is_archive {
                manager.expect_delete(root_id);
            }
            transformed_node_id = complete_store(
                &manager,
                Some(root_id),
                transformed_node_id,
                true,
                &[root_id],
                &updates,
                0,
                is_archive,
            );
            manager.wait_for_unlock(transformed_node_id);
        });
    }

    #[rstest_reuse::apply(is_archive)]
    fn store_applies_storeaction_reparent_and_processes_new_parent_node_next(
        #[case] is_archive: bool,
    ) {
        let (manager, log, root_id, root_id_lock) = boilerplate();
        let key1 = [1; 32];
        let key2 = [2; 32];
        let updates = KeyedUpdateBatch::from_key_value_pairs(&[(key1, VALUE), (key2, VALUE)]);
        thread::scope(|s| {
            s.spawn(|| {
                let root_id_guard = root_id_lock.write().unwrap();
                store(root_id_guard, &updates, &*manager, &log, is_archive).unwrap();
            });

            manager.expect_write_access(root_id, vec![]);
            let new_parent_node = manager.make();
            let mut new_parent_node_id = new_parent_node.id();
            manager.expect(
                root_id,
                RcNodeExpectation::NextStoreAction {
                    updates: updates.clone(),
                    depth: 0,
                    self_id: root_id,
                    result: StoreAction::HandleReparent(new_parent_node.clone_non_rc()),
                },
            );
            manager.expect_add(new_parent_node);
            manager.expect_write_access(new_parent_node_id, vec![root_id]);
            manager.wait_for_unlock(root_id);
            new_parent_node_id = complete_store(
                &manager,
                Some(root_id),
                new_parent_node_id,
                true,
                &[root_id],
                &updates,
                0,
                is_archive,
            );
            manager.wait_for_unlock(new_parent_node_id);
        });
    }
}
