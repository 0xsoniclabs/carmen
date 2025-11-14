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
    database::managed_trie::{ManagedTrieNode, managed_trie_node::LookupResult},
    error::{BTResult, Error},
    node_manager::NodeManager,
    types::{Key, Value},
};

/// Looks up the value associated with `key` in the managed trie rooted at `root_id`.
///
/// At most two nodes are read-locked at any given time during tree traversal, allowing for
/// concurrent lookup/store operations on different parts of the tree.
pub fn lookup<T: ManagedTrieNode>(
    root_id: T::Id,
    key: &Key,
    manager: &impl NodeManager<Id = T::Id, Node = T>,
) -> BTResult<Value, Error> {
    let mut current_lock = manager.get_read_access(root_id)?;
    let mut depth = 0;

    loop {
        match current_lock.lookup(key, depth)? {
            LookupResult::Value(v) => return Ok(v),
            LookupResult::Node(node_id) => {
                let next_lock = manager.get_read_access(node_id)?;
                current_lock = next_lock;
                depth += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, thread};

    use super::*;
    use crate::database::managed_trie::test_utils::{RcNodeExpectation, RcNodeManager};

    #[test]
    fn lookup_returns_value_from_node() {
        let manager = Arc::new(RcNodeManager::new());
        let key = Key::from([7u8; 32]);
        let value = Value::from([42u8; 32]);
        let root_id = manager.insert(manager.make());

        thread::scope(|s| {
            s.spawn(|| {
                let result = lookup(root_id, &key, &*manager);
                assert_eq!(result, Ok(value));
            });

            manager.expect_read_access(root_id, vec![]);
            manager.expect(
                root_id,
                RcNodeExpectation::Lookup {
                    key,
                    depth: 0,
                    result: LookupResult::Value(value),
                },
            );
        });
    }

    #[test]
    fn lookup_keeps_lock_on_parent_while_locking_child() {
        let manager = Arc::new(RcNodeManager::new());
        let key = Key::from([7u8; 32]);
        let value = Value::from([42u8; 32]);
        let root_id = manager.insert(manager.make());
        let child_id = manager.insert(manager.make());

        thread::scope(|s| {
            s.spawn(|| {
                let result = lookup(root_id, &key, &*manager);
                assert_eq!(result, Ok(value));
            });

            manager.expect_read_access(root_id, vec![]);
            manager.expect(
                root_id,
                RcNodeExpectation::Lookup {
                    key,
                    depth: 0,
                    result: LookupResult::Node(child_id),
                },
            );
            // Root should be locked here
            manager.expect_read_access(child_id, vec![root_id]);
            manager.expect(
                child_id,
                RcNodeExpectation::Lookup {
                    key,
                    depth: 1,
                    result: LookupResult::Value(value),
                },
            );
        });
    }

    #[test]
    fn lookup_releases_grandparent_lock() {
        let manager = Arc::new(RcNodeManager::new());
        let key = Key::from([7u8; 32]);
        let value = Value::from([42u8; 32]);
        let root_id = manager.insert(manager.make());
        let child_id = manager.insert(manager.make());
        let grandchild_id = manager.insert(manager.make());

        thread::scope(|s| {
            s.spawn(|| {
                let result = lookup(root_id, &key, &*manager);
                assert_eq!(result, Ok(value));
            });

            manager.expect_read_access(root_id, vec![]);
            manager.expect(
                root_id,
                RcNodeExpectation::Lookup {
                    key,
                    depth: 0,
                    result: LookupResult::Node(child_id),
                },
            );
            manager.expect_read_access(child_id, vec![root_id]);
            manager.expect(
                child_id,
                RcNodeExpectation::Lookup {
                    key,
                    depth: 1,
                    result: LookupResult::Node(grandchild_id),
                },
            );
            // At this point the root should no longer be locked
            manager.expect_read_access(grandchild_id, vec![child_id]);
            manager.expect(
                grandchild_id,
                RcNodeExpectation::Lookup {
                    key,
                    depth: 2,
                    result: LookupResult::Value(value),
                },
            );
        });
    }
}
