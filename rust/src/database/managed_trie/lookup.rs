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
    error::Error,
    node_manager::NodeManager,
    types::{Key, Value},
};

pub fn lookup<T: ManagedTrieNode>(
    root_id: T::Id,
    key: &Key,
    manager: &impl NodeManager<Id = T::Id, NodeType = T>,
) -> Result<Value, Error> {
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
