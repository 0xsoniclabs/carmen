// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::sync::{RwLock, RwLockReadGuard};

use dashmap::DashSet;

/// The type has interior mutability to allow concurrent updates.
pub struct TrieUpdateLog<ID> {
    pub dirty_nodes_by_level: RwLock<Vec<DashSet<ID>>>,
}

impl<ID: Eq + std::hash::Hash> TrieUpdateLog<ID> {
    pub fn new() -> Self {
        let mut dirty_nodes_by_level = Vec::new();
        dirty_nodes_by_level.resize_with(256, || DashSet::new());
        TrieUpdateLog {
            dirty_nodes_by_level: RwLock::new(dirty_nodes_by_level),
        }
    }

    pub fn add(&self, level: u8, id: ID) {
        let guard = self.access_level(level);
        guard[level as usize].insert(id);
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn delete(&self, level: u8, id: ID) {
        let guard = self.access_level(level);
        guard[level as usize].remove(&id);
    }

    pub fn move_down(&self, id: ID, from_level: u8) {
        let guard = self.access_level(from_level + 1);
        let from_level = from_level as usize;
        if guard[from_level].remove(&id).is_some() {
            guard[from_level + 1].insert(id);
        }
    }

    pub fn count(&self) -> usize {
        let guard = self.dirty_nodes_by_level.read().unwrap();
        guard.iter().map(DashSet::len).sum()
    }

    pub fn clear(&self) {
        self.dirty_nodes_by_level.write().unwrap().clear();
    }

    fn access_level(&self, level: u8) -> RwLockReadGuard<'_, Vec<DashSet<ID>>> {
        let guard = self.dirty_nodes_by_level.read().unwrap();
        if guard.len() > level as usize {
            return guard;
        }

        drop(guard);
        let mut guard = self.dirty_nodes_by_level.write().unwrap();
        guard.resize_with((level as usize) + 1, || DashSet::new());
        drop(guard);
        self.dirty_nodes_by_level.read().unwrap()
    }
}
