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
    dirty_nodes_by_level: RwLock<Vec<DashSet<ID>>>,
}

impl<ID: Copy + Eq + std::hash::Hash> TrieUpdateLog<ID> {
    pub fn new() -> Self {
        TrieUpdateLog {
            dirty_nodes_by_level: RwLock::new(Vec::new()),
        }
    }

    pub fn add(&self, level: usize, id: ID) {
        let guard = self.access_level(level);
        guard[level].insert(id);
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn delete(&self, level: usize, id: ID) {
        let guard = self.access_level(level);
        guard[level].remove(&id);
    }

    pub fn move_down(&self, id: ID, from_level: usize) {
        let guard = self.access_level(from_level + 1);
        if guard[from_level].remove(&id).is_some() {
            guard[from_level + 1].insert(id);
        }
    }

    pub fn count(&self) -> usize {
        let guard = self.dirty_nodes_by_level.read().unwrap();
        guard.iter().map(DashSet::len).sum()
    }

    pub fn levels(&self) -> usize {
        let guard = self.dirty_nodes_by_level.read().unwrap();
        guard.len()
    }

    pub fn dirty_nodes(&self, level: usize) -> Vec<ID> {
        let guard = self.access_level(level);
        guard[level].iter().map(|entry| *entry.key()).collect()
    }

    pub fn clear(&self) {
        self.dirty_nodes_by_level.write().unwrap().clear();
    }

    fn access_level(&self, level: usize) -> RwLockReadGuard<'_, Vec<DashSet<ID>>> {
        let guard = self.dirty_nodes_by_level.read().unwrap();
        if guard.len() > level {
            return guard;
        }

        drop(guard);
        let mut guard = self.dirty_nodes_by_level.write().unwrap();
        guard.resize_with((level) + 1, || DashSet::new());
        drop(guard);
        self.dirty_nodes_by_level.read().unwrap()
    }
}
