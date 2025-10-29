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

/// A log of nodes that are being modified during updates to a managed trie, allowing for the batch
/// recomputation of commitments afterwards. Nodes are identified by an `ID` type, and are organized
/// by their depth in the trie.
///
/// The log has interior mutability to allow concurrent updates.
pub struct TrieUpdateLog<ID> {
    dirty_nodes_by_level: RwLock<Vec<DashSet<ID>>>,
}

#[cfg_attr(not(test), expect(unused))]
impl<ID: Copy + Eq + std::hash::Hash> TrieUpdateLog<ID> {
    /// Creates a new, empty [`TrieUpdateLog`].
    pub fn new() -> Self {
        TrieUpdateLog {
            dirty_nodes_by_level: RwLock::new(Vec::new()),
        }
    }

    /// Marks the node with the given `id` at the specified `depth` as dirty,
    /// indicating that it has been modified and its commitment needs to be updated.
    pub fn mark_dirty(&self, depth: usize, id: ID) {
        let guard = self.access_level(depth);
        guard[depth].insert(id);
    }

    /// Deletes the node with the given `id` at the specified `depth` from the log.
    #[allow(clippy::needless_pass_by_value)]
    pub fn delete(&self, depth: usize, id: ID) {
        let guard = self.access_level(depth);
        guard[depth].remove(&id);
    }

    /// Moves the node with the given `id` from the specified `from_depth` to the next level down.
    pub fn move_down(&self, from_depth: usize, id: ID) {
        let guard = self.access_level(from_depth + 1);
        if guard[from_depth].remove(&id).is_some() {
            guard[from_depth + 1].insert(id);
        }
    }

    /// Counts the total number of dirty nodes across all levels.
    pub fn count(&self) -> usize {
        let guard = self.dirty_nodes_by_level.read().unwrap();
        guard.iter().map(DashSet::len).sum()
    }

    /// Returns the number of levels for which dirty nodes have been logged.
    pub fn levels(&self) -> usize {
        let guard = self.dirty_nodes_by_level.read().unwrap();
        guard.len()
    }

    /// Returns a vector of all dirty node IDs at the specified `depth`.
    pub fn dirty_nodes(&self, depth: usize) -> Vec<ID> {
        let guard = self.access_level(depth);
        guard[depth].iter().map(|entry| *entry.key()).collect()
    }

    /// Clears all dirty nodes from the log.
    pub fn clear(&self) {
        self.dirty_nodes_by_level.write().unwrap().clear();
    }

    /// Ensures that the dirty nodes vector has at least `level + 1` entries,
    /// and returns a read guard to it.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_empty_log() {
        let log: TrieUpdateLog<u32> = TrieUpdateLog::new();
        let levels = log.dirty_nodes_by_level.read().unwrap();
        assert_eq!(levels.len(), 0);
    }

    #[test]
    fn mark_dirty_marks_id_as_dirty_on_specific_level() {
        let log: TrieUpdateLog<u32> = TrieUpdateLog::new();

        log.mark_dirty(2, 42);
        let levels = log.dirty_nodes_by_level.read().unwrap();
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0].len(), 0);
        assert_eq!(levels[1].len(), 0);
        assert_eq!(levels[2].len(), 1);
        assert!(levels[2].contains(&42));
        drop(levels);

        log.mark_dirty(3, 33);
        let levels = log.dirty_nodes_by_level.read().unwrap();
        assert_eq!(levels.len(), 4);
        assert_eq!(levels[3].len(), 1);
        assert!(levels[3].contains(&33));
        drop(levels);

        log.mark_dirty(3, 33);
        let levels = log.dirty_nodes_by_level.read().unwrap();
        assert_eq!(levels.len(), 4);
        assert_eq!(levels[3].len(), 1);
        assert!(levels[3].contains(&33));
        drop(levels);
    }

    #[test]
    fn marking_same_id_on_same_level_multiple_times_is_no_op() {
        let log: TrieUpdateLog<u32> = TrieUpdateLog::new();

        log.mark_dirty(1, 7);
        log.mark_dirty(1, 7);
        log.mark_dirty(1, 7);

        let levels = log.dirty_nodes_by_level.read().unwrap();
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[1].len(), 1);
        assert!(levels[1].contains(&7));
    }

    #[test]
    fn delete_removes_id_from_specific_level() {
        let log: TrieUpdateLog<u32> = TrieUpdateLog::new();

        log.mark_dirty(0, 1);
        log.mark_dirty(0, 2);
        log.mark_dirty(0, 3);

        log.delete(0, 2);

        let levels = log.dirty_nodes_by_level.read().unwrap();
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].len(), 2);
        assert!(levels[0].contains(&1));
        assert!(!levels[0].contains(&2));
        assert!(levels[0].contains(&3));
        drop(levels);

        log.delete(1, 3); // Wrong level, should be no-op
        let levels = log.dirty_nodes_by_level.read().unwrap();
        assert_eq!(levels.len(), 2); // Deleting from non-existing level creates it
        assert_eq!(levels[0].len(), 2);
        assert_eq!(levels[1].len(), 0);
        assert!(levels[0].contains(&3));
    }

    #[test]
    fn move_down_moves_id_from_one_level_to_the_next() {
        let log: TrieUpdateLog<u32> = TrieUpdateLog::new();

        log.mark_dirty(0, 10);
        log.move_down(0, 10);
        let levels = log.dirty_nodes_by_level.read().unwrap();
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].len(), 0);
        assert_eq!(levels[1].len(), 1);
        assert!(levels[1].contains(&10));
        drop(levels);

        // Moving non-existing id is no-op
        log.move_down(0, 20);
        let levels = log.dirty_nodes_by_level.read().unwrap();
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].len(), 0);
        assert_eq!(levels[1].len(), 1);
        assert!(levels[1].contains(&10));
    }

    #[test]
    fn count_counts_all_dirty_ids_across_levels() {
        let log: TrieUpdateLog<u32> = TrieUpdateLog::new();

        assert_eq!(log.count(), 0);

        log.mark_dirty(0, 1);
        log.mark_dirty(0, 2);
        assert_eq!(log.count(), 2);

        log.mark_dirty(1, 3);
        assert_eq!(log.count(), 3);

        log.mark_dirty(2, 4);
        log.mark_dirty(2, 5);
        log.mark_dirty(2, 6);
        assert_eq!(log.count(), 6);

        log.delete(0, 1);
        assert_eq!(log.count(), 5);

        log.move_down(2, 5);
        assert_eq!(log.count(), 5);
    }

    #[test]
    fn levels_returns_number_of_levels() {
        let log: TrieUpdateLog<u32> = TrieUpdateLog::new();

        assert_eq!(log.levels(), 0);

        log.mark_dirty(0, 1);
        assert_eq!(log.levels(), 1);

        log.mark_dirty(2, 2);
        assert_eq!(log.levels(), 3);

        log.mark_dirty(1, 3);
        assert_eq!(log.levels(), 3);
    }

    #[test]
    fn dirty_nodes_returns_all_dirty_ids_on_level() {
        let log: TrieUpdateLog<u32> = TrieUpdateLog::new();

        let dirty = log.dirty_nodes(0);
        assert!(dirty.is_empty());

        log.mark_dirty(1, 10);
        log.mark_dirty(1, 20);
        log.mark_dirty(1, 30);
        log.mark_dirty(2, 40);

        let mut dirty = log.dirty_nodes(1);
        dirty.sort();
        assert_eq!(dirty, [10, 20, 30]);

        let dirty = log.dirty_nodes(2);
        assert_eq!(dirty, [40]);
    }

    #[test]
    fn clear_clears_all_dirty_nodes_on_all_levels() {
        let log: TrieUpdateLog<u32> = TrieUpdateLog::new();

        log.mark_dirty(0, 1);
        log.mark_dirty(1, 2);
        log.mark_dirty(2, 3);

        log.clear();

        assert_eq!(log.count(), 0);
    }
}
