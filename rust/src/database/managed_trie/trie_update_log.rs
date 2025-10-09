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

// TODO: Do we have to deal with deletions?
// TODO: We need to properly handle leaf nodes that get pushed down
pub struct TrieUpdateLog<IdType> {
    pub dirty_nodes_by_level: Vec<HashMap<IdType, [u8; 32]>>,
}

impl<IdType: Eq + std::hash::Hash> TrieUpdateLog<IdType> {
    pub fn new() -> Self {
        TrieUpdateLog {
            dirty_nodes_by_level: Vec::new(),
        }
    }

    pub fn delete(&mut self, id: IdType) {
        for level in &mut self.dirty_nodes_by_level {
            level.remove(&id);
        }
    }

    pub fn add(&mut self, level: u8, id: IdType, child_idx: u8) {
        let level = level as usize;
        self.dirty_nodes_by_level
            .resize_with(self.dirty_nodes_by_level.len().max(level + 1), HashMap::new);
        self.dirty_nodes_by_level[level].entry(id).or_default()[(child_idx / 8) as usize] |=
            1 << (child_idx % 8);
    }

    pub fn clear(&mut self) {
        self.dirty_nodes_by_level.clear();
    }
}
