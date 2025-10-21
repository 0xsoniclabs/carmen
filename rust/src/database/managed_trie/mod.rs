// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

mod lookup;
mod managed_trie_node;
mod store;
mod trie_update_log;

pub use lookup::lookup;
pub use managed_trie_node::{CanStoreResult, LookupResult, ManagedTrieNode, UnionManagedTrieNode};
pub use store::store;
pub use trie_update_log::TrieUpdateLog;
use zerocopy::{FromBytes, Immutable, IntoBytes, Unaligned};

use crate::types::Value;

/// A commitment together with a dirty flag indicating whether it needs to be recomputed.
///
/// NOTE: While this type is meant to be part of trie nodes, a dirty commitment should never
/// be persisted to disk. The dirty flag is nevertheless part of the on-disk representation,
/// so that the entire node can be transmuted to/from bytes using zerocopy.
/// Related issue: https://github.com/0xsoniclabs/sonic-admin/issues/373
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned)]
#[repr(C)]
pub struct CachedCommitment<C> {
    pub commitment: C,
    // bool does not implement FromBytes, so we use u8 instead
    pub dirty: u8,

    // FIXME Just hacking - these are only needed for Verkle leaf nodes
    // TODO: Also store scalars?
    pub c1: C,
    pub c2: C,
    // TODO Naming
    pub committed_values: [Value; 256],
    pub committed_used_bits: [u8; 256 / 8],
    // FIXME Just a hack - we could also use a bitmap for this, or store Option<Value> in
    //       committed_values
    pub changed_slots: [u8; 256],
}

impl<C> CachedCommitment<C> {
    pub fn store(&mut self, index: usize, prev_value: Value) {
        if self.changed_slots[index] == 0 {
            self.changed_slots[index] = 1;
            self.committed_values[index] = prev_value;
            self.dirty = 1;
        }
    }
}

impl<C: Default> Default for CachedCommitment<C> {
    fn default() -> Self {
        Self {
            commitment: C::default(),
            dirty: 0,
            c1: C::default(),
            c2: C::default(),
            committed_values: [Value::default(); 256],
            committed_used_bits: [0u8; 256 / 8],
            changed_slots: [0u8; 256],
        }
    }
}

impl<C: Copy> CachedCommitment<C> {
    pub fn commitment(&self) -> C {
        self.commitment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default, PartialEq, Eq)]
    struct DummyCommitment(u32);

    #[test]
    fn cached_commitment_default_returns_clean_cache_with_default_commitment() {
        let cache: CachedCommitment<DummyCommitment> = CachedCommitment::default();
        assert_eq!(cache.commitment, DummyCommitment::default());
        assert_eq!(cache.dirty, 0);
    }
}
