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

// TODO Test: Default commitment cache is clean
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned,
)]
#[repr(C)]
pub struct CachedCommitment<C> {
    pub commitment: C,
    // bool does not implement FromBytes, so we use u8 instead
    pub dirty: u8,
}

impl<C: Copy> CachedCommitment<C> {
    pub fn commitment(&self) -> C {
        self.commitment
    }
}
