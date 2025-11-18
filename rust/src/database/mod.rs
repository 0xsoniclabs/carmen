// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

mod managed_trie;
pub mod verkle;

pub use managed_trie::ManagedTrieNode;
pub use verkle::{
    CrateCryptoInMemoryVerkleTrie, ManagedVerkleTrie, SimpleInMemoryVerkleTrie,
    VerkleTrieCarmenState,
};
