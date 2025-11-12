// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

mod compute_commitment;
mod crypto;
mod embedding;
mod state;
#[cfg(test)]
mod test_utils;
pub mod variants;
mod verkle_trie;

pub use state::VerkleTrieCarmenState;
pub use variants::{CrateCryptoInMemoryVerkleTrie, SimpleInMemoryVerkleTrie};
