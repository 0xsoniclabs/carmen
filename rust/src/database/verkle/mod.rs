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
mod variants;
mod verkle_trie;

// TODO: Not needed once nodes are moved into this module (same for CachedCommitment)
pub use crypto::Commitment;
pub use state::VerkleTrieCarmenState;
pub use variants::FakeCache; // TODO: Remove
pub use variants::{CachedCommitment, ManagedVerkleTrie, SimpleInMemoryVerkleTrie};
