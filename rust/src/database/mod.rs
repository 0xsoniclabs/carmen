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

use crate::error::{BTResult, Error};

/// A trait for accepting a visitor.
pub trait AcceptVisitor {
    type Node;

    fn accept(&self, visitor: &mut impl NodeVisitor<Self::Node>) -> BTResult<(), Error>;
}

/// A trait for visiting trie nodes.
#[cfg_attr(test, mockall::automock, allow(clippy::disallowed_types))]
pub trait NodeVisitor<N> {
    fn visit(&mut self, node: &N, level: u64) -> BTResult<(), Error>;
}
