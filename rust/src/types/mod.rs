// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

pub use commitment::*;
pub use id::*;
pub use node::*;
pub use update::Update;

mod commitment;
mod id;
mod node;
mod update;

/// The Carmen live state implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveImpl {
    Memory = 0,
    File = 1,
    LevelDb = 2,
}

/// The Carmen archive state implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveImpl {
    None = 0,
    LevelDb = 1,
    Sqlite = 2,
}

/// An account address.
pub type Address = [u8; 20];

/// A key in the state trie.
pub type Key = [u8; 32];

/// A value in the state trie.
pub type Value = [u8; 32];

/// A hash.
pub type Hash = [u8; 32];

/// An 256-bit integer.
pub type U256 = [u8; 32];

/// An account nonce.
/// Carmen does not do any numeric operations on nonce. By using [`[u8; 8]`] instead of [`u64`], we
/// don't require 8 byte alignment.
pub type Nonce = [u8; 8];
