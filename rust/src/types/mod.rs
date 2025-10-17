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
pub use disk_representable::{DiskRepresentable, DiskRepresentableByType};
pub use node_size::*;
pub use update::{BalanceUpdate, CodeUpdate, NonceUpdate, SlotUpdate, Update};

mod commitment;
mod disk_representable;
mod node_size;
mod update;

// TODO unsafe??
pub trait AllVariants {
    fn all_variants() -> &'static [(Self, &'static str)]
    where
        Self: Sized;
}

pub trait TreeId {
    type NodeType;

    /// Creates a new [`NodeId`] from a [`u64`] index and a [`NodeType`].
    /// The index must be smaller than 2^46.
    fn from_idx_and_node_type(idx: u64, node_type: Self::NodeType) -> Self;

    /// Converts the [`NodeId`] to a [`u64`] index, stripping the prefix.
    /// The index is guaranteed to be smaller than 2^46.
    fn to_index(self) -> u64;

    /// Converts the [`NodeId`] to a [`NodeType`], if the prefix is valid.
    fn to_node_type(self) -> Option<Self::NodeType>;
}

pub trait ToNodeType {
    type NodeType;

    fn to_node_type(&self) -> Self::NodeType;
}

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
