// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use crate::{
    database::verkle::crypto::Commitment,
    error::Error,
    types::{Key, Value},
};

/// An implementation of the Verkle trie authenticated storage, as specified by Ethereum.
///
/// Verkle tries provide basic [`Key`]-[`Value`] storage of fixed-length keys and values,
/// and the ability to compute a cryptographic commitment of the trie's state using
/// the Pedersen commitment scheme.
///
/// The trait prescribes interior mutability through shared references,
/// allowing for safe concurrent access.
#[cfg_attr(not(test), expect(unused))]
pub trait VerkleTrie: Send + Sync {
    /// Retrieves the value associated with the given key.
    /// Returns the default [`Value`] if the key does not exist.
    fn get(&self, key: &Key) -> Result<Value, Error>;

    /// Sets the value for the given key.
    fn set(&self, key: &Key, value: &Value) -> Result<(), Error>;

    /// Computes and returns the current root commitment of the trie.
    /// The commitment can be used as cryptographic proof of the trie's state,
    /// i.e., all contained key-value pairs.
    fn commit(&self) -> Commitment;
}
