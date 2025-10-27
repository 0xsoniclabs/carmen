// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use crate::types::Value;

/// An abstract interface for commitments in managed trie nodes.
///
/// Implementors of this trait receive notifications about modifications
/// to children and slots, allowing them to update their commitments accordingly.
#[cfg_attr(not(test), expect(unused))]
pub trait TrieCommitment {
    /// Indicates that the child at the given index has been modified.
    fn modify_child(&mut self, index: usize);

    /// Indicates that the value at the given index has been modified.
    /// The previous value is provided for computing the delta in vector commitment schemes.
    fn store(&mut self, index: usize, prev: Value);
}
