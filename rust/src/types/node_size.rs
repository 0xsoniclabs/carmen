// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

/// A trait to determine the size of a node in a trie.
pub trait NodeSize {
    /// Returns the size of the node in bytes.
    fn node_byte_size(&self) -> usize;

    /// Returns the minimum size of a non-empty node in bytes.
    fn min_non_empty_node_size() -> usize;
}
