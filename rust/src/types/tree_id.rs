// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use crate::types::ToNodeType;

/// A trait for types that are used as IDs for nodes in a tree structure.
pub trait TreeId: ToNodeType {
    /// Creates a new ID from a [`u64`] index and a node type.
    fn from_idx_and_node_type(idx: u64, node_type: Self::NodeType) -> Self;

    /// Converts the ID to a [`u64`] index, stripping the prefix.
    fn to_index(self) -> u64;
}
