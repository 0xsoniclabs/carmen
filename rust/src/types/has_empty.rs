// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

/// Trait for node types that have an empty variant.
pub trait HasEmptyNode {
    /// Returns true if the node is the empty node variant.
    fn is_empty_node(&self) -> bool;

    /// Creates an empty node.
    fn empty_node() -> Self
    where
        Self: Sized;
}

/// Trait for ID types that identify nodes with an empty variant.
pub trait HasEmptyId {
    /// Returns true if the ID is the empty ID variant.
    fn is_empty_id(&self) -> bool;

    /// Creates an ID for an empty node.
    fn empty_id() -> Self
    where
        Self: Sized;
}
