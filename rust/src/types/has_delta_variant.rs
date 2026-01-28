// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use crate::error::{BTResult, Error};

/// Trait for node types that have a delta variant.
pub trait HasDeltaVariant {
    /// The ID type for this node type.
    type Id;

    /// Returns the ID of the base node if it is required to initialize this node or `None`.
    fn needs_delta_base(&self) -> Option<Self::Id>;

    /// Copies all data that is part of the in-memory but not the on-disk representation of this
    /// delta node from a base node into this delta node.
    /// If this functions gets called from a non-delta node, it should do nothing.
    fn copy_from_delta_base(&mut self, base: &Self) -> BTResult<(), Error>;
}
