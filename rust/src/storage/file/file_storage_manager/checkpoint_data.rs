// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::storage::file::from_to_file::FromToFile;

/// Data stored in the checkpoint file.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
#[repr(C)]
pub struct CheckpointData {
    /// The checkpoint number.
    pub checkpoint_number: u64,
    /// The number of root IDs that are part of this checkpoint.
    pub root_id_count: u64,
}

impl FromToFile for CheckpointData {}
