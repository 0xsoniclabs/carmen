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

/// Metadata stored in the metadata file.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
#[repr(C)]
pub struct NodeFileStorageMetadata {
    /// The checkpoint number.
    pub checkpoint: u64,
    /// The number of frozen nodes that can not be modified because they are part of this
    /// checkpoint.
    pub frozen_nodes: u64,
    /// The number of frozen reuse indices that can not be reused because they are part of this
    /// checkpoint.
    pub frozen_reuse_indices: u64,
}

impl FromToFile for NodeFileStorageMetadata {}
