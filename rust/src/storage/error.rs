// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use thiserror::Error;

/// The error type for storage operations.
#[derive(Debug, Error)]
pub enum Error {
    #[error("not found")]
    NotFound,
    #[error("id / node type mismatch")]
    IdNodeTypeMismatch,
    #[error("id encodes a non-existing node type")]
    InvalidId,
    #[error("invalid file size")]
    DatabaseCorruption,
    #[error("IO error in storage: {0}")]
    Io(#[from] std::io::Error),
}
