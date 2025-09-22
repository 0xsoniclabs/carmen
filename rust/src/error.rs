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

use crate::storage;

/// The top level error type for Carmen .
/// This type is returned to the ffi interface and converted there.
#[derive(Debug, Error, PartialEq)]
pub enum Error {
    /// An unsupported schema version was provided.
    #[error("unsupported schema version: {0}")]
    UnsupportedSchema(u8),
    /// An unsupported operation was attempted.
    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),
    #[error("storage error: {0}")]
    Storage(#[from] storage::Error),
    #[error("Cache error: {0}")]
    Cache(String),
}
