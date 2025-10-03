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
    #[error("the node is frozen and cannot be modified")]
    Frozen,
    #[error("id / node type mismatch")]
    IdNodeTypeMismatch,
    #[error("id encodes a non-existing node type")]
    InvalidId,
    #[error("checkpoint creation failed")]
    Checkpoint,
    #[error("invalid file size")]
    DatabaseCorruption,
    #[error("IO error in storage: {0}")]
    Io(#[from] std::io::Error),
}

impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Error::NotFound, Error::NotFound)
            | (Error::Frozen, Error::Frozen)
            | (Error::IdNodeTypeMismatch, Error::IdNodeTypeMismatch)
            | (Error::InvalidId, Error::InvalidId)
            | (Error::Checkpoint, Error::Checkpoint)
            | (Error::DatabaseCorruption, Error::DatabaseCorruption) => true,
            (Error::Io(a), Error::Io(b)) => {
                a.kind() == b.kind()
                    && a.raw_os_error() == b.raw_os_error()
                    && a.to_string() == b.to_string()
            }
            (
                Error::NotFound
                | Error::Frozen
                | Error::IdNodeTypeMismatch
                | Error::InvalidId
                | Error::Checkpoint
                | Error::DatabaseCorruption
                | Error::Io(_),
                _,
            ) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implements_partial_eq() {
        let err1 = Error::NotFound;
        let err2 = Error::NotFound;
        let err3 = Error::IdNodeTypeMismatch;
        let err4 = Error::InvalidId;
        let err5 = Error::DatabaseCorruption;
        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
        assert_ne!(err1, err4);
        assert_ne!(err1, err5);

        let io_err1 = Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        let io_err2 = Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        let io_err3 = Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "another message",
        ));
        let io_err4 = Error::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "permission denied",
        ));
        assert_ne!(io_err1, err1);
        assert_eq!(io_err1, io_err2);
        assert_ne!(io_err1, io_err3);
        assert_ne!(io_err1, io_err4);

        let io_err5 = Error::Io(std::io::Error::from_raw_os_error(3));
        let io_err6 = Error::Io(std::io::Error::from_raw_os_error(4));
        assert_ne!(io_err1, io_err5);
        assert_ne!(io_err5, io_err6);
        assert_ne!(io_err5, io_err6);
    }
}
