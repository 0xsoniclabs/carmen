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
