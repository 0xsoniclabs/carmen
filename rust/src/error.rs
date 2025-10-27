// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{backtrace::Backtrace, error::Error as StdError, ops::Deref};

use thiserror::Error;

use crate::storage;

/// The top level error type for Carmen .
/// This type is returned to the ffi interface and converted there.
#[derive(Debug, Error, PartialEq)]
pub enum Error {
    /// An unsupported schema version was provided.
    #[error("unsupported schema version: {0}")]
    UnsupportedSchema(u8),
    /// An unsupported implementation was requested.
    #[error("unsupported implementation: {0}")]
    UnsupportedImplementation(String),
    /// An unsupported operation was attempted.
    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),
    #[error("storage error: {0}")]
    Storage(#[from] storage::Error),
    #[error("node manager error: {0}")]
    NodeManager(String),
    /// An illegal concurrent operation was attempted.
    /// Not all illegal concurrent operations can be detected.
    #[error("illegal concurrent operation: {0}")]
    IllegalConcurrentOperation(String),
}

pub type BTResult<T, E> = Result<T, BTError<E>>;

/// A wrapper around an error that captures a backtrace when created.
pub struct BTError<E> {
    /// [`Backtrace`]` alone is already 48 bytes large. Together with [`E`], the size will be at
    /// least 56 bytes, likely more. By boxing the inner struct, the size of [`BTError`]` is
    /// reduced to 8 bytes, which in turn reduces the size of [`Result<T, BTError<E>>`].
    inner: Box<BTErrorInner<E>>,
}

/// The inner struct of [`BTError`] which holds the actual error and the backtrace.
struct BTErrorInner<E> {
    error: E,
    backtrace: Backtrace,
}

// Ideally, we would have `impl<E> From<E> for BTError<E>` but that causes conflicts with other
// implementations which essentially provide instantiations of `impl<E, F: Into<E>> From<F> for
// BTError<E>`. Without the latter one the `?` stops working, which is very undesirable. Therefore,
// we provide the non-generic implementations which we actually need.

impl<F: Into<Error>> From<F> for BTError<Error> {
    fn from(error: F) -> Self {
        Self {
            inner: Box::new(BTErrorInner {
                error: error.into(),
                backtrace: Backtrace::force_capture(),
            }),
        }
    }
}

impl From<BTError<storage::Error>> for BTError<Error> {
    fn from(error: BTError<storage::Error>) -> Self {
        Self {
            inner: Box::new(BTErrorInner {
                error: error.inner.error.into(),
                backtrace: error.inner.backtrace,
            }),
        }
    }
}

impl<F: Into<storage::Error>> From<F> for BTError<storage::Error> {
    fn from(error: F) -> Self {
        Self {
            inner: Box::new(BTErrorInner {
                error: error.into(),
                backtrace: Backtrace::force_capture(),
            }),
        }
    }
}

impl From<BTError<std::io::Error>> for BTError<storage::Error> {
    fn from(error: BTError<std::io::Error>) -> Self {
        Self {
            inner: Box::new(BTErrorInner {
                error: error.inner.error.into(),
                backtrace: error.inner.backtrace,
            }),
        }
    }
}

impl<F: Into<std::io::Error>> From<F> for BTError<std::io::Error> {
    fn from(error: F) -> Self {
        Self {
            inner: Box::new(BTErrorInner {
                error: error.into(),
                backtrace: Backtrace::force_capture(),
            }),
        }
    }
}

impl<F: Into<String>> From<F> for BTError<String> {
    fn from(error: F) -> Self {
        Self {
            inner: Box::new(BTErrorInner {
                error: error.into(),
                backtrace: Backtrace::force_capture(),
            }),
        }
    }
}

impl<E> Deref for BTError<E> {
    type Target = E;

    fn deref(&self) -> &Self::Target {
        &self.inner.error
    }
}

impl<E: std::fmt::Debug> std::fmt::Debug for BTError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}\nBacktrace:\n{}",
            self.inner.error, self.inner.backtrace
        )
    }
}

impl<E: std::fmt::Display> std::fmt::Display for BTError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner.error)
    }
}

impl<E: StdError> StdError for BTError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.inner.error.source()
    }
}

impl<E: PartialEq> PartialEq for BTError<E> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.error == other.inner.error
    }
}

impl<E: PartialEq> PartialEq<E> for BTError<E> {
    fn eq(&self, other: &E) -> bool {
        &self.inner.error == other
    }
}

impl<E> BTError<E> {
    /// Returns the backtrace captured when this error was created.
    pub fn backtrace(&self) -> &Backtrace {
        &self.inner.backtrace
    }

    /// Returns the inner error.
    pub fn into_inner(self) -> E {
        self.inner.error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bterror_of_error__can_be_constructed_from_anything_that_is_convertible_into_error() {
        let error: storage::Error = storage::Error::NotFound; // storage::Error implements Into<Error>
        let _: BTError<Error> = error.into();
    }

    #[test]
    fn bterror_of_error__can_be_constructed_from__bterror_of_storage_error() {
        let error: BTError<storage::Error> = BTError {
            inner: Box::new(BTErrorInner {
                error: storage::Error::NotFound,
                backtrace: Backtrace::force_capture(),
            }),
        };
        let _: BTError<Error> = error.into();
    }

    #[test]
    fn bterror_of_storage_error__can_be_constructed_from_anything_than_is_convertible_into_storage_error()
     {
        let error: std::io::Error = std::io::Error::from(std::io::ErrorKind::AddrInUse); // std::io::Error implements Into<storage::Error>
        let _: BTError<storage::Error> = error.into();
    }

    #[test]
    fn bterror_of_storage_error__can_be_constructed_from__bterror_of_io_error() {
        let error: BTError<std::io::Error> = BTError {
            inner: Box::new(BTErrorInner {
                error: std::io::Error::from(std::io::ErrorKind::AddrInUse),
                backtrace: Backtrace::force_capture(),
            }),
        };
        let _: BTError<storage::Error> = error.into();
    }

    #[test]
    fn bterror_of_io_error__can_be_constructed_from_anything_than_is_convertible_into_io_error() {
        let error = std::io::ErrorKind::AddrInUse; // std::io::ErrorKind implements Into<std::io::Error>
        let _: BTError<std::io::Error> = error.into();
    }

    #[test]
    fn bterror_of_string__can_be_constructed_from_anything_than_is_convertible_into_string() {
        let error = "test"; // str implements Into<String>
        let _: BTError<String> = error.into();
    }

    #[test]
    fn dereferences_to_inner_error() {
        let error = Error::UnsupportedSchema(0);
        let bt_error: BTError<_> = error.into();
        let inner: &Error = bt_error.deref();

        assert_eq!(&bt_error.inner.error, inner);
    }

    #[test]
    fn debug_prints_inner_error_and_backtrace() {
        let error = Error::UnsupportedSchema(0);
        let bt_error: BTError<_> = error.into();

        assert_eq!(
            format!("{bt_error:?}"),
            format!(
                "{:?}\nBacktrace:\n{}",
                bt_error.inner.error, bt_error.inner.backtrace
            )
        );
    }

    #[test]
    fn display_forwards_to_inner_error() {
        let error = Error::UnsupportedSchema(0);
        let bt_error: BTError<_> = error.into();

        assert_eq!(format!("{bt_error}"), format!("{}", bt_error.inner.error));
    }

    #[test]
    fn error_forwards_source_to_inner_error() {
        let error = Error::Storage(storage::Error::NotFound);
        let bt_error: BTError<_> = error.into();

        assert_eq!(
            bt_error.source().unwrap().to_string(),
            bt_error.inner.error.source().unwrap().to_string()
        );
    }

    #[test]
    fn partial_eq_compares_to_inner_error() {
        let error = Error::UnsupportedSchema(0);
        let bt_error: BTError<_> = error.into();

        assert_eq!(bt_error, Error::UnsupportedSchema(0));
        assert_ne!(bt_error, Error::UnsupportedSchema(1));
    }

    #[test]
    fn backtrace_returns_captured_backtrace() {
        let backtrace_string = b().unwrap_err().backtrace().to_string();
        let lines: Vec<_> = backtrace_string.lines().collect();
        assert_eq!(
            lines[0],
            "   0: <carmen_rust::error::BTError<carmen_rust::error::Error> as core::convert::From<F>>::from",
        );
        assert!(lines[1].starts_with("             at ./src/error.rs"));
        assert_eq!(
            lines[2],
            "   1: <core::result::Result<T,F> as core::ops::try_trait::FromResidual<core::result::Result<core::convert::Infallible,E>>>::from_residual"
        );
        assert_eq!(lines[4], "   2: carmen_rust::error::tests::b",);
        assert!(lines[5].starts_with("             at ./src/error.rs"));
    }

    fn a() -> Result<(), Error> {
        Err(Error::UnsupportedSchema(0))
    }

    fn b() -> BTResult<(), Error> {
        a()?;
        Ok(())
    }
}
