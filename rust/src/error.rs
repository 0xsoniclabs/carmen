// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

#[cfg(test)]
use std::{ops::Deref, sync::Mutex};

use thiserror::Error;

use crate::storage;

/// The top level error type for Carmen .
/// This type is returned to the ffi interface and converted there.
#[derive(Debug, Error)]
pub enum Error {
    /// An unsupported schema version was provided.
    #[error("unsupported schema version: {0}")]
    UnsupportedSchema(u8),
    /// An unsupported operation was attempted.
    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),
    #[error("storage error: {0}")]
    Storage(#[from] storage::Error),
}

/// A thread-safe state for storing an error.
#[cfg(test)]
#[derive(Debug, Default)]
pub struct ErrorState {
    error: Mutex<Option<Error>>,
}

#[cfg(test)]
impl ErrorState {
    /// Create a new instance of `ErrorState` with no error.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the error if no error has been registered yet.
    pub fn register(&self, error: Error) {
        let mut guard = self.error.lock().unwrap();
        if guard.is_none() {
            *guard = Some(error);
        }
    }

    /// Retrieve the error, if any.
    pub fn get(&self) -> impl Deref<Target = Option<Error>> {
        self.error.lock().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_state_new_creates_state_with_no_error() {
        let state = ErrorState::new();
        let guard = state.get();
        assert!(guard.is_none());
    }

    #[test]
    fn error_state_register_sets_error_if_none_exists() {
        let state = ErrorState::new();
        state.register(Error::UnsupportedSchema(1));
        {
            let guard = state.get();
            assert!(guard.is_some());
            assert!(matches!(
                guard.as_ref().unwrap(),
                Error::UnsupportedSchema(1)
            ));
        }

        // Registering another error should not change the existing one
        state.register(Error::UnsupportedOperation("test".to_string()));
        {
            let guard = state.get();
            assert!(guard.is_some());
            assert!(matches!(
                guard.as_ref().unwrap(),
                Error::UnsupportedSchema(1)
            ));
        }
    }
}
