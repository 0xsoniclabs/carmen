// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

mod exported;

#[allow(non_upper_case_globals, non_camel_case_types, non_snake_case, unused)]
mod bindings {
    use crate::error::Error;

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    impl From<Error> for Result {
        fn from(error: Error) -> Self {
            eprintln!("Returning error via FFI: {error:?}");

            match error {
                Error::UnsupportedSchema(_) => Result_kResult_UnsupportedSchema,
                Error::UnsupportedOperation(_) => Result_kResult_UnsupportedOperation,
                Error::UnsupportedImplementation(_) => Result_kResult_UnsupportedImplementation,
                Error::Storage(
                    crate::storage::Error::NotFound
                    | crate::storage::Error::IdNodeTypeMismatch
                    | crate::storage::Error::InvalidId,
                )
                | Error::NodeManager(_) => Result_kResult_InternalError,
                Error::Storage(crate::storage::Error::DatabaseCorruption) => {
                    Result_kResult_CorruptedDatabase
                }
                Error::Storage(crate::storage::Error::Io(_)) => Result_kResult_IOError,
            }
        }
    }
}
