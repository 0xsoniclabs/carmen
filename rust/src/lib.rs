// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::mem::MaybeUninit;

use crate::{error::Error, types::*};

mod error;
mod ffi;
mod storage;
pub mod types;

/// Opens a new [CarmenDb] database object based on the provided implementation maintaining
/// its data in the given directory. If the directory does not exist, it is
/// created. If it is empty, a new, empty state is initialized. If it contains
/// state information, the information is loaded.
pub fn open_carmen_db(
    schema: u8,
    _live_impl: LiveImpl,
    _archive_impl: ArchiveImpl,
    _directory: &[u8],
) -> Result<Box<dyn CarmenDb>, Error> {
    if schema != 6 {
        return Err(Error::UnsupportedSchema(schema));
    }
    // here we would open the specific live and archive implementations of CarmenDb based on the
    // state and archive impls.
    Ok(Box::new(CarmenS6Db))
}

/// The safe Carmen database interface.
/// This is the safe interface which gets called from the exported FFI functions.
#[cfg_attr(test, mockall::automock)]
pub trait CarmenDb: Send + Sync {
    /// Flushes all committed state information to disk to guarantee permanent
    /// storage. All internally cached modifications are synced to disk.
    fn flush(&self) -> Result<(), Error>;

    /// Closes this state, releasing all IO handles and locks on external resources.
    fn close(&self) -> Result<(), Error>;

    /// Returns a handle to the live state. The resulting state must be released and must not
    /// outlive the life time of the database.
    fn get_live_state(&self) -> Result<Box<dyn CarmenState>, Error>;

    /// Returns a handle to an archive state reflecting the state at the given block height. The
    /// resulting state must be released and must not outlive the life time of the
    /// provided state.
    fn get_archive_state(&self, block: u64) -> Result<Box<dyn CarmenState>, Error>;

    /// Returns a summary of the used memory.
    fn get_memory_footprint(&self) -> Result<Box<str>, Error>;
}

/// The safe Carmen state interface.
/// This is the safe interface which gets called from the exported FFI functions.
#[cfg_attr(test, mockall::automock)]
pub trait CarmenState: Send + Sync {
    /// Checks if the given account exists.
    fn account_exists(&self, addr: &Address) -> Result<bool, Error>;

    /// Returns the balance of the given account.
    fn get_balance(&self, addr: &Address) -> Result<U256, Error>;

    /// Returns the nonce of the given account.
    fn get_nonce(&self, addr: &Address) -> Result<u64, Error>;

    /// Returns the value of storage location (addr,key) in the given state.
    fn get_storage_value(&self, addr: &Address, key: &Key) -> Result<Value, Error>;

    /// Returns the code stored under the given address.
    fn get_code(&self, addr: &Address, code_buf: &mut [MaybeUninit<u8>]) -> Result<usize, Error>;

    /// Returns the hash of the code stored under the given address.
    fn get_code_hash(&self, addr: &Address) -> Result<Hash, Error>;

    /// Returns the code length stored under the given address.
    fn get_code_len(&self, addr: &Address) -> Result<u32, Error>;

    /// Returns a global state hash of the given state.
    fn get_hash(&self) -> Result<Hash, Error>;

    /// Applies the provided block update to the maintained state.
    #[allow(clippy::needless_lifetimes)] // using an elided lifetime here breaks automock
    fn apply_block_update<'u>(&self, block: u64, update: Update<'u>) -> Result<(), Error>;
}

/// The `S6` implementation of [`CarmenDb`].
pub struct CarmenS6Db;

#[allow(unused_variables)]
impl CarmenDb for CarmenS6Db {
    fn flush(&self) -> Result<(), Error> {
        unimplemented!()
    }

    fn close(&self) -> Result<(), Error> {
        unimplemented!()
    }

    fn get_live_state(&self) -> Result<Box<dyn CarmenState>, Error> {
        unimplemented!()
    }

    fn get_archive_state(&self, block: u64) -> Result<Box<dyn CarmenState>, Error> {
        unimplemented!()
    }

    fn get_memory_footprint(&self) -> Result<Box<str>, Error> {
        unimplemented!()
    }
}

/// The `S6` live state implementation of [`CarmenState`].
pub struct LiveState;

#[allow(unused_variables)]
impl CarmenState for LiveState {
    fn account_exists(&self, addr: &Address) -> Result<bool, Error> {
        unimplemented!()
    }

    fn get_balance(&self, addr: &Address) -> Result<U256, Error> {
        unimplemented!()
    }

    fn get_nonce(&self, addr: &Address) -> Result<u64, Error> {
        unimplemented!()
    }

    fn get_storage_value(&self, addr: &Address, key: &Key) -> Result<Value, Error> {
        unimplemented!()
    }

    fn get_code(&self, addr: &Address, code_buf: &mut [MaybeUninit<u8>]) -> Result<usize, Error> {
        unimplemented!()
    }

    fn get_code_hash(&self, addr: &Address) -> Result<Hash, Error> {
        unimplemented!()
    }

    fn get_code_len(&self, addr: &Address) -> Result<u32, Error> {
        unimplemented!()
    }

    fn get_hash(&self) -> Result<Hash, Error> {
        unimplemented!()
    }

    fn apply_block_update(&self, block: u64, update: Update) -> Result<(), Error> {
        unimplemented!()
    }
}

/// The `S6` archive state implementation of [`CarmenState`].
pub struct ArchiveState;

#[allow(unused_variables)]
impl CarmenState for ArchiveState {
    fn account_exists(&self, addr: &Address) -> Result<bool, Error> {
        unimplemented!()
    }

    fn get_balance(&self, addr: &Address) -> Result<U256, Error> {
        unimplemented!()
    }

    fn get_nonce(&self, addr: &Address) -> Result<u64, Error> {
        unimplemented!()
    }

    fn get_storage_value(&self, addr: &Address, key: &Key) -> Result<Value, Error> {
        unimplemented!()
    }

    fn get_code(&self, addr: &Address, code_buf: &mut [MaybeUninit<u8>]) -> Result<usize, Error> {
        unimplemented!()
    }

    fn get_code_hash(&self, addr: &Address) -> Result<Hash, Error> {
        unimplemented!()
    }

    fn get_code_len(&self, addr: &Address) -> Result<u32, Error> {
        unimplemented!()
    }

    fn get_hash(&self) -> Result<Hash, Error> {
        unimplemented!()
    }

    fn apply_block_update(&self, block: u64, update: Update) -> Result<(), Error> {
        Err(Error::UnsupportedOperation(
            "Archive state does not support applying block updates".to_string(),
        ))
    }
}
