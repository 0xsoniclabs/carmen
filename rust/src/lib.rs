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
mod types;

/// Opens a new [CarmenS6Db] database object based on the provided implementation maintaining
/// its data in the given directory. If the directory does not exist, it is
/// created. If it is empty, a new, empty state is initialized. If it contains
/// state information, the information is loaded.
pub fn open_carmen_s6_db(
    _schema: u8,
    _state: StateImpl,
    _archive: ArchiveImpl,
    _directory: &[u8],
) -> Result<Box<dyn CarmenS6Db>, Error> {
    // here we would choose the specific implementation of CarmenS6Db based on the state and
    // archive.
    Ok(Box::new(DbState))
}

/// The safe Carmen S6 interface.
/// This is the safe interface which gets called from the exported FFI functions.
#[cfg_attr(test, mockall::automock)]
pub trait CarmenS6Db {
    /// Flushes all committed state information to disk to guarantee permanent
    /// storage. All internally cached modifications are synced to disk.
    fn flush(&mut self) -> Result<(), Error>;

    /// Closes this state, releasing all IO handles and locks on external resources.
    fn close(&mut self) -> Result<(), Error>;

    /// Creates a state snapshot reflecting the state at the given block height. The
    /// resulting state must be released and must not outlive the life time of the
    /// provided state.
    fn get_archive_state(&mut self, block: u64) -> Result<Box<dyn CarmenS6Db>, Error>;

    /// Returns the current state of the given account.
    fn get_account_state(&mut self, addr: &Address) -> Result<AccountState, Error>;

    /// Returns the balance of the given account.
    fn get_balance(&mut self, addr: &Address) -> Result<U256, Error>;

    /// Returns the nonce of the given account.
    fn get_nonce(&mut self, addr: &Address) -> Result<u64, Error>;

    /// Returns the value of storage location (addr,key) in the given state.
    fn get_storage_value(&mut self, addr: &Address, key: &mut Key) -> Result<Value, Error>;

    /// Returns the code stored under the given address.
    fn get_code(
        &mut self,
        addr: &Address,
        code_buf: &mut [MaybeUninit<u8>],
    ) -> Result<usize, Error>;

    /// Returns the hash of the code stored under the given address.
    fn get_code_hash(&mut self, addr: &Address) -> Result<Hash, Error>;

    /// Returns the code length stored under the given address.
    fn get_code_len(&mut self, addr: &Address) -> Result<u32, Error>;

    /// Returns a global state hash of the given state.
    fn get_hash(&mut self) -> Result<Hash, Error>;

    /// Returns a summary of the used memory.
    fn get_memory_footprint(&mut self) -> Result<Box<str>, Error>;

    /// Applies the provided block update to the maintained state.
    #[allow(clippy::needless_lifetimes)] // using an elided lifetime here breaks automock
    fn apply_block_update<'u>(&mut self, block: u64, update: Update<'u>) -> Result<(), Error>;
}

/// The main implementation of [`CarmenS6Db`].
pub struct DbState;

#[allow(unused_variables)]
impl CarmenS6Db for DbState {
    fn flush(&mut self) -> Result<(), Error> {
        unimplemented!()
    }

    fn close(&mut self) -> Result<(), Error> {
        unimplemented!()
    }

    fn get_archive_state(&mut self, block: u64) -> Result<Box<dyn CarmenS6Db>, Error> {
        unimplemented!()
    }

    fn get_account_state(&mut self, addr: &Address) -> Result<AccountState, Error> {
        unimplemented!()
    }

    fn get_balance(&mut self, addr: &Address) -> Result<U256, Error> {
        unimplemented!()
    }

    fn get_nonce(&mut self, addr: &Address) -> Result<u64, Error> {
        unimplemented!()
    }

    fn get_storage_value(&mut self, addr: &Address, key: &mut Key) -> Result<Value, Error> {
        unimplemented!()
    }

    fn get_code(
        &mut self,
        addr: &Address,
        code_buf: &mut [MaybeUninit<u8>],
    ) -> Result<usize, Error> {
        unimplemented!()
    }

    fn get_code_hash(&mut self, addr: &Address) -> Result<Hash, Error> {
        unimplemented!()
    }

    fn get_code_len(&mut self, addr: &Address) -> Result<u32, Error> {
        unimplemented!()
    }

    fn get_hash(&mut self) -> Result<Hash, Error> {
        unimplemented!()
    }

    fn get_memory_footprint(&mut self) -> Result<Box<str>, Error> {
        unimplemented!()
    }

    fn apply_block_update(&mut self, block: u64, update: Update) -> Result<(), Error> {
        unimplemented!()
    }
}
