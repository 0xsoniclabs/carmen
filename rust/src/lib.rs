// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.
#![cfg_attr(test, allow(non_snake_case))]

use std::{mem::MaybeUninit, ops::Deref};

use crate::{
    database::VerkleTrieCarmenState,
    error::{BTResult, Error},
    sync::Arc,
    types::*,
};

mod database;
pub mod error;
mod ffi;
mod node_manager;
pub mod storage;
pub mod sync;
pub mod types;
mod utils;

/// Opens a new [CarmenDb] database object based on the provided implementation maintaining
/// its data in the given directory. If the directory does not exist, it is
/// created. If it is empty, a new, empty state is initialized. If it contains
/// state information, the information is loaded.
pub fn open_carmen_db(
    schema: u8,
    live_impl: LiveImpl,
    archive_impl: ArchiveImpl,
    _directory: &[u8],
) -> BTResult<Box<dyn CarmenDb>, Error> {
    if schema != 6 {
        return Err(Error::UnsupportedSchema(schema).into());
    }

    if !matches!(archive_impl, ArchiveImpl::None) {
        return Err(
            Error::UnsupportedImplementation("archive is not yet supported".to_owned()).into(),
        );
    }

    match live_impl {
        LiveImpl::Memory => Ok(Box::new(CarmenS6Db::new(VerkleTrieCarmenState::<
            database::SimpleInMemoryVerkleTrie,
        >::new()))),
        LiveImpl::File => Err(Error::UnsupportedImplementation(
            "file-based live state is not yet supported".to_owned(),
        )
        .into()),
        LiveImpl::LevelDb => Err(Error::UnsupportedImplementation(
            "LevelDB-based live state is not supported".to_owned(),
        )
        .into()),
    }
}

/// The safe Carmen database interface.
/// This is the safe interface which gets called from the exported FFI functions.
#[cfg_attr(test, mockall::automock)]
pub trait CarmenDb: Send + Sync {
    /// Creates a new checkpoint by persisting all state information to disk to guarantee permanent
    /// storage.
    fn checkpoint(&self) -> BTResult<(), Error>;

    /// Creates a new checkpoint and then closes this state, releasing all IO handles and locks on
    /// external resources.
    // TODO: Do not create a checkpoint if in an error state (https://github.com/0xsoniclabs/sonic-admin/issues/378)
    fn close(&self) -> BTResult<(), Error>;

    /// Returns a handle to the live state. The resulting state must be released and must not
    /// outlive the life time of the database.
    fn get_live_state(&self) -> BTResult<Box<dyn CarmenState>, Error>;

    /// Returns a handle to an archive state reflecting the state at the given block height. The
    /// resulting state must be released and must not outlive the life time of the
    /// provided state.
    fn get_archive_state(&self, block: u64) -> BTResult<Box<dyn CarmenState>, Error>;

    /// Returns a summary of the used memory.
    fn get_memory_footprint(&self) -> BTResult<Box<str>, Error>;
}

/// The safe Carmen state interface.
/// This is the safe interface which gets called from the exported FFI functions.
#[cfg_attr(test, mockall::automock)]
pub trait CarmenState: Send + Sync {
    /// Checks if the given account exists.
    fn account_exists(&self, addr: &Address) -> BTResult<bool, Error>;

    /// Returns the balance of the given account.
    fn get_balance(&self, addr: &Address) -> BTResult<U256, Error>;

    /// Returns the nonce of the given account.
    fn get_nonce(&self, addr: &Address) -> BTResult<Nonce, Error>;

    /// Returns the value of storage location (addr,key) in the given state.
    fn get_storage_value(&self, addr: &Address, key: &Key) -> BTResult<Value, Error>;

    /// Retrieves the code stored under the given address and stores it in `code_buf`.
    /// Returns the number of bytes written to `code_buf`.
    fn get_code(&self, addr: &Address, code_buf: &mut [MaybeUninit<u8>]) -> BTResult<usize, Error>;

    /// Returns the hash of the code stored under the given address.
    fn get_code_hash(&self, addr: &Address) -> BTResult<Hash, Error>;

    /// Returns the code length stored under the given address.
    fn get_code_len(&self, addr: &Address) -> BTResult<u32, Error>;

    /// Returns a global state hash of the given state.
    fn get_hash(&self) -> BTResult<Hash, Error>;

    /// Applies the provided block update to the maintained state.
    #[allow(clippy::needless_lifetimes)] // using an elided lifetime here breaks automock
    fn apply_block_update<'u>(&self, block: u64, update: Update<'u>) -> BTResult<(), Error>;
}

// TODO: Get rid of this once we no longer store an Arc<CarmenState> in CarmenS6Db
impl<T: CarmenState> CarmenState for Arc<T> {
    fn account_exists(&self, addr: &Address) -> BTResult<bool, Error> {
        self.deref().account_exists(addr)
    }

    fn get_balance(&self, addr: &Address) -> BTResult<U256, Error> {
        self.deref().get_balance(addr)
    }

    fn get_nonce(&self, addr: &Address) -> BTResult<Nonce, Error> {
        self.deref().get_nonce(addr)
    }

    fn get_storage_value(&self, addr: &Address, key: &Key) -> BTResult<Value, Error> {
        self.deref().get_storage_value(addr, key)
    }

    fn get_code(&self, addr: &Address, code_buf: &mut [MaybeUninit<u8>]) -> BTResult<usize, Error> {
        self.deref().get_code(addr, code_buf)
    }

    fn get_code_hash(&self, addr: &Address) -> BTResult<Hash, Error> {
        self.deref().get_code_hash(addr)
    }

    fn get_code_len(&self, addr: &Address) -> BTResult<u32, Error> {
        self.deref().get_code_len(addr)
    }

    fn get_hash(&self) -> BTResult<Hash, Error> {
        self.deref().get_hash()
    }

    #[allow(clippy::needless_lifetimes)]
    fn apply_block_update<'u>(&self, block: u64, update: Update<'u>) -> BTResult<(), Error> {
        self.deref().apply_block_update(block, update)
    }
}

/// The `S6` implementation of [`CarmenDb`].
pub struct CarmenS6Db<LS: CarmenState> {
    live_state: Arc<LS>,
}

impl<LS: CarmenState> CarmenS6Db<LS> {
    /// Creates a new [CarmenS6Db] with the provided live state.
    pub fn new(live_state: LS) -> Self {
        Self {
            live_state: Arc::new(live_state),
        }
    }
}

#[allow(unused_variables)]
impl<LS: CarmenState + 'static> CarmenDb for CarmenS6Db<LS> {
    fn checkpoint(&self) -> BTResult<(), Error> {
        // No-op for in-memory state
        // TODO: Handle for storage-based implementation
        Ok(())
    }

    fn close(&self) -> BTResult<(), Error> {
        // No-op for in-memory state
        // TODO: Handle for storage-based implementation
        Ok(())
    }

    fn get_live_state(&self) -> BTResult<Box<dyn CarmenState>, Error> {
        Ok(Box::new(self.live_state.clone()))
    }

    fn get_archive_state(&self, block: u64) -> BTResult<Box<dyn CarmenState>, Error> {
        unimplemented!()
    }

    fn get_memory_footprint(&self) -> BTResult<Box<str>, Error> {
        Err(
            Error::UnsupportedOperation("get_memory_footprint is not yet implemented".to_string())
                .into(),
        )
    }
}
