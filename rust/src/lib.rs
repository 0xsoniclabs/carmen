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
#![cfg_attr(
    feature = "shuttle",
    deny(clippy::disallowed_types, clippy::disallowed_methods)
)]

use std::{mem::MaybeUninit, ops::Deref, path::Path};

pub use crate::types::{ArchiveImpl, BalanceUpdate, LiveImpl, Update};
use crate::{
    database::{
        ManagedTrieNode, VerkleTrieCarmenState,
        verkle::variants::managed::{
            FullLeafNode, InnerNode, SparseLeafNode, VerkleNode, VerkleNodeFileStorageManager,
        },
    },
    error::{BTResult, Error},
    node_manager::cached_node_manager::CachedNodeManager,
    storage::{
        Storage,
        file::{NoSeekFile, NodeFileStorage},
        storage_with_flush_buffer::StorageWithFlushBuffer,
    },
    sync::{Arc, Mutex},
    types::*,
};

pub mod database;
pub mod error;
mod ffi;
pub mod node_manager;
pub mod statistics;
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
    live_impl: &[u8],
    archive_impl: &[u8],
    directory: &Path,
) -> BTResult<Box<dyn CarmenDb>, Error> {
    if schema != 6 {
        return Err(Error::UnsupportedSchema(schema).into());
    }

    let _archive_dir = directory.join("archive");
    if !matches!(archive_impl, b"none" | b"") {
        return Err(
            Error::UnsupportedImplementation("archive is not yet supported".to_owned()).into(),
        );
    }

    let live_dir = directory.join("live");
    match live_impl {
        b"memory" => Ok(Box::new(CarmenS6InMemoryDb::new(VerkleTrieCarmenState::<
            database::SimpleInMemoryVerkleTrie,
        >::new()))),
        b"crate-crypto-memory" => Ok(Box::new(CarmenS6InMemoryDb::new(VerkleTrieCarmenState::<
            database::CrateCryptoInMemoryVerkleTrie,
        >::new()))),
        b"file" => {
            type FileStorage = VerkleNodeFileStorageManager<
                NodeFileStorage<InnerNode, NoSeekFile>,
                NodeFileStorage<SparseLeafNode<1>, NoSeekFile>,
                NodeFileStorage<SparseLeafNode<2>, NoSeekFile>,
                NodeFileStorage<SparseLeafNode<21>, NoSeekFile>,
                NodeFileStorage<SparseLeafNode<64>, NoSeekFile>,
                NodeFileStorage<SparseLeafNode<141>, NoSeekFile>,
                NodeFileStorage<FullLeafNode, NoSeekFile>,
            >;
            let storage = StorageWithFlushBuffer::<FileStorage>::open(&live_dir)?;
            let is_pinned = |node: &VerkleNode| node.get_commitment().is_dirty();
            // TODO: The cache size is arbitrary, base this on a configurable memory limit instead
            // https://github.com/0xsoniclabs/sonic-admin/issues/382
            let manager = Arc::new(CachedNodeManager::new(1_000_000, storage, is_pinned));
            Ok(Box::new(CarmenS6FileBasedDb::new(
                manager.clone(),
                VerkleTrieCarmenState::<database::ManagedVerkleTrie<_>>::try_new(manager)?,
            )))
        }
        b"ldb" => Err(Error::UnsupportedImplementation(
            "LevelDB-based live state is not supported".to_owned(),
        )
        .into()),
        _ => Err(Error::UnsupportedImplementation("unknown live implementation".to_owned()).into()),
    }
}

/// The safe Carmen database interface.
/// This is the safe interface which gets called from the exported FFI functions.
#[cfg_attr(test, mockall::automock, allow(clippy::disallowed_types))]
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
#[cfg_attr(test, mockall::automock, allow(clippy::disallowed_types))]
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

/// An implementation of [`CarmenState`] for `Arc<T>` where `T: CarmenState`,
/// required so we can hand out multiple references to a single state instance
/// on [`CarmenDb::get_live_state`].
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

/// An in-memory `S6` implementation of [`CarmenDb`].
///
/// Does not support closing or checkpointing.
pub struct CarmenS6InMemoryDb<LS: CarmenState> {
    live_state: Arc<LS>,
}

impl<LS: CarmenState> CarmenS6InMemoryDb<LS> {
    /// Creates a new [CarmenS6InMemoryDb] with the provided live state.
    /// The live state is expected to be an in-memory implementation.
    /// No lifecycle methods for closing or checkpointing will be invoked.
    pub fn new(live_state: LS) -> Self {
        Self {
            live_state: Arc::new(live_state),
        }
    }
}

impl<LS: CarmenState + 'static> CarmenDb for CarmenS6InMemoryDb<LS> {
    fn checkpoint(&self) -> BTResult<(), Error> {
        // No-op for in-memory state
        Ok(())
    }

    fn close(&self) -> BTResult<(), Error> {
        // No-op for in-memory state
        Ok(())
    }

    fn get_live_state(&self) -> BTResult<Box<dyn CarmenState>, Error> {
        Ok(Box::new(self.live_state.clone()))
    }

    fn get_archive_state(&self, _block: u64) -> BTResult<Box<dyn CarmenState>, Error> {
        unimplemented!()
    }

    fn get_memory_footprint(&self) -> BTResult<Box<str>, Error> {
        Err(
            Error::UnsupportedOperation("get_memory_footprint is not yet implemented".to_string())
                .into(),
        )
    }
}

/// A file-based `S6` implementation of [`CarmenDb`].
pub struct CarmenS6FileBasedDb<S: Storage, LS: CarmenState> {
    // We currently have to bend over backwards to satisfy the CarmenDb interface
    // TODO: Clean this up on FFI level https://github.com/0xsoniclabs/sonic-admin/issues/474
    manager: Mutex<Option<Arc<CachedNodeManager<S>>>>,
    live_state: Mutex<Option<Arc<LS>>>,
}

impl<S: Storage, LS: CarmenState> CarmenS6FileBasedDb<S, LS> {
    /// Creates a new [`CarmenS6FileBasedDb`] with the provided node manager and live state.
    pub fn new(manager: Arc<CachedNodeManager<S>>, live_state: LS) -> Self {
        Self {
            manager: Mutex::new(Some(manager)),
            live_state: Mutex::new(Some(Arc::new(live_state))),
        }
    }
}

impl<S, LS> CarmenDb for CarmenS6FileBasedDb<S, LS>
where
    S: Storage,
    S::Item: Default + Send + Sync,
    S::Id: Eq + std::hash::Hash + Send + Sync,
    LS: CarmenState + 'static,
{
    fn checkpoint(&self) -> BTResult<(), Error> {
        // TODO: Support checkpoints for archive
        Err(
            Error::UnsupportedOperation("cannot create checkpoint for live state".to_owned())
                .into(),
        )
    }

    fn close(&self) -> BTResult<(), Error> {
        let mut manager = self.manager.lock().unwrap();
        let manager = manager.take().ok_or(Error::CorruptedState(
            "database is already closed".to_owned(),
        ))?;
        // Release live state first, since it holds a reference to the manager
        self.live_state.lock().unwrap().take();
        let manager = Arc::into_inner(manager).ok_or_else(|| {
            Error::CorruptedState("node manager reference count is not 1 on close".to_owned())
        })?;
        manager.close()?;
        Ok(())
    }

    fn get_live_state(&self) -> BTResult<Box<dyn CarmenState>, Error> {
        if let Some(live_state) = &*self.live_state.lock().unwrap() {
            Ok(Box::new(live_state.clone()))
        } else {
            Err(Error::CorruptedState("live state has been closed".to_owned()).into())
        }
    }

    fn get_archive_state(&self, _block: u64) -> BTResult<Box<dyn CarmenState>, Error> {
        unimplemented!()
    }

    fn get_memory_footprint(&self) -> BTResult<Box<str>, Error> {
        Err(
            Error::UnsupportedOperation("get_memory_footprint is not yet implemented".to_string())
                .into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use crypto_bigint::U256;

    use super::*;
    use crate::{
        error::BTError,
        utils::test_dir::{Permissions, TestDir},
    };

    #[test]
    fn file_based_verkle_trie_implementation_supports_closing_and_reopening() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let db = open_carmen_db(6, b"file", b"none", dir.path()).unwrap();

        let addr = [4; 20];
        let balance = U256::from(42u64).to_be_bytes();
        db.get_live_state()
            .unwrap()
            .apply_block_update(
                0,
                Update {
                    balances: &[BalanceUpdate { addr, balance }],
                    ..Default::default()
                },
            )
            .unwrap();

        db.close().unwrap();

        let db = open_carmen_db(6, b"file", b"none", dir.path()).unwrap();
        let received = db.get_live_state().unwrap().get_balance(&addr).unwrap();
        assert_eq!(received, balance);
    }

    #[test]
    fn carmen_s6_file_based_db_checkpoint_returns_error() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let db = open_carmen_db(6, b"file", b"none", dir.path()).unwrap();

        let result = db.checkpoint();
        assert_eq!(
            result,
            Err(
                Error::UnsupportedOperation("cannot create checkpoint for live state".to_owned())
                    .into()
            )
        );
    }

    #[test]
    fn carmen_s6_file_based_db_close_fails_if_already_closed() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let db = open_carmen_db(6, b"file", b"none", dir.path()).unwrap();

        db.close().unwrap();
        let result = db.close();
        assert_eq!(
            result,
            Err(Error::CorruptedState("database is already closed".to_owned()).into())
        );
    }

    #[test]
    fn carmen_s6_file_based_db_close_fails_if_node_manager_refcount_not_one() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let db = open_carmen_db(6, b"file", b"none", dir.path()).unwrap();
        let _live_state = db.get_live_state().unwrap();

        let result = db.close();
        assert_eq!(
            result,
            Err(
                Error::CorruptedState("node manager reference count is not 1 on close".to_owned())
                    .into()
            )
        );
    }

    #[test]
    fn carmen_s6_file_based_db_get_live_state_fails_if_closed() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let db = open_carmen_db(6, b"file", b"none", dir.path()).unwrap();

        db.close().unwrap();
        let result = db.get_live_state().map_err(BTError::into_inner);
        assert!(matches!(
            result,
            Err(Error::CorruptedState(e)) if e == "live state has been closed"
        ));
    }
}
