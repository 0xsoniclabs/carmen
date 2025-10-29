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

use std::{mem::MaybeUninit, ops::Deref, path::PathBuf, sync::Arc};

pub use crate::types::{ArchiveImpl, BalanceUpdate, LiveImpl, Update};
use crate::{
    database::{
        VerkleTrieCarmenState,
        managed_trie::ManagedTrieNode,
        verkle::variants::managed::{
            FullLeafNode, InnerNode, Node, NodeFileStorageManager, SparseLeafNode,
        },
    },
    error::Error,
    node_manager::cached_node_manager::CachedNodeManager,
    statistics::{
        Statistics,
        formatters::{
            StatisticsFormatter, csv_writer::CSVWriter,
            writer_with_indentation::WriterWithIndentation,
        },
    },
    storage::{
        Storage,
        file::{NoSeekFile, NodeFileStorage},
    },
    types::*,
};

mod database;
mod error;
mod ffi;
mod node_manager;
pub mod statistics;
pub mod storage;
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
    directory: &[u8],
) -> Result<Box<dyn CarmenDb>, Error> {
    if schema != 6 {
        return Err(Error::UnsupportedSchema(schema));
    }

    if !matches!(archive_impl, ArchiveImpl::None) {
        return Err(Error::UnsupportedImplementation(
            "archive is not yet supported".to_owned(),
        ));
    }

    match live_impl {
        // FIXME: Reusing memory for now
        LiveImpl::Memory => {
            type FileStorage = NodeFileStorageManager<
                NodeFileStorage<InnerNode, NoSeekFile>,
                NodeFileStorage<SparseLeafNode<2>, NoSeekFile>,
                NodeFileStorage<FullLeafNode, NoSeekFile>,
            >;
            eprintln!("Opening storage at {}", str::from_utf8(directory).unwrap());
            // let storage = StorageWithFlushBuffer::<FileStorage>::open(&PathBuf::from(
            //     str::from_utf8(directory).unwrap(),
            // ))
            // .unwrap();
            let storage =
                FileStorage::open(&PathBuf::from(str::from_utf8(directory).unwrap())).unwrap();

            let is_pinned = |node: &Node| node.get_commitment().is_dirty();

            let manager = Arc::new(CachedNodeManager::new(1_000_000, storage, is_pinned));

            Ok(Box::new(CarmenS6Db::new(VerkleTrieCarmenState::<
                database::ManagedVerkleTrie<_>,
            >::new(manager))))
        }
        LiveImpl::File => Err(Error::UnsupportedImplementation(
            "file-based live state is not yet supported".to_owned(),
        )),
        LiveImpl::LevelDb => Err(Error::UnsupportedImplementation(
            "LevelDB-based live state is not supported".to_owned(),
        )),
    }
}

/// The safe Carmen database interface.
/// This is the safe interface which gets called from the exported FFI functions.
#[cfg_attr(test, mockall::automock)]
pub trait CarmenDb: Send + Sync {
    /// Creates a new checkpoint by persisting all state information to disk to guarantee permanent
    /// storage.
    fn checkpoint(&self) -> Result<(), Error>;

    /// Creates a new checkpoint and then closes this state, releasing all IO handles and locks on
    /// external resources.
    // TODO: Do not create a checkpoint if in an error state (https://github.com/0xsoniclabs/sonic-admin/issues/378)
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
    fn get_nonce(&self, addr: &Address) -> Result<Nonce, Error>;

    /// Returns the value of storage location (addr,key) in the given state.
    fn get_storage_value(&self, addr: &Address, key: &Key) -> Result<Value, Error>;

    /// Retrieves the code stored under the given address and stores it in `code_buf`.
    /// Returns the number of bytes written to `code_buf`.
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

    fn get_statistics(&self) -> Result<Statistics, Error>;
}

// TODO: Get rid of this once we no longer store an Arc<CarmenState> in CarmenS6Db
impl<T: CarmenState> CarmenState for Arc<T> {
    fn account_exists(&self, addr: &Address) -> Result<bool, Error> {
        self.deref().account_exists(addr)
    }

    fn get_balance(&self, addr: &Address) -> Result<U256, Error> {
        self.deref().get_balance(addr)
    }

    fn get_nonce(&self, addr: &Address) -> Result<Nonce, Error> {
        self.deref().get_nonce(addr)
    }

    fn get_storage_value(&self, addr: &Address, key: &Key) -> Result<Value, Error> {
        self.deref().get_storage_value(addr, key)
    }

    fn get_code(&self, addr: &Address, code_buf: &mut [MaybeUninit<u8>]) -> Result<usize, Error> {
        self.deref().get_code(addr, code_buf)
    }

    fn get_code_hash(&self, addr: &Address) -> Result<Hash, Error> {
        self.deref().get_code_hash(addr)
    }

    fn get_code_len(&self, addr: &Address) -> Result<u32, Error> {
        self.deref().get_code_len(addr)
    }

    fn get_hash(&self) -> Result<Hash, Error> {
        self.deref().get_hash()
    }

    #[allow(clippy::needless_lifetimes)]
    fn apply_block_update<'u>(&self, block: u64, update: Update<'u>) -> Result<(), Error> {
        self.deref().apply_block_update(block, update)
    }

    fn get_statistics(&self) -> Result<Statistics, Error> {
        self.deref().get_statistics()
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
    fn checkpoint(&self) -> Result<(), Error> {
        // No-op for in-memory state
        // TODO: Handle for storage-based implementation
        Ok(())
    }

    fn close(&self) -> Result<(), Error> {
        // No-op for in-memory state
        // TODO: Handle for storage-based implementation
        let file = std::fs::File::create("carmen_stats.txt").unwrap();
        let mut formatters = [
            Box::new(WriterWithIndentation::new(file)) as Box<dyn StatisticsFormatter>,
            Box::new(CSVWriter {}) as Box<dyn StatisticsFormatter>,
        ];
        self.get_live_state()?
            .get_statistics()?
            .print(&mut formatters)
            .unwrap();
        Ok(())
    }

    fn get_live_state(&self) -> Result<Box<dyn CarmenState>, Error> {
        Ok(Box::new(self.live_state.clone()))
    }

    fn get_archive_state(&self, block: u64) -> Result<Box<dyn CarmenState>, Error> {
        unimplemented!()
    }

    fn get_memory_footprint(&self) -> Result<Box<str>, Error> {
        Err(Error::UnsupportedOperation(
            "get_memory_footprint is not yet implemented".to_string(),
        ))
    }
}
