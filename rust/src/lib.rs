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

#[cfg(feature = "storage-statistics")]
use crate::statistics::storage::StorageOperationLogger;
pub use crate::types::{ArchiveImpl, BalanceUpdate, LiveImpl, Update};
use crate::{
    database::{
        ManagedTrieNode, ManagedVerkleTrie, VerkleTrieCarmenState,
        verkle::{
            StateMode,
            variants::managed::{
                FullInnerNode, FullLeafNode, InnerDeltaNode, LeafDeltaNode, SparseInnerNode,
                SparseLeafNode, VerkleNode, VerkleNodeFileStorageManager, VerkleNodeId,
            },
        },
    },
    error::{BTResult, Error},
    node_manager::cached_node_manager::CachedNodeManager,
    storage::{
        DbOpenMode, RootIdProvider, Storage,
        file::{NoSeekFile, NodeFileStorage},
        storage_with_flush_buffer::StorageWithFlushBuffer,
    },
    sync::Arc,
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

pub type VerkleStorageManager = VerkleNodeFileStorageManager<
    NodeFileStorage<SparseInnerNode<9>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<15>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<21>, NoSeekFile>,
    NodeFileStorage<FullInnerNode, NoSeekFile>,
    NodeFileStorage<InnerDeltaNode, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<1>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<2>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<5>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<18>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<146>, NoSeekFile>,
    NodeFileStorage<FullLeafNode, NoSeekFile>,
    NodeFileStorage<LeafDeltaNode, NoSeekFile>,
>;

type VerkleStorage = StorageWithFlushBuffer<VerkleStorageManager>;

/// Opens a new [CarmenDb] database object based on the provided implementation maintaining
/// its data in the given directory. If the directory does not exist, it is
/// created. If it is empty, a new, empty state is initialized. If it contains
/// state information, the information is loaded.
pub fn open_carmen_db(
    schema: u8,
    live_impl: &[u8],
    archive_impl: &[u8],
    directory: &Path,
    db_open_mode: DbOpenMode,
) -> BTResult<Box<dyn CarmenDb>, Error> {
    if schema != 6 {
        return Err(Error::UnsupportedSchema(schema).into());
    }

    match (live_impl, archive_impl) {
        // (b"memory", b"none" | b"") => {
        //     Ok(Box::new(CarmenS6InMemoryDb::new(VerkleTrieCarmenState::<
        //         database::SimpleInMemoryVerkleTrie,
        //     >::new_live())))
        // }
        (b"crate-crypto-memory", b"none" | b"") => {
            let live_dir = directory.join("live");
            Ok(Box::new(CarmenS6InMemoryDb::new(VerkleTrieCarmenState::<
                database::CrateCryptoInMemoryVerkleTrie,
            >::new_live(live_dir))))
        }
        (b"file", b"none"| b"") => {
            let live_dir = directory.join("live");
            let storage = VerkleStorage::open(&live_dir, db_open_mode)?;
            #[cfg(feature = "storage-statistics")]
            let storage = StorageOperationLogger::try_new(storage, Path::new("."))?;

            let is_pinned = |node: &VerkleNode| !node.get_commitment().is_clean();
            // TODO: The cache size is arbitrary, base this on a configurable memory limit instead
            // https://github.com/0xsoniclabs/sonic-admin/issues/382
            let manager = Arc::new(CachedNodeManager::new(1_000_000, storage, is_pinned));
            Ok(Box::new(CarmenS6FileBasedDb::new(
                manager.clone(),
                VerkleTrieCarmenState::<ManagedVerkleTrie<_>>::try_new(manager, StateMode::Live)?,
            )))
        }
        (b"file", b"file") => {
            let archive_dir = directory.join("archive");
            let storage = VerkleStorage::open(&archive_dir, db_open_mode)?;
            let is_pinned = |node: &VerkleNode| !node.get_commitment().is_clean();
            // TODO: The cache size is arbitrary, base this on a configurable memory limit instead
            // https://github.com/0xsoniclabs/sonic-admin/issues/382
            let manager = Arc::new(CachedNodeManager::new(1_000_000, storage, is_pinned));
            Ok(Box::new(CarmenS6FileBasedDb::new(
                manager.clone(),
                VerkleTrieCarmenState::<ManagedVerkleTrie<_>>::try_new(manager, StateMode::EvolvingArchive)?
            )))
        }
        _ => Err(Error::UnsupportedImplementation(format!(
            "the combination of live implementation `{}` and archive implementation `{}` is not supported",
            String::from_utf8_lossy(live_impl),
            String::from_utf8_lossy(archive_impl)
        ))
        .into()),
    }
}

/// The safe Carmen database interface.
/// This is the safe interface which gets called from the exported FFI functions.
#[cfg_attr(test, mockall::automock, allow(clippy::disallowed_types))]
pub trait CarmenDb: Send + Sync {
    /// Creates a new checkpoint by persisting all state information to disk to guarantee permanent
    /// storage.
    fn checkpoint(&self) -> BTResult<(), Error>;

    /// Closes this database, releasing all resources and causing its destruction.
    fn close(self: Box<Self>) -> BTResult<(), Error>;

    /// Returns a handle to the live state. The resulting state must be released and must not
    /// outlive the life time of the database.
    fn get_live_state(&self) -> BTResult<Box<dyn CarmenState>, Error>;

    /// Returns a handle to an archive state reflecting the state at the given block height. The
    /// resulting state must be released and must not outlive the life time of the
    /// provided state.
    fn get_archive_state(&self, block: u64) -> BTResult<Box<dyn CarmenState>, Error>;

    /// Retrieves the last block number of the blockchain.
    fn get_archive_block_height(&self) -> BTResult<Option<u64>, Error>;

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

pub trait IsArchive {
    /// Returns true if this is an archive state.
    fn is_archive(&self) -> bool;
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

impl CarmenDb
    for CarmenS6InMemoryDb<VerkleTrieCarmenState<database::CrateCryptoInMemoryVerkleTrie>>
{
    fn checkpoint(&self) -> BTResult<(), Error> {
        // No-op for in-memory state
        Ok(())
    }

    fn close(self: Box<Self>) -> BTResult<(), Error> {
        self.live_state.flush();
        Ok(())
    }

    fn get_live_state(&self) -> BTResult<Box<dyn CarmenState>, Error> {
        Ok(Box::new(self.live_state.clone()))
    }

    fn get_archive_state(&self, _block: u64) -> BTResult<Box<dyn CarmenState>, Error> {
        unimplemented!()
    }

    fn get_archive_block_height(&self) -> BTResult<Option<u64>, Error> {
        Err(Error::UnsupportedOperation(
            "get_archive_block_height is not supported for in-memory databases".to_string(),
        )
        .into())
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
    manager: Arc<CachedNodeManager<S>>,
    live_state: Arc<LS>,
}

impl<S: Storage, LS: CarmenState> CarmenS6FileBasedDb<S, LS> {
    /// Creates a new [`CarmenS6FileBasedDb`] with the provided node manager and live state.
    pub fn new(manager: Arc<CachedNodeManager<S>>, live_state: LS) -> Self {
        Self {
            manager,
            live_state: Arc::new(live_state),
        }
    }
}

impl<S, LS> CarmenDb for CarmenS6FileBasedDb<S, LS>
where
    S: Storage<Id = VerkleNodeId, Item = VerkleNode> + RootIdProvider<Id = VerkleNodeId> + 'static,
    LS: CarmenState + IsArchive + 'static,
{
    fn checkpoint(&self) -> BTResult<(), Error> {
        // TODO: Support checkpoints for archive
        Err(
            Error::UnsupportedOperation("cannot create checkpoint for live state".to_owned())
                .into(),
        )
    }

    fn close(self: Box<Self>) -> BTResult<(), Error> {
        // Ensure that we have no dirty commitments before flushing to disk
        self.live_state.get_hash()?;

        // Release live state first, since it holds a reference to the manager
        drop(self.live_state);
        let manager = Arc::into_inner(self.manager).ok_or_else(|| {
            Error::CorruptedState("node manager reference count is not 1 on close".to_owned())
        })?;
        manager.close()?;
        Ok(())
    }

    fn get_live_state(&self) -> BTResult<Box<dyn CarmenState>, Error> {
        Ok(Box::new(self.live_state.clone()))
    }

    fn get_archive_state(&self, block: u64) -> BTResult<Box<dyn CarmenState>, Error> {
        if !self.live_state.is_archive() {
            return Err(Error::UnsupportedOperation(
                "creating an archive state failed: the database was opened in live only mode"
                    .into(),
            )
            .into());
        }
        Ok(Box::new(Arc::new(VerkleTrieCarmenState::<_>::try_new(
            self.manager.clone(),
            StateMode::Archive(block),
        )?)))
    }

    fn get_archive_block_height(&self) -> BTResult<Option<u64>, Error> {
        if !self.live_state.is_archive() {
            return Err(Error::UnsupportedOperation(
                "get_archive_block_height is not supported for live only databases".to_string(),
            )
            .into());
        }
        Ok(self.manager.highest_block_number()?)
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
    use zerocopy::transmute;

    use super::*;
    use crate::utils::test_dir::{Permissions, TestDir};

    #[rstest_reuse::template]
    #[rstest::rstest]
    #[case::live(b"none")]
    #[case::archive(b"file")]
    fn archive_impl(#[case] archive_impl: &[u8]) {}

    #[rstest_reuse::apply(archive_impl)]
    fn file_based_verkle_trie_implementation_supports_closing_and_reopening(
        #[case] archive_impl: &[u8],
        #[values(DbOpenMode::ReadOnly, DbOpenMode::ReadWrite)] db_open_mode: DbOpenMode,
    ) {
        // This test writes to 512 leaf nodes. In two leaf nodes only one slot gets set, in two leaf
        // nodes two slots get set and so on.
        // This makes sure that no matter the variants of sparse leaf nodes that are used for
        // storage optimization, there will always be at least two nodes for each variant.

        // Skip the first 256 indices to avoid special casing in embedding, where the first leaf
        // only stores 64 values, and the second second 192.
        let key_indices_offset: u16 = 256;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let db =
            open_carmen_db(6, b"file", archive_impl, dir.path(), DbOpenMode::ReadWrite).unwrap();

        let mut slot_updates = Vec::new();
        for address_idx in 0..256 * 2 {
            for key_idx in key_indices_offset..=key_indices_offset + address_idx {
                let mut addr = [0; 20];
                addr[..2].copy_from_slice(&address_idx.to_be_bytes());
                let key = U256::from(key_idx);
                slot_updates.push(SlotUpdate {
                    addr,
                    key: key.to_be_bytes(),
                    value: key.to_be_bytes(),
                });
            }
        }
        let update = Update {
            slots: &slot_updates,
            ..Default::default()
        };

        db.get_live_state()
            .unwrap()
            .apply_block_update(0, update)
            .unwrap();

        db.close().unwrap();

        dir.set_permissions(db_open_mode.to_permissions()).unwrap();
        let db = open_carmen_db(6, b"file", archive_impl, &dir, db_open_mode).unwrap();
        let live = db.get_live_state().unwrap();
        for address_idx in 0..2 * 256 {
            for key_idx in key_indices_offset..=key_indices_offset + address_idx {
                let mut addr = [0; 20];
                addr[..2].copy_from_slice(&address_idx.to_be_bytes());
                let key = U256::from(key_idx);
                assert_eq!(
                    live.get_storage_value(&addr, &key.to_be_bytes()).unwrap(),
                    key.to_be_bytes()
                );
            }
        }
    }

    #[test]
    fn file_based_verkle_trie_implementation_supports_archive_state_semantics() {
        let addr = [1; 20];
        let balance1 = transmute!([[0u8; 16], [2; 16]]);
        let balance2 = transmute!([[0u8; 16], [3; 16]]);

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let db = open_carmen_db(6, b"file", b"file", &dir, DbOpenMode::ReadWrite).unwrap();
        let live_state = db.get_live_state().unwrap();

        live_state
            .apply_block_update(
                0,
                Update {
                    balances: &[BalanceUpdate {
                        addr,
                        balance: balance1,
                    }],
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(live_state.get_balance(&addr).unwrap(), balance1);

        live_state
            .apply_block_update(
                1,
                Update {
                    balances: &[BalanceUpdate {
                        addr,
                        balance: balance2,
                    }],
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(live_state.get_balance(&addr).unwrap(), balance2);

        let archive_state = db.get_archive_state(0).unwrap();
        assert_eq!(archive_state.get_balance(&addr).unwrap(), balance1);
        let archive_state = db.get_archive_state(1).unwrap();
        assert_eq!(archive_state.get_balance(&addr).unwrap(), balance2);
    }

    #[rstest_reuse::apply(archive_impl)]
    fn carmen_s6_file_based_db_checkpoint_returns_error(#[case] archive_impl: &[u8]) {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let db = open_carmen_db(6, b"file", archive_impl, &dir, DbOpenMode::ReadWrite).unwrap();

        let result = db.checkpoint();
        assert_eq!(
            result,
            Err(
                Error::UnsupportedOperation("cannot create checkpoint for live state".to_owned())
                    .into()
            )
        );
    }

    #[rstest_reuse::apply(archive_impl)]
    fn carmen_s6_file_based_db_close_fails_if_node_manager_refcount_not_one(
        #[case] archive_impl: &[u8],
    ) {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let db = open_carmen_db(6, b"file", archive_impl, &dir, DbOpenMode::ReadWrite).unwrap();
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
    fn carmen_s6_file_based_db_get_archive_block_height_fails_in_live_only_mode() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let db = open_carmen_db(6, b"file", b"none", &dir, DbOpenMode::ReadWrite).unwrap();
        let result = db.get_archive_block_height();
        assert_eq!(
            result,
            Err(Error::UnsupportedOperation(
                "get_archive_block_height is not supported for live only databases".to_string(),
            )
            .into())
        );
    }
}
