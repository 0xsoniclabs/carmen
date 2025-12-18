// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.
#![allow(dead_code)]

use std::{fs::OpenOptions, path::Path};

pub use self::error::Error;
use crate::error::BTResult;

mod error;
pub mod file;
pub mod storage_with_flush_buffer;
#[cfg(test)]
use tests::all_db_modes;

/// The mode in which the database can be opened.
#[derive(Debug, Clone, Copy)]
pub enum DbMode {
    ReadOnly,
    ReadWrite,
}

impl DbMode {
    /// Returns if the database mode allows write access.
    pub fn has_write_access(&self) -> bool {
        matches!(self, DbMode::ReadWrite)
    }

    /// Returns if the database mode is read-only.
    pub fn read_only(&self) -> bool {
        matches!(self, DbMode::ReadOnly)
    }

    /// Converts the database mode to the corresponding [`std::fs::OpenOptions`] instance for
    /// opening files.
    /// Files are never truncated. In `ReadWrite` mode, files
    /// are created if they do not exist.
    pub fn to_open_options(&self) -> OpenOptions {
        let mut options = OpenOptions::new();
        match self {
            DbMode::ReadOnly => options.create(false).truncate(false).read(true),
            DbMode::ReadWrite => options.create(true).truncate(false).read(true).write(true),
        };
        options
    }
}

/// A trait for storage backends that can store and retrieve items by their IDs.
/// This is used for multiple layers of the storage system, but with different types for
/// `Id` and `Item`.
pub trait Storage: Send + Sync {
    /// The type of the ID used to identify `Self::Item` in the storage.
    type Id: Copy;
    /// The type of the item stored in the storage.
    type Item;

    /// Opens the storage backend at the given path in the given mode and restores the state of the
    /// last committed checkpoint.
    /// Depending on the implementation, the path is required to be a directory or a
    /// file.
    fn open(path: &Path, db_mode: DbMode) -> BTResult<Self, Error>
    where
        Self: Sized;

    /// Returns the item with the given ID.
    fn get(&self, id: Self::Id) -> BTResult<Self::Item, Error>;

    /// Reserves a new ID for the given item.
    /// IDs are only unique at a give point in time, but may be reused if this item is deleted.
    fn reserve(&self, item: &Self::Item) -> Self::Id;

    /// Stores the item with the given ID.
    fn set(&self, id: Self::Id, item: &Self::Item) -> BTResult<(), Error>;

    /// Deletes the item with the given ID.
    /// The ID may be reused in the future.
    fn delete(&self, id: Self::Id) -> BTResult<(), Error>;

    /// Closes the storage, flushing any pending changes to disk.
    fn close(self) -> BTResult<(), Error>;
}

/// An entity which can create durable checkpoints of its state that can be restored later.
/// This trait is used for entities which may hold volatile state in memory, but which are not
/// directly responsible for storing that state durably. Therefore, they only need to flush their
/// state to the underlying layer and call `checkpoint` on that layer, but are not participating in
/// the two-phase commit protocol. The only exception is the lowest layer which implements
/// [`Checkpointable`]. This layer acts as the checkpoint coordinator. The layers below it should
/// implement [`CheckpointParticipant`].
/// Users of this trait need to ensure that when a checkpoint is requested, there is no other
/// operation (read, write or other checkpoint) in progress.
pub trait Checkpointable: Send + Sync {
    /// Creates a checkpoint which is guaranteed to be durable and returns the checkpoint number.
    fn checkpoint(&self) -> BTResult<u64, Error>;

    /// Restores the state on disk to the state at the given checkpoint.
    /// If this operation succeeds, the next open call should be able to open the state without
    /// any errors.
    fn restore(path: &Path, checkpoint: u64) -> BTResult<(), Error>;
}

/// An entity which participates in a two-phase commit protocol.
/// This trait is used for entities that are responsible for storing part of the overall state
/// durably on disk.
/// Users of this trait need to ensure that the order of operations is a valid sequence, and that
/// there is no other operation (read, write or other checkpoint) in progress.
pub trait CheckpointParticipant {
    /// Checks that `checkpoint` is the latest checkpoint the participant has committed to.
    fn ensure(&self, checkpoint: u64) -> BTResult<(), Error>;

    /// Prepares the given checkpoint, but does not commit to it yet.
    /// A prepared checkpoint can be either committed or aborted.
    fn prepare(&self, checkpoint: u64) -> BTResult<(), Error>;

    /// Commits the given checkpoint, making it durable. The checkpoint must have been
    /// prepared before. Failing to commit a prepared checkpoint is an irrecoverable error.
    fn commit(&self, checkpoint: u64) -> BTResult<(), Error>;

    /// Aborts the given checkpoint, reverting any preparations done in the prepare phase. The node
    /// data itself is not discarded. The checkpoint must have been prepared before.
    fn abort(&self, checkpoint: u64) -> BTResult<(), Error>;

    /// Restores the state on disk to the state at the given checkpoint.
    /// If this operation succeeds, the next open call should be able to open the state without
    /// any errors.
    fn restore(path: &Path, checkpoint: u64) -> BTResult<(), Error>
    where
        Self: Sized;
}

/// An entity which can provide and store IDs of a tree's root node at different block numbers.
pub trait RootIdProvider {
    /// The type of the ID used to identify nodes.
    type Id;

    /// Returns the root ID for the given block number.
    fn get_root_id(&self, block_number: u64) -> BTResult<Self::Id, Error>;

    /// Sets the root ID for the given block number. The block number must greater than all block
    /// numbers previously passed to `set_root_id`.
    fn set_root_id(&self, block_number: u64, id: Self::Id) -> BTResult<(), Error>;

    /// Returns the highest block number for which a root ID is stored.
    fn highest_block_number(&self) -> BTResult<Option<u64>, Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::test_dir::{Permissions, TestDir};

    #[test]
    fn db_mode_to_open_options_returns_correct_options() {
        let tmp_dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let non_existing_file = tmp_dir.path().join("non_existing_file");
        let existing_file = tmp_dir.path().join("existing_file");
        std::fs::write(&existing_file, b"").unwrap();

        let read_options = DbMode::ReadOnly.to_open_options();
        assert!(read_options.open(&non_existing_file).is_err());
        assert!(read_options.open(&existing_file).is_ok());

        let read_write_options = DbMode::ReadWrite.to_open_options();
        assert!(read_write_options.open(&non_existing_file).is_ok());
        // File is created if it does not exist with ReadWrite mode
        assert!(std::fs::exists(non_existing_file).unwrap());
        assert!(read_write_options.open(&existing_file).is_ok());
    }

    #[rstest_reuse::template]
    #[rstest::rstest]
    #[case::read(DbMode::ReadOnly)]
    #[case::read_write(DbMode::ReadWrite)]
    pub fn all_db_modes(#[case] db_mode: DbMode) {}
}
