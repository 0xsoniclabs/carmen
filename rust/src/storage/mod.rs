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

use std::path::Path;

pub use self::error::Error;

mod error;
pub mod file;
pub mod storage_with_flush_buffer;

/// A trait for storage backends that can store and retrieve items by their IDs.
/// This is used for multiple layers of the storage system, but with different types for
/// `Id` and `Item`.
pub trait Storage {
    /// The type of the ID used to identify `Self::Item` in the storage.
    type Id: Copy;
    /// The type of the item stored in the storage.
    type Item;

    /// Opens the storage backend at the given path and restores the state of the last committed
    /// checkpoint.
    /// Depending on the implementation, the path is required to be a directory or a
    /// file.
    fn open(path: &Path) -> Result<Self, Error>
    where
        Self: Sized;

    /// Returns the item with the given ID.
    fn get(&self, id: Self::Id) -> Result<Self::Item, Error>;

    /// Reserves a new ID for the given item.
    /// IDs are only unique at a give point in time, but may be reused if this item is deleted.
    fn reserve(&self, item: &Self::Item) -> Self::Id;

    /// Stores the item with the given ID.
    fn set(&self, id: Self::Id, item: &Self::Item) -> Result<(), Error>;

    /// Deletes the item with the given ID.
    /// The ID may be reused in the future.
    fn delete(&self, id: Self::Id) -> Result<(), Error>;
}

/// An entity which can create durable checkpoints of its state.
/// This trait is used for entities which may hold volatile state in memory, but which are not
/// directly responsible for storing that state durably. Therefore, they only need to flush their
/// state to the underlying layer and call `checkpoint` on that layer, but are not participating in
/// the two-phase commit protocol. The only except is the lowers layer which implements
/// [`Checkpointable`]. This layer acts as the checkpoint coordinator. The layers below it should
/// implement [`CheckpointParticipant`].
pub trait Checkpointable {
    /// Create a checkpoint which is guaranteed to be durable.
    fn checkpoint(&self) -> Result<(), Error>;
}

/// An entity which participates in a two-phase commit protocol.
/// This trait is used for entities that are responsible for storing part of the overall state
/// durably on disk.
pub trait CheckpointParticipant {
    /// Checks that `checkpoint` is the latest checkpoint the participant has committed to.
    fn ensure(&self, checkpoint: u64) -> Result<(), Error>;

    /// Prepares the given checkpoint, but does not commit to it yet.
    /// A prepared checkpoint can be either committed or aborted.
    fn prepare(&self, checkpoint: u64) -> Result<(), Error>;

    /// Commits the given checkpoint, making it durable. The checkpoint must have been
    /// prepared before. Failing to commit a prepared checkpoint is an irrecoverable error.
    fn commit(&self, checkpoint: u64) -> Result<(), Error>;

    /// Aborts the given checkpoint, reverting any preparations done in the prepare phase. The node
    /// data itself is not discarded. The checkpoint must have been prepared before.
    fn abort(&self, checkpoint: u64) -> Result<(), Error>;
}
