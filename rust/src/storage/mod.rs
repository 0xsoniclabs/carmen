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

pub mod error;
pub mod file;
pub mod storage_with_flush_buffer;


/// A trait for storage backends that can store and retrieve items by their IDs.
/// This is used for multiple layers of the storage system, but with different types for
/// `Id` and `Item`.
#[cfg_attr(test, mockall::automock(type Id = i32; type Item = i32;))]
pub trait Storage {
    /// The type of the ID used to identify `Self::Item` in the storage.
    type Id: Copy;
    /// The type of the item stored in the storage.
    type Item;

    /// Opens the storage backend at the given path.
    /// Depending on the implementation, the path is required to be a directory or a file.
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

    /// Flushes all changes to the storage.
    fn flush(&self) -> Result<(), Error>;
}
