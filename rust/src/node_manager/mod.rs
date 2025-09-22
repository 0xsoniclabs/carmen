// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{
    ops::{Deref, DerefMut},
    sync::{RwLockReadGuard, RwLockWriteGuard},
};

use crate::error::Error;
pub mod cached_node_manager;

/// A collection of thread-safe *nodes* that dereference to [`NodeManager::NodeType`].
///
/// Nodes are uniquely identified by a [`NodeManager::Id`].
/// Nodes ownership is held by the [`NodeManager`] implementation and can be accessed through read
/// or write locks with the [`NodeManager::get_read_access`] and [`NodeManager::get_write_access`]
/// methods.
/// Calling a `get_*` method with the same ID twice is guaranteed to yield the same item.
/// IDs are managed by the pool itself, which hands out new IDs upon insertion of an item.
/// IDs are not globally unique and may be reused after deletion.
///
/// The concrete type returned by the [`NodeManager`] may not be [`NodeManager::NodeType`] but
/// instead a wrapper type which dereferences to [`NodeManager::NodeType`]. This abstraction allows
/// for the pool to associate metadata with each item, for example to implement smart cache
/// eviction.
#[allow(dead_code)]
pub trait NodeManager {
    /// The id type used to identify items in the pool.
    type Id;
    /// The node type indexed by the pool, which is specialized depending on the trie
    /// implementation.
    type NodeType;

    /// Adds the item in the node manager and returns an ID for it.
    fn add(&self, item: Self::NodeType) -> Result<Self::Id, Error>;

    /// Retrieves and lock an item from the node manager with read access, if it exists. Returns
    /// [`crate::storage::Error::NotFound`] otherwise.
    fn get_read_access(
        &self,
        id: Self::Id,
    ) -> Result<RwLockReadGuard<'_, impl Deref<Target = Self::NodeType>>, Error>;

    /// Retrieves and lock an item from the node manager with write access, if it exists. Returns
    /// [`crate::storage::Error::NotFound`] otherwise.
    fn get_write_access(
        &self,
        id: Self::Id,
    ) -> Result<RwLockWriteGuard<'_, impl DerefMut<Target = Self::NodeType>>, Error>;

    /// Deletes an item with the given ID from the pool
    /// The ID may be reused in the future, when creating a new item by calling
    /// [`NodeManager::add`].
    fn delete(&self, id: Self::Id) -> Result<(), Error>;

    /// Flushes all pending operations to the underlying storage layer (if one exists).
    fn flush(&self) -> Result<(), Error>;
}
