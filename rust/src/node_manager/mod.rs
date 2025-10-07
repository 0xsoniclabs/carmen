// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::ops::{Deref, DerefMut};

use crate::{error::Error, sync::*};
pub mod cached_node_manager;

/// A collection of thread-safe *nodes* that dereference to [`NodeManager::NodeType`].
///
/// Nodes are uniquely identified by a [`NodeManager::Id`] and are owned by the node manager.
/// They can be accessed through read or write locks with the [`NodeManager::get_read_access`] and
/// [`NodeManager::get_write_access`] methods.
/// IDs are managed by the node manager itself, which hands out new IDs upon insertion of a node.
/// IDs are not globally unique and may be reused after deletion.
///
/// The concrete type returned by the [`NodeManager`] may not be [`NodeManager::NodeType`] but
/// instead a wrapper type which dereferences to [`NodeManager::NodeType`]. This abstraction allows
/// for the node manager to associate metadata with each node, for example to implement smart cache
/// eviction.
pub trait NodeManager {
    /// The ID type used to identify nodes in the node manager.
    type Id;
    /// The node type indexed by the node manager.
    type NodeType;

    /// Adds the given node to the node manager and returns an ID for it.
    fn add(&self, node: Self::NodeType) -> Result<Self::Id, Error>;

    /// Returns a read guard for a node in the node manager, if it exists. Returns
    /// [`crate::storage::Error::NotFound`] otherwise.
    fn get_read_access(
        &self,
        id: Self::Id,
    ) -> Result<RwLockReadGuard<'_, impl Deref<Target = Self::NodeType>>, Error>;

    /// Returns a write guard for a node in the node manager, if it exists. Returns
    /// [`crate::storage::Error::NotFound`] otherwise.
    fn get_write_access(
        &self,
        id: Self::Id,
    ) -> Result<RwLockWriteGuard<'_, impl DerefMut<Target = Self::NodeType>>, Error>;

    /// Deletes a node with the given ID from the node manager.
    /// The ID may be reused in the future, when adding new nodes using
    /// [`NodeManager::add`].
    fn delete(&self, id: Self::Id) -> Result<(), Error>;
}
