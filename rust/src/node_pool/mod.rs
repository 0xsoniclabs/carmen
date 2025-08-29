// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::sync::{Arc, LockResult, RwLock, RwLockReadGuard};

use crate::{
    error::Error,
    types::{Node, NodeId},
};

#[allow(dead_code)]
/// An abstraction for a thread-safe pool of nodes.
pub trait NodePool<T> {
    /// Retrieves an entry from the pool.
    fn get(&self, id: NodeId) -> Result<NodePoolEntry<T>, Error>;

    /// Stores the value in the pool and reserves an ID for it.
    fn set(&self, value: Node) -> Result<NodeId, Error>;

    /// Deletes the entry with the given ID from the pool
    /// The ID may be reused in the future.
    fn delete(&self, id: NodeId) -> Result<(), Error>;

    /// Flushes all pool elements
    fn flush(&self) -> Result<(), Error>;
}

#[allow(dead_code)]
/// A node pool entry that can be safely shared across threads.
#[derive(Debug)]
pub struct NodePoolEntry<T>(Arc<RwLock<T>>);

impl<T> NodePoolEntry<T> {
    #[allow(dead_code)]
    /// Creates a new pool entry with the given [`NodePoolEntry`].
    pub fn new(value: Arc<RwLock<T>>) -> Self {
        Self(value)
    }

    /// Acquires a read lock on the entry.
    #[allow(dead_code)]
    pub fn read(&self) -> LockResult<RwLockReadGuard<'_, T>> {
        self.0.read()
    }

    /// Acquires a write lock on the entry.
    #[allow(dead_code)]
    pub fn write(&self) -> LockResult<std::sync::RwLockWriteGuard<'_, T>> {
        self.0.write()
    }
}
