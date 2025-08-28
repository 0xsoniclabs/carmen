use std::sync::{Arc, LockResult, RwLock};

use crate::{
    error::Error,
    types::{Node, NodeId},
};

pub mod node_pool_with_storage;

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

/// A node pool entry that can be safely shared across threads.
#[derive(Debug)]
pub struct NodePoolEntry<T>(Arc<RwLock<T>>);

impl<T> NodePoolEntry<T> {
    /// Creates a new pool entry with the given [`NodePoolEntry`].
    pub fn new(value: Arc<RwLock<T>>) -> Self {
        Self(value)
    }

    /// Acquires a read lock on the entry.
    pub fn read(&self) -> LockResult<std::sync::RwLockReadGuard<'_, T>> {
        self.0.read()
    }

    /// Acquires a write lock on the entry.
    pub fn write(&self) -> LockResult<std::sync::RwLockWriteGuard<'_, T>> {
        self.0.write()
    }
}
