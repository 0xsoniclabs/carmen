use std::{
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock},
};

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
}

impl<T> Deref for NodePoolEntry<T> {
    type Target = Arc<RwLock<T>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for NodePoolEntry<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
