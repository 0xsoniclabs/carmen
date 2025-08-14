use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::{
    cache::NodeCacheEntryImpl,
    storage::{Error, Storage},
    types::{NodeId, NodeType},
};

/// A simple in-memory [`Storage`] implementation
pub struct MemoryStorage<Id, T> {
    storage: Arc<Mutex<HashMap<Id, T>>>,
    next_id: AtomicU64,
}

impl<Id: Copy, T: Default + Clone> MemoryStorage<Id, T> {
    /// Creates a new in-memory storage with the given data as storage backend.
    /// Previous data are cleared.
    pub fn new_with_data(data: Arc<Mutex<HashMap<Id, T>>>) -> Self {
        data.lock().unwrap().clear();
        Self {
            storage: data,
            next_id: AtomicU64::new(0),
        }
    }

    /// Creates a new in-memory storage
    pub fn new() -> Self {
        Self::new_with_data(Arc::new(Mutex::new(HashMap::new())))
    }
}

impl Storage for MemoryStorage<NodeId, Arc<NodeCacheEntryImpl>> {
    type Id = NodeId;
    type Item = Arc<NodeCacheEntryImpl>;

    fn get(&self, id: Self::Id) -> Result<Self::Item, Error> {
        let storage = self.storage.lock().unwrap();
        if storage.contains_key(&id) {
            Ok(storage[&id].clone())
        } else {
            Err(Error::NotFound)
        }
    }

    /// Reserve a new ID for the given element.
    fn reserve(&self, node: &Self::Item) -> Self::Id {
        loop {
            let val = self.next_id.load(Ordering::Relaxed);
            let id =
                self.next_id
                    .compare_exchange(val, val + 1, Ordering::SeqCst, Ordering::SeqCst);
            if id.is_ok() {
                return NodeId::from_idx_and_node_type(
                    id.unwrap(),
                    NodeType::from(&node.read().unwrap().value),
                );
            }
        }
    }

    /// No-op
    fn flush(&self) -> Result<(), Error> {
        Ok(())
    }

    /// Sets the node with the given ID.
    /// The ID must have been reserved before through the [`Self::reserve`] function.
    fn set(&self, id: Self::Id, node: &Self::Item) -> Result<(), Error> {
        match self.next_id.load(Ordering::Relaxed).checked_sub(1) {
            None => return Err(Error::NotFound), // No IDs reserved yet
            Some(idx) if idx < id.to_index() => return Err(Error::NotFound),
            _ => {} // idx >= id.to_index()
        };
        let mut storage = self.storage.lock().unwrap();
        let _ = storage.insert(id, node.clone());
        Ok(())
    }

    /// Deletes the node with the given ID.
    /// If the ID does not exist, it is a no-op.
    fn delete(&self, id: Self::Id) -> Result<(), crate::storage::Error> {
        let mut storage = self.storage.lock().unwrap();
        let _ = storage.remove(&id);
        Ok(())
    }

    fn open(_path: &std::path::Path) -> Result<Self, crate::storage::Error>
    where
        Self: Sized,
    {
        unimplemented!()
    }
}
