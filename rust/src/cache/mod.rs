use crate::error::Error;
use quick_cache::Lifecycle;
use quick_cache::UnitWeighter;
use quick_cache::sync::GuardResult;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::RandomState;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Copy)]
pub struct Id(u64);

impl Id {
    fn new(id: u64) -> Self {
        Id(id)
    }

    fn to_u64(self) -> u64 {
        self.0
    }

    fn to_usize(self) -> usize {
        self.0 as usize
    }
}

impl Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A trait representing a cache for nodes of type `T` indexed by `K`.
pub trait Cache<K, V> {
    fn get(&self, id: K) -> Result<Arc<Mutex<CacheEntry<V>>>, Error>;

    fn reserve(&self, node: &V) -> Id;

    fn set(&mut self, id: K, node: V) -> Result<(), Error>;

    fn flush(&mut self) -> Result<(), Error>;
}

pub trait Storage<T> {
    fn get(&self, id: Id) -> Result<T, Error>;

    fn reserve(&self, node: &T) -> Id;

    fn set(&self, id: Id, node: &T) -> Result<(), Error>;

    fn flush(&self) -> Result<(), Error>;
}

static UPDATE_MODE: AtomicBool = AtomicBool::new(false);

struct NodeCacheLifecycle<T> {
    storage: Arc<Box<dyn Storage<T>>>,
}

impl<T> Clone for NodeCacheLifecycle<T> {
    fn clone(&self) -> Self {
        NodeCacheLifecycle {
            storage: self.storage.clone(),
        }
    }
}

impl<V: Clone + Debug> Lifecycle<Id, Arc<Mutex<CacheEntry<V>>>> for NodeCacheLifecycle<V> {
    fn begin_request(&self) -> Self::RequestState {
        // No state needed for this implementation
    }

    /// Stores the entry in the storage when it is evicted if it's dirty.
    fn on_evict(&self, _state: &mut Self::RequestState, key: Id, val: Arc<Mutex<CacheEntry<V>>>) {
        let entry = val.lock().unwrap(); // How to handle lock errors?
        if !entry.dirty {
            return;
        }
        self.storage.set(key, &entry.value).unwrap(); //FIXME: when would it fail?
    }

    /// Checks if the entry cannot be evicted.
    /// The only case in which is true is when the database is in update mode.
    fn is_pinned(&self, _key: &Id, _val: &Arc<Mutex<CacheEntry<V>>>) -> bool {
        UPDATE_MODE.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Type used to store state during a request.
    type RequestState = ();
}

/// An entry in the cache that holds a value of type `T` and a flag indicating if it is dirty.
/// The dirty flag indicates if the entry has been modified while in cache and needs to be flushed to the storage when evicted.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CacheEntry<T> {
    value: T,
    dirty: bool,
}

/// A cache for elements of type `T` indexed by `Id` that uses a storage backend to retrieve and persist data.
pub struct NodeCache<T> {
    cache: quick_cache::sync::Cache<
        Id,
        Arc<Mutex<CacheEntry<T>>>,
        UnitWeighter,
        RandomState,
        NodeCacheLifecycle<T>,
    >,
    storage: Arc<Box<dyn Storage<T>>>,
}

impl<T: Clone + Debug> NodeCache<T> {
    /// Creates a new [`NodeCache`] with the given storage and estimated capacity.
    /// The `estimated_capacity` is used to optimize the cache's internal structure.
    pub fn try_new(storage: Box<dyn Storage<T>>, estimated_capacity: usize) -> Result<Self, Error> {
        let storage = Arc::new(storage);
        let options = quick_cache::OptionsBuilder::new()
            .estimated_items_capacity(estimated_capacity)
            .weight_capacity(estimated_capacity as u64) // Using a weight capacity of 1 per item
            .build()
            .map_err(|e| Error::Custom(e.to_string()))?;
        Ok(Self {
            cache: quick_cache::sync::Cache::with_options(
                options,
                UnitWeighter,
                RandomState::default(),
                NodeCacheLifecycle {
                    storage: storage.clone(),
                },
            ),
            storage,
        })
    }
}

impl<V> Cache<Id, V> for NodeCache<V>
where
    V: Clone + Debug,
{
    /// Retrieves an entry from the cache.
    /// If the entry is not in the cache, it retrieves it from the storage.
    fn get(&self, id: Id) -> Result<Arc<Mutex<CacheEntry<V>>>, Error> {
        self.cache.get_or_insert_with(&id, || {
            let entry = self.storage.get(id)?;
            Ok(Arc::new(Mutex::new(CacheEntry {
                value: entry,
                dirty: false,
            })))
        })
    }

    /// Reserves a new ID for the node.
    /// This ID can be used to set or retrieve the node from the cache.
    fn reserve(&self, node: &V) -> Id {
        self.storage.reserve(node)
    }

    /// Sets a node in the cache and marks it as dirty.
    /// If the node already exists, it updates its value.
    fn set(&mut self, id: Id, node: V) -> Result<(), Error> {
        match self.cache.get_value_or_guard(&id, None) {
            GuardResult::Value(entry) => {
                let mut entry = entry.lock().unwrap();
                entry.value = node;
                entry.dirty = true;
                Ok(())
            }
            GuardResult::Guard(guard) => {
                guard
                    .insert(Arc::new(Mutex::new(CacheEntry {
                        value: node,
                        dirty: true,
                    })))
                    .map_err(|_| Error::Custom("Failed to insert into cache".to_string()))?;
                Ok(())
            }
            GuardResult::Timeout => unreachable!(), // There is no timeout
        }
    }

    /// Flushes all dirty entries in the cache to the storage.
    fn flush(&mut self) -> Result<(), Error> {
        for (id, entry) in self.cache.iter() {
            let mut entry = entry.lock().map_err(|e| Error::Custom(e.to_string()))?; //TODO: error handling
            if entry.dirty {
                self.storage.set(id, &entry.value)?;
                entry.dirty = false;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    #[test]
    fn node_cache_try_new_creates_node_cache() {
        let storage = Box::new(StubStorage::<String, 100>::new());
        let cache = NodeCache::try_new(storage, 10);
        assert!(cache.is_ok());
    }

    #[test]
    fn node_cache_get_returns_entry_from_storage() {
        let storage = Box::new(StubStorage::<u64, 100>::new());
        for node in 0..100 {
            let pos = storage.reserve(&node);
            storage.set(pos, &node).unwrap();
        }

        let cache = NodeCache::try_new(storage, 10).unwrap();
        for i in 0..100 {
            let entry = cache.get(Id::new(i)).unwrap();
            let entry = entry.lock().unwrap();
            assert_eq!(entry.value, i);
        }
    }

    #[test]
    fn node_cache_set_sets_elements_in_cache() {
        let storage = Box::new(StubStorage::<u64, 100>::new());
        let mut cache = NodeCache::try_new(storage, 10).unwrap();
        for i in 0..10 {
            let id = cache.reserve(&i);
            cache.set(id, i).unwrap();
            let entry = cache.get(id).unwrap();
            let entry = entry.lock().unwrap();
            assert_eq!(entry.value, i);
            assert!(entry.dirty);
        }
    }

    #[test]
    fn node_cache_set_updates_existing_entry() {
        let storage = Box::new(StubStorage::<u64, 10>::new());
        let mut cache = NodeCache::try_new(storage, 10).unwrap();
        let node = 5;
        let id = cache.reserve(&node);
        cache.set(id, node).unwrap();
        let entry = cache.get(id).unwrap();
        {
            let entry = entry.lock().unwrap();
            assert_eq!(entry.value, 5);
            assert!(entry.dirty);
        }
        // Update the value
        cache.set(id, 10).unwrap();
        let entry = entry.lock().unwrap();
        assert_eq!(entry.value, 10);
    }

    #[test]
    fn node_cache_flush_saves_dirty_entries_to_storage() {
        let data = Arc::new(Mutex::new(vec![0; 100]));
        let storage = Box::new(StubStorage::<u64, 100>::new_with_vec(data.clone()));
        let mut cache = NodeCache::try_new(storage, 10).unwrap();
        for i in 0..10 {
            let id = cache.reserve(&i);
            cache.set(id, i).unwrap();
        }
        cache.flush().unwrap();
        for i in 0..10u64 {
            assert_eq!(data.lock().unwrap()[i as usize], i);
        }
    }

    #[test]
    fn node_cache_stores_data_in_storage_on_evict() {
        let data = Arc::new(Mutex::new(vec![0; 100]));
        let storage = Box::new(StubStorage::<u64, 100>::new_with_vec(data.clone()));
        let mut cache = NodeCache::try_new(storage, 10).unwrap();
        // Fill the cache
        for i in 0..10 {
            let id = cache.reserve(&i);
            cache.set(id, i).unwrap();
        }
        // Trigger eviction by adding more items
        for i in 10..20 {
            let id = cache.reserve(&i);
            cache.set(id, i).unwrap();
        }
        // The last 10 items should be evicted because of infinite reuse distance
        for i in 10..20u64 {
            assert_eq!(data.lock().unwrap()[i as usize], i);
        }
    }

    struct StubStorage<T, const N: u64> {
        storage: Arc<Mutex<Vec<T>>>,
        cur_size: AtomicU64,
    }

    impl<T: Default + Clone, const N: u64> StubStorage<T, N> {
        pub fn new() -> Self {
            Self {
                storage: Arc::new(Mutex::new(vec![T::default(); N as usize])),
                cur_size: AtomicU64::new(0),
            }
        }

        pub fn new_with_vec(data: Arc<Mutex<Vec<T>>>) -> Self {
            let size = data.lock().unwrap().len() as u64;
            assert!(size <= N, "Data size exceeds maximum capacity");
            Self {
                storage: data,
                cur_size: AtomicU64::new(0),
            }
        }

        pub fn get_max_size(&self) -> u64 {
            N
        }
    }

    impl<T: Default + Display, const N: u64> Storage<T> for StubStorage<T, N>
    where
        T: Clone,
    {
        fn get(&self, id: Id) -> Result<T, Error> {
            if id.to_u64() < self.cur_size.load(Ordering::Relaxed) {
                Ok(self.storage.lock().unwrap()[id.to_usize()].clone())
            } else {
                Err(format!("No entry found for ID: {}", id).into())
            }
        }

        fn reserve(&self, _node: &T) -> Id {
            loop {
                let val = self.cur_size.load(Ordering::Relaxed);
                if val >= N {
                    panic!("Storage is full, cannot reserve more nodes");
                }

                let res = self.cur_size.compare_exchange(
                    val,
                    val + 1,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                );
                if res.is_ok() {
                    return Id::new(val);
                }
            }
        }

        fn flush(&self) -> Result<(), Error> {
            Ok(()) // Stub implementation
        }

        fn set(&self, id: Id, node: &T) -> Result<(), Error> {
            if id.to_u64() < self.cur_size.load(Ordering::Relaxed) {
                self.storage.lock().unwrap()[id.to_usize()] = node.clone();
                Ok(())
            } else {
                Err(format!("Cannot insert element {} at pos {}", node, id).into())
            }
        }
    }
}
