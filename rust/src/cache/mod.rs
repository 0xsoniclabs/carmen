use crate::error::Error;
use crate::error::ErrorState;
use crate::storage::Storage;
use crate::types::Node;
use crate::types::NodeId;
use quick_cache::Lifecycle;
use quick_cache::UnitWeighter;
use std::fmt::Debug;
use std::hash::RandomState;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::LockResult;
use std::sync::RwLock;
use std::sync::RwLockReadGuard;
use std::sync::RwLockWriteGuard;

/// A trait representing a cache.
pub trait Cache {
    /// The type of the ID used to identify [`Self::Item`] in the cache.
    type Id;
    /// The type of the payload stored in the cache.
    type ItemPayload;
    /// The type stored in the cache.
    type StoredItem;

    /// Retrieves an entry from the cache.
    fn get(&self, id: Self::Id) -> Result<Self::StoredItem, Error>;

    /// Stores the value in the cache and reserves an ID for it.
    fn set(&self, value: Self::ItemPayload) -> Result<Self::Id, Error>;

    /// Flushes all cache elements
    fn flush(&self) -> Result<(), Error>;
}

/// A [`quick_cache::Lifecycle`] implementation for the [`NodeCache`].
/// It handles two main tasks:
/// - Storing entries in the storage backend when they are evicted from the cache if dirty.
/// - Pinning entries that are currently is use outside of the cache
struct NodeCacheLifecycle<S: Storage> {
    storage: Arc<S>,
    error: Arc<ErrorState>,
}

impl<S: Storage> Clone for NodeCacheLifecycle<S> {
    fn clone(&self) -> Self {
        NodeCacheLifecycle {
            storage: self.storage.clone(),
            error: self.error.clone(),
        }
    }
}

impl<S: Storage<Id = NodeId, Item = Arc<NodeCacheEntryImpl>>>
    Lifecycle<NodeId, Arc<NodeCacheEntryImpl>> for NodeCacheLifecycle<S>
{
    /// No-op for beginning a request, as we don't need to store any state.
    fn begin_request(&self) -> Self::RequestState {}

    /// Stores the entry in the storage when it is evicted if it's dirty.
    /// In case of an error, it registers the error in the `ErrorState`.
    fn on_evict(&self, _state: &mut Self::RequestState, key: NodeId, val: Arc<NodeCacheEntryImpl>) {
        if !val.read().unwrap().dirty {
            return;
        }
        self.storage.set(key, &val).unwrap_or_else(|e| {
            self.error.register(Error::Storage(e));
        });
    }

    /// Checks if the entry is pinned and cannot be evicted.
    /// An entry is pinned if another thread holds a reference to it (strong count > 1)
    fn is_pinned(&self, _key: &NodeId, val: &Arc<NodeCacheEntryImpl>) -> bool {
        Arc::strong_count(val) > 1
    }

    /// Type used to store state during a request.
    /// We don't store any state, set as unit type.
    type RequestState = ();
}

/// The payload stored in the [`NodeCache`], which wraps a [`Node`] with a **dirty** flag.
/// - The **dirty** flag indicates if the entry has been modified and needs to be flushed to the storage when evicted.
#[derive(Debug)]
pub struct NodeCacheEntryPayload {
    pub value: Node,
    dirty: bool,
}

/// The entry stored in the [`NodeCache`],
/// which wraps a [`NodeCacheEntryPayload`] and provides thread-safe access.
#[derive(Debug)]
pub struct NodeCacheEntryImpl {
    value: RwLock<NodeCacheEntryPayload>,
}

impl NodeCacheEntryImpl {
    /// Creates a new cache entry with the given value.
    pub fn new(value: Node) -> Self {
        NodeCacheEntryImpl {
            value: RwLock::new(NodeCacheEntryPayload {
                value,
                dirty: false,
            }),
        }
    }

    /// Provides read access to the inner [`NodeCacheEntryPayload`].
    pub fn read(&self) -> LockResult<RwLockReadGuard<'_, NodeCacheEntryPayload>> {
        self.value.read()
    }

    /// Provides write access to the [`NodeCacheEntryPayload`]
    /// and sets the **dirty** flag
    pub fn write(&self) -> LockResult<RwLockWriteGuard<'_, NodeCacheEntryPayload>> {
        let mut value = self.value.write()?;
        value.dirty = true;
        Ok(value)
    }
}

/// The public interface for a [`NodeCache`] entry.
/// The inner implementation can be access through the [`Deref`] trait,
/// NOTE: this type does not implement `Clone` to avoid cloning the inner [`Arc`].
#[derive(Debug)]
pub struct NodeCacheEntry {
    value: Arc<NodeCacheEntryImpl>,
}

impl NodeCacheEntry {
    /// Creates a new [`NodeCacheEntry`] with the given value.
    pub fn new(value: Arc<NodeCacheEntryImpl>) -> Self {
        NodeCacheEntry { value }
    }
}

impl Deref for NodeCacheEntry {
    type Target = NodeCacheEntryImpl;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

/// A thread-safe cache for nodes, which uses a [`Storage`] backend to persist and retrieve entries.
/// It provides the following set of guarantees:
/// - Entries ownership is hold by the cache
/// - Returned entries are always contained in the cache
/// - A returned entry will never be evicted as long as a reference to it exists outside of the cache
pub struct NodeCache<S: Storage> {
    // Read lock for concurrent access to the cache elements.
    // Write lock for flushing to ensure no other accesses.
    #[allow(clippy::type_complexity)]
    cache: RwLock<
        quick_cache::sync::Cache<
            NodeId,
            Arc<NodeCacheEntryImpl>,
            UnitWeighter,
            RandomState,
            NodeCacheLifecycle<S>,
        >,
    >,
    storage: Arc<S>,
}

impl<S: Storage<Id = NodeId, Item = Arc<NodeCacheEntryImpl>>> NodeCache<S> {
    /// Creates a new [`NodeCache`] with the given [`Storage`] backend and estimated capacity.
    /// An error state is also provided to store any errors that may occur during cache operations.
    pub fn try_new(
        storage: S,
        estimated_capacity: usize,
        error: Arc<ErrorState>,
    ) -> Result<Self, Error> {
        let storage = Arc::new(storage);
        let options = quick_cache::OptionsBuilder::new()
            .estimated_items_capacity(estimated_capacity)
            .weight_capacity(estimated_capacity as u64) // Using a weight capacity of 1 per item
            .build()
            .map_err(|e| Error::Cache(e.to_string()))?;
        Ok(Self {
            cache: RwLock::new(quick_cache::sync::Cache::with_options(
                options,
                UnitWeighter,
                RandomState::default(),
                NodeCacheLifecycle {
                    storage: storage.clone(),
                    error,
                },
            )),
            storage,
        })
    }
}

impl<S> Cache for NodeCache<S>
where
    S: Storage<Id = NodeId, Item = Arc<NodeCacheEntryImpl>>,
{
    type Id = NodeId;
    type StoredItem = NodeCacheEntry;
    type ItemPayload = Node;

    /// Retrieves a [`NodeCacheEntry`] from the cache with the specified ID.
    /// If the entry is not in the cache, it's retrieved from the storage.
    fn get(&self, id: Self::Id) -> Result<Self::StoredItem, Error> {
        let cache = self.cache.read().unwrap();
        let entry = cache.get(&id);
        // NOTE: Our formatting options make the suggested `if let` statement very ugly IMO.
        #[allow(clippy::single_match_else)]
        let entry = match entry {
            Some(entry) => entry,
            None => {
                // Retrieve from storage if not in cache
                let entry = self.storage.get(id)?;
                cache.insert(id, entry.clone()); // NOTE: cloning the Arc here to avoid eviction
                entry
            }
        };

        Ok(NodeCacheEntry::new(entry))
    }

    /// Stores the value in the cache and reserves an ID for it.
    fn set(&self, node: Node) -> Result<Self::Id, Error> {
        let entry = Arc::new(NodeCacheEntryImpl::new(node));
        let id = self.storage.reserve(&entry);
        {
            let mut entry_guard = entry.write().unwrap();
            entry_guard.dirty = true; // Dirty as it's a new entry
        }
        self.cache.read().unwrap().insert(id, entry);
        Ok(id)
    }

    /// Flushes all dirty entries to the [`Storage`].
    fn flush(&self) -> Result<(), Error> {
        for (id, entry) in self.cache.write().unwrap().iter() {
            let entry_guard = entry.read().unwrap();
            if entry_guard.dirty {
                drop(entry_guard);
                self.storage.set(id, &entry)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use mockall::{
        Sequence,
        predicate::{always, eq},
    };

    use super::*;
    use crate::{
        storage::{self, MockStorage},
        types::NodeType,
    };

    #[test]
    fn node_cache_lifecycle_on_evict_saves_dirty_entries() {
        // Dirty element is saved
        {
            let mut storage = MockStorage::new();
            let entry = Arc::new(NodeCacheEntryImpl::new(Node::Empty));
            let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            entry.write().unwrap().dirty = true; // Mark as dirty
            storage
                .expect_set()
                .times(1)
                .with(eq(id), eq(entry.clone()))
                .returning(|_, _| Ok(()));
            let lifecycle = NodeCacheLifecycle {
                storage: Arc::new(storage),
                error: Arc::new(ErrorState::default()),
            };

            lifecycle.on_evict(&mut (), id, entry);
        }
        // Clean element is not saved
        {
            let mut storage = MockStorage::new();
            let entry = Arc::new(NodeCacheEntryImpl::new(Node::Empty));
            let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            entry.write().unwrap().dirty = false; // Mark as clean
            // Expect no call to storage set
            storage.expect_set().never().with(always(), always());
            let lifecycle = NodeCacheLifecycle {
                storage: Arc::new(storage),
                error: Arc::new(ErrorState::default()),
            };

            lifecycle.on_evict(&mut (), id, entry);
        }
    }

    #[test]
    fn node_cache_lifecycle_on_evict_fails_on_storage_error_and_stores_error() {
        let mut storage = MockStorage::new();
        let entry = Arc::new(NodeCacheEntryImpl::new(Node::Empty));
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        drop(entry.write().unwrap()); // Trigger storage set
        storage
            .expect_set()
            .times(1)
            .with(eq(id), eq(entry.clone()))
            .returning(|_, _| Err(storage::Error::NotFound));
        let error_state = Arc::new(ErrorState::default());
        let lifecycle = NodeCacheLifecycle {
            storage: Arc::new(storage),
            error: error_state.clone(),
        };

        lifecycle.on_evict(&mut (), id, entry);

        let error_state = error_state.get();
        assert!(error_state.is_some());
        assert!(matches!(
            error_state.as_ref(),
            Some(Error::Storage(storage::Error::NotFound))
        ));
    }

    #[test]
    fn node_cache_lifecycle_is_pinned_checks_strong_count() {
        let lifecycle = NodeCacheLifecycle {
            storage: Arc::new(MockStorage::new()),
            error: Arc::new(ErrorState::default()),
        };
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let entry = Arc::new(NodeCacheEntryImpl::new(Node::Empty));
        // Strong count == 1 (not pinned)
        {
            assert!(!lifecycle.is_pinned(&id, &entry));
        }
        // Strong count > 1 (pinned)
        {
            let entry_clone = entry.clone();
            assert!(lifecycle.is_pinned(&id, &entry_clone));
        }
    }

    #[test]
    fn inner_node_cache_entry_write_sets_dirty_flag() {
        let entry = NodeCacheEntryImpl::new(Node::Empty);
        let guard = entry.write().unwrap();
        assert!(guard.dirty);
    }

    #[test]
    fn node_cache_try_new_creates_node_cache() {
        let storage = MockStorage::new();
        let cache = NodeCache::try_new(storage, 10, Arc::new(ErrorState::default()));
        assert!(cache.is_ok());
        let cache = cache.unwrap();
        assert_eq!(cache.cache.read().unwrap().capacity(), 10);
    }

    #[test]
    fn node_cache_get_returns_existing_entry_from_storage_if_not_in_cache() {
        let expected_entry = Arc::new(NodeCacheEntryImpl::new(Node::Empty));
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let mut storage = MockStorage::new();
        storage.expect_get().with(eq(id)).returning({
            let expected_entry = expected_entry.clone();
            move |_| Ok(expected_entry.clone())
        });

        let cache = NodeCache::try_new(storage, 10, Arc::new(ErrorState::default())).unwrap();
        let entry = cache.get(id).unwrap();
        assert_eq!(*expected_entry, *entry);
    }

    #[test]
    fn node_cache_get_returns_error_if_node_id_does_not_exist() {
        let mut storage = MockStorage::new();
        storage
            .expect_get()
            .returning(|_| Err(storage::Error::NotFound));

        let cache = NodeCache::try_new(storage, 10, Arc::new(ErrorState::default())).unwrap();
        let res = cache
            .get(NodeId::from_idx_and_node_type(0, NodeType::Empty))
            .expect_err("get should fail");
        assert!(matches!(res, Error::Storage(storage::Error::NotFound)));
    }

    #[test]
    fn node_cache_get_always_insert_element_in_cache() {
        const NUM_ELEMENTS: u64 = 10;
        let mut storage = MockStorage::new();
        let mut sequence = Sequence::new();
        for i in 0..NUM_ELEMENTS + 1 {
            // 1 element more than capacity
            storage
                .expect_get()
                .times(1)
                .in_sequence(&mut sequence)
                .with(eq(NodeId::from_idx_and_node_type(i, NodeType::Empty)))
                .returning(move |_| Ok(Arc::new(NodeCacheEntryImpl::new(Node::Empty))));
        }
        storage
            .expect_set()
            .times(1)
            .with(
                eq(NodeId::from_idx_and_node_type(
                    // Last element - 1 will be evicted because of infinite reuse distance
                    NUM_ELEMENTS - 1,
                    NodeType::Empty,
                )),
                always(),
            )
            .returning(|_, _| Ok(()));

        let cache = NodeCache::try_new(
            storage,
            NUM_ELEMENTS as usize,
            Arc::new(ErrorState::default()),
        )
        .expect("cache should be created");

        for i in 0..NUM_ELEMENTS {
            let id = NodeId::from_idx_and_node_type(i, NodeType::Empty);
            let entry = cache.get(id).expect("get should succeed");
            let entry_guard = entry.write().unwrap(); // Write lock to trigger storage set on eviction
            assert_eq!(entry_guard.value, Node::Empty);
            assert!(entry_guard.dirty); // Should be dirty after insertion
        }

        // Insert one element more than capacity, triggering eviction of the precedent element.
        let id = NodeId::from_idx_and_node_type(NUM_ELEMENTS, NodeType::Empty);
        let _ = cache.get(id).expect("get should succeed");
    }

    #[test]
    fn node_cache_set_inserts_elements_in_cache() {
        let mut storage = MockStorage::new();
        storage
            .expect_reserve()
            .returning(|_| NodeId::from_idx_and_node_type(42, NodeType::Empty));
        let cache = NodeCache::try_new(storage, 10, Arc::new(ErrorState::default())).unwrap();
        let node = Node::Empty;
        let id = cache.set(node).expect("set should succeed");
        assert_eq!(id, NodeId::from_idx_and_node_type(42, NodeType::Empty));
        let entry = cache.get(id).expect("get should succeed");
        let entry_guard = entry.read().unwrap();
        assert_eq!(entry_guard.value, Node::Empty);
        assert!(entry_guard.dirty);
    }

    #[test]
    fn node_cache_flush_saves_dirty_entries_to_storage() {
        const NUM_ELEMENTS: u64 = 10;
        let data = Arc::new(Mutex::new(vec![]));
        let mut storage = MockStorage::new();
        let mut sequence = Sequence::new();
        for i in 0..NUM_ELEMENTS {
            storage
                .expect_reserve()
                .times(1)
                .in_sequence(&mut sequence)
                .returning({
                    let data = data.clone();
                    move |node| {
                        data.lock().unwrap().push(node.clone());
                        NodeId::from_idx_and_node_type(i, NodeType::Empty)
                    }
                });
            storage
                .expect_set()
                .times(1)
                .with(
                    eq(NodeId::from_idx_and_node_type(i, NodeType::Empty)),
                    eq(Arc::new(NodeCacheEntryImpl::new(Node::Empty))),
                )
                .returning(move |_, value| {
                    value.write().unwrap().dirty = false; // Mark as clean after saving
                    Ok(())
                });
        }

        let cache = NodeCache::try_new(
            storage,
            NUM_ELEMENTS as usize,
            Arc::new(ErrorState::default()),
        )
        .expect("cache should be created");

        for i in 0..NUM_ELEMENTS {
            let node = Node::Empty;
            let id = cache.set(node).expect("set should succeed");
            assert_eq!(id, NodeId::from_idx_and_node_type(i, NodeType::Empty));
            let entry = cache.get(id).expect("get should succeed");
            let entry_guard = entry.read().unwrap();
            assert_eq!(entry_guard.value, Node::Empty);
            assert!(entry_guard.dirty);
        }

        let result = cache.flush();
        assert!(result.is_ok());

        // Check all entries are clean after flush, therefore all the expectations have been fulfilled
        assert_eq!(data.lock().unwrap().len(), NUM_ELEMENTS as usize);
        for entry in data.lock().unwrap().iter() {
            let entry_guard = entry.read().unwrap();
            assert!(!entry_guard.dirty, "Entry should be clean after flush");
        }

        // calling flush again should not access the storage
        let result = cache.flush();
        assert!(result.is_ok());
    }

    #[test]
    fn node_cache_stores_data_in_storage_on_evict() {
        const NUM_ELEMENTS: u64 = 1;
        let mut storage = MockStorage::new();
        let mut sequence = Sequence::new();
        storage
            .expect_reserve()
            .times(1)
            .in_sequence(&mut sequence)
            .returning(move |_| NodeId::from_idx_and_node_type(0, NodeType::Empty));

        // With unit-size cache, each element is immediately evicted
        storage
            .expect_set()
            .times(1)
            .with(
                eq(NodeId::from_idx_and_node_type(0, NodeType::Empty)),
                eq(Arc::new(NodeCacheEntryImpl::new(Node::Empty))),
            )
            .returning(|_, value| {
                value.write().unwrap().dirty = false; // Mark as clean after saving
                Ok(())
            });

        let cache = NodeCache::try_new(
            storage,
            NUM_ELEMENTS as usize,
            Arc::new(ErrorState::default()),
        )
        .expect("cache should be created");
        for _ in 0..NUM_ELEMENTS {
            let node = Node::Empty;
            let _ = cache.set(node).expect("set should succeed");
        }
    }

    impl PartialEq for NodeCacheEntryImpl {
        fn eq(&self, other: &Self) -> bool {
            let self_guard = self.value.read().unwrap();
            let other_guard = other.value.read().unwrap();
            self_guard.value == other_guard.value
        }
    }
}
