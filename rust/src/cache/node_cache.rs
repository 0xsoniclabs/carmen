use std::{
    hash::RandomState,
    ops::Deref,
    sync::{Arc, RwLock},
};

use quick_cache::{Lifecycle, UnitWeighter};

#[cfg(test)]
use crate::cache::DeleteStatusMsg;
use crate::{
    cache::Cache,
    error::{Error, ErrorState},
    storage::Storage,
    types::{CacheEntry, CacheEntryImpl, CachedNode, Node, NodeId},
};

/// A thread-safe cache for nodes, which uses a [`Storage`] backend to persist and retrieve entries.
/// It provides the following set of guarantees:
/// - Entries ownership is hold by the cache
/// - Returned entries are always contained in the cache
/// - An entry will never be evicted as long as a reference to it exists outside of the cache
#[derive(Clone)]
pub struct NodeCache<S: Storage> {
    inner: Arc<NodeCacheImpl<S>>,
}

impl<S> NodeCache<S>
where
    S: Storage<Id = NodeId, Item = CacheEntryImpl>,
{
    /// Creates a new [`NodeCacheImpl`] with the given [`Storage`] backend and estimated capacity.
    /// An error state is also provided to store any errors that may occur during cache operations.
    pub fn try_new(
        storage: S,
        estimated_capacity: usize,
        error: Arc<ErrorState>,
    ) -> Result<Self, Error> {
        Ok(NodeCache {
            inner: Arc::new(NodeCacheImpl::try_new(storage, estimated_capacity, error)?),
        })
    }
}

impl<S: Storage> Deref for NodeCache<S> {
    type Target = NodeCacheImpl<S>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// The internal implementation of the [`NodeCache`], which implements the [`Cache`] trait.
/// It uses [`quick_cache::sync::Cache`] as the cache backend and a [`Storage`] object to retrieve
/// and persist entries.
pub struct NodeCacheImpl<S: Storage> {
    // Read lock for concurrent access to the cache elements.
    // Write lock for flushing to ensure no other accesses.
    #[allow(clippy::type_complexity)]
    cache: RwLock<
        quick_cache::sync::Cache<
            NodeId,                // The cache key type
            CacheEntryImpl,        // The type of the item stored in the cache
            UnitWeighter,          // Each element is treated the same on eviction
            RandomState,           // Default hasher
            NodeCacheLifecycle<S>, // Lifecycle to handle eviction and storage
        >,
    >,
    storage: Arc<S>, // The storage backend to persist entries
}

impl<S: Storage<Id = NodeId, Item = CacheEntryImpl>> NodeCacheImpl<S> {
    /// Initializes a new [`NodeCacheImpl`] by creating a [`quick_cache::sync::Cache`] with the
    /// given estimated capacity, a [`UnitWeighter`] for eviction, and a [`NodeCacheLifecycle`]
    /// to handle eviction and storage operations. It also stores a reference to the provided
    /// [`Storage`] backend and an [`ErrorState`] to register any errors that may occur during cache
    /// operations.
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
                NodeCacheLifecycle::new(storage.clone(), error),
            )),
            storage,
        })
    }
}

impl<S> Cache for NodeCacheImpl<S>
where
    S: Storage<Id = NodeId, Item = CacheEntryImpl>,
{
    type Key = NodeId;
    type Item = CacheEntry;
    type ItemPayload = Node;

    /// Retrieves a [`CacheEntryImpl`] from the cache with the specified ID.
    /// If the entry is not in the cache, it's retrieved from the underlying storage.
    fn get(&self, id: Self::Key) -> Result<Self::Item, Error> {
        let cache = self.cache.read().unwrap();
        let entry = cache.get(&id);
        // NOTE: Our rustfmt conf make the suggested `if let` statement very ugly IMO.
        #[allow(clippy::single_match_else)]
        let entry = match entry {
            Some(entry) => entry,
            None => {
                // Retrieve from storage if not in cache
                let entry = self.storage.get(id)?;
                cache.insert(id, entry.clone()); // NOTE: cloning the Arc to avoid eviction
                entry
            }
        };

        Ok(CacheEntry(entry))
    }

    /// Stores the node in the cache and reserves a [`NodeId`] for it.
    fn set(&self, node: Node) -> Result<Self::Key, Error> {
        let entry = CacheEntryImpl::new(RwLock::new(CachedNode::new_dirty(node)));
        let id = self.storage.reserve(&entry);
        self.cache.read().unwrap().insert(id, entry);
        Ok(id)
    }

    /// Flushes all dirty entries to the internal [`Storage`].
    fn flush(&self) -> Result<(), Error> {
        for (id, entry) in self.cache.write().unwrap().iter() {
            let entry_guard = entry.read().unwrap();
            if entry_guard.dirty() {
                drop(entry_guard);
                self.storage.set(id, &entry)?;
            }
        }
        Ok(())
    }

    /// Deletes the entry with the given ID from the cache and storage.
    /// This method waits until there are no other references to the entry before removing it from
    /// the cache.
    /// While this operation is in progress, the cache cannot be accessed to ensure the same ID is
    /// queried
    fn delete(
        &self,
        id: Self::Key,
        #[cfg(test)] _test_notify: Option<std::sync::mpsc::Sender<DeleteStatusMsg>>,
    ) -> Result<(), Error> {
        // We acquire a write lock to ensure no get operation on the same ID happens while deleting.
        // Clippy complains we use the lock only for reading, which is expected in this case.
        #[allow(clippy::readonly_write_lock)]
        let cache = self.cache.write().unwrap();
        if cache.contains_key(&id) {
            // Safe to unwrap as we just checked the key exists and we hold a write lock on the
            // cache
            let item = cache.get(&id).unwrap();
            // Wait until there is no other reference to the item
            #[cfg(test)]
            let mut sent = false;
            while Arc::strong_count(&item) > 2 {
                #[cfg(test)]
                if !sent {
                    if let Some(_test_notify) = &_test_notify {
                        _test_notify.send(DeleteStatusMsg::Waiting).unwrap();
                    }
                    sent = true;
                }
                // Two references: cache and local variable
                std::thread::yield_now();
            }
            // Now no one holds a reference to the item
            cache.remove(&id);
            #[cfg(test)]
            if let Some(_test_notify) = _test_notify {
                _test_notify.send(DeleteStatusMsg::CacheDeleted).unwrap();
            }
        }
        self.storage.delete(id)?;
        Ok(())
    }
}

/// A [`quick_cache::Lifecycle`] implementation for the [`NodeCacheImpl`].
/// It handles two tasks:
/// - Storing entries in the storage backend when they are evicted from the cache if dirty.
/// - Preventing eviction of entries that are currently in use outside of the cache.
///
/// It stores a reference to an [`ErrorState`] to register any errors that occur during storage
/// operations as [`quick_cache::Lifecycle`] methods do not return.
struct NodeCacheLifecycle<S: Storage> {
    storage: Arc<S>,
    error: Arc<ErrorState>,
}

impl<S: Storage> NodeCacheLifecycle<S> {
    /// Creates a new [`NodeCacheLifecycle`] with the given storage backend and error state
    /// reference.
    pub fn new(storage: Arc<S>, error: Arc<ErrorState>) -> Self {
        NodeCacheLifecycle { storage, error }
    }
}

impl<S: Storage> Clone for NodeCacheLifecycle<S> {
    fn clone(&self) -> Self {
        NodeCacheLifecycle {
            storage: self.storage.clone(),
            error: self.error.clone(),
        }
    }
}

impl<S: Storage<Id = NodeId, Item = CacheEntryImpl>> Lifecycle<NodeId, CacheEntryImpl>
    for NodeCacheLifecycle<S>
{
    /// No-op for beginning a request, as we don't need to store any state.
    fn begin_request(&self) -> Self::RequestState {}

    /// Stores the entry in the storage when if it's dirty.
    /// In case of an error, it registers it in the [`ErrorState`].
    fn on_evict(&self, _state: &mut Self::RequestState, key: NodeId, val: CacheEntryImpl) {
        if !val.read().unwrap().dirty() {
            return;
        }
        self.storage.set(key, &val).unwrap_or_else(|e| {
            self.error.register(Error::Storage(e));
        });
    }

    /// Checks if the entry is pinned and cannot be evicted.
    /// An entry is pinned if another thread holds a reference to it (strong count > 1)
    fn is_pinned(&self, _key: &NodeId, val: &CacheEntryImpl) -> bool {
        Arc::strong_count(val) > 1
    }

    /// Type used to store state during a request.
    /// We don't store any state, set as unit type.
    type RequestState = ();
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
    fn node_cache_try_new_creates_node_cache() {
        let cache = NodeCache::try_new(MockStorage::new(), 10, Arc::new(ErrorState::default()));
        assert!(cache.is_ok());
        let cache = cache.unwrap();
        assert_eq!(cache.deref().cache.read().unwrap().capacity(), 10);
    }

    #[test]
    fn node_cache_impl_try_new_creates_node_cache() {
        let storage = MockStorage::new();
        let cache = NodeCacheImpl::try_new(storage, 10, Arc::new(ErrorState::default()));
        assert!(cache.is_ok());
        let cache = cache.unwrap();
        assert_eq!(cache.cache.read().unwrap().capacity(), 10);
    }

    #[test]
    fn node_cache_impl_get_returns_existing_entry_from_storage_if_not_in_cache() {
        let expected_entry = CacheEntryImpl::new(RwLock::new(CachedNode::new_clean(Node::Empty)));
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let mut storage = MockStorage::new();
        storage.expect_get().with(eq(id)).returning({
            let expected_entry = expected_entry.clone();
            move |_| Ok(expected_entry.clone())
        });

        let cache = NodeCacheImpl::try_new(storage, 10, Arc::new(ErrorState::default())).unwrap();
        let entry = cache.get(id).unwrap();
        assert!(are_cache_entry_equals(&entry, &expected_entry));
    }

    #[test]
    fn node_cache_impl_get_returns_error_if_node_id_does_not_exist() {
        let mut storage = MockStorage::new();
        storage
            .expect_get()
            .returning(|_| Err(storage::Error::NotFound));

        let cache = NodeCacheImpl::try_new(storage, 10, Arc::new(ErrorState::default())).unwrap();
        let res = cache
            .get(NodeId::from_idx_and_node_type(0, NodeType::Empty))
            .expect_err("get should fail");
        assert!(matches!(res, Error::Storage(storage::Error::NotFound)));
    }

    #[test]
    fn node_cache_impl_get_always_insert_element_in_cache() {
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
                .returning(move |_| {
                    Ok(CacheEntryImpl::new(RwLock::new(CachedNode::new_dirty(
                        Node::Empty,
                    ))))
                });
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
            .returning(|_, _| Ok(())); //

        let cache = NodeCacheImpl::try_new(
            storage,
            NUM_ELEMENTS as usize,
            Arc::new(ErrorState::default()),
        )
        .expect("cache should be created");

        for i in 0..NUM_ELEMENTS {
            let id = NodeId::from_idx_and_node_type(i, NodeType::Empty);
            let entry = cache.get(id).expect("get should succeed");
            drop(entry.write().unwrap()); // Write lock to trigger storage set on eviction
            assert!(are_cache_entry_equals(
                &entry,
                &CacheEntryImpl::new(RwLock::new(CachedNode::new_dirty(Node::Empty)))
            ));
        }

        // Insert one element more than capacity, triggering eviction of the precedent element.
        let id = NodeId::from_idx_and_node_type(NUM_ELEMENTS, NodeType::Empty);
        let _ = cache.get(id).expect("get should succeed");
    }

    #[test]
    fn node_cache_impl_set_inserts_elements_in_cache() {
        let mut storage = MockStorage::new();
        storage
            .expect_reserve()
            .returning(|_| NodeId::from_idx_and_node_type(42, NodeType::Empty));
        storage.expect_get().never(); // Cache get should not call storage get
        let cache = NodeCacheImpl::try_new(storage, 10, Arc::new(ErrorState::default())).unwrap();
        let node = Node::Empty;
        let id = cache.set(node).expect("set should succeed");
        assert_eq!(id, NodeId::from_idx_and_node_type(42, NodeType::Empty));
        let entry = cache.get(id).expect("get should succeed");
        assert!(are_cache_entry_equals(
            &entry,
            &CacheEntryImpl::new(RwLock::new(CachedNode::new_dirty(Node::Empty)))
        ));
    }

    #[test]
    fn node_cache_impl_flush_saves_dirty_entries_to_storage() {
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
                .withf(move |idx, value| {
                    *idx == NodeId::from_idx_and_node_type(i, NodeType::Empty)
                        && are_cache_entry_equals(
                            value,
                            &CacheEntryImpl::new(RwLock::new(CachedNode::new_dirty(Node::Empty))),
                        )
                })
                .returning(move |_, _| Ok(()));
        }

        let cache = NodeCacheImpl::try_new(
            storage,
            NUM_ELEMENTS as usize,
            Arc::new(ErrorState::default()),
        )
        .expect("cache should be created");

        for _ in 0..NUM_ELEMENTS {
            let _ = cache.set(Node::Empty).expect("set should succeed");
        }

        cache.flush().expect("flush should succeed");
    }

    #[test]
    fn node_cache_impl_stores_data_in_storage_on_evict() {
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
            .withf(move |idx, value| {
                *idx == NodeId::from_idx_and_node_type(0, NodeType::Empty)
                    && are_cache_entry_equals(
                        value,
                        &CacheEntryImpl::new(RwLock::new(CachedNode::new_dirty(Node::Empty))),
                    )
            })
            .returning(|_, _| Ok(()));

        let cache = NodeCacheImpl::try_new(storage, 1, Arc::new(ErrorState::default()))
            .expect("cache should be created");
        // Insert an element, which will be immediately evicted and stored
        let _ = cache.set(Node::Empty).expect("set should succeed");
    }

    #[test]
    fn node_cache_impl_delete_removes_entry_from_cache_and_storage() {
        let mut storage = MockStorage::new();
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let entry = CacheEntryImpl::new(RwLock::new(CachedNode::new_clean(Node::Empty)));
        storage
            .expect_delete()
            .times(1)
            .with(eq(id))
            .returning(|_| Ok(()));

        let cache = NodeCacheImpl::try_new(storage, 2, Arc::new(ErrorState::default())).unwrap();
        cache.cache.write().unwrap().insert(id, entry);

        cache.delete(id, None).expect("delete should succeed");
        assert!(cache.cache.read().unwrap().get(&id).is_none());
    }

    #[test]
    fn node_cache_impl_delete_waits_for_no_references_to_entry() {
        let mut storage = MockStorage::new();
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let entry = CacheEntryImpl::new(RwLock::new(CachedNode::new_clean(Node::Empty)));
        storage
            .expect_delete()
            .times(1)
            .with(eq(id))
            .returning(|_| Ok(()));

        let cache = NodeCacheImpl::try_new(storage, 2, Arc::new(ErrorState::default())).unwrap();
        cache.cache.write().unwrap().insert(id, entry.clone()); //Insert a clone to hold a local reference

        // Wraps to the cache into an Arc to share it across threads
        let cache_arc = Arc::new(cache);
        let (tx, rx) = std::sync::mpsc::channel::<DeleteStatusMsg>();
        let delete_thread = {
            let cache_arc = cache_arc.clone();
            std::thread::spawn(move || {
                cache_arc
                    .delete(id, Some(tx))
                    .expect("delete should succeed");
            })
        };

        let notify_res = rx.recv().expect("should receive waiting message");
        assert!(matches!(notify_res, DeleteStatusMsg::Waiting));
        // Drop reference to entry to allow deletion
        drop(entry);
        let notify_res = rx.recv().expect("should receive cache deleted message");
        assert!(matches!(notify_res, DeleteStatusMsg::CacheDeleted));
        delete_thread.join().expect("delete thread should finish");
        assert!(cache_arc.cache.read().unwrap().get(&id).is_none());
    }

    #[test]
    fn node_cache_impl_delete_fails_on_storage_error() {
        let mut storage = MockStorage::new();
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        storage
            .expect_delete()
            .times(1)
            .with(eq(id))
            .returning(|_| Err(storage::Error::NotFound));

        let cache = NodeCacheImpl::try_new(storage, 2, Arc::new(ErrorState::default())).unwrap();
        cache.cache.write().unwrap().insert(
            id,
            CacheEntryImpl::new(RwLock::new(CachedNode::new_clean(Node::Empty))),
        );

        let res = cache.delete(id, None);
        assert!(res.is_err());
        assert!(matches!(
            res.unwrap_err(),
            Error::Storage(storage::Error::NotFound)
        ));
    }
    #[test]
    fn node_cache_lifecycle_on_evict_stores_dirty_entries() {
        // Dirty element is stored
        {
            let mut storage = MockStorage::new();
            let entry = CacheEntryImpl::new(RwLock::new(CachedNode::new_dirty(Node::Empty)));
            let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            storage
                .expect_set()
                .times(1)
                .withf({
                    let entry = entry.clone();
                    move |i, e| *i == id && are_cache_entry_equals(e, &entry)
                })
                .returning(|_, _| Ok(()));
            let lifecycle = NodeCacheLifecycle {
                storage: Arc::new(storage),
                error: Arc::new(ErrorState::default()),
            };

            lifecycle.on_evict(&mut (), id, entry);
        }
        // Clean element is not stored
        {
            let mut storage = MockStorage::new();
            let entry = CacheEntryImpl::new(RwLock::new(CachedNode::new_clean(Node::Empty)));
            let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            // Expect no call to storage set
            storage.expect_set().never();
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
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let entry = CacheEntryImpl::new(RwLock::new(CachedNode::new_dirty(Node::Empty)));
        storage
            .expect_set()
            .times(1)
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
    fn node_cache_lifecycle_is_pinned_checks_entry_number_of_references() {
        let lifecycle = NodeCacheLifecycle {
            storage: Arc::new(MockStorage::new()),
            error: Arc::new(ErrorState::default()),
        };
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let entry = CacheEntryImpl::new(RwLock::new(CachedNode::new_clean(Node::Empty)));
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

    /// Helper function to compare two `CacheEntryImpl` instances for equality.
    fn are_cache_entry_equals(entry1: &CacheEntryImpl, entry2: &CacheEntryImpl) -> bool {
        let guard1 = entry1.read().unwrap();
        let guard2 = entry2.read().unwrap();
        *guard1 == *guard2
    }
}
