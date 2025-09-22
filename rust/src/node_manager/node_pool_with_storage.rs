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
    hash::RandomState,
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock},
};

use quick_cache::{Lifecycle, UnitWeighter};

use crate::{
    error::Error,
    pool::{NodeManager, PoolItem},
    storage::Storage,
    types::{Node, NodeId},
};

/// A [`Node`] with additional metadata about the node lifecycle.
/// [`NodeWithMetadata`] automatically dereferences to `Node` via the [`Deref`] trait.
/// The node's status is set to [`NodeStatus::Dirty`] when a mutable reference is requested.
/// Accessing a deleted node will panic.
#[derive(Debug, PartialEq, Eq)]
pub struct NodeWithMetadata {
    value: Node,
    status: NodeStatus,
}

/// The status of a [`NodeWithMetadata`].
/// It can be:
/// - `Clean`: the node is in sync with the storage
/// - `Dirty`: the node has been modified and needs to be flushed to storage
/// - `Deleted`: the node has been deleted and should not be used anymore
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeStatus {
    Clean,
    Dirty,
    Deleted,
}

impl NodeWithMetadata {
    /// Creates a new [`NodeWithMetadata`].
    pub fn new(value: Node) -> Self {
        NodeWithMetadata {
            value,
            status: NodeStatus::Clean,
        }
    }
}

impl Deref for NodeWithMetadata {
    type Target = Node;

    fn deref(&self) -> &Self::Target {
        if self.status == NodeStatus::Deleted {
            panic!("Attempted to access a deleted node");
        }
        &self.value
    }
}

impl DerefMut for NodeWithMetadata {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.status == NodeStatus::Deleted {
            panic!("Attempted to access a deleted node");
        }
        self.status = NodeStatus::Dirty; // Mark as dirty on mutable borrow
        &mut self.value
    }
}
/// A thread-safe [`Pool`], which uses a [`Storage`] backend to persist and retrieve entries.
/// Elements are cached in an internal in-memory to provide fast access to frequently used nodes.
/// It provides the following set of guarantees:
/// - Entries ownership is hold by the node pool
/// - Returned entries are always contained in the pool
/// - An entry will never be evicted as long as a reference to it exists outside of the pool, unless
///   it's explicitly deleted
#[derive(Clone)]
pub struct NodePoolWithStorage<S: Storage> {
    inner: Arc<NodePoolWithStorageImpl<S>>,
}

impl<S> NodePoolWithStorage<S>
where
    S: Storage<Id = NodeId, Item = Node>,
{
    /// Creates a new [`NodePoolWithStorageImpl`] with the given [`Storage`] backend and estimated
    /// capacity.
    /// An [`ErrorState`] is also provided to store any errors that may occur during
    /// cache operations.
    pub fn try_new(storage: S, estimated_capacity: usize) -> Result<Self, Error> {
        Ok(NodePoolWithStorage {
            inner: Arc::new(NodePoolWithStorageImpl::try_new(
                storage,
                estimated_capacity,
            )?),
        })
    }
}

impl<S: Storage> Deref for NodePoolWithStorage<S> {
    type Target = NodePoolWithStorageImpl<S>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// The internal implementation of the [`NodePoolWithStorage`], which implements the [`Pool`]
/// trait.
/// It internally holds a [`quick_cache::sync::Cache`] as an in-memory cache, and a reference to the
/// provided [`Storage`] backend to persist entries.
pub struct NodePoolWithStorageImpl<S: Storage> {
    cache: quick_cache::sync::Cache<
        NodeId,                          // The cache key type
        Arc<RwLock<NodeWithMetadata>>,   // The type of the item stored in the cache
        UnitWeighter,                    // Each element is treated the same on eviction
        RandomState,                     // Default hasher
        NodePoolWithStorageLifecycle<S>, // Lifecycle to handle eviction and storage
    >,
    storage: Arc<S>, // The storage backend to persist entries
}

impl<S: Storage<Id = NodeId, Item = Node>> NodePoolWithStorageImpl<S> {
    /// Initializes a new [`NodePoolWithStorageImpl`] by creating a [`quick_cache::sync::Cache`]
    /// with the given estimated capacity, a [`UnitWeighter`] for eviction, and a
    /// [`NodePoolWithStorageLifecycle`] to handle eviction and storage operations. It also stores a
    /// reference to the provided [`Storage`] backend and an [`ErrorState`] to register any
    /// errors that may occur during cache operations.
    pub fn try_new(storage: S, estimated_capacity: usize) -> Result<Self, Error> {
        let storage = Arc::new(storage);
        let options = quick_cache::OptionsBuilder::new()
            .estimated_items_capacity(estimated_capacity)
            .weight_capacity(estimated_capacity as u64) // Using a weight capacity of 1 per item
            .build()
            .map_err(|e| Error::Cache(e.to_string()))?;
        Ok(Self {
            cache: quick_cache::sync::Cache::with_options(
                options,
                UnitWeighter,
                RandomState::default(),
                NodePoolWithStorageLifecycle::new(storage.clone()),
            ),
            storage,
        })
    }
}

impl<S> NodeManager for NodePoolWithStorageImpl<S>
where
    S: Storage<Id = NodeId, Item = Node>,
{
    type Id = NodeId;
    type NodeType = Node;

    /// Retrieves an entry from the pool with the specified ID.
    /// If the entry is not in the pool, it's retrieved from the underlying storage.
    fn get(
        &self,
        id: Self::Id,
    ) -> Result<PoolItem<impl DerefMut<Target = Self::NodeType> + Send + Sync + 'static>, Error>
    {
        let entry = self.cache.get(&id);
        // NOTE: Our rustfmt conf make the suggested `if let` statement very ugly IMO.
        #[allow(clippy::single_match_else)]
        let entry = match entry {
            Some(entry) => entry,
            None => {
                // Retrieve from storage if not in cache
                let entry = Arc::new(RwLock::new(NodeWithMetadata::new(
                    self.storage.get(id)?,
                    NodeStatus::Clean,
                )));
                self.cache.insert(id, entry.clone()); // NOTE: cloning the Arc to avoid eviction
                entry
            }
        };

        Ok(PoolItem::new(entry))
    }

    /// Adds the node in the pool and reserves a [`NodeId`] for it.
    fn add(&self, node: Self::NodeType) -> Result<Self::Id, Error> {
        let id = self.storage.reserve(&node);
        let entry = Arc::new(RwLock::new(NodeWithMetadata::new(node, NodeStatus::Dirty)));
        self.cache.insert(id, entry);
        Ok(id)
    }

    /// Flushes all dirty entries to the internal [`Storage`].
    /// The flushed status is everything that is in the pool between the start and the end of the
    /// flush.
    ///  There are three possible scenarios that may happen during flushing:
    /// - Deleted entry: it will not appear in the flush. This is not an issue as it has been
    ///   deleted anyway.
    /// - Evicted entry: entries are stored on eviction, therefore it has already been flushed
    /// - New elements added to the pool while flushing: those elements may or may not be flushed,
    ///   but they will be flushed in the next flush call
    fn flush(&self) -> Result<(), Error> {
        for (id, entry) in self.cache.iter() {
            let mut entry_guard = entry.write().unwrap();
            if entry_guard.status == NodeStatus::Dirty {
                self.storage.set(id, &entry_guard.value)?;
                entry_guard.status = NodeStatus::Clean;
            }
        }
        self.storage.flush()?;
        Ok(())
    }

    /// Deletes the entry with the given ID from the pool and storage.
    /// NOTE: deleting an entry does not guarantee that no references to it exist outside of the
    /// pool. Instead, the entry status is set to [`NodeStatus::Deleted`] to prevent further usage.
    fn delete(&self, id: Self::Id) -> Result<(), Error> {
        if self.cache.contains_key(&id) {
            let item = self.cache.get(&id).unwrap();
            item.write().unwrap().status = NodeStatus::Deleted;
            self.cache.remove(&id);
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
struct NodePoolWithStorageLifecycle<S: Storage> {
    storage: Arc<S>,
}

impl<S: Storage> NodePoolWithStorageLifecycle<S> {
    /// Creates a new [`NodeCacheLifecycle`] with the given storage backend and error state
    /// reference.
    pub fn new(storage: Arc<S>) -> Self {
        NodePoolWithStorageLifecycle { storage }
    }
}

impl<S: Storage> Clone for NodePoolWithStorageLifecycle<S> {
    fn clone(&self) -> Self {
        NodePoolWithStorageLifecycle {
            storage: self.storage.clone(),
        }
    }
}

impl<S: Storage<Id = NodeId, Item = Node>> Lifecycle<NodeId, Arc<RwLock<NodeWithMetadata>>>
    for NodePoolWithStorageLifecycle<S>
{
    /// No-op for beginning a request, as we don't need to store any state.
    fn begin_request(&self) -> Self::RequestState {}

    /// Stores the entry in the storage when if it's dirty.
    /// In case of an error, it registers it in the [`ErrorState`].
    fn on_evict(
        &self,
        _state: &mut Self::RequestState,
        key: NodeId,
        val: Arc<RwLock<NodeWithMetadata>>,
    ) {
        let val = val.read().unwrap();
        if val.status != NodeStatus::Dirty {
            return;
        }
        self.storage
            .set(key, &val.value)
            .unwrap_or_else(|e| panic!("Failed to store evicted node {}: {}", key.to_index(), e));
    }

    /// Checks if the entry is pinned and cannot be evicted.
    /// An entry is pinned if another thread holds a reference to it (strong count > 1)
    fn is_pinned(&self, _key: &NodeId, val: &Arc<RwLock<NodeWithMetadata>>) -> bool {
        Arc::strong_count(val) > 1
    }

    /// Type used to store state during a request.
    /// We don't store any state, set as unit type.
    type RequestState = ();
}

#[cfg(test)]
mod tests {
    use std::{panic::catch_unwind, sync::Mutex};

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
    fn node_pool_try_new_creates_node_pool() {
        let cache = NodePoolWithStorage::try_new(MockStorage::new(), 10);
        assert!(cache.is_ok());
        let cache = cache.unwrap();
        assert_eq!(cache.deref().cache.capacity(), 10);
    }

    #[test]
    fn node_pool_with_storage_impl_try_new_creates_node_pool() {
        let storage = MockStorage::new();
        let cache = NodePoolWithStorageImpl::try_new(storage, 10);
        assert!(cache.is_ok());
        let cache = cache.unwrap();
        assert_eq!(cache.cache.capacity(), 10);
    }

    #[test]
    fn node_pool_with_storage_impl_get_returns_existing_entry_from_storage_if_not_in_cache() {
        let expected_entry = Node::Empty;
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let mut storage = MockStorage::new();
        storage.expect_get().with(eq(id)).returning({
            let expected_entry = expected_entry.clone();
            move |_| Ok(expected_entry.clone())
        });

        let cache = NodePoolWithStorageImpl::try_new(storage, 10).unwrap();
        let entry = cache.get(id).unwrap();
        assert!(**entry.read().unwrap() == expected_entry);
    }

    #[test]
    fn node_pool_with_storage_impl_add_inserts_elements_in_cache() {
        let mut storage = MockStorage::new();
        storage
            .expect_reserve()
            .returning(|_| NodeId::from_idx_and_node_type(42, NodeType::Empty));
        storage.expect_get().never(); // Cache get should not call storage get
        let cache = NodePoolWithStorageImpl::try_new(storage, 10).unwrap();
        let node = Node::Empty;
        let id = cache.add(node).expect("set should succeed");
        assert_eq!(id, NodeId::from_idx_and_node_type(42, NodeType::Empty));
        let entry = cache.get(id).expect("get should succeed");
        assert!(**entry.read().unwrap() == Node::Empty);
    }

    #[test]
    fn node_pool_with_storage_impl_get_returns_error_if_node_id_does_not_exist() {
        let mut storage = MockStorage::new();
        storage
            .expect_get()
            .returning(|_| Err(storage::Error::NotFound));

        let cache = NodePoolWithStorageImpl::try_new(storage, 10).unwrap();
        let res = cache.get(NodeId::from_idx_and_node_type(0, NodeType::Empty));
        assert!(res.is_err());
        assert!(matches!(
            res.err().unwrap(),
            Error::Storage(storage::Error::NotFound)
        ));
    }

    #[test]
    fn node_pool_with_storage_impl_get_always_insert_element_in_cache() {
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
                .returning(move |_| Ok(Node::Empty));
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

        let cache = NodePoolWithStorageImpl::try_new(storage, NUM_ELEMENTS as usize)
            .expect("cache should be created");

        for i in 0..NUM_ELEMENTS {
            let id = NodeId::from_idx_and_node_type(i, NodeType::Empty);
            let entry = cache.get(id).expect("get should succeed");
            {
                let _ = &mut **entry.write().unwrap(); // Mutable borrow to mark as dirty
            }
            assert!(cache.cache.get(&id).is_some());
        }

        // Insert one element more than capacity, triggering eviction of the precedent element.
        let id = NodeId::from_idx_and_node_type(NUM_ELEMENTS, NodeType::Empty);
        let _ = cache.get(id).expect("get should succeed");
    }

    #[test]
    fn node_pool_with_storage_impl_flush_saves_dirty_entries_to_storage() {
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
                        && value == &Node::Empty
                })
                .returning(move |_, _| Ok(()));
        }
        storage.expect_flush().times(1).returning(|| Ok(()));

        let cache = NodePoolWithStorageImpl::try_new(storage, NUM_ELEMENTS as usize)
            .expect("cache should be created");

        for _ in 0..NUM_ELEMENTS {
            let _ = cache.add(Node::Empty).expect("set should succeed");
        }

        cache.flush().expect("flush should succeed");
    }

    #[test]
    fn node_pool_with_storage_impl_stores_data_in_storage_on_evict() {
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
                *idx == NodeId::from_idx_and_node_type(0, NodeType::Empty) && value == &Node::Empty
            })
            .returning(|_, _| Ok(()));

        let cache = NodePoolWithStorageImpl::try_new(storage, 1).expect("cache should be created");
        // Insert an element, which will be immediately evicted and stored
        let _ = cache.add(Node::Empty).expect("set should succeed");
    }

    #[test]
    fn node_pool_with_storage_impl_delete_removes_entry_from_cache_and_storage() {
        let mut storage = MockStorage::new();
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let entry = Arc::new(RwLock::new(NodeWithMetadata::new(
            Node::Empty,
            NodeStatus::Clean,
        )));
        storage
            .expect_delete()
            .times(1)
            .with(eq(id))
            .returning(|_| Ok(()));

        let cache = NodePoolWithStorageImpl::try_new(storage, 2).unwrap();
        cache.cache.insert(id, entry);

        cache.delete(id).expect("delete should succeed");
        assert!(cache.cache.get(&id).is_none());
    }

    #[test]
    fn node_pool_with_storage_impl_delete_sets_deleted_status_if_in_cache() {
        let mut storage = MockStorage::new();
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let entry = Arc::new(RwLock::new(NodeWithMetadata::new(
            Node::Empty,
            NodeStatus::Clean,
        )));
        storage
            .expect_delete()
            .times(1)
            .with(eq(id))
            .returning(|_| Ok(()));

        let cache = NodePoolWithStorageImpl::try_new(storage, 2).unwrap();
        cache.cache.insert(id, entry.clone());

        cache.delete(id).expect("delete should succeed");
        assert!(cache.cache.get(&id).is_none());
        assert!(entry.read().unwrap().status == NodeStatus::Deleted);
        // assert!(entry.read().unwrap().get().is_err());
        // assert!(entry.write().unwrap().get_mut().is_err());
    }

    #[test]
    fn node_pool_with_storage_impl_delete_fails_on_storage_error() {
        let mut storage = MockStorage::new();
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        storage
            .expect_delete()
            .times(1)
            .with(eq(id))
            .returning(|_| Err(storage::Error::NotFound));

        let cache = NodePoolWithStorageImpl::try_new(storage, 2).unwrap();
        cache.cache.insert(
            id,
            Arc::new(RwLock::new(NodeWithMetadata::new(
                Node::Empty,
                NodeStatus::Clean,
            ))),
        );

        let res = cache.delete(id);
        assert!(res.is_err());
        assert!(matches!(
            res.unwrap_err(),
            Error::Storage(storage::Error::NotFound)
        ));
    }
    #[test]
    fn node_pool_lifecycle_on_evict_stores_dirty_entries() {
        // Dirty element is stored
        {
            let mut storage = MockStorage::new();
            let entry = Arc::new(RwLock::new(NodeWithMetadata::new(
                Node::Empty,
                NodeStatus::Dirty,
            )));
            let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            storage.expect_set().times(1).returning(|_, _| Ok(()));
            let lifecycle = NodePoolWithStorageLifecycle {
                storage: Arc::new(storage),
            };

            lifecycle.on_evict(&mut (), id, entry);
        }
        // Clean element is not stored
        {
            let mut storage = MockStorage::new();
            let entry = Arc::new(RwLock::new(NodeWithMetadata::new(
                Node::Empty,
                NodeStatus::Clean,
            )));
            let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            // Expect no call to storage set
            storage.expect_set().never();
            let lifecycle = NodePoolWithStorageLifecycle {
                storage: Arc::new(storage),
            };

            lifecycle.on_evict(&mut (), id, entry);
        }
    }

    #[test]
    fn node_pool_lifecycle_on_evict_fails_on_storage_error_and_panics() {
        let mut storage = MockStorage::new();
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let entry = Arc::new(RwLock::new(NodeWithMetadata::new(
            Node::Empty,
            NodeStatus::Dirty,
        )));
        storage
            .expect_set()
            .times(1)
            .returning(|_, _| Err(storage::Error::NotFound));
        let lifecycle = NodePoolWithStorageLifecycle {
            storage: Arc::new(storage),
        };

        let res = catch_unwind(|| {
            lifecycle.on_evict(&mut (), id, entry);
        });

        assert!(res.is_err());
    }

    #[test]
    fn node_pool_lifecycle_is_pinned_checks_entry_number_of_references() {
        let lifecycle = NodePoolWithStorageLifecycle {
            storage: Arc::new(MockStorage::new()),
        };
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let entry = Arc::new(RwLock::new(NodeWithMetadata::new(
            Node::Empty,
            NodeStatus::Clean,
        )));
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
    fn node_with_metadata_sets_dirty_flag_on_deref_mut() {
        let mut cached_node = NodeWithMetadata::new(Node::Empty);
        assert_eq!(cached_node.status, NodeStatus::Clean);
        let _ = cached_node.deref();
        assert_eq!(cached_node.status, NodeStatus::Clean);
        let _ = cached_node.deref_mut();
        assert_eq!(cached_node.status, NodeStatus::Dirty);
    }

    #[test]
    #[should_panic = "Attempted to access a deleted node"]
    fn node_with_metadata_deref_mut_panics_on_deleted_node() {
        let cached_node = NodeWithMetadata {
            value: Node::Empty,
            status: NodeStatus::Deleted,
        };
        let _ = cached_node.deref();
    }

    #[test]
    #[should_panic = "Attempted to access a deleted node"]
    fn node_with_metadata_deref_panics_on_deleted_node() {
        let mut cached_node = NodeWithMetadata {
            value: Node::Empty,
            status: NodeStatus::Deleted,
        };
        let _ = cached_node.deref_mut();
    }
}
