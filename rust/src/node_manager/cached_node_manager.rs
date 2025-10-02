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
    cmp::Eq,
    hash::{Hash, RandomState},
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use dashmap::DashSet;
use quick_cache::{Lifecycle, UnitWeighter};

use crate::{
    error::Error,
    node_manager::NodeManager,
    storage::{Checkpointable, Storage},
    types::Node,
};

/// A wrapper which dereferences to [`Node`] and additionally stores its dirty status,
/// indicating whether it needs to be flushed to storage.
/// The node's status is set to dirty when a mutable reference is requested.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeWithMetadata<N> {
    node: N,
    is_dirty: bool,
}

impl<N> Deref for NodeWithMetadata<N> {
    type Target = N;

    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

impl<N> DerefMut for NodeWithMetadata<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.is_dirty = true; // Mark as dirty on mutable borrow
        &mut self.node
    }
}

pub struct CachedNodeManager<K, N, S> {
    // A fixed-size container that acts as the owner of all nodes.
    // This allows us to hand out read/write guards from a shared reference to the node manager.
    // Wrapped in an Arc so that it can be shared with [`ItemLifecycle`].
    nodes: Arc<[RwLock<NodeWithMetadata<N>>]>,
    /// The cache managing the key-to-slot mapping as well as item eviction
    cache: quick_cache::sync::Cache<
        K,                // key type to identify cached values
        usize,            // value type representing slots in `Self::nodes`
        UnitWeighter,     // all values are considered to cost the same
        RandomState,      // default hasher
        ItemLifecycle<N>, // tracks and reports evicted items
    >,
    free_slots: DashSet<usize>, // set of free slots in [`Self::nodes`]
    //storage for managing IDs, fetching missing nodes, and saving evicted nodes to
    storage: S,
}

impl<K: Eq + Hash + Copy, S, N> CachedNodeManager<K, N, S>
where
    S: Storage<Id = K, Item = N>,
    N: Default,
{
    /// Creates a new [`CachedNodeManager`] with the given capacity and storage backend.
    pub fn new(capacity: usize, storage: S) -> Self {
        // We allocate a slot for one additional node. This way, when the cache is full, we always
        // have a free slot we can use to insert a new item into the cache and force the
        // eviction of an old one.
        let num_nodes = capacity + 1;
        let nodes: Arc<[_]> = (0..num_nodes)
            .map(|_| {
                // Pre-allocate with default values. This requires `N: Default`.
                RwLock::new(NodeWithMetadata {
                    node: N::default(),
                    is_dirty: false,
                })
            })
            .collect();

        let options = quick_cache::OptionsBuilder::new()
            .estimated_items_capacity(capacity)
            .weight_capacity(capacity as u64) // unit weight per value
            .build()
            .expect("failed to build cache options. Did you provide all the required options?");

        CachedNodeManager {
            nodes: nodes.clone(),
            storage,
            cache: quick_cache::sync::Cache::with_options(
                options,
                UnitWeighter,
                RandomState::default(),
                ItemLifecycle { nodes },
            ),
            free_slots: DashSet::from_iter(0..num_nodes),
        }
    }

    /// Persists an evicted item to storage if it is dirty and frees up the item's storage slot.
    fn on_evict(&self, entry: (K, usize)) -> Result<(), Error> {
        let (key, pos) = entry;
        // Get exclusive write access to the node before storing it
        // to ensure that no other thread has a reference to it and
        // avoid risking to lose data.
        #[allow(clippy::readonly_write_lock)]
        let mut guard = self.nodes[pos].write().unwrap();
        if guard.is_dirty {
            self.storage.set(key, &guard)?;
        }
        **guard = N::default(); // reset node to default value to release storage
        self.free_slots.insert(pos);
        Ok(())
    }

    /// Insert an item in the node manager, reusing a free slot if available or evicting an
    /// existing item if the cache is full.
    /// Returns the position of the inserted node in the `nodes` vector.
    fn insert(&self, key: K, node: NodeWithMetadata<N>) -> Result<usize, Error> {
        // While there should always be at least one free position available, there is a interval in
        // which the list may be empty while inserting a new item and evicting an old one.
        // In that case, we loop until a free position is available.
        let pos = loop {
            let pos = self.free_slots.iter().next().map(|p| *p);
            if let Some(pos) = pos
                && let Some(pos) = self.free_slots.remove(&pos)
            {
                break pos;
            }
        };

        let mut guard = self.nodes[pos].write().unwrap();
        *guard = node;

        // Insert a new item in cache, evict an old item if necessary
        let evicted = self.cache.insert_with_lifecycle(key, pos);
        if let Some(evicted) = evicted {
            // If the evicted item was dirty, persist it to storage
            self.on_evict(evicted)?;
        }
        Ok(pos)
    }
}

impl<K: Eq + Hash + Copy, N, S> NodeManager for CachedNodeManager<K, N, S>
where
    S: Storage<Id = K, Item = N>,
    N: Default,
{
    type Id = K;
    type NodeType = N;

    fn add(&self, node: Self::NodeType) -> Result<Self::Id, Error> {
        let id = self.storage.reserve(&node);
        self.insert(
            id,
            NodeWithMetadata {
                node,
                is_dirty: true,
            },
        )?;
        Ok(id)
    }

    fn get_read_access(
        &self,
        id: Self::Id,
    ) -> Result<RwLockReadGuard<'_, impl Deref<Target = Self::NodeType>>, Error> {
        if let Some(pos) = self.cache.get(&id) {
            let guard = self.nodes[pos].read().unwrap();
            // The node may have been deleted from the cache in the meantime, check again.
            if self.cache.get(&id).is_some() {
                return Ok(guard);
            }
        }
        let node = self.storage.get(id)?;
        let pos = self.insert(
            id,
            NodeWithMetadata {
                node,
                is_dirty: false,
            },
        )?;
        Ok(self.nodes[pos].read().unwrap())
    }

    fn get_write_access(
        &self,
        id: Self::Id,
    ) -> Result<RwLockWriteGuard<'_, impl DerefMut<Target = Self::NodeType>>, Error> {
        if let Some(pos) = self.cache.get(&id) {
            let guard = self.nodes[pos].write().unwrap();
            // The node may have been deleted from the cache in the meantime, check again.
            if self.cache.get(&id).is_some() {
                return Ok(guard);
            }
        }
        let node = self.storage.get(id)?;
        let pos = self.insert(
            id,
            NodeWithMetadata {
                node,
                is_dirty: false,
            },
        )?;
        Ok(self.nodes[pos].write().unwrap())
    }

    fn delete(&self, id: Self::Id) -> Result<(), Error> {
        if let Some(pos) = self.cache.get(&id) {
            // Get exclusive write access before dropping the node
            // to ensure that no other thread is holding a reference to it.
            let mut guard = self.nodes[pos].write().unwrap();
            self.cache.remove(&id);
            **guard = N::default(); // reset node to default value to release storage
            self.free_slots.insert(pos);
        }
        self.storage.delete(id)?;
        Ok(())
    }
}

impl<K: Eq + Hash + Copy, N, S> Checkpointable for CachedNodeManager<K, N, S>
where
    S: Storage<Id = K, Item = N> + Checkpointable,
    N: Default,
{
    fn checkpoint(&self) -> Result<(), crate::storage::Error> {
        for (id, pos) in self.cache.iter() {
            let mut entry_guard = self.nodes[pos].write().unwrap();
            // Skip unused slots.
            if self.free_slots.contains(&pos) {
                continue;
            }
            if entry_guard.is_dirty {
                self.storage.set(id, &entry_guard.node)?;
                entry_guard.is_dirty = false;
            }
        }
        self.storage.checkpoint()?;
        Ok(())
    }
}

/// Manages the lifecycle of cached items, preventing eviction of items currently in use.
pub struct ItemLifecycle<N> {
    nodes: Arc<[RwLock<NodeWithMetadata<N>>]>,
}

impl<K: Eq + Hash + Copy, N> Lifecycle<K, usize> for ItemLifecycle<N> {
    type RequestState = Option<(K, usize)>;

    /// Checks if an item can be evicted from the cache.
    /// An item is considered pinned if another thread holds a lock to it
    fn is_pinned(&self, _key: &K, value: &usize) -> bool {
        // NOTE: Another thread may try to acquire a write lock on this node after the function
        // returns, but that should be fine as the the shard containing the item should
        // remain write-locked for the entire eviction process.
        self.nodes[*value].try_write().is_err()
    }

    /// No-op
    fn begin_request(&self) -> Self::RequestState {
        None
    }

    /// Records the key and value of an evicted item in the request state.
    /// This is useful for inspecting which items were evicted after an insertion.
    fn on_evict(&self, state: &mut Self::RequestState, _key: K, value: usize) {
        *state = Some((_key, value));
    }
}

impl<N> Clone for ItemLifecycle<N> {
    fn clone(&self) -> Self {
        ItemLifecycle {
            nodes: self.nodes.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{path::Path, sync::Mutex};

    use mockall::{
        Sequence, mock,
        predicate::{always, eq, ne},
    };

    use super::*;
    use crate::{
        storage::{self},
        types::{NodeId, NodeType},
    };

    #[test]
    fn cached_node_manager_new_creates_node_manager() {
        let storage = MockCachedNodeManagerStorage::new();
        let manager =
            CachedNodeManager::<NodeId, Node, MockCachedNodeManagerStorage>::new(10, storage);
        assert_eq!(manager.cache.capacity(), 10);
        assert_eq!(manager.nodes.len(), 11);
        assert_eq!(
            manager.free_slots.len(),
            11, // All slots + 1
        );
    }

    #[test]
    fn cached_node_manager_evict_saves_dirty_nodes_in_storage() {
        let id1 = NodeId::from_idx_and_node_type(0, NodeType::Leaf2);
        let id2 = NodeId::from_idx_and_node_type(1, NodeType::Leaf2);
        let mut storage = MockCachedNodeManagerStorage::new();
        storage
            .expect_set()
            .times(1)
            .with(eq(id1), always())
            .returning(|_, _| Ok(()));

        let manager = CachedNodeManager::new(10, storage);
        // Manually insert two nodes
        *manager.nodes[0].write().unwrap() = NodeWithMetadata {
            node: Node::Leaf2(Box::default()),
            is_dirty: true,
        };
        *manager.nodes[1].write().unwrap() = NodeWithMetadata {
            node: Node::Leaf2(Box::default()),
            is_dirty: false,
        };
        manager.cache.insert(id1, 0);
        manager.cache.insert(id2, 1);

        // This should be evicted as it is dirty.
        manager.on_evict((id1, 0)).unwrap();
        assert!(manager.free_slots.contains(&0));
        assert!(**manager.nodes[0].read().unwrap() == Node::default()); // node reset to default
        // This should not be evicted as it is clean.
        manager.on_evict((id2, 1)).unwrap();
        assert!(manager.free_slots.contains(&1));
    }

    #[test]
    fn cached_node_manager_insert_inserts_items() {
        // Cache is not full
        {
            let mut storage = MockCachedNodeManagerStorage::new();
            storage.expect_set().never();
            let manager = CachedNodeManager::new(10, storage);
            let expected = NodeWithMetadata {
                node: Node::Leaf2(Box::default()),
                is_dirty: true,
            };
            let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            let pos = manager.insert(id, expected.clone()).unwrap();
            let guard = manager.nodes[pos].read().unwrap();
            assert_eq!(*guard, expected);
            assert!(!manager.free_slots.contains(&pos));
        }
        // Cache is full, empty list is empty
        {
            let mut storage = MockCachedNodeManagerStorage::new();
            storage.expect_set().times(1).returning(|_, _| Ok(()));
            // Create manager that only fits two nodes.
            let manager = CachedNodeManager::new(2, storage);
            let expected_node = NodeWithMetadata {
                node: Node::Empty,
                is_dirty: true,
            };
            let id1 = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            let id2 = NodeId::from_idx_and_node_type(1, NodeType::Empty);
            let id3 = NodeId::from_idx_and_node_type(2, NodeType::Empty);
            let pos1 = manager.insert(id1, expected_node.clone()).unwrap();
            let guard = manager.nodes[pos1].read().unwrap();
            assert_eq!(*guard, expected_node);
            drop(guard);
            let pos2 = manager.insert(id2, expected_node.clone()).unwrap();
            let guard = manager.nodes[pos2].read().unwrap();
            assert_eq!(*guard, expected_node);
            drop(guard);
            let pos3 = manager.insert(id3, expected_node.clone()).unwrap();
            let guard = manager.nodes[pos3].read().unwrap();
            assert_eq!(*guard, expected_node);
            assert!(manager.free_slots.len() == 1);
        }
    }

    #[test]
    fn cached_node_manager_add_reserves_id_and_inserts_nodes() {
        let expected_id = NodeId::from_idx_and_node_type(42, NodeType::Leaf2);
        let node = Node::Leaf256(Box::default());
        let mut storage = MockCachedNodeManagerStorage::new();
        storage.expect_reserve().returning(move |_| expected_id);
        storage.expect_get().never(); // Shouldn't query storage on add
        let manager = CachedNodeManager::new(10, storage);
        let id = manager.add(node.clone()).unwrap();
        assert_eq!(id, expected_id);
        let pos = manager.cache.get(&id).unwrap();
        assert!(manager.nodes[pos].read().unwrap().is_dirty);
        assert_eq!(manager.nodes[pos].read().unwrap().node, node);
    }

    #[rstest_reuse::apply(get_method)]
    fn cached_node_manager_get_methods_return_cached_entry(#[case] get_method: GetMethod) {
        let expected_entry = Node::Empty;
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let mut storage = MockCachedNodeManagerStorage::new();
        storage.expect_get().never(); // Shouldn't query storage if entry is in cache
        let manager = CachedNodeManager::new(10, storage);
        let _ = manager
            .insert(
                id,
                NodeWithMetadata {
                    node: expected_entry.clone(),
                    is_dirty: false,
                },
            )
            .unwrap();
        let entry = get_method(&manager, id).unwrap();
        assert!(entry == expected_entry);
    }

    #[rstest_reuse::apply(get_method)]
    fn cached_node_manager_get_methods_return_existing_entry_from_storage_if_not_in_cache(
        #[case] get_method: GetMethod,
    ) {
        let expected_entry = Node::Empty;
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let mut storage = MockCachedNodeManagerStorage::new();
        storage.expect_get().times(1).with(eq(id)).returning({
            let expected_entry = expected_entry.clone();
            move |_| Ok(expected_entry.clone())
        });

        let manager = CachedNodeManager::new(10, storage);
        let entry = get_method(&manager, id).unwrap();
        assert!(entry == expected_entry);
    }

    #[rstest_reuse::apply(get_method)]
    fn cached_node_manager_get_methods_returns_error_if_node_id_does_not_exist(
        #[case] get_method: GetMethod,
    ) {
        let mut storage = MockCachedNodeManagerStorage::new();
        storage
            .expect_get()
            .returning(|_| Err(storage::Error::NotFound));

        let manager = CachedNodeManager::new(10, storage);
        let res = get_method(&manager, NodeId::from_idx_and_node_type(0, NodeType::Empty));
        assert!(res.is_err());
        assert!(matches!(
            res.err().unwrap(),
            Error::Storage(storage::Error::NotFound)
        ));
    }

    #[rstest_reuse::apply(get_method)]
    fn cached_node_manager_get_methods_always_insert_node_in_cache_when_retrieved_from_storage(
        #[case] get_method: GetMethod,
    ) {
        const NUM_NODES: u64 = 10;
        let mut storage = MockCachedNodeManagerStorage::new();
        let mut sequence = Sequence::new();
        for i in 0..NUM_NODES + 1 {
            // 1 item more than capacity
            storage
                .expect_get()
                .times(1)
                .in_sequence(&mut sequence)
                .with(eq(NodeId::from_idx_and_node_type(i, NodeType::Empty)))
                .returning(move |_| Ok(Node::Empty));
        }
        storage
            .expect_set()
            .with(
                ne(NodeId::from_idx_and_node_type(
                    // The NUM_NODES-th node will be evicted because of infinite reuse distance
                    NUM_NODES,
                    NodeType::Empty,
                )),
                always(),
            )
            .returning(|_, _| Ok(()));

        let manager = CachedNodeManager::new(NUM_NODES as usize, storage);

        for i in 0..NUM_NODES {
            let id = NodeId::from_idx_and_node_type(i, NodeType::Empty);
            let mut entry = manager.get_write_access(id).unwrap();
            {
                let _: &mut Node = &mut entry; // Mutable borrow to mark as dirty
            }
            assert!(manager.cache.get(&id).is_some());
        }

        // Retrieving and insert one item more than capacity, triggering eviction of the
        // precedent item.
        let id = NodeId::from_idx_and_node_type(NUM_NODES, NodeType::Empty);
        let _unused = get_method(&manager, id).unwrap();
    }

    #[test]
    fn cached_node_manager_checkpoint_saves_dirty_nodes_to_storage() {
        const NUM_NODES: u64 = 10;
        let data = Arc::new(Mutex::new(vec![]));
        let mut storage = MockCachedNodeManagerStorage::new();
        let mut sequence = Sequence::new();
        for i in 0..NUM_NODES {
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
                    eq(Node::Empty),
                )
                .returning(move |_, _| Ok(()));
        }
        storage.expect_checkpoint().times(1).returning(|| Ok(()));

        let manager = CachedNodeManager::new(NUM_NODES as usize, storage);
        for _ in 0..NUM_NODES {
            // Newly added nodes are always dirty
            let _ = manager.add(Node::Empty).unwrap();
        }
        manager.checkpoint().expect("checkpoint should succeed");
    }

    #[test]
    fn cached_node_manager_delete_removes_entry_from_cache_and_storage() {
        let mut storage = MockCachedNodeManagerStorage::new();
        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);
        let entry = Node::Inner(Box::default());
        storage.expect_reserve().times(1).returning(move |_| id);
        storage
            .expect_delete()
            .times(1)
            .with(eq(id))
            .returning(|_| Ok(()));

        let manager = CachedNodeManager::new(2, storage);
        let _ = manager.add(entry).unwrap();
        let _ = manager.cache.get(&id).expect("entry should be in cache");
        manager.delete(id).unwrap();
        assert!(manager.cache.get(&id).is_none());
        assert!(
            manager.free_slots.contains(&0),
            "Node position 0 should be in free list after deletion"
        );
        assert!(**manager.nodes[0].read().unwrap() == Node::default()); // node reset to default
    }

    #[test]
    fn cached_node_manager_delete_fails_on_storage_error() {
        let mut storage = MockCachedNodeManagerStorage::new();
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        storage.expect_reserve().times(1).returning(move |_| id);
        storage
            .expect_delete()
            .times(1)
            .with(eq(id))
            .returning(|_| Err(storage::Error::NotFound));

        let manager = CachedNodeManager::new(2, storage);
        let _ = manager.add(Node::Empty).unwrap();
        let res = manager.delete(id);
        assert!(res.is_err());
        assert!(matches!(
            res.unwrap_err(),
            Error::Storage(storage::Error::NotFound)
        ));
    }

    #[test]
    fn item_lifecycle_is_pinned_checks_lock_and_pinned_pos() {
        let nodes = Arc::from([RwLock::new(NodeWithMetadata {
            node: Node::Empty,
            is_dirty: false,
        })]);
        let lifecycle = ItemLifecycle { nodes };

        // Element is not pinned if it can be locked and position is not PINNED_POS
        assert!(!lifecycle.is_pinned(&0, &0));

        // Element is pinned if it cannot be locked (another thread holds a lock)
        let _guard = lifecycle.nodes[0].write().unwrap(); // Lock item at pos 0
        assert!(lifecycle.is_pinned(&0, &0));
    }

    #[test]
    fn item_lifecycle_on_evict_records_evicted_items() {
        let nodes: Arc<[RwLock<NodeWithMetadata<Node>>]> = Arc::from(vec![].into_boxed_slice());
        let lifecycle = ItemLifecycle { nodes };
        let mut state = lifecycle.begin_request();
        assert!(state.is_none());
        lifecycle.on_evict(&mut state, 42, 0);
        assert_eq!(state, Some((42, 0)));
    }

    #[test]
    fn node_with_metadata_sets_dirty_flag_on_deref_mut() {
        let mut node = NodeWithMetadata {
            node: Node::Empty,
            is_dirty: false,
        };
        assert!(!node.is_dirty);
        let _ = node.deref();
        assert!(!node.is_dirty);
        let _ = node.deref_mut();
        assert!(node.is_dirty);
    }

    mock! {
        pub CachedNodeManagerStorage {}

        impl Checkpointable for CachedNodeManagerStorage {
            fn checkpoint(&self) -> Result<(), storage::Error>;
        }

        impl Storage for CachedNodeManagerStorage {
            type Id = NodeId;
            type Item = Node;

            fn open(_path: &Path) -> Result<Self, storage::Error>
            where
                Self: Sized;

            fn get(&self, _id: <Self as Storage>::Id) -> Result<<Self as Storage>::Item, storage::Error>;

            fn reserve(&self, _item: &<Self as Storage>::Item) -> <Self as Storage>::Id;

            fn set(&self, _id: <Self as Storage>::Id, _item: &<Self as Storage>::Item) -> Result<(), storage::Error>;

            fn delete(&self, _id: <Self as Storage>::Id) -> Result<(), storage::Error>;
        }
    }

    /// Type alias for a closure that calls either `get_read_access` or `get_write_access`
    type GetMethod = fn(
        &CachedNodeManager<NodeId, Node, MockCachedNodeManagerStorage>,
        NodeId,
    ) -> Result<Node, Error>;

    /// Reusable rstest template to test both `get_read_access` and `get_write_access`
    #[rstest_reuse::template]
    #[rstest::rstest]
    #[case::get_read_access((|manager, id| {
        let guard = manager.get_read_access(id)?;
        Ok((**guard).clone())
    }) as GetMethod)]
    #[case::get_write_access((|manager, id| {
        let guard = manager.get_write_access(id)?;
        Ok((**guard).clone())
    }) as GetMethod)]
    fn get_method(#[case] f: GetMethod) {}
}
