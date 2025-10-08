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
    mem,
    ops::{Deref, DerefMut},
    vec::Vec,
};

use dashmap::DashSet;
use quick_cache::{Lifecycle, OptionsBuilder, Weighter};

use crate::{
    error::Error,
    node_manager::NodeManager,
    storage::{Checkpointable, Storage},
    sync::*,
    types::{Node, NodeSize},
};

/// A wrapper which dereferences to [`Node`] and additionally stores its dirty status,
/// indicating whether it needs to be flushed to storage.
/// The node's status is set to dirty when a mutable reference is requested.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
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

/// A node manager that caches nodes in memory, with a underlying storage backend.
/// Nodes are retrieved from the underlying storage if they are not present in the cache, and store
/// them back when evicted if they have been modified.
/// It only guarantees that no access to a deleted node is possible.
pub struct CachedNodeManager<K, N, S> {
    // A fixed-size container that acts as the owner of all nodes.
    // This allows us to hand out read/write guards from a shared reference to the node manager.
    // Wrapped in an Arc so that it can be shared with [`ItemLifecycle`].
    nodes: Arc<[RwLock<NodeWithMetadata<N>>]>,
    /// The cache managing the key-to-slot mapping as well as item eviction
    cache: quick_cache::sync::Cache<
        K,                // key type to identify cached values
        usize,            // value type representing slots in `Self::nodes`
        ByteSizeWeighter, // all values are considered to cost the same
        RandomState,      // default hasher
        ItemLifecycle<N>, // tracks and reports evicted items
    >,
    free_slots: DashSet<usize>, // set of free slots in [`Self::nodes`]
    //storage for managing IDs, fetching missing nodes, and saving evicted nodes to
    storage: S,
}

impl<K, S, N> CachedNodeManager<K, N, S>
where
    S: Storage<Id = K, Item = N>,
    N: Default + NodeSize,
    K: Eq + Hash + Copy + NodeSize + std::fmt::Debug,
{
    /// Creates a new [`CachedNodeManager`] with the given capacity (in bytes) and storage backend.
    /// The cache capacity is set to `(num_nodes - 1)` to ensure that there is always at least one
    /// free slot available in the node manager. Useful for creating node managers with smaller
    pub fn new(bytes_capacity: usize, storage: S) -> Self {
        // Requires at least 1MB
        if bytes_capacity < 1024 * 1024 {
            panic!("Node manager < 1 MiB. Please provide a larger capacity.");
        }
        // NOTE: Here we could just take into consideration the element footprint and ignore the
        // cache overhead. This would increase the number of nodes we can store and increase the
        // maximum cache weight capacity.
        let min_node_byte_size =
            N::min_non_empty_node_size() + Self::get_element_memory_footprint();
        // We allocate a slot for one additional node. This way, when the cache is full, we always
        // have a free slot we can use to insert a new item into the cache and force the
        // eviction of an old one.
        let num_nodes = bytes_capacity / min_node_byte_size + 1; // + 1 to be sure there is always gonna be a free node available

        // TODO: Benchmark different shard size as the default value is overestimated.
        Self::new_with_options(num_nodes, None, None, storage)
    }

    /// Creates a new [`CachedNodeManager`] with the given number of nodes, number of shards, and
    /// hot allocation percentage.
    /// The number of shards is an estimation: it will be rounded up to the next power of two and it
    /// may be smaller to ensure that each shard has enough capacity (depending on the internal
    /// implementation of `quick_cache`). The cache capacity is set to `(num_nodes - 1)` to
    /// ensure that there is always at least one free slot available in the node manager. Useful
    /// for creating node managers with smaller cache capacity and predictable eviction
    /// behavior.
    fn new_with_options(
        num_nodes: usize,
        num_shards: Option<usize>,
        hot_allocation: Option<f64>,
        storage: S,
    ) -> Self {
        let nodes: Arc<[_]> = (0..num_nodes)
            .map(|_| {
                // Pre-allocate with default values. This requires `N: Default`.
                RwLock::new(NodeWithMetadata {
                    node: N::default(),
                    is_dirty: false,
                })
            })
            .collect();

        // NOTE: cache_weight needs to be <= num_nodes * N::min_non_empty_node_size() to ensure that
        // the cache weight is exhausted when inserting only nodes with the minimum size.
        let cache_weight = (num_nodes - 1) * N::min_non_empty_node_size();
        let mut options = OptionsBuilder::new();
        options
            .weight_capacity(cache_weight as u64)
            .estimated_items_capacity(num_nodes - 1);
        if let Some(num_shards) = num_shards {
            options.shards(num_shards);
        };
        if let Some(hot_allocation) = hot_allocation {
            options.hot_allocation(hot_allocation);
        }
        let options = options
            .build()
            .expect("failed to build cache options. Did you provide all the required options?");

        CachedNodeManager {
            nodes: nodes.clone(),
            storage,
            cache: quick_cache::sync::Cache::with_options(
                options,
                ByteSizeWeighter::default(),
                RandomState::default(),
                ItemLifecycle { nodes },
            ),
            free_slots: DashSet::from_iter(0..num_nodes),
        }
    }

    /// Persists an evicted item to storage if it is dirty and frees up the item's storage slot.
    fn on_evict(&self, entry: &[(K, usize)]) -> Result<(), Error> {
        for (key, pos) in entry {
            // Get exclusive write access to the node before storing it
            // to ensure that no other thread has a reference to it and
            // avoid risking to lose data.
            #[allow(clippy::readonly_write_lock)]
            let mut guard = self.nodes[*pos].write().unwrap();
            if guard.is_dirty {
                self.storage.set(*key, &guard)?;
            }
            **guard = N::default(); // reset node to default value to release storage
            self.free_slots.insert(*pos);
        }
        Ok(())
    }

    /// Insert an item in the node manager, reusing a free slot if available or evicting an
    /// existing item if the cache is full.
    /// Returns the position of the inserted node in the `nodes` vector.
    fn insert(&self, key: K, node: NodeWithMetadata<N>) -> Result<usize, Error> {
        // While there should always be at least one free slot available, there is a interval in
        // which the set may be empty while another thread is inserting a new item and before it has
        // evicted an old one. In that case, we loop until a free slot is available.
        let pos = loop {
            let pos = self.free_slots.iter().next().map(|p| *p);
            if let Some(pos) = pos
                && let Some(pos) = self.free_slots.remove(&pos)
            {
                break pos;
            }
            hint::spin_loop();
        };
        let mut guard = self.nodes[pos].write().unwrap();
        *guard = node;
        drop(guard); // release lock before inserting in cache
        // Insert a new item in cache, evict an old item if necessary
        let evicted = self.cache.insert_with_lifecycle(key, pos);
        self.on_evict(evicted.as_slice())?;
        Ok(pos)
    }

    /// Returns the memory required to store a node, including wrappers and cache overhead.
    /// For the cache overhead, see [quick_cache docs](https://docs.rs/quick_cache/latest/quick_cache/#approximate-memory-usage)
    pub fn get_element_memory_footprint() -> usize {
        mem::size_of::<RwLock<NodeWithMetadata<N>>>() // Stored wrapper in the nodes vector
    + (((mem::size_of::<K>() + mem::size_of::<usize>() + 21) as f64 * 1.5).floor() as usize).next_power_of_two() // Cache overhead per item
    }
}

impl<K, N, S> NodeManager for CachedNodeManager<K, N, S>
where
    S: Storage<Id = K, Item = N>,
    N: Default + NodeSize,
    K: Eq + Hash + Copy + NodeSize + std::fmt::Debug,
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
            loop {
                let guard = self.nodes[pos].read().unwrap();
                // The node may have been deleted from the cache in the meantime and/or re-inserted
                if let Some(new_pos) = self.cache.peek(&id) {
                    if new_pos == pos {
                        return Ok(guard);
                    } else {
                        continue; // The node was re-inserted in the cache in a different position, retry
                    }
                }
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
            loop {
                let guard = self.nodes[pos].write().unwrap();
                // The node may have been deleted from the cache in the meantime and/or re-inserted
                if let Some(new_pos) = self.cache.peek(&id) {
                    if new_pos == pos {
                        return Ok(guard);
                    } else {
                        continue; // The node was re-inserted in the cache in a different position, retry.
                    }
                }
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

impl<K, N, S> Checkpointable for CachedNodeManager<K, N, S>
where
    S: Storage<Id = K, Item = N> + Checkpointable,
    N: Default,
    K: Eq + Hash + Copy + NodeSize,
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
    type RequestState = Vec<(K, usize)>;

    /// Checks if an item can be evicted from the cache.
    /// An item is considered pinned if another thread holds a lock to it
    fn is_pinned(&self, _key: &K, value: &usize) -> bool {
        // NOTE: Another thread may try to acquire a write lock on this node after the function
        // returns, but that should be fine as the the shard containing the item should
        // remain write-locked for the entire eviction process.
        self.nodes[*value].try_write().is_err()
    }

    /// Initializes the vector used to track evicted items during a request.
    fn begin_request(&self) -> Self::RequestState {
        Vec::new()
    }

    /// Records the key and value of an evicted item in the request state.
    /// This is useful for inspecting which items were evicted after an insertion.
    fn on_evict(&self, state: &mut Self::RequestState, key: K, value: usize) {
        state.push((key, value));
    }
}

impl<N> Clone for ItemLifecycle<N> {
    fn clone(&self) -> Self {
        ItemLifecycle {
            nodes: self.nodes.clone(),
        }
    }
}

/// A `quick_cache` weighter that weights elements based on their byte size.
#[derive(Debug, Clone, Default)]
pub struct ByteSizeWeighter {}

impl<K> Weighter<K, usize> for ByteSizeWeighter
where
    K: Eq + Hash + Copy + NodeSize,
{
    fn weight(&self, key: &K, _value: &usize) -> u64 {
        // Weighing using the key allows us to avoid storing a reference to the node vector in the
        // weighter and locking the node to obtain the byte size, which may deadlock.
        // NOTE: This requires the key type to encode the node type in it.
        key.node_byte_size() as u64
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use mockall::{
        Sequence, mock,
        predicate::{always, eq, ne},
    };

    use super::*;
    use crate::{
        storage::{self},
        types::{NodeId, NodeType},
    };

    type TestCachedNodeManager = CachedNodeManager<NodeId, Node, MockCachedNodeManagerStorage>;
    const ONE_MB: usize = 1024 * 1024;

    #[test]
    fn cached_node_manager_new_creates_node_manager() {
        let capacity = ONE_MB;
        let storage = MockCachedNodeManagerStorage::new();
        let manager = TestCachedNodeManager::new(capacity, storage);
        let total_node_size: usize = (Node::min_non_empty_node_size()
            + mem::size_of::<RwLock<NodeWithMetadata<Node>>>())
            * manager.nodes.len();
        assert!(
            total_node_size
                + ((size_of::<NodeId>() + 21 + size_of::<usize>())
                    * ((manager.nodes.len() - 1) as f64 * 1.5) as usize)
                    .next_power_of_two()
                <= capacity
        );
    }

    #[test]
    #[should_panic]
    fn cached_node_manager_new_fails_if_capacity_too_small() {
        let storage = MockCachedNodeManagerStorage::new();
        let _ = TestCachedNodeManager::new(1024, storage);
    }

    #[test]
    fn cached_node_manager_does_not_exceed_capacity() {
        let max_capacity = ONE_MB;
        let cache_capacity = get_cache_capacity(max_capacity) as usize;
        let cases = [
            vec![NodeType::Leaf2],
            vec![NodeType::Leaf256],
            vec![NodeType::Inner],
            vec![NodeType::Leaf2, NodeType::Leaf256, NodeType::Inner],
        ];

        for case in cases {
            let mut storage = MockCachedNodeManagerStorage::new();
            storage.expect_set().returning(move |_, _| Ok(()));
            let num_nodes = cache_capacity
                / (case
                    .iter()
                    .map(|c| c.node_byte_size() / case.len())
                    .sum::<usize>());
            let manager = TestCachedNodeManager::new(max_capacity, storage);

            for i in 0..(num_nodes * 2) {
                let node_type = case[i % case.len()];
                let id = NodeId::from_idx_and_node_type(i as u64, node_type);
                let _ = manager
                    .insert(
                        id,
                        NodeWithMetadata {
                            node: match node_type {
                                NodeType::Leaf2 => Node::Leaf2(Box::default()),
                                NodeType::Leaf256 => Node::Leaf256(Box::default()),
                                NodeType::Inner => Node::Inner(Box::default()),
                                NodeType::Empty => unreachable!(),
                            },
                            is_dirty: true,
                        },
                    )
                    .unwrap();
            }
            // Iterating over the cache gives the actual elements stored in the manager
            let cache_elements_weight = manager
                .cache
                .iter()
                .map(|(id, _)| {
                    id.node_byte_size() + TestCachedNodeManager::get_element_memory_footprint()
                })
                .sum::<usize>();
            // Size of unused elements in the node vector
            let vector_overhead = (manager.nodes.len() - manager.cache.len())
                * (size_of::<RwLock<NodeWithMetadata<Node>>>() + size_of::<Box<Node>>());
            assert!(cache_elements_weight + vector_overhead <= max_capacity);
        }
    }

    #[test]
    fn cached_node_manager_new_with_options_creates_node_manager_with_provided_options() {
        let num_shards = 1;
        let hot_allocation = 1.0;
        let num_nodes = num_shards * 32; // 32 items per shard seems what quick_cache requires for each shard.
        let mut storage = MockCachedNodeManagerStorage::new();
        storage.expect_set().never();
        let manager = TestCachedNodeManager::new_with_options(
            num_nodes,
            Some(num_shards),
            Some(hot_allocation),
            storage,
        );
        assert_eq!(manager.cache.num_shards(), num_shards);
        // quick_cache doesn't allow querying the hot allocation directly, so we check it by filling
        // up the cache and observing the eviction behavior. With hot_allocation = 1.0, no eviction
        // should happen.
        // The only reliable way to test this is to also have a shard capacity of 1, otherwise key
        // collisions may cause evictions on the same shard.
        let node = NodeWithMetadata {
            node: Node::Leaf2(Box::default()),
            is_dirty: true, // mark as dirty to trigger eviction
        };
        for i in 0..manager.nodes.len() - 1 {
            let _ = manager.insert(
                NodeId::from_idx_and_node_type(i as u64, NodeType::Leaf2),
                node.clone(),
            );
        }
        assert_eq!(manager.cache.len(), manager.nodes.len() - 1);
    }

    #[test]
    fn cached_node_manager_always_has_free_positions_in_free_list() {
        let min_size_node_type = NodeType::Leaf2;
        let max_capacity = ONE_MB;
        let mut storage = MockCachedNodeManagerStorage::new();
        storage.expect_set().returning(move |_, _| Ok(()));
        let manager = TestCachedNodeManager::new(max_capacity, storage);

        for i in 0..manager.nodes.len() {
            let id = NodeId::from_idx_and_node_type(i as u64, min_size_node_type);
            let _ = manager
                .insert(
                    id,
                    NodeWithMetadata {
                        node: Node::Leaf2(Box::default()),
                        is_dirty: true,
                    },
                )
                .unwrap();
        }
        assert!(
            !manager.free_slots.is_empty(),
            "There should always be at least one free position in the free list"
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
        let manager = TestCachedNodeManager::new(ONE_MB, storage);

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

        manager
            .on_evict(&[(id1, 0)])
            .expect("should be evicted because it's dirty");
        assert!(manager.free_slots.contains(&0));
        assert!(**manager.nodes[0].read().unwrap() == Node::default());
        manager
            .on_evict(&[(id2, 1)])
            .expect("should not be evicted because it's clean");
        assert!(manager.free_slots.contains(&1));
    }

    #[test]
    fn cached_node_manager_insert_inserts_items() {
        let expected_node = NodeWithMetadata {
            node: Node::Leaf2(Box::default()),
            is_dirty: true,
        };
        // Cache is not full
        {
            let mut storage = MockCachedNodeManagerStorage::new();
            storage.expect_set().never();
            let manager = TestCachedNodeManager::new(ONE_MB, storage);

            let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            let pos = manager.insert(id, expected_node.clone()).unwrap();
            let guard = manager.nodes[pos].read().unwrap();
            assert_eq!(*guard, expected_node);
            assert!(!manager.free_slots.contains(&pos));
        }
        // Cache is full
        {
            let mut storage = MockCachedNodeManagerStorage::new();
            storage.expect_set().returning(move |_, _| Ok(()));
            let manager = TestCachedNodeManager::new(ONE_MB, storage);

            for i in 0..manager.nodes.len() + 1 {
                let id = NodeId::from_idx_and_node_type(i as u64, NodeType::Leaf2);
                let pos = manager.insert(id, expected_node.clone()).unwrap();
                let guard = manager.nodes[pos].read().unwrap();
                assert_eq!(*guard, expected_node);
            }
        }
    }

    #[test]
    fn cached_node_manager_add_reserves_id_and_inserts_nodes() {
        let expected_id = NodeId::from_idx_and_node_type(42, NodeType::Leaf2);
        let node = Node::Leaf256(Box::default());
        let mut storage = MockCachedNodeManagerStorage::new();
        storage.expect_reserve().returning(move |_| expected_id);
        storage.expect_get().never(); // Shouldn't query storage on add
        let manager = TestCachedNodeManager::new(ONE_MB, storage);

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
        let manager = TestCachedNodeManager::new(ONE_MB, storage);

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
        let manager = TestCachedNodeManager::new(ONE_MB, storage);

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
        let manager = TestCachedNodeManager::new(ONE_MB, storage);

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
        let max_capacity = ONE_MB;
        let cache_capacity = get_cache_capacity(max_capacity) as usize;
        let cases = [
            vec![NodeType::Leaf2],
            vec![NodeType::Leaf256],
            vec![NodeType::Inner],
            vec![NodeType::Leaf2, NodeType::Leaf256, NodeType::Inner],
        ];

        for case in cases {
            let mut storage = MockCachedNodeManagerStorage::new();
            let mut sequence = Sequence::new();
            let evicted_nodes = Arc::new(Mutex::new(vec![]));
            let num_nodes = cache_capacity
                / case
                    .iter()
                    .map(|c| c.node_byte_size() / case.len())
                    .sum::<usize>();

            for i in 0..num_nodes + 1 {
                let node_type = case[i % case.len()];
                // 1 item more than capacity
                storage
                    .expect_get()
                    .times(1)
                    .in_sequence(&mut sequence)
                    .with(eq(NodeId::from_idx_and_node_type(i as u64, node_type)))
                    .returning(move |_| {
                        Ok(match node_type {
                            NodeType::Leaf2 => Node::Leaf2(Box::default()),
                            NodeType::Leaf256 => Node::Leaf256(Box::default()),
                            NodeType::Inner => Node::Inner(Box::default()),
                            NodeType::Empty => unreachable!(),
                        })
                    });
            }
            // Expect all nodes except `num_nodes` to be saved to storage
            storage
                .expect_set()
                .with(
                    ne(NodeId::from_idx_and_node_type(
                        num_nodes as u64,
                        case[num_nodes as usize % case.len()],
                    )),
                    always(),
                )
                .returning({
                    let evicted_nodes = evicted_nodes.clone();
                    move |id, _| {
                        evicted_nodes.lock().unwrap().push(id);
                        Ok(())
                    }
                });
            let manager = TestCachedNodeManager::new(ONE_MB, storage);

            for i in 0..num_nodes {
                let id = NodeId::from_idx_and_node_type(i as u64, case[i % case.len()]);
                let mut entry = manager.get_write_access(id).unwrap();
                {
                    let _: &mut Node = &mut entry; // Mutable borrow to mark as dirty
                }
                assert!(manager.cache.get(&id).is_some());
            }

            // Retrieving and insert one item more than capacity, triggering eviction of one item
            // but not of the current one
            let id = NodeId::from_idx_and_node_type(num_nodes as u64, case[num_nodes % case.len()]);
            let _unused = get_method(&manager, id).unwrap();
        }
    }

    #[test]
    fn cached_node_manager_checkpoint_saves_dirty_nodes_to_storage() {
        const NUM_NODES: u64 = 10;
        let mut storage = MockCachedNodeManagerStorage::new();
        let mut sequence = Sequence::new();
        for i in 0..NUM_NODES {
            storage
                .expect_reserve()
                .times(1)
                .in_sequence(&mut sequence)
                .returning(move |_| NodeId::from_idx_and_node_type(i, NodeType::Leaf2));
            storage
                .expect_set()
                .times(1)
                .with(
                    eq(NodeId::from_idx_and_node_type(i, NodeType::Leaf2)),
                    always(),
                )
                .returning(move |_, _| Ok(()));
        }
        storage.expect_checkpoint().times(1).returning(|| Ok(()));
        let manager = TestCachedNodeManager::new(ONE_MB, storage);

        for _ in 0..NUM_NODES {
            // Newly added nodes are always dirty
            let _ = manager.add(Node::Leaf2(Box::default())).unwrap();
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
        let manager = TestCachedNodeManager::new(ONE_MB, storage);

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

        let manager = TestCachedNodeManager::new(ONE_MB, storage);
        let _ = manager.add(Node::Empty).unwrap();
        let res = manager.delete(id);
        assert!(res.is_err());
        assert!(matches!(
            res.unwrap_err(),
            Error::Storage(storage::Error::NotFound)
        ));
    }

    #[rstest_reuse::apply(get_method)]
    fn shuttle__cached_node_manager_never_returns_a_reference_to_a_non_existing_node_on_delete(
        #[case] get_method: GetMethod,
    ) {
        run_shuttle_check(
            move || {
                let capacity = ONE_MB;
                let id = NodeId::from_idx_and_node_type(0, NodeType::Leaf2);
                let node = NodeWithMetadata {
                    node: Node::Leaf2(Box::default()),
                    is_dirty: false,
                };
                let mut storage = MockCachedNodeManagerStorage::new();
                let is_deleted = Arc::new(AtomicBool::new(false));
                storage.expect_delete().times(1).with(eq(id)).returning({
                    let is_deleted = is_deleted.clone();
                    move |_| {
                        is_deleted.store(true, Ordering::SeqCst);
                        Ok(())
                    }
                });
                storage.expect_get().times(0..=1).with(eq(id)).returning({
                    let is_deleted = is_deleted.clone();
                    move |_| {
                        if is_deleted.load(Ordering::SeqCst) {
                            Err(storage::Error::NotFound)
                        } else {
                            Ok(Node::Leaf2(Box::default()))
                        }
                    }
                });
                let manager = Arc::new(TestCachedNodeManager::new(capacity, storage));
                *manager.nodes[0].write().unwrap() = node.clone();
                manager.cache.insert(id, 0);

                // now we spawn two threads: one that deletes the node, and one that tries to get
                // a reference to it.
                // the condition we want to test is that the get thread never returns a reference to
                // a deleted (aka. empty) node.

                let delete_thread = thread::spawn({
                    let manager = manager.clone();
                    move || {
                        manager.delete(id).unwrap();
                    }
                });

                let get_thread = thread::spawn({
                    let manager = manager.clone();
                    move || {
                        let res = get_method(&manager, id);
                        if let Ok(guard) = res {
                            assert!(guard != Node::default());
                        }
                    }
                });

                delete_thread.join().unwrap();
                get_thread.join().unwrap();
            },
            1000,
        );
    }

    #[rstest_reuse::apply(get_method)]
    fn shuttle__cached_node_manager_never_returns_a_reference_to_a_non_existing_node_on_evict(
        #[case] get_method: GetMethod,
    ) {
        run_shuttle_check(
            move || {
                let num_elements: usize = 3;
                // Evict the element previous to the last one
                let id_to_evict =
                    NodeId::from_idx_and_node_type(num_elements as u64 - 2, NodeType::Leaf2);
                let mut storage = MockCachedNodeManagerStorage::new();
                let is_evicted = Arc::new(AtomicBool::new(false));
                storage.expect_set().times(1).returning({
                    let is_evicted = is_evicted.clone();
                    move |id, _| {
                        if id == id_to_evict {
                            is_evicted.store(true, Ordering::SeqCst);
                        }
                        Ok(())
                    }
                });
                storage
                    .expect_get()
                    .times(0..=1)
                    .with(eq(id_to_evict))
                    .returning({
                        let is_evicted = is_evicted.clone();
                        move |_| {
                            if is_evicted.load(Ordering::SeqCst) {
                                Err(storage::Error::NotFound)
                            } else {
                                Ok(Node::Leaf2(Box::default()))
                            }
                        }
                    });
                storage.expect_reserve().times(3).returning({
                    let counter = std::sync::atomic::AtomicU64::new(0);
                    move |_| {
                        NodeId::from_idx_and_node_type(
                            counter.fetch_add(1, Ordering::SeqCst),
                            NodeType::Leaf2,
                        )
                    }
                });
                // Manually construct a CachedNodeManager with a single-shard, small cache to
                // control element eviction
                let manager = Arc::new(TestCachedNodeManager::new_with_options(
                    num_elements,
                    Some(1),
                    Some(1.0),
                    storage,
                ));

                // now we spawn two threads: one that adds a new node to a full cache, which will
                // cause eviction of an existing node, and one that tries to get a reference to the
                // node being evicted. The condition we want to test is that the get thread never
                // returns a reference to an evicted (aka. empty) node.
                manager.add(Node::Leaf2(Box::default())).unwrap(); // Fill cache
                manager.add(Node::Leaf2(Box::default())).unwrap(); // Fill cache

                let add_thread = thread::spawn({
                    let manager = manager.clone();
                    move || {
                        let _ = manager.add(Node::Leaf2(Box::default())).unwrap();
                    }
                });

                let get_thread = thread::spawn({
                    let manager = manager.clone();
                    move || {
                        let res = get_method(&manager, id_to_evict);
                        if let Ok(guard) = res {
                            assert!(guard != Node::default());
                        }
                    }
                });

                add_thread.join().unwrap();
                get_thread.join().unwrap();
            },
            1000,
        );
    }

    #[rstest_reuse::apply(get_method)]
    #[test]
    fn shuttle__cached_node_manager_never_returns_a_reference_to_a_non_existing_node_on_get(
        #[case] get_method: GetMethod,
    ) {
        run_shuttle_check(
            move || {
                eprintln!("\n\nNew iteration\n");
                let num_elements: usize = 3;
                // Evict the element previous to the last one
                let id_to_evict =
                    NodeId::from_idx_and_node_type(num_elements as u64 - 2, NodeType::Leaf2);
                let id_to_insert =
                    NodeId::from_idx_and_node_type(num_elements as u64, NodeType::Leaf2);
                let mut storage = MockCachedNodeManagerStorage::new();
                let is_evicted = Arc::new(AtomicBool::new(false));
                storage.expect_set().times(1).returning({
                    let is_evicted = is_evicted.clone();
                    move |id, _| {
                        if id == id_to_evict {
                            is_evicted.store(true, Ordering::SeqCst);
                        }
                        Ok(())
                    }
                });
                storage.expect_get().times(1..=2).returning({
                    let is_evicted = is_evicted.clone();
                    move |id| {
                        if id == id_to_insert {
                            Ok(Node::Leaf2(Box::default()))
                        } else if is_evicted.load(Ordering::SeqCst) {
                            Err(storage::Error::NotFound)
                        } else {
                            Ok(Node::Leaf2(Box::default()))
                        }
                    }
                });
                storage.expect_reserve().times(2).returning({
                    let counter = std::sync::atomic::AtomicU64::new(0);
                    move |_| {
                        NodeId::from_idx_and_node_type(
                            counter.fetch_add(1, Ordering::SeqCst),
                            NodeType::Leaf2,
                        )
                    }
                });
                // Manually construct a CachedNodeManager with a single-shard, small cache to
                // control element eviction
                let manager = Arc::new(TestCachedNodeManager::new_with_options(
                    num_elements,
                    Some(1),
                    Some(1.0),
                    storage,
                ));

                // now we spawn two threads: one that tries to get an existing element in the cache,
                // and other one that tries to get an element that needs to be queried from the
                // storage. The condition we want to test is that the get thread never
                // returns a reference to an evicted (aka. empty) node.
                manager.add(Node::Leaf2(Box::default())).unwrap(); // Fill cache
                manager.add(Node::Leaf2(Box::default())).unwrap(); // Fill cache

                let get_storage_thread = thread::spawn({
                    let manager = manager.clone();
                    move || {
                        let _ = get_method(&manager, id_to_insert).unwrap();
                    }
                });

                let get_existing_thread = thread::spawn({
                    let manager = manager.clone();
                    move || {
                        let res = get_method(&manager, id_to_evict);
                        if let Ok(guard) = res {
                            assert!(guard != Node::default());
                        }
                    }
                });

                get_storage_thread.join().unwrap();
                get_existing_thread.join().unwrap();
            },
            1000,
        );
    }

    #[test]
    fn shuttle__cached_node_manager_insertion_always_terminate() {
        // The idea of this test is to check that inserting in a full cache with multiple threads
        // doesn't lead to a deadlock when there is contention on the only available free slot.
        // Therefore this test needs only to terminate to be considered successful.
        run_shuttle_check(
            move || {
                let num_elements = 2;
                let mut storage = MockCachedNodeManagerStorage::new();
                storage.expect_set().returning(move |_, _| Ok(()));
                storage.expect_reserve().returning({
                    let counter = std::sync::atomic::AtomicU64::new(0);
                    move |_| {
                        NodeId::from_idx_and_node_type(
                            counter.fetch_add(1, Ordering::SeqCst),
                            NodeType::Leaf2,
                        )
                    }
                });

                let manager = Arc::new(TestCachedNodeManager::new_with_options(
                    num_elements,
                    Some(1),
                    Some(1.0),
                    storage,
                ));
                for _ in 0..num_elements - 1 {
                    manager.add(Node::Leaf2(Box::default())).unwrap();
                }

                // Now we spawn two threads trying to insert an item in a full cache, causing
                // contention on the only available free slot.
                let mut handles = vec![];
                for _ in 0..2 {
                    let manager = manager.clone();
                    handles.push(thread::spawn(move || {
                        let _ = manager.add(Node::Leaf2(Box::default())).unwrap();
                    }));
                }

                for handle in handles {
                    handle.join().unwrap();
                }
            },
            1000,
        );
    }

    #[track_caller]
    fn run_shuttle_check(test: impl Fn() + Send + Sync + 'static, _num_iter: usize) {
        #[cfg(feature = "shuttle")]
        shuttle::check_random(test, _num_iter);
        #[cfg(not(feature = "shuttle"))]
        test();
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
        assert!(state.is_empty());
        lifecycle.on_evict(&mut state, 42, 0);
        lifecycle.on_evict(&mut state, 84, 1);
        assert!(state.len() == 2);
        assert_eq!(state[0], (42, 0));
        assert_eq!(state[1], (84, 1));
    }

    #[test]
    fn size_weighter_weight_returns_node_size() {
        let cases = [
            Node::Empty,
            Node::Leaf2(Box::default()),
            Node::Leaf256(Box::default()),
            Node::Inner(Box::default()),
        ];

        for node in cases {
            let weighter = ByteSizeWeighter::default();
            let key = NodeId::from_idx_and_node_type(0, NodeType::from_node(&node));
            let weight = weighter.weight(&key, &0);
            assert_eq!(weight as usize, key.node_byte_size());
        }
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

            fn get(&self, _id: <Self as Storage>::Id) -> Result<<Self as Storage>::Item,
    storage::Error>;

            fn reserve(&self, _item: &<Self as Storage>::Item) -> <Self as Storage>::Id;

            fn set(&self, _id: <Self as Storage>::Id, _item: &<Self as Storage>::Item) ->
    Result<(), storage::Error>;

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

    /// Helper function to get the actual cache capacity for a given byte size.
    fn get_cache_capacity(size: usize) -> u64 {
        TestCachedNodeManager::new(size, MockCachedNodeManagerStorage::new())
            .cache
            .capacity()
    }
}
