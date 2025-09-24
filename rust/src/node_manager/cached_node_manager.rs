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
    collections::VecDeque,
    hash::{Hash, RandomState},
    ops::{Deref, DerefMut},
    sync::{
        Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard,
        atomic::{AtomicUsize, Ordering},
    },
    vec::Vec,
};

use quick_cache::{Lifecycle, UnitWeighter};

use crate::{error::Error, node_manager::NodeManager, storage::Storage, types::Node};

/// A [`Node`] with a **status** attribute to store if it needs to be flushed to storage.
/// [`NodeWithMetadata`] automatically dereferences to `Node` via the [`Deref`] trait.
/// The node's status is set to [`NodeStatus::Dirty`] when a mutable reference is requested.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeWithMetadata<N> {
    node: N,
    status: NodeStatus,
}

impl<N> NodeWithMetadata<N> {
    /// Returns true if the node needs to be stored in the storage backend.
    pub fn should_store(&self) -> bool {
        self.status == NodeStatus::Dirty
    }
}

/// The status of a [`NodeWithMetadata`].
/// It can be:
/// - `Clean`: the node is in sync with the storage
/// - `Dirty`: the node has been modified and needs to be flushed to storage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeStatus {
    Clean,
    Dirty,
}

impl<N> Deref for NodeWithMetadata<N> {
    type Target = N;

    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

impl<N> DerefMut for NodeWithMetadata<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.status = NodeStatus::Dirty; // Mark as dirty on mutable borrow
        &mut self.node
    }
}

pub struct CachedNodeManager<K, N, S>
where
    S: Storage<Id = K, Item = N>,
{
    // A fixed-size container that acts as the owner of all nodes.
    // This allows us to hand out read/write guards from a shared reference to the node manager.
    // Wrapped in an Arc so that it can be shared with [`ItemLifecycle`].
    nodes: Arc<[RwLock<NodeWithMetadata<N>>]>,
    // cache, managing the key to node position mapping as well as the item eviction
    cache: quick_cache::sync::Cache<
        K,                // key type to identify cached values
        usize,            // value type to identify node positions in the nodes vector
        UnitWeighter,     // all values are considered to cost the same
        RandomState,      // default hasher
        ItemLifecycle<N>, // tracks and reports evicted items
    >,
    free_list: Mutex<VecDeque<usize>>, // free list of available node positions
    next_empty: AtomicUsize,           // next empty position in nodes vector
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
        let mut nodes = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            // Pre-allocate with default values. This requires W: Default.
            nodes.push(RwLock::new(NodeWithMetadata {
                node: N::default(),
                status: NodeStatus::Clean,
            }));
        }
        let nodes: Arc<[RwLock<NodeWithMetadata<N>>]> = Arc::from(nodes.into_boxed_slice());

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
            free_list: Mutex::new(VecDeque::new()),
            next_empty: AtomicUsize::new(0),
        }
    }

    /// Evicts an entry from the cache, storing it in the storage if `storage_filter` returns
    /// true.
    /// NOTE: this may be done in a separate thread
    fn evict(
        &self,
        entry: (K, usize),
        free_list_guard: &mut std::sync::MutexGuard<'_, VecDeque<usize>>,
    ) -> Result<(), Error> {
        let (key, pos) = entry;
        // If the cache was full, we had to insert an item with the actual key and pos
        // PINNED_POS to trigger eviction. When inserting the the correct key and pos,
        // quick_cache returns the old pos as an evicted item. Therefore we have to skip it
        // here
        if pos == ItemLifecycle::<N>::PINNED_POS {
            return Ok(()); // skip pinned items
        }
        // Get exclusive write access to the node before storing it
        // to ensure that no other thread has a reference to it and
        // avoid risking to lose data.
        #[allow(clippy::readonly_write_lock)]
        let guard = self.nodes[pos].write().unwrap();
        if guard.should_store() {
            self.storage.set(key, &guard)?;
        }
        free_list_guard.push_back(pos);
        Ok(())
    }

    /// Insert an item in the node manager, reusing a free slot if available or evicting an
    /// existing item if the cache is full.
    /// Returns the position of the inserted node in the `nodes` vector.
    fn insert(&self, key: K, node: NodeWithMetadata<N>) -> Result<usize, Error> {
        let pos = if let Some(pos) = self.free_list.lock().unwrap().pop_front() {
            pos
        } else if let pos = self.next_empty.fetch_add(1, Ordering::Relaxed)
            && pos < self.nodes.len()
        {
            // This is not gonna overflow in practice
            pos
        } else {
            // The cache is full, we need to evict an item to make space.
            // The default behavior of quick_cache is to immediately evict every new item
            // until it is seen for a second time. To avoid this behavior, we first insert the
            // special `PINNED_POS` value which is always pinned by
            // [`ItemLifecycle`].
            let evicted = self
                .cache
                .insert_with_lifecycle(key, ItemLifecycle::<N>::PINNED_POS)
                .ok_or(Error::NodeManager(
                    "no available space in cache".to_string(),
                ))?;
            let mut free_list_guard = self.free_list.lock().unwrap();
            self.evict(evicted, &mut free_list_guard)?;

            // Now, there should be a position in the free list. If not, the
            // cache eviction failed (e.g. since all items are pinned) and
            // the insertion cannot proceed.
            // Note: this should never happen as we checked before that the eviction returned an
            // item.
            free_list_guard.pop_front().ok_or(Error::NodeManager(
                "no available space in cache".to_string(),
            ))?
        };

        let mut guard = self.nodes[pos].write().unwrap();
        *guard = node;

        // Insert a new item in cache, evict an old item if necessary
        let evicted = self.cache.insert_with_lifecycle(key, pos);
        if let Some(evicted) = evicted {
            let mut free_list_guard = self.free_list.lock().unwrap();
            self.evict(evicted, &mut free_list_guard)?;
        }
        Ok(pos)
    }
}

impl<K: Eq + Hash + Copy, S> NodeManager for CachedNodeManager<K, Node, S>
where
    S: Storage<Id = K, Item = Node>,
{
    type Id = K;
    type NodeType = Node;

    fn add(&self, node: Self::NodeType) -> Result<Self::Id, Error> {
        let id = self.storage.reserve(&node);
        self.insert(
            id,
            NodeWithMetadata {
                node,
                status: NodeStatus::Dirty,
            },
        )?;
        Ok(id)
    }

    fn get_read_access(
        &self,
        id: Self::Id,
    ) -> Result<RwLockReadGuard<'_, impl Deref<Target = Self::NodeType>>, Error> {
        if let Some(pos) = self.cache.get(&id) {
            Ok(self.nodes[pos].read().unwrap())
        } else {
            // Get node from storage, or return `Error::NotFound` if it doesn't exist.
            let node = self.storage.get(id)?;
            let pos = self.insert(
                id,
                NodeWithMetadata {
                    node,
                    status: NodeStatus::Clean,
                },
            )?;
            Ok(self.nodes[pos].read().unwrap())
        }
    }

    fn get_write_access(
        &self,
        id: Self::Id,
    ) -> Result<RwLockWriteGuard<'_, impl DerefMut<Target = Self::NodeType>>, Error> {
        if let Some(pos) = self.cache.get(&id) {
            Ok(self.nodes[pos].write().unwrap())
        } else {
            // Get node from storage, or return `Error::NotFound` if it doesn't exist.
            let node = self.storage.get(id)?;
            let pos = self.insert(
                id,
                NodeWithMetadata {
                    node,
                    status: NodeStatus::Clean,
                },
            )?;
            Ok(self.nodes[pos].write().unwrap())
        }
    }

    fn delete(&self, id: Self::Id) -> Result<(), Error> {
        if let Some(pos) = self.cache.get(&id) {
            // Get exclusive write access before dropping the node
            // to ensure that no other thread is holding a reference to it.
            let _guard = self.nodes[pos].write().unwrap();
            self.cache.remove(&id);
            let mut free_list = self.free_list.lock().unwrap();
            free_list.push_back(pos);
        }
        self.storage.delete(id)?;
        Ok(())
    }

    fn flush(&self) -> Result<(), crate::error::Error> {
        for (id, pos) in self.cache.iter() {
            let mut entry_guard = self.nodes[pos].write().unwrap();
            // Skip deleted nodes. We expect the free list to be short, so this should be cheap.
            if self.free_list.lock().unwrap().contains(&pos) {
                continue;
            }
            if entry_guard.status == NodeStatus::Dirty {
                self.storage.set(id, &entry_guard.node)?;
                entry_guard.status = NodeStatus::Clean;
            }
        }
        self.storage.flush()?;
        Ok(())
    }
}

/// Manages the lifecycle of cached items, preventing eviction of items currently in use.
pub struct ItemLifecycle<N> {
    nodes: Arc<[RwLock<NodeWithMetadata<N>>]>,
}

impl<W> ItemLifecycle<W> {
    /// Dummy position that is always considered as pinned by [`ItemLifecycle`].
    /// Used to push out other cache entries in case the cache is full.
    const PINNED_POS: usize = usize::MAX;
}

impl<K: Eq + Hash + Copy, N> Lifecycle<K, usize> for ItemLifecycle<N> {
    type RequestState = Option<(K, usize)>;

    /// Checks if an item can be evicted from the cache.
    /// An item is considered pinned if:
    /// - Another thread holds a lock to it
    /// - Its position is set to `PINNED_POS`, which is a dummy position used to explicitly mark
    ///   item as pinned during insertion.
    fn is_pinned(&self, _key: &K, value: &usize) -> bool {
        // NOTE: Another thread may try to acquire a write lock on this node after the function
        // returns, but that should be fine as the the shard containing the item should
        // remain write-locked for the entire eviction process.
        *value == Self::PINNED_POS || self.nodes[*value].try_write().is_err()
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
        predicate::{always, eq},
    };

    use super::*;
    use crate::{
        storage::{self},
        types::{NodeId, NodeType},
    };

    #[test]
    fn cached_node_manager_new_creates_node_manager() {
        let storage = MockCachedNodeManagerStorage::new();
        let cache =
            CachedNodeManager::<NodeId, Node, MockCachedNodeManagerStorage>::new(10, storage);
        assert_eq!(cache.cache.capacity(), 10);
    }

    #[test]
    fn cached_node_manager_evict_stores_entries_in_storage() {
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let mut storage = MockCachedNodeManagerStorage::new();
        storage.expect_reserve().times(1).returning(move |_| id);
        storage
            .expect_set()
            .times(1)
            .with(eq(id), eq(&Node::Empty))
            .returning(|_, _| Ok(()));

        let cache = CachedNodeManager::new(10, storage);
        cache.add(Node::Empty).unwrap();
        // Get mutable reference to `Node` to mark it as dirty
        {
            let _ = &mut **cache.get_write_access(id).unwrap();
        }
        let mut free_list_guard = cache.free_list.lock().unwrap();
        cache.evict((id, 0), &mut free_list_guard).unwrap();
        assert!(free_list_guard.pop_front() == Some(0));
        // Set the status to clean to avoid eviction
        let mut node_guard = cache.nodes[0].write().unwrap();
        node_guard.status = NodeStatus::Clean;
        drop(node_guard);
        cache.evict((id, 0), &mut free_list_guard).unwrap(); // should not store again
        assert!(free_list_guard.pop_front() == Some(0));
        cache
            .evict(
                (id, ItemLifecycle::<Node>::PINNED_POS),
                &mut free_list_guard,
            )
            .unwrap(); // should not store pinned items
        assert!(free_list_guard.is_empty());
    }

    #[test]
    fn cached_node_manager_insert_inserts_items() {
        // Cache is not full, empty list is empty
        {
            let mut storage = MockCachedNodeManagerStorage::new();
            storage.expect_set().never();
            let manager = CachedNodeManager::new(10, storage);
            let expected = NodeWithMetadata {
                node: Node::Empty,
                status: NodeStatus::Dirty,
            };
            let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            let pos = manager.insert(id, expected.clone()).unwrap();
            assert_eq!(pos, 0);
            let guard = manager.nodes[pos].read().unwrap();
            assert_eq!(*guard, expected);
        }
        // Cache is full, empty list is empty
        {
            let mut storage = MockCachedNodeManagerStorage::new();
            storage.expect_set().times(1).returning(|_, _| Ok(()));
            let manager = CachedNodeManager::new(1, storage);
            let expected_node = NodeWithMetadata {
                node: Node::Empty,
                status: NodeStatus::Dirty,
            };
            let id1 = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            let id2 = NodeId::from_idx_and_node_type(1, NodeType::Empty);
            let pos1 = manager.insert(id1, expected_node.clone()).unwrap();
            assert_eq!(pos1, 0);
            let guard = manager.nodes[pos1].read().unwrap();
            assert_eq!(*guard, expected_node);
            drop(guard);
            let pos2 = manager.insert(id2, expected_node.clone()).unwrap();
            assert_eq!(pos2, 0); // same position as first node, which was evicted
            let guard = manager.nodes[pos2].read().unwrap();
            assert_eq!(*guard, expected_node);
        }
        // Cache is not full, empty list is not empty
        {
            let mut storage = MockCachedNodeManagerStorage::new();
            storage.expect_set().never();
            let manager = CachedNodeManager::new(10, storage);
            let expected_node = NodeWithMetadata {
                node: Node::Empty,
                status: NodeStatus::Dirty,
            };
            let id1 = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            let id2 = NodeId::from_idx_and_node_type(1, NodeType::Empty);
            let pos1 = manager.insert(id1, expected_node.clone()).unwrap();
            assert_eq!(pos1, 0);
            let guard = manager.nodes[pos1].read().unwrap();
            assert_eq!(*guard, expected_node);
            drop(guard);
            manager.free_list.lock().unwrap().push_back(pos1); // manually free the position
            let pos2 = manager.insert(id2, expected_node.clone()).unwrap();
            assert_eq!(pos2, 0); // same position as first node, which was freed
            let guard = manager.nodes[pos2].read().unwrap();
            assert_eq!(*guard, expected_node);
        }
    }

    #[test]
    fn cached_node_manager_add_inserts_nodes() {
        let mut storage = MockCachedNodeManagerStorage::new();
        storage
            .expect_reserve()
            .returning(|_| NodeId::from_idx_and_node_type(42, NodeType::Empty));
        storage.expect_get().never(); // Shouldn't query storage on add
        let manager = CachedNodeManager::new(10, storage);
        let node = Node::Empty;
        let id = manager.add(node).unwrap();
        assert_eq!(id, NodeId::from_idx_and_node_type(42, NodeType::Empty));
        let entry = manager.get_read_access(id).unwrap();
        assert!(**entry == Node::Empty);
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
                    status: NodeStatus::Clean,
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
        const NUM_ELEMENTS: u64 = 10;
        let mut storage = MockCachedNodeManagerStorage::new();
        let mut sequence = Sequence::new();
        for i in 0..NUM_ELEMENTS + 1 {
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
            .times(1)
            .with(
                eq(NodeId::from_idx_and_node_type(
                    // NUM_ELEMENT - 1 will be evicted because of infinite reuse distance
                    NUM_ELEMENTS - 1,
                    NodeType::Empty,
                )),
                always(),
            )
            .returning(|_, _| Ok(()));

        let manager = CachedNodeManager::new(NUM_ELEMENTS as usize, storage);

        for i in 0..NUM_ELEMENTS {
            let id = NodeId::from_idx_and_node_type(i, NodeType::Empty);
            let mut entry = manager.get_write_access(id).unwrap();
            {
                let _ = &mut **entry; // Mutable borrow to mark as dirty
            }
            assert!(manager.cache.get(&id).is_some());
        }

        // Retrieving and insert one item more than capacity, triggering eviction of the
        // precedent item.
        let id = NodeId::from_idx_and_node_type(NUM_ELEMENTS, NodeType::Empty);
        let _unused = get_method(&manager, id).unwrap();
    }

    #[test]
    fn cached_node_manager_flush_saves_dirty_nodes_to_storage() {
        const NUM_ELEMENTS: u64 = 10;
        let data = Arc::new(Mutex::new(vec![]));
        let mut storage = MockCachedNodeManagerStorage::new();
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

        let manager = CachedNodeManager::new(NUM_ELEMENTS as usize, storage);
        for _ in 0..NUM_ELEMENTS {
            let _ = manager.add(Node::Empty).unwrap();
        }
        manager.flush().expect("flush should succeed");
    }

    #[test]
    fn cached_node_manager_delete_removes_entry_from_cache_and_storage() {
        let mut storage = MockCachedNodeManagerStorage::new();
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let entry = Node::Inner(Box::default());
        storage.expect_reserve().times(1).returning(move |_| id);
        storage
            .expect_delete()
            .times(1)
            .with(eq(id))
            .returning(|_| Ok(()));

        let manager = CachedNodeManager::new(2, storage);
        let _ = manager.add(entry).unwrap();
        manager.delete(id).unwrap();
        assert!(manager.cache.get(&id).is_none());
        // First node should be inserted at pos 0
        assert!(manager.free_list.lock().unwrap().contains(&0));
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
    fn cached_node_manager_stores_data_in_storage_on_evict() {
        let mut storage = MockCachedNodeManagerStorage::new();
        let mut sequence = Sequence::new();
        storage
            .expect_reserve()
            .times(1)
            .in_sequence(&mut sequence)
            .returning(move |_| NodeId::from_idx_and_node_type(0, NodeType::Empty));
        storage
            .expect_reserve()
            .times(1)
            .in_sequence(&mut sequence)
            .returning(move |_| NodeId::from_idx_and_node_type(1, NodeType::Empty));
        // With unit-size cache, each item is immediately evicted
        storage
            .expect_set()
            .times(1)
            .withf(move |idx, value| {
                *idx == NodeId::from_idx_and_node_type(0, NodeType::Empty) && value == &Node::Empty
            })
            .returning(|_, _| Ok(()));

        let manager = CachedNodeManager::new(1, storage);
        // Insert two nodes to trigger the eviction of the first one
        let _ = manager.add(Node::Empty).unwrap();
        let id = manager.add(Node::Inner(Box::default())).unwrap();

        let entry = manager.get_read_access(id).unwrap();
        assert!(**entry == Node::Inner(Box::default()));
    }

    #[test]
    fn item_lifecycle_is_pinned_checks_lock_and_pinned_pos() {
        let nodes = Arc::from([RwLock::new(NodeWithMetadata {
            node: Node::Empty,
            status: NodeStatus::Clean,
        })]);
        let lifecycle = ItemLifecycle { nodes };

        // Element is not pinned if it can be locked and position is not PINNED_POS
        assert!(!lifecycle.is_pinned(&0, &0));

        // Element is pinned if its position is PINNED_POS
        assert!(lifecycle.is_pinned(&0, &ItemLifecycle::<Node>::PINNED_POS));

        // Element is pinned if it cannot be locked (another thread holds a lock)
        let _guard = lifecycle.nodes[0].write().unwrap(); // Lock item at pos 1
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
        let mut cached_node = NodeWithMetadata {
            node: Node::Empty,
            status: NodeStatus::Clean,
        };
        assert!(cached_node.status != NodeStatus::Dirty);
        let _ = cached_node.deref();
        assert!(cached_node.status == NodeStatus::Clean);
        let _ = cached_node.deref_mut();
        assert!(cached_node.status == NodeStatus::Dirty);
    }

    #[test]
    fn node_with_metadata_storage_filter_returns_true_if_dirty() {
        let mut cached_node = NodeWithMetadata {
            node: Node::Empty,
            status: NodeStatus::Clean,
        };
        assert!(!cached_node.should_store());
        let _ = cached_node.deref_mut();
        assert!(cached_node.should_store());
    }

    mock! {
        pub CachedNodeManagerStorage {}

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

            fn flush(&self) -> Result<(), storage::Error>;
        }
    }

    type GetMethod = fn(
        &CachedNodeManager<NodeId, Node, MockCachedNodeManagerStorage>,
        NodeId,
    ) -> Result<Node, Error>;

    #[rstest_reuse::template]
    #[rstest::rstest]
    #[case::get_read_access((|cache, id| {
        let guard = cache.get_read_access(id)?;
        Ok((**guard).clone())
    }) as GetMethod)]
    #[case::get_write_access((|cache, id| {
        let guard = cache.get_write_access(id)?;
        Ok((**guard).clone())
    }) as GetMethod)]
    fn get_method(#[case] f: GetMethod) {}
}
