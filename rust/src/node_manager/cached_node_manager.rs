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
    sync::{Arc, Mutex, RwLock, RwLockReadGuard, atomic::AtomicUsize},
    vec::Vec,
};

use quick_cache::{Lifecycle, UnitWeighter};

use crate::{
    error::Error,
    node_manager::NodeManager,
    storage::{self, Storage},
    types::Node,
};

/// A [`Node`] with a **status** to store metadata about the node lifecycle.
/// [`NodeWithMetadata`] automatically dereferences to `Node` via the [`Deref`] trait.
/// The node's status is set to [`NodeStatus::Dirty`] when a mutable reference is requested.
/// Accessing a deleted node will panic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeWithMetadata {
    value: Node,
    status: NodeStatus,
}

impl Default for NodeWithMetadata {
    fn default() -> Self {
        NodeWithMetadata {
            value: Node::Empty,
            status: NodeStatus::Clean,
        }
    }
}

/// The status of a [`NodeWithMetadata`].
/// It can be:
/// - `Clean`: the node is in sync with the storage
/// - `Dirty`: the node has been modified and needs to be flushed to storage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    Clean,
    Dirty,
}

impl NodeWithMetadata {
    /// Creates a new [`NodeWithMetadata`] with the given [`Node`] and status.
    pub fn new(value: Node, status: NodeStatus) -> Self {
        NodeWithMetadata { value, status }
    }
}

impl Deref for NodeWithMetadata {
    type Target = Node;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl DerefMut for NodeWithMetadata {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.status = NodeStatus::Dirty; // Mark as dirty on mutable borrow
        &mut self.value
    }
}

/// Dummy position used to notify the cache that the (key, value) pair is pinned and must not be
/// evicted.
const PINNED_POS: usize = usize::MAX;

pub struct CachedNodeManager<K, W, S>
where
    S: Storage<Id = K, Item = W::Target>,
    W: DerefMut,
{
    elements: Arc<Vec<RwLock<W>>>, // the owner of all values
    // cache, managing the key to element position mapping as well as the element eviction
    cache: quick_cache::sync::Cache<
        K,                   // key type to identify cached elements
        usize,               // value type to identify element positions in the elements vector
        UnitWeighter,        // all elements are considered to cost the same
        RandomState,         // default hasher
        ElementLifecycle<W>, // tracks and reports evicted elements
    >,
    free_list: Mutex<VecDeque<usize>>, // free list of available element positions
    next_empty: AtomicUsize,           // next empty position in elements vector
    //storage for managing IDs, fetching missing elements, and saving evicted elements to
    storage: S,
}

impl<K: Eq + Hash + Copy, S, W> CachedNodeManager<K, W, S>
where
    S: Storage<Id = K, Item = W::Target>,
    W: DerefMut + Default,
{
    /// Creates a new [`CachedNodeManager`] with the given capacity and storage backend.
    pub fn new(capacity: usize, storage: S) -> Self {
        let mut elements = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            // Pre-allocate with default values. This requires W: Default.
            elements.push(RwLock::new(W::default()));
        }
        let elements = Arc::new(elements);

        let options = quick_cache::OptionsBuilder::new()
            .estimated_items_capacity(capacity)
            .weight_capacity(capacity as u64) // unit weight per element
            .build()
            .expect("failed to build cache options. Did you provide all the required options?");

        CachedNodeManager {
            elements: elements.clone(),
            storage,
            cache: quick_cache::sync::Cache::with_options(
                options,
                UnitWeighter,
                RandomState::default(),
                ElementLifecycle {
                    elements: elements.clone(),
                },
            ),
            free_list: Mutex::new(VecDeque::new()),
            next_empty: AtomicUsize::new(0),
        }
    }

    /// Evicts an entry from the cache, storing it in the storage if `storage_filter` returns
    /// true.
    /// NOTE: this may be done in a separate thread
    fn evict(&self, entry: Option<(K, usize)>, storage_filter: impl Fn(&W) -> bool) {
        if let Some((key, pos)) = entry {
            // If the cache was full, we had to insert an element with the actual key and pos
            // PINNED_POS to trigger eviction. When inserting the the correct key and pos,
            // quick_cache returns the old pos as an evicted element. Therefore we have to skip it
            // here
            if pos == PINNED_POS {
                return;
            }
            let guard = self.elements[pos].write().unwrap();
            if !storage_filter(&guard) {
                return; // skip elements that should not be stored
            }
            self.storage.set(key, &guard).unwrap();
            self.free_list.lock().unwrap().push_back(pos);
        }
    }

    fn insert_into_free_slot(
        &self,
        key: K,
        item: W,
        storage_filter: impl Fn(&W) -> bool,
    ) -> Result<usize, Error> {
        let mut pos = self
            .free_list
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| {
                self.next_empty
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            });
        if pos >= self.elements.len() {
            self.next_empty
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

            // The cache is full, we need to evict an element to make space.
            // Element is inserted as pinned to avoid being immediately evicted
            let evicted = self.cache.insert_with_lifecycle(key, PINNED_POS);
            self.evict(evicted, &storage_filter);

            // Now, there should be an element in the free list. If not, the
            // cache eviction failed (e.g. since all elements are pinned) and
            // the insertion cannot proceed.
            pos = self
                .free_list
                .lock()
                .unwrap()
                .pop_front()
                .ok_or(storage::Error::NotFound)?;
        }
        let mut guard = self.elements[pos]
            .write()
            .map_err(|_| storage::Error::NotFound)?;
        *guard = item;

        // Include new element in cache, evict old elements if necessary.
        let evicted = self.cache.insert_with_lifecycle(key, pos);
        self.evict(evicted, storage_filter);
        Ok(pos)
    }
}

impl<K: Eq + Hash + Copy, S> NodeManager for CachedNodeManager<K, NodeWithMetadata, S>
where
    S: Storage<Id = K, Item = Node>,
{
    type Id = K;
    type NodeType = Node;

    fn add(&self, item: Self::NodeType) -> Result<Self::Id, Error> {
        let id = self.storage.reserve(&item);
        self.insert_into_free_slot(id, NodeWithMetadata::new(item, NodeStatus::Dirty), |n| {
            n.status == NodeStatus::Dirty
        })?;
        Ok(id)
    }

    fn get_read_access(
        &self,
        id: Self::Id,
    ) -> Result<RwLockReadGuard<'_, impl Deref<Target = Self::NodeType>>, Error> {
        match self.cache.get(&id) {
            Some(pos) => return Ok(self.elements[pos].read().unwrap()),
            None => {
                let item = self.storage.get(id)?; // check if element exists in storage
                let pos = self.insert_into_free_slot(
                    id,
                    NodeWithMetadata::new(item, NodeStatus::Clean),
                    |n| n.status == NodeStatus::Dirty,
                )?;
                return Ok(self.elements[pos].read().unwrap());
            }
        }
    }

    fn get_write_access(
        &self,
        id: Self::Id,
    ) -> Result<std::sync::RwLockWriteGuard<'_, impl DerefMut<Target = Self::NodeType>>, Error>
    {
        match self.cache.get(&id) {
            Some(pos) => Ok(self.elements[pos].write().unwrap()),
            None => {
                let item = self.storage.get(id)?; // check if element exists in storage
                let pos = self.insert_into_free_slot(
                    id,
                    NodeWithMetadata::new(item, NodeStatus::Clean),
                    |n| n.status == NodeStatus::Dirty,
                )?;
                Ok(self.elements[pos].write().unwrap())
            }
        }
    }

    fn delete(&self, id: Self::Id) -> Result<(), Error> {
        if let Some(pos) = self.cache.get(&id) {
            // get exclusive write access before dropping the element
            let _guard = self.elements[pos].write().unwrap();
            self.cache.remove(&id);
            let mut free_list = self.free_list.lock().unwrap();
            free_list.push_back(pos);
        }
        self.storage
            .delete(id)
            .map_err(|_| Error::Storage(storage::Error::NotFound))?;
        Ok(())
    }

    fn flush(&self) -> Result<(), crate::error::Error> {
        for (id, pos) in self.cache.iter() {
            let mut entry_guard = self.elements[pos].write().unwrap();
            if self.free_list.lock().unwrap().contains(&pos) {
                continue; // skip deleted elements
            }
            if entry_guard.status == NodeStatus::Dirty {
                self.storage.set(id, &entry_guard.value)?;
                entry_guard.status = NodeStatus::Clean;
            }
        }
        self.storage.flush()?;
        Ok(())
    }
}

/// Manages the lifecycle of cached elements, preventing eviction of elements currently in use.
pub struct ElementLifecycle<W> {
    elements: Arc<Vec<RwLock<W>>>,
}

impl<K: Eq + Hash + Copy, W> Lifecycle<K, usize> for ElementLifecycle<W> {
    type RequestState = Option<(K, usize)>;

    /// Checks if an element can be evicted from the cache.
    /// An element is considered pinned if:
    /// - Another thread holds a lock to it
    /// - Its position is set to `PINNED_POS`, which is a dummy position used to explicitly mark
    ///   elements as pinned during insertion.
    fn is_pinned(&self, _key: &K, value: &usize) -> bool {
        // NOTE: another thread may ask for a write lock while this is checked, but that should be
        // fine as the the shard containing the evicting element is write locked at this stage.
        *value == PINNED_POS || self.elements[*value].try_write().is_err()
    }

    /// No-op
    fn begin_request(&self) -> Self::RequestState {
        None
    }

    /// Records the key and value of an evicted element in the request state.
    /// This is useful for inspecting which elements were evicted after an insertion.
    fn on_evict(&self, state: &mut Self::RequestState, _key: K, _value: usize) {
        *state = Some((_key, _value));
    }
}

impl<W> Clone for ElementLifecycle<W> {
    fn clone(&self) -> Self {
        ElementLifecycle {
            elements: self.elements.clone(),
        }
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
        types::{InnerNode, NodeId, NodeType},
    };

    #[test]
    fn cached_node_manager_new_creates_node_manager() {
        let storage = MockStorage::new();
        let cache = CachedNodeManager::<NodeId, NodeWithMetadata, MockStorage>::new(10, storage);
        assert_eq!(cache.cache.capacity(), 10);
    }

    #[test]
    fn cached_node_manager_evict_stores_entries_in_storage() {
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let mut storage = MockStorage::new();
        storage.expect_reserve().times(1).returning(move |_| id);
        storage
            .expect_set()
            .times(1)
            .with(eq(id), eq(&Node::Empty))
            .returning(|_, _| Ok(()));

        let cache = CachedNodeManager::new(10, storage);
        cache.add(Node::Empty).expect("set should succeed");
        cache.evict(Some((id, 0)), |_| true);
        cache.evict(Some((id, 0)), |_| false); // should not store again
        cache.evict(Some((id, PINNED_POS)), |_| true); // should not store pinned element
    }

    #[test]
    fn cached_node_manager_insert_into_free_slot_inserts_elements_in_cache() {
        // Cache is not full, empty list is empty
        {
            let cache = CachedNodeManager::new(10, MockStorage::new());
            let node = NodeWithMetadata::new(Node::Empty, NodeStatus::Dirty);
            let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            let pos = cache
                .insert_into_free_slot(id, node, |_| true)
                .expect("insert should succeed");
            assert_eq!(pos, 0);
        }
        // Cache is full, empty list is empty
        {
            let mut storage = MockStorage::new();
            storage.expect_set().times(1).returning(|_, _| Ok(()));
            let cache = CachedNodeManager::new(1, storage);
            let node = NodeWithMetadata::new(Node::Empty, NodeStatus::Dirty);
            let id1 = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            let id2 = NodeId::from_idx_and_node_type(1, NodeType::Empty);
            let pos1 = cache
                .insert_into_free_slot(id1, node.clone(), |_| true)
                .expect("insert should succeed");
            assert_eq!(pos1, 0);
            let pos2 = cache
                .insert_into_free_slot(id2, node, |_| true)
                .expect("insert should succeed");
            assert_eq!(pos2, 0); // same position as first element, which was evicted
        }
        // Cache is not full, empty list is not empty
        {
            let cache = CachedNodeManager::new(10, MockStorage::new());
            let node = NodeWithMetadata::new(Node::Empty, NodeStatus::Dirty);
            let id1 = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            let id2 = NodeId::from_idx_and_node_type(1, NodeType::Empty);
            let pos1 = cache
                .insert_into_free_slot(id1, node.clone(), |_| true)
                .expect("insert should succeed");
            assert_eq!(pos1, 0);
            cache.free_list.lock().unwrap().push_back(pos1); // manually free the position
            let pos2 = cache
                .insert_into_free_slot(id2, node, |_| true)
                .expect("insert should succeed");
            assert_eq!(pos2, 0); // same position as first element, which was freed
        }
    }

    #[test]
    fn cached_node_manager_add_inserts_elements_in_cache() {
        let mut storage = MockStorage::new();
        storage
            .expect_reserve()
            .returning(|_| NodeId::from_idx_and_node_type(42, NodeType::Empty));
        storage.expect_get().never(); // Cache get should not call storage get
        let cache = CachedNodeManager::new(10, storage);
        let node = Node::Empty;
        let id = cache.add(node).expect("set should succeed");
        assert_eq!(id, NodeId::from_idx_and_node_type(42, NodeType::Empty));
        let entry = cache.get_read_access(id).expect("get should succeed");
        assert!(**entry == Node::Empty);
    }

    #[rstest_reuse::apply(get_method)]
    fn cached_node_manager_get_methods_return_cached_entry(#[case] get_method: GetMethod) {
        let expected_entry = Node::Empty;
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let mut storage = MockStorage::new();
        storage.expect_get().never(); // Cache get should not call storage get
        let cache = CachedNodeManager::new(10, storage);
        let _ = cache
            .insert_into_free_slot(
                id,
                NodeWithMetadata::new(expected_entry.clone(), NodeStatus::Clean),
                |_| true,
            )
            .expect("insert should succeed");
        {
            let entry = get_method(&cache, id).expect("get should succeed");
            assert!(entry == expected_entry);
        }
    }

    #[rstest_reuse::apply(get_method)]
    fn cached_node_manager_get_methods_return_existing_entry_from_storage_if_not_in_cache(
        #[case] get_method: GetMethod,
    ) {
        let expected_entry = Node::Empty;
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let mut storage = MockStorage::new();
        storage.expect_get().times(1).with(eq(id)).returning({
            let expected_entry = expected_entry.clone();
            move |_| Ok(expected_entry.clone())
        });

        let cache = CachedNodeManager::new(10, storage);
        let entry = get_method(&cache, id).expect("get should succeed");
        assert!(entry == expected_entry);
    }

    #[rstest_reuse::apply(get_method)]
    fn cached_node_manager_get_methods_returns_error_if_node_id_does_not_exist(
        #[case] get_method: GetMethod,
    ) {
        let mut storage = MockStorage::new();
        storage
            .expect_get()
            .returning(|_| Err(storage::Error::NotFound));

        let cache = CachedNodeManager::new(10, storage);
        let res = get_method(&cache, NodeId::from_idx_and_node_type(0, NodeType::Empty));
        assert!(res.is_err());
        assert!(matches!(
            res.err().unwrap(),
            Error::Storage(storage::Error::NotFound)
        ));
    }

    #[rstest_reuse::apply(get_method)]
    fn cached_node_manager_get_methods_always_insert_element_in_cache(
        #[case] get_method: GetMethod,
    ) {
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

        let cache = CachedNodeManager::new(NUM_ELEMENTS as usize, storage);

        for i in 0..NUM_ELEMENTS {
            let id = NodeId::from_idx_and_node_type(i, NodeType::Empty);
            let mut entry = cache.get_write_access(id).expect("get should succeed");
            {
                let _ = &mut **entry; // Mutable borrow to mark as dirty
            }
            assert!(cache.cache.get(&id).is_some());
        }

        // Retrieving and insert one element more than capacity, triggering eviction of the
        // precedent element.
        let id = NodeId::from_idx_and_node_type(NUM_ELEMENTS, NodeType::Empty);
        let _unused = get_method(&cache, id).expect("get should succeed");
    }

    #[test]
    fn cached_node_manager_flush_saves_dirty_entries_to_storage() {
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

        let cache = CachedNodeManager::new(NUM_ELEMENTS as usize, storage);

        for _ in 0..NUM_ELEMENTS {
            let _ = cache.add(Node::Empty).expect("set should succeed");
        }

        cache.flush().expect("flush should succeed");
    }

    #[test]
    fn cached_node_manager_delete_removes_entry_from_cache_and_storage() {
        let mut storage = MockStorage::new();
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        let entry = Node::Inner(Box::new(InnerNode::default()));
        storage.expect_reserve().times(1).returning(move |_| id);
        storage
            .expect_delete()
            .times(1)
            .with(eq(id))
            .returning(|_| Ok(()));

        let cache = CachedNodeManager::new(2, storage);
        let _ = cache.add(entry).expect("add should succeed");
        cache.delete(id).expect("delete should succeed");
        assert!(cache.cache.get(&id).is_none());
        // First element should be inserted at pos 0
        assert!(cache.free_list.lock().unwrap().contains(&0));
    }

    #[test]
    fn cached_node_manager_delete_fails_on_storage_error() {
        let mut storage = MockStorage::new();
        let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
        storage.expect_reserve().times(1).returning(move |_| id);
        storage
            .expect_delete()
            .times(1)
            .with(eq(id))
            .returning(|_| Err(storage::Error::NotFound));

        let cache = CachedNodeManager::new(2, storage);
        let _ = cache.add(Node::Empty).expect("add should succeed");
        let res = cache.delete(id);
        assert!(res.is_err());
        assert!(matches!(
            res.unwrap_err(),
            Error::Storage(storage::Error::NotFound)
        ));
    }

    #[test]
    fn cached_node_manager_stores_data_in_storage_on_evict() {
        let mut storage = MockStorage::new();
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
        // With unit-size cache, each element is immediately evicted
        storage
            .expect_set()
            .times(1)
            .withf(move |idx, value| {
                *idx == NodeId::from_idx_and_node_type(0, NodeType::Empty) && value == &Node::Empty
            })
            .returning(|_, _| Ok(()));

        let cache = CachedNodeManager::new(1, storage);
        // Insert two elements to trigger the eviction of the first one
        let _ = cache.add(Node::Empty).expect("set should succeed");
        let id = cache
            .add(Node::Inner(Box::new(InnerNode::default())))
            .expect("set should succeed");

        let entry = cache.get_read_access(id).expect("get should succeed");
        assert!(**entry == Node::Inner(Box::new(InnerNode::default())));
    }

    #[test]
    fn element_lifecycle_is_pinned_checks_lock_and_pinned_pos() {
        let elements = Arc::new(vec![RwLock::new(1)]);
        let lifecycle = ElementLifecycle { elements };

        // Element is not pinned if it can be locked and position is not PINNED_POS
        assert!(!lifecycle.is_pinned(&0, &0));

        // Element is pinned if its position is PINNED_POS
        assert!(lifecycle.is_pinned(&0, &PINNED_POS));

        // Element is pinned if it cannot be locked (another thread holds a lock)
        let _guard = lifecycle.elements[0].write().unwrap(); // Lock element at pos 1
        assert!(lifecycle.is_pinned(&0, &0));
    }

    #[test]
    fn element_lifecycle_on_evict_records_evicted_element() {
        let elements = Arc::new(vec![RwLock::new(1)]);
        let lifecycle = ElementLifecycle { elements };
        let mut state = lifecycle.begin_request();
        assert!(state.is_none());
        lifecycle.on_evict(&mut state, 42, 0);
        assert_eq!(state, Some((42, 0)));
    }

    #[test]
    fn node_with_metadata_sets_dirty_flag_on_deref_mut() {
        let mut cached_node = NodeWithMetadata::new(Node::Empty, NodeStatus::Clean);
        assert!(cached_node.status != NodeStatus::Dirty);
        let _ = cached_node.deref();
        assert!(cached_node.status == NodeStatus::Clean);
        let _ = cached_node.deref_mut();
        assert!(cached_node.status == NodeStatus::Dirty);
    }

    type GetMethod = fn(
        &CachedNodeManager<NodeId, NodeWithMetadata, MockStorage>,
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
