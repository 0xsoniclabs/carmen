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
    hash::Hash,
    ops::{Deref, DerefMut},
};

use dashmap::DashSet;
use quick_cache::{
    Lifecycle, OptionsBuilder, Weighter,
    sync::{GuardResult, PlaceholderGuard},
};

use crate::{
    error::Error,
    node_manager::{
        NodeManager,
        lock_cache::{LockCache, OnEvict},
    },
    storage::{Checkpointable, Storage},
    sync::*,
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

struct EvictToStorage<S> {
    storage: S,
}

impl<S> OnEvict<S::Id, NodeWithMetadata<S::Item>> for EvictToStorage<S>
where
    S: Storage + Send + Sync,
{
    fn on_evict(&self, key: S::Id, value: NodeWithMetadata<S::Item>) {
        if value.is_dirty {
            self.storage.set(key, &value).unwrap(); // FIXME
        }
    }
}

pub struct CachedNodeManager<K, N, S>
where
    S: Storage<Id = K, Item = N>,
{
    nodes: LockCache<K, NodeWithMetadata<N>>,
    //storage for managing IDs, fetching missing nodes, and saving evicted nodes to
    storage: Arc<EvictToStorage<S>>, // TODO Rename
}

impl<K: Eq + Hash + Copy + std::fmt::Debug, S, N> CachedNodeManager<K, N, S>
where
    S: Storage<Id = K, Item = N> + Send + Sync + 'static,
    N: Default + Clone,
{
    /// Creates a new [`CachedNodeManager`] with the given capacity and storage backend.
    pub fn new(capacity: usize, storage: S) -> Self {
        let storage = Arc::new(EvictToStorage { storage });
        CachedNodeManager {
            nodes: LockCache::new(
                capacity,
                storage.clone() as Arc<dyn OnEvict<K, NodeWithMetadata<N>>>,
            ),
            storage,
        }
    }
}

impl<K: Eq + Hash + Copy + std::fmt::Debug, N, S> NodeManager for CachedNodeManager<K, N, S>
where
    S: Storage<Id = K, Item = N> + 'static,
    N: Default + Clone,
{
    type Id = K;
    type NodeType = N;

    fn add(&self, node: Self::NodeType) -> Result<Self::Id, Error> {
        let id = self.storage.storage.reserve(&node);
        let _guard = self.nodes.get_read_access_or_insert(id, move || {
            Ok(NodeWithMetadata {
                node: node.clone(),
                is_dirty: true,
            })
        })?;
        Ok(id)
    }

    fn get_read_access(
        &self,
        id: Self::Id,
    ) -> Result<RwLockReadGuard<'_, impl Deref<Target = Self::NodeType>>, Error> {
        let lock = self.nodes.get_read_access_or_insert(id, || {
            let node = self.storage.storage.get(id)?;
            Ok(NodeWithMetadata {
                node,
                is_dirty: false,
            })
        })?;
        Ok(lock)
    }

    fn get_write_access(
        &self,
        id: Self::Id,
    ) -> Result<RwLockWriteGuard<'_, impl DerefMut<Target = Self::NodeType>>, Error> {
        let lock = self.nodes.get_write_access_or_insert(id, || {
            let node = self.storage.storage.get(id)?;
            Ok(NodeWithMetadata {
                node,
                is_dirty: false,
            })
        })?;
        Ok(lock)
    }

    fn delete(&self, id: Self::Id) -> Result<(), Error> {
        self.nodes.remove(id);
        Ok(())
    }
}

impl<K: Eq + Hash + Copy + std::fmt::Debug, N, S> Checkpointable for CachedNodeManager<K, N, S>
where
    S: Storage<Id = K, Item = N> + Checkpointable,
    N: Default + Clone,
{
    fn checkpoint(&self) -> Result<(), crate::storage::Error> {
        for (id, lock) in self.nodes.iter() {
            let mut guard = lock.write().unwrap();
            if guard.is_dirty {
                self.storage.storage.set(id, &guard.node)?;
                guard.is_dirty = false;
            }
        }
        self.storage.storage.checkpoint()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // use std::{path::Path, sync::Mutex};

    // use mockall::{
    //     Sequence, mock,
    //     predicate::{always, eq, ne},
    // };

    // use super::*;
    // use crate::{
    //     database::verkle::variants::managed::{EmptyNode, Node, NodeId, NodeType},
    //     storage::{self},
    // };

    // #[test]
    // fn cached_node_manager_new_creates_node_manager() {
    //     let storage = MockCachedNodeManagerStorage::new();
    //     let manager =
    //         CachedNodeManager::<NodeId, Node, MockCachedNodeManagerStorage>::new(10, storage);
    //     assert_eq!(manager.cache.capacity(), 10);
    //     assert_eq!(manager.nodes.len(), 11);
    //     assert_eq!(
    //         manager.free_slots.len(),
    //         11, // All slots + 1
    //     );
    // }

    // #[test]
    // fn cached_node_manager_evict_saves_dirty_nodes_in_storage() {
    //     let id1 = NodeId::from_idx_and_node_type(0, NodeType::Leaf2);
    //     let id2 = NodeId::from_idx_and_node_type(1, NodeType::Leaf2);
    //     let mut storage = MockCachedNodeManagerStorage::new();
    //     storage
    //         .expect_set()
    //         .times(1)
    //         .with(eq(id1), always())
    //         .returning(|_, _| Ok(()));

    //     let manager = CachedNodeManager::new(10, storage);
    //     // Manually insert two nodes
    //     *manager.nodes[0].write().unwrap() = NodeWithMetadata {
    //         node: Node::Leaf2(Box::default()),
    //         is_dirty: true,
    //     };
    //     *manager.nodes[1].write().unwrap() = NodeWithMetadata {
    //         node: Node::Leaf2(Box::default()),
    //         is_dirty: false,
    //     };
    //     manager.cache.insert(id1, 0);
    //     manager.cache.insert(id2, 1);

    //     // This should be evicted as it is dirty.
    //     manager.on_evict((id1, 0)).unwrap();
    //     assert!(manager.free_slots.contains(&0));
    //     assert!(**manager.nodes[0].read().unwrap() == Node::default()); // node reset to default
    //     // This should not be evicted as it is clean.
    //     manager.on_evict((id2, 1)).unwrap();
    //     assert!(manager.free_slots.contains(&1));
    // }

    // #[test]
    // fn cached_node_manager_insert_inserts_items() {
    //     // Cache is not full
    //     {
    //         let mut storage = MockCachedNodeManagerStorage::new();
    //         storage.expect_set().never();
    //         let manager = CachedNodeManager::new(10, storage);
    //         let expected = NodeWithMetadata {
    //             node: Node::Leaf2(Box::default()),
    //             is_dirty: true,
    //         };
    //         let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
    //         let pos = manager.insert(id, expected.clone()).unwrap();
    //         let guard = manager.nodes[pos].read().unwrap();
    //         assert_eq!(*guard, expected);
    //         assert!(!manager.free_slots.contains(&pos));
    //     }
    //     // Cache is full, empty list is empty
    //     {
    //         let mut storage = MockCachedNodeManagerStorage::new();
    //         storage.expect_set().times(1).returning(|_, _| Ok(()));
    //         // Create manager that only fits two nodes.
    //         let manager = CachedNodeManager::new(2, storage);
    //         let expected_node = NodeWithMetadata {
    //             node: Node::Empty(EmptyNode),
    //             is_dirty: true,
    //         };
    //         let id1 = NodeId::from_idx_and_node_type(0, NodeType::Empty);
    //         let id2 = NodeId::from_idx_and_node_type(1, NodeType::Empty);
    //         let id3 = NodeId::from_idx_and_node_type(2, NodeType::Empty);
    //         let pos1 = manager.insert(id1, expected_node.clone()).unwrap();
    //         let guard = manager.nodes[pos1].read().unwrap();
    //         assert_eq!(*guard, expected_node);
    //         drop(guard);
    //         let pos2 = manager.insert(id2, expected_node.clone()).unwrap();
    //         let guard = manager.nodes[pos2].read().unwrap();
    //         assert_eq!(*guard, expected_node);
    //         drop(guard);
    //         let pos3 = manager.insert(id3, expected_node.clone()).unwrap();
    //         let guard = manager.nodes[pos3].read().unwrap();
    //         assert_eq!(*guard, expected_node);
    //         assert!(manager.free_slots.len() == 1);
    //     }
    // }

    // #[test]
    // fn cached_node_manager_add_reserves_id_and_inserts_nodes() {
    //     let expected_id = NodeId::from_idx_and_node_type(42, NodeType::Leaf2);
    //     let node = Node::Leaf256(Box::default());
    //     let mut storage = MockCachedNodeManagerStorage::new();
    //     storage.expect_reserve().returning(move |_| expected_id);
    //     storage.expect_get().never(); // Shouldn't query storage on add
    //     let manager = CachedNodeManager::new(10, storage);
    //     let id = manager.add(node.clone()).unwrap();
    //     assert_eq!(id, expected_id);
    //     let pos = manager.cache.get(&id).unwrap();
    //     assert!(manager.nodes[pos].read().unwrap().is_dirty);
    //     assert_eq!(manager.nodes[pos].read().unwrap().node, node);
    // }

    // #[rstest_reuse::apply(get_method)]
    // fn cached_node_manager_get_methods_return_cached_entry(#[case] get_method: GetMethod) {
    //     let expected_entry = Node::Empty(EmptyNode);
    //     let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
    //     let mut storage = MockCachedNodeManagerStorage::new();
    //     storage.expect_get().never(); // Shouldn't query storage if entry is in cache
    //     let manager = CachedNodeManager::new(10, storage);
    //     let _ = manager
    //         .insert(
    //             id,
    //             NodeWithMetadata {
    //                 node: expected_entry.clone(),
    //                 is_dirty: false,
    //             },
    //         )
    //         .unwrap();
    //     let entry = get_method(&manager, id).unwrap();
    //     assert!(entry == expected_entry);
    // }

    // #[rstest_reuse::apply(get_method)]
    // fn cached_node_manager_get_methods_return_existing_entry_from_storage_if_not_in_cache(
    //     #[case] get_method: GetMethod,
    // ) {
    //     let expected_entry = Node::Empty(EmptyNode);
    //     let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
    //     let mut storage = MockCachedNodeManagerStorage::new();
    //     storage.expect_get().times(1).with(eq(id)).returning({
    //         let expected_entry = expected_entry.clone();
    //         move |_| Ok(expected_entry.clone())
    //     });

    //     let manager = CachedNodeManager::new(10, storage);
    //     let entry = get_method(&manager, id).unwrap();
    //     assert!(entry == expected_entry);
    // }

    // #[rstest_reuse::apply(get_method)]
    // fn cached_node_manager_get_methods_returns_error_if_node_id_does_not_exist(
    //     #[case] get_method: GetMethod,
    // ) {
    //     let mut storage = MockCachedNodeManagerStorage::new();
    //     storage
    //         .expect_get()
    //         .returning(|_| Err(storage::Error::NotFound));

    //     let manager = CachedNodeManager::new(10, storage);
    //     let res = get_method(&manager, NodeId::from_idx_and_node_type(0, NodeType::Empty));
    //     assert!(res.is_err());
    //     assert!(matches!(
    //         res.err().unwrap(),
    //         Error::Storage(storage::Error::NotFound)
    //     ));
    // }

    // #[rstest_reuse::apply(get_method)]
    // fn cached_node_manager_get_methods_always_insert_node_in_cache_when_retrieved_from_storage(
    //     #[case] get_method: GetMethod,
    // ) {
    //     const NUM_NODES: u64 = 10;
    //     let mut storage = MockCachedNodeManagerStorage::new();
    //     let mut sequence = Sequence::new();
    //     for i in 0..NUM_NODES + 1 {
    //         // 1 item more than capacity
    //         storage
    //             .expect_get()
    //             .times(1)
    //             .in_sequence(&mut sequence)
    //             .with(eq(NodeId::from_idx_and_node_type(i, NodeType::Empty)))
    //             .returning(move |_| Ok(Node::Empty(EmptyNode)));
    //     }
    //     storage
    //         .expect_set()
    //         .with(
    //             ne(NodeId::from_idx_and_node_type(
    //                 // The NUM_NODES-th node will be evicted because of infinite reuse distance
    //                 NUM_NODES,
    //                 NodeType::Empty,
    //             )),
    //             always(),
    //         )
    //         .returning(|_, _| Ok(()));

    //     let manager = CachedNodeManager::new(NUM_NODES as usize, storage);

    //     for i in 0..NUM_NODES {
    //         let id = NodeId::from_idx_and_node_type(i, NodeType::Empty);
    //         let mut entry = manager.get_write_access(id).unwrap();
    //         {
    //             let _: &mut Node = &mut entry; // Mutable borrow to mark as dirty
    //         }
    //         assert!(manager.cache.get(&id).is_some());
    //     }

    //     // Retrieving and insert one item more than capacity, triggering eviction of the
    //     // precedent item.
    //     let id = NodeId::from_idx_and_node_type(NUM_NODES, NodeType::Empty);
    //     let _unused = get_method(&manager, id).unwrap();
    // }

    // #[test]
    // fn cached_node_manager_checkpoint_saves_dirty_nodes_to_storage() {
    //     const NUM_NODES: u64 = 10;
    //     let data = Arc::new(Mutex::new(vec![]));
    //     let mut storage = MockCachedNodeManagerStorage::new();
    //     let mut sequence = Sequence::new();
    //     for i in 0..NUM_NODES {
    //         storage
    //             .expect_reserve()
    //             .times(1)
    //             .in_sequence(&mut sequence)
    //             .returning({
    //                 let data = data.clone();
    //                 move |node| {
    //                     data.lock().unwrap().push(node.clone());
    //                     NodeId::from_idx_and_node_type(i, NodeType::Empty)
    //                 }
    //             });
    //         storage
    //             .expect_set()
    //             .times(1)
    //             .with(
    //                 eq(NodeId::from_idx_and_node_type(i, NodeType::Empty)),
    //                 eq(Node::Empty(EmptyNode)),
    //             )
    //             .returning(move |_, _| Ok(()));
    //     }
    //     storage.expect_checkpoint().times(1).returning(|| Ok(()));

    //     let manager = CachedNodeManager::new(NUM_NODES as usize, storage);
    //     for _ in 0..NUM_NODES {
    //         // Newly added nodes are always dirty
    //         let _ = manager.add(Node::Empty(EmptyNode)).unwrap();
    //     }
    //     manager.checkpoint().expect("checkpoint should succeed");
    // }

    // #[test]
    // fn cached_node_manager_delete_removes_entry_from_cache_and_storage() {
    //     let mut storage = MockCachedNodeManagerStorage::new();
    //     let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);
    //     let entry = Node::Inner(Box::default());
    //     storage.expect_reserve().times(1).returning(move |_| id);
    //     storage
    //         .expect_delete()
    //         .times(1)
    //         .with(eq(id))
    //         .returning(|_| Ok(()));

    //     let manager = CachedNodeManager::new(2, storage);
    //     let _ = manager.add(entry).unwrap();
    //     let _ = manager.cache.get(&id).expect("entry should be in cache");
    //     manager.delete(id).unwrap();
    //     assert!(manager.cache.get(&id).is_none());
    //     assert!(
    //         manager.free_slots.contains(&0),
    //         "Node position 0 should be in free list after deletion"
    //     );
    //     assert!(**manager.nodes[0].read().unwrap() == Node::default()); // node reset to default
    // }

    // #[test]
    // fn cached_node_manager_delete_fails_on_storage_error() {
    //     let mut storage = MockCachedNodeManagerStorage::new();
    //     let id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
    //     storage.expect_reserve().times(1).returning(move |_| id);
    //     storage
    //         .expect_delete()
    //         .times(1)
    //         .with(eq(id))
    //         .returning(|_| Err(storage::Error::NotFound));

    //     let manager = CachedNodeManager::new(2, storage);
    //     let _ = manager.add(Node::Empty(EmptyNode)).unwrap();
    //     let res = manager.delete(id);
    //     assert!(res.is_err());
    //     assert!(matches!(
    //         res.unwrap_err(),
    //         Error::Storage(storage::Error::NotFound)
    //     ));
    // }

    // #[test]
    // fn item_lifecycle_is_pinned_checks_lock_and_pinned_pos() {
    //     let nodes = Arc::from([RwLock::new(NodeWithMetadata {
    //         node: Node::Empty,
    //         is_dirty: false,
    //     })]);
    //     let lifecycle = ItemLifecycle { nodes };

    //     // Element is not pinned if it can be locked and position is not PINNED_POS
    //     assert!(!lifecycle.is_pinned(&0, &0));

    //     // Element is pinned if it cannot be locked (another thread holds a lock)
    //     let _guard = lifecycle.nodes[0].write().unwrap(); // Lock item at pos 0
    //     assert!(lifecycle.is_pinned(&0, &0));
    // }

    // #[test]
    // fn item_lifecycle_on_evict_records_evicted_items() {
    //     let nodes: Arc<[RwLock<NodeWithMetadata<Node>>]> = Arc::from(vec![].into_boxed_slice());
    //     let lifecycle = ItemLifecycle { nodes };
    //     let mut state = lifecycle.begin_request();
    //     assert!(state.is_none());
    //     lifecycle.on_evict(&mut state, 42, 0);
    //     assert_eq!(state, Some((42, 0)));
    // }

    // #[test]
    // fn node_with_metadata_sets_dirty_flag_on_deref_mut() {
    //     let mut node = NodeWithMetadata {
    //         node: Node::Empty,
    //         is_dirty: false,
    //     };
    //     assert!(!node.is_dirty);
    //     let _ = node.deref();
    //     assert!(!node.is_dirty);
    //     let _ = node.deref_mut();
    //     assert!(node.is_dirty);
    // }

    // mock! {
    //     pub CachedNodeManagerStorage {}

    //     impl Checkpointable for CachedNodeManagerStorage {
    //         fn checkpoint(&self) -> Result<(), storage::Error>;
    //     }

    //     impl Storage for CachedNodeManagerStorage {
    //         type Id = NodeId;
    //         type Item = Node;

    //         fn open(_path: &Path) -> Result<Self, storage::Error>
    //         where
    //             Self: Sized;

    //         fn get(&self, _id: <Self as Storage>::Id) -> Result<<Self as Storage>::Item,
    // storage::Error>;

    //         fn reserve(&self, _item: &<Self as Storage>::Item) -> <Self as Storage>::Id;

    //         fn set(&self, _id: <Self as Storage>::Id, _item: &<Self as Storage>::Item) ->
    // Result<(), storage::Error>;

    //         fn delete(&self, _id: <Self as Storage>::Id) -> Result<(), storage::Error>;
    //     }
    // }

    // /// Type alias for a closure that calls either `get_read_access` or `get_write_access`
    // type GetMethod = fn(
    //     &CachedNodeManager<NodeId, Node, MockCachedNodeManagerStorage>,
    //     NodeId,
    // ) -> Result<Node, Error>;

    // /// Reusable rstest template to test both `get_read_access` and `get_write_access`
    // #[rstest_reuse::template]
    // #[rstest::rstest]
    // #[case::get_read_access((|manager, id| {
    //     let guard = manager.get_read_access(id)?;
    //     Ok((**guard).clone())
    // }) as GetMethod)]
    // #[case::get_write_access((|manager, id| {
    //     let guard = manager.get_write_access(id)?;
    //     Ok((**guard).clone())
    // }) as GetMethod)]
    // fn get_method(#[case] f: GetMethod) {}
}
