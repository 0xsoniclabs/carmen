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
    path::Path,
};

use crate::{
    error::{BTResult, Error},
    node_manager::{
        NodeManager,
        lock_cache::{EvictionHooks, LockCache},
    },
    storage::{Checkpointable, RootIdProvider, Storage},
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
    types::{HasDeltaVariant, HasEmptyId, HasEmptyNode},
};

/// A wrapper which dereferences to `N` and additionally stores its dirty status,
/// indicating whether it needs to be flushed to storage.
/// The node status is set to dirty when a mutable reference is requested.
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

/// A wrapper around a storage backend that implements the [`EvictionHooks`] trait
struct StorageEvictionHandler<S: Storage> {
    storage: S,
    is_pinned_predicate: fn(&S::Item) -> bool,
}

impl<S> EvictionHooks for StorageEvictionHandler<S>
where
    S: Storage,
{
    type Key = S::Id;
    type Value = NodeWithMetadata<S::Item>;

    fn is_pinned(&self, _key: &Self::Key, value: &Self::Value) -> bool {
        (self.is_pinned_predicate)(value)
    }

    /// Stores the evicted node in the underlying storage if it is dirty.
    fn on_evict(&self, key: S::Id, node: NodeWithMetadata<S::Item>) -> BTResult<(), Error> {
        if node.is_dirty {
            return self.storage.set(key, &node).map_err(Into::into);
        }
        Ok(())
    }
}

impl<S: Storage> Deref for StorageEvictionHandler<S> {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.storage
    }
}

/// A node manager that caches nodes in memory, with a underlying storage backend.
///
/// Nodes are retrieved from the underlying storage if they are not present in the cache, and stored
/// back when they get evicted and they have been modified.
pub struct CachedNodeManager<S>
where
    S: Storage,
{
    // Cache for storing nodes in memory.
    nodes: LockCache<S::Id, NodeWithMetadata<S::Item>>,
    // Storage for managing IDs, fetching missing nodes, and storing evicted nodes.
    storage: Arc<StorageEvictionHandler<S>>,
    // We store a single empty node to avoid frequent additions/deletions in the cache.
    // Every read/write access to an empty node ID returns this instance.
    empty_node: RwLock<NodeWithMetadata<S::Item>>,
    empty_id: S::Id,
}

impl<S> CachedNodeManager<S>
where
    S: Storage + 'static,
    S::Id: Eq + Hash + Copy + HasEmptyId,
    S::Item: Default + HasEmptyNode,
{
    /// Creates a new [`CachedNodeManager`] with the given capacity, storage backend, and pin
    /// predicate.
    ///
    /// The pin predicate can be used to prevent nodes from being evicted from the cache.
    /// It will only be called if a node is otherwise eligible for eviction.
    pub fn new(capacity: usize, storage: S, is_pinned_predicate: fn(&S::Item) -> bool) -> Self {
        let storage = Arc::new(StorageEvictionHandler {
            storage,
            is_pinned_predicate,
        });
        CachedNodeManager {
            nodes: LockCache::new(
                capacity,
                storage.clone()
                    as Arc<dyn EvictionHooks<Key = S::Id, Value = NodeWithMetadata<S::Item>>>,
            ),
            storage,
            empty_node: RwLock::new(NodeWithMetadata {
                node: S::Item::empty_node(),
                is_dirty: false,
            }),
            empty_id: S::Id::empty_id(),
        }
    }

    /// Returns the capacity of the node manager's internal cache.
    pub fn capacity(&self) -> u64 {
        self.nodes.capacity()
    }
}

impl<S> CachedNodeManager<S>
where
    S: Storage,
    S::Id: Eq + Hash,
    S::Item: Default,
{
    /// Consumes the node manager, flushes all dirty nodes and calls [`Storage::close`] on the
    /// underlying storage.
    pub fn close(self) -> BTResult<(), Error> {
        for (id, mut guard) in self.nodes.iter_write() {
            // TODO: Make sure we don't flush nodes with dirty commitments
            // https://github.com/0xsoniclabs/sonic-admin/issues/483
            if guard.is_dirty {
                self.storage.storage.set(id, &guard.node)?;
                guard.is_dirty = false;
            }
        }

        // Drop cache since it holds a reference to the storage through the eviction hooks.
        drop(self.nodes);

        let storage = Arc::into_inner(self.storage).ok_or_else(|| {
            Error::CorruptedState("storage reference count is not 1 on close".into())
        })?;
        storage.storage.close().map_err(Into::into)
    }
}

impl<S> NodeManager for CachedNodeManager<S>
where
    S: Storage + 'static,
    S::Id: Eq + Hash + Copy + HasEmptyId,
    S::Item: Default + HasEmptyNode + HasDeltaVariant<Id = S::Id>,
{
    type Id = S::Id;
    type Node = S::Item;

    fn add(&self, node: Self::Node) -> BTResult<Self::Id, Error> {
        if node.is_empty_node() {
            return Ok(self.empty_id);
        }

        let id = self.storage.reserve(&node);
        let _guard = self.nodes.get_read_access_or_insert(id, move || {
            Ok(NodeWithMetadata {
                node,
                is_dirty: true,
            })
        })?;
        Ok(id)
    }

    /// Returns a read guard for a node in the node manager. If the node is not present in the
    /// cache, it is fetched from the underlying storage and cached.
    /// If the node does not exist in storage, returns [`crate::storage::Error::NotFound`].
    fn get_read_access(
        &self,
        id: Self::Id,
    ) -> BTResult<RwLockReadGuard<'_, impl Deref<Target = Self::Node>>, Error> {
        if id.is_empty_id() {
            return Ok(self.empty_node.read().unwrap());
        }

        let lock = self.nodes.get_read_access_or_insert(id, || {
            let mut node = self.storage.storage.get(id)?;
            if let Some(full_id) = node.needs_full() {
                let full = self.get_read_access(full_id)?;
                node.copy_from_base(&**full)?;
            }
            Ok(NodeWithMetadata {
                node,
                is_dirty: false,
            })
        })?;
        Ok(lock)
    }

    /// Returns a write guard for a node in the node manager. If the node is not present in the
    /// cache, it is fetched from the underlying storage and cached.
    /// If the node does not exist in storage, returns [`crate::storage::Error::NotFound`].
    fn get_write_access(
        &self,
        id: Self::Id,
    ) -> BTResult<RwLockWriteGuard<'_, impl DerefMut<Target = Self::Node>>, Error> {
        if id.is_empty_id() {
            return Ok(self.empty_node.write().unwrap());
        }

        let lock = self.nodes.get_write_access_or_insert(id, || {
            let mut node = self.storage.storage.get(id)?;
            if let Some(full_id) = node.needs_full() {
                let full = self.get_read_access(full_id)?;
                node.copy_from_base(&**full)?;
            }
            Ok(NodeWithMetadata {
                node,
                is_dirty: false,
            })
        })?;
        Ok(lock)
    }

    /// Deletes a node with the given ID from the node manager and the underlying storage.
    /// No concurrent calls to [`get_read_access`](Self::get_read_access) or
    /// [`get_write_access`](Self::get_write_access) must be made for the same ID.
    /// It is not safe to call this function multiple times for the same ID, unless allowed by `S`.
    fn delete(&self, id: Self::Id) -> BTResult<(), Error> {
        if id.is_empty_id() {
            return Ok(());
        }

        self.nodes.remove(id)?;
        self.storage.delete(id)?;
        Ok(())
    }
}

impl<S> Checkpointable for CachedNodeManager<S>
where
    S: Storage + 'static + Checkpointable,
    S::Id: Eq + Hash + Copy + Send + Sync + HasEmptyId,
    S::Item: Default + Clone + Send + Sync + HasEmptyNode,
{
    fn checkpoint(&self) -> BTResult<u64, crate::storage::Error> {
        for (id, mut guard) in self.nodes.iter_write() {
            if guard.is_dirty {
                self.storage.storage.set(id, &guard.node)?;
                guard.is_dirty = false;
            }
        }
        self.storage.storage.checkpoint()
    }

    fn restore(path: &Path, checkpoint: u64) -> BTResult<(), crate::storage::Error> {
        S::restore(path, checkpoint)
    }
}

impl<S> RootIdProvider for CachedNodeManager<S>
where
    S: Storage + RootIdProvider,
{
    type Id = <S as RootIdProvider>::Id;

    fn get_root_id(&self, block_number: u64) -> BTResult<Self::Id, crate::storage::Error> {
        self.storage.get_root_id(block_number)
    }

    fn set_root_id(&self, block_number: u64, id: Self::Id) -> BTResult<(), crate::storage::Error> {
        self.storage.set_root_id(block_number, id)
    }

    fn highest_block_number(&self) -> BTResult<Option<u64>, crate::storage::Error> {
        self.storage.highest_block_number()
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use mockall::{
        Sequence, mock,
        predicate::{always, eq},
    };

    use super::*;
    use crate::{
        error::BTError,
        storage,
        types::tree_id::TreeId,
        utils::test_nodes::{DELTA_TEST_NODE, FULL_TEST_NODE_ID, TestNode, TestNodeId},
    };

    /// Helper function to return a [`storage::Error::NotFound`] wrapped in an [`Error`]
    fn not_found() -> BTResult<NodeWithMetadata<TestNode>, Error> {
        Err(Error::Storage(storage::Error::NotFound).into())
    }

    /// Helper function to insert a node into the cache.
    fn cache_insert(
        manager: &CachedNodeManager<MockCachedNodeManagerStorage>,
        id: TestNodeId,
        node: TestNode,
        is_dirty: bool,
    ) {
        let _unused = manager
            .nodes
            .get_read_access_or_insert(id, move || Ok(NodeWithMetadata { node, is_dirty }))
            .unwrap();
    }

    #[test]
    fn cached_node_manager_add_reserves_id_and_inserts_nodes() {
        let expected_id = 0;
        let node = 123;
        let mut storage = MockCachedNodeManagerStorage::new();
        storage.expect_reserve().returning(move |_| expected_id);
        storage.expect_get().never(); // Shouldn't query storage on add
        let manager = CachedNodeManager::new(10, storage, pin_nothing);
        let id = manager.add(node).unwrap();
        assert_eq!(id, expected_id);
        let node_res = manager
            .nodes
            .get_read_access_or_insert(id, not_found)
            .unwrap();
        assert!(node_res.is_dirty);
        assert_eq!(node_res.node, node);
    }

    #[test]
    fn cached_node_manager_add_returns_shared_empty_id_for_empty_node() {
        let mut storage = MockCachedNodeManagerStorage::new();
        storage.expect_reserve().never(); // Shouldn't reserve ID for empty node
        storage.expect_get().never(); // Shouldn't query storage on add
        let manager = CachedNodeManager::new(10, storage, pin_nothing);
        let id = manager.add(TestNode::empty_node()).unwrap();
        assert!(id.is_empty_id());
    }

    #[rstest_reuse::apply(get_method)]
    fn cached_node_manager_get_methods_return_cached_entry(#[case] get_method: GetMethod) {
        let id = 0;
        let expected_entry = 123;
        let mut storage = MockCachedNodeManagerStorage::new();
        storage.expect_get().never(); // Shouldn't query storage if entry is in cache
        let manager = CachedNodeManager::new(10, storage, pin_nothing);

        cache_insert(&manager, id, expected_entry, true);
        let entry = get_method(&manager, id).unwrap();
        assert!(entry == expected_entry);
    }

    #[rstest_reuse::apply(get_method)]
    fn cached_node_manager_get_methods_return_existing_entry_from_storage_if_not_in_cache(
        #[case] get_method: GetMethod,
    ) {
        let id = 0;
        let expected_entry = 123;
        let mut storage = MockCachedNodeManagerStorage::new();
        storage
            .expect_get()
            .times(1)
            .with(eq(id))
            .returning(move |_| Ok(expected_entry));

        let manager = CachedNodeManager::new(10, storage, pin_nothing);
        let entry = get_method(&manager, id).unwrap();
        assert!(entry == expected_entry);
    }

    #[rstest_reuse::apply(get_method)]
    fn cached_node_manager_get_methods_load_full_node_from_cache_when_reading_delta_node_from_storage(
        #[case] get_method: GetMethod,
    ) {
        let id = 0;
        let mut storage = MockCachedNodeManagerStorage::new();
        let mut sequence = Sequence::new();
        storage
            .expect_get()
            .times(1)
            .with(eq(id))
            .returning(move |_| Ok(DELTA_TEST_NODE))
            .in_sequence(&mut sequence);
        storage
            .expect_get()
            .times(1)
            .with(eq(FULL_TEST_NODE_ID))
            .returning(move |_| Ok(0))
            .in_sequence(&mut sequence);

        let manager = CachedNodeManager::new(10, storage, pin_nothing);
        let entry = get_method(&manager, id).unwrap();
        assert!(entry == DELTA_TEST_NODE);
    }

    #[rstest_reuse::apply(get_method)]
    fn cached_node_manager_get_methods_return_shared_empty_node_for_empty_id(
        #[case] get_method: GetMethod,
    ) {
        let mut storage = MockCachedNodeManagerStorage::new();
        storage.expect_get().never(); // Shouldn't query storage for empty ID
        let manager = CachedNodeManager::new(10, storage, pin_nothing);
        let entry = get_method(&manager, TestNodeId::empty_id()).unwrap();
        assert!(entry.is_empty_node());
    }

    #[rstest_reuse::apply(get_method)]
    fn cached_node_manager_get_methods_returns_error_if_node_id_does_not_exist(
        #[case] get_method: GetMethod,
    ) {
        let mut storage = MockCachedNodeManagerStorage::new();
        storage
            .expect_get()
            .returning(|_| Err(storage::Error::NotFound.into()));

        let manager = CachedNodeManager::new(10, storage, pin_nothing);
        let res = get_method(&manager, 0);
        assert!(matches!(
            res.map_err(BTError::into_inner),
            Err(Error::Storage(storage::Error::NotFound))
        ));
    }

    #[test]
    fn cached_node_manager_saves_dirty_nodes_in_storage_on_eviction() {
        // Dirty entries are stored
        {
            let mut storage = MockCachedNodeManagerStorage::new();
            storage
                .expect_set()
                .times(1)
                .with(always(), always()) // we can't make assumptions on which node will be evicted
                .returning(|_, _| Ok(()));
            let manager = CachedNodeManager::new(2, storage, pin_nothing);

            cache_insert(&manager, 0, 123, true);
            cache_insert(&manager, 1, 456, true);
            // Trigger eviction with next insertion
            cache_insert(&manager, 2, 789, true);
        }
        // Clean entries are not stored
        {
            let mut storage = MockCachedNodeManagerStorage::new();
            storage.expect_set().never();
            let manager = CachedNodeManager::new(2, storage, pin_nothing);

            cache_insert(&manager, 0, 123, false);
            cache_insert(&manager, 1, 456, false);
            // Trigger eviction with next insertion
            cache_insert(&manager, 2, 789, false);
        }
    }

    #[test]
    fn cached_node_manager_checkpoint_saves_dirty_nodes_to_storage() {
        const NUM_NODES: u32 = 10;
        let node = 123;
        let mut storage = MockCachedNodeManagerStorage::new();
        for i in 0..NUM_NODES {
            storage
                .expect_set()
                .times(1)
                .with(eq(i), eq(node))
                .returning(move |_, _| Ok(()));
        }
        storage.expect_checkpoint().times(1).returning(|| Ok(1));

        let manager = CachedNodeManager::new(NUM_NODES as usize, storage, pin_nothing);
        for i in 0..NUM_NODES {
            cache_insert(&manager, i, 123, true);
        }
        manager.checkpoint().expect("checkpoint should succeed");
    }

    #[test]
    fn cached_node_manager_restore_calls_restore_on_underlying_storage() {
        let ctx = MockCachedNodeManagerStorage::restore_context();
        ctx.expect()
            .with(eq(Path::new("/path_of_restore_test")), eq(1))
            .returning(|_, _| Ok(()))
            .times(1);

        CachedNodeManager::<MockCachedNodeManagerStorage>::restore(
            Path::new("/path_of_restore_test"),
            1,
        )
        .unwrap();
    }

    #[test]
    fn cached_node_manager_delete_removes_entry_from_cache_and_storage() {
        let mut storage = MockCachedNodeManagerStorage::new();
        let id = 0;
        let entry = 123;
        storage
            .expect_delete()
            .times(1)
            .with(eq(id))
            .returning(|_| Ok(()));
        let manager = CachedNodeManager::new(2, storage, pin_nothing);

        cache_insert(&manager, id, entry, true);
        // Check the element is in the manager
        assert_eq!(manager.nodes.iter_write().count(), 1);
        manager.delete(id).unwrap();
        assert_eq!(manager.nodes.iter_write().count(), 0);
    }

    #[test]
    fn cached_node_manager_delete_on_empty_id_is_noop() {
        let mut storage = MockCachedNodeManagerStorage::new();
        storage.expect_delete().never();

        let manager = CachedNodeManager::new(2, storage, pin_nothing);
        // Shouldn't error
        manager.delete(TestNodeId::empty_id()).unwrap();
    }

    #[test]
    fn cached_node_manager_delete_fails_on_storage_error() {
        let mut storage = MockCachedNodeManagerStorage::new();
        let id = 0;
        storage
            .expect_delete()
            .times(1)
            .with(eq(id))
            .returning(|_| Err(storage::Error::NotFound.into()));

        let manager = CachedNodeManager::new(2, storage, pin_nothing);
        cache_insert(&manager, id, 123, true);
        let res = manager.delete(id);
        assert!(matches!(
            res.map_err(BTError::into_inner),
            Err(Error::Storage(storage::Error::NotFound))
        ));
    }

    #[test]
    fn cached_node_manager_capacity_returns_cache_capacity() {
        let storage = MockCachedNodeManagerStorage::new();
        let capacity = 42;
        let manager = CachedNodeManager::new(capacity, storage, pin_nothing);
        assert_eq!(manager.capacity(), capacity as u64);
    }

    #[test]
    fn cached_node_manager_forwards_root_id_provider_calls() {
        let expected_id = 42;
        let block_number = 7;
        let mut storage = MockCachedNodeManagerStorage::new();
        storage
            .expect_get_root_id()
            .times(1)
            .with(eq(block_number))
            .returning(move |_| Ok(expected_id));
        storage
            .expect_set_root_id()
            .times(1)
            .with(eq(block_number), eq(expected_id))
            .returning(|_, _| Ok(()));

        let manager = CachedNodeManager::new(10, storage, pin_nothing);
        let received_id = manager.get_root_id(block_number).unwrap();
        assert_eq!(received_id, expected_id);

        manager.set_root_id(block_number, expected_id).unwrap();
    }

    #[test]
    fn cached_node_manager_close_flushes_dirty_nodes_and_calls_close_on_underlying_storage() {
        let mut storage = MockCachedNodeManagerStorage::new();
        let mut seq = mockall::Sequence::new();
        storage
            .expect_set()
            .times(1)
            .in_sequence(&mut seq)
            .with(eq(3), eq(777))
            .returning(move |_, _| Ok(()));
        storage
            .expect_close()
            .times(1)
            .in_sequence(&mut seq)
            .returning(|| Ok(()));

        let manager = CachedNodeManager::new(10, storage, pin_nothing);
        cache_insert(&manager, 3, 777, true);
        cache_insert(&manager, 4, 321, false);

        manager.close().unwrap();
    }

    #[test]
    fn get_root_id_calls_get_root_id_on_underlying_storage_layer() {
        let block_number = 1;
        let root_id = TestNodeId::from_idx_and_node_kind(1, ());

        let mut mock_storage = MockCachedNodeManagerStorage::new();
        mock_storage
            .expect_get_root_id()
            .with(eq(block_number))
            .returning(move |_| Ok(root_id))
            .times(1);

        let manager = CachedNodeManager::new(10, mock_storage, pin_nothing);

        assert_eq!(manager.get_root_id(block_number), Ok(root_id));
    }

    #[test]
    fn set_root_id_calls_set_root_id_on_underlying_storage_layer() {
        let block_number = 1;
        let root_id = TestNodeId::from_idx_and_node_kind(1, ());

        let mut mock_storage = MockCachedNodeManagerStorage::new();
        mock_storage
            .expect_set_root_id()
            .with(eq(block_number), eq(root_id))
            .returning(|_, _| Ok(()))
            .times(1);

        let manager = CachedNodeManager::new(10, mock_storage, pin_nothing);

        assert!(manager.set_root_id(block_number, root_id).is_ok(),);
    }

    #[test]
    fn get_highest_block_number_calls_get_highest_block_number_on_underlying_storage_layer() {
        let highest_block_number = Some(1);

        let mut mock_storage = MockCachedNodeManagerStorage::new();
        mock_storage
            .expect_highest_block_number()
            .returning(move || Ok(highest_block_number))
            .times(1);

        let manager = CachedNodeManager::new(10, mock_storage, pin_nothing);

        assert_eq!(manager.highest_block_number(), Ok(highest_block_number));
    }

    #[test]
    fn node_with_metadata_sets_dirty_flag_on_deref_mut() {
        let mut node = NodeWithMetadata {
            node: 0,
            is_dirty: false,
        };
        assert!(!node.is_dirty);
        let _ = node.deref();
        assert!(!node.is_dirty);
        let _ = node.deref_mut();
        assert!(node.is_dirty);
    }

    #[test]
    fn storage_eviction_handler_calls_is_pinned_predicate() {
        let mut storage = MockCachedNodeManagerStorage::new();
        storage.expect_set().never();
        let handler = StorageEvictionHandler {
            storage,
            is_pinned_predicate: move |node: &i32| node == &123,
        };

        let node1 = &NodeWithMetadata {
            node: 123,
            is_dirty: false,
        };
        let node2 = &NodeWithMetadata {
            node: 456,
            is_dirty: false,
        };

        assert!(handler.is_pinned(&0, node1));
        assert!(!handler.is_pinned(&1, node2));
    }

    #[test]
    fn storage_eviction_handler_on_evict_saves_dirty_nodes() {
        let mut storage = MockCachedNodeManagerStorage::new();
        storage
            .expect_set()
            .times(1)
            .with(eq(0), eq(123))
            .returning(|_, _| Ok(()));
        let handler = StorageEvictionHandler {
            storage,
            is_pinned_predicate: pin_nothing,
        };
        let dirty_node = NodeWithMetadata {
            node: 123,
            is_dirty: true,
        };
        handler.on_evict(0, dirty_node).unwrap();
        // Clean nodes don't trigger storage set
        let clean_node = NodeWithMetadata {
            node: 456,
            is_dirty: false,
        };
        handler.on_evict(1, clean_node).unwrap();
    }

    #[test]
    fn storage_eviction_handler_on_evict_fails_on_storage_error() {
        let mut storage = MockCachedNodeManagerStorage::new();
        storage
            .expect_set()
            .returning(|_, _| Err(storage::Error::NotFound.into()));
        let handler = StorageEvictionHandler {
            storage,
            is_pinned_predicate: pin_nothing,
        };
        let res = handler.on_evict(
            0,
            NodeWithMetadata {
                node: 123,
                is_dirty: true,
            },
        );
        assert!(res.is_err());
        assert!(matches!(
            res.map_err(BTError::into_inner),
            Err(Error::Storage(storage::Error::NotFound))
        ));
    }

    /// Default predicate that never pins any node.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn pin_nothing(_node: &TestNode) -> bool {
        false
    }

    #[allow(clippy::disallowed_types)]
    mod mock {
        use super::*;
        mock! {
            pub CachedNodeManagerStorage {}

            impl Checkpointable for CachedNodeManagerStorage {
                fn checkpoint(&self) -> BTResult<u64, storage::Error>;

                fn restore(path: &Path, checkpoint: u64) -> BTResult<(), storage::Error>;
            }

            impl RootIdProvider for CachedNodeManagerStorage {
                type Id = TestNodeId;

                fn get_root_id(
                    &self,
                    block_number: u64,
                ) -> BTResult<<Self as RootIdProvider>::Id, storage::Error>;

                fn set_root_id(
                    &self,
                    block_number: u64,
                    id: <Self as RootIdProvider>::Id,
                ) -> BTResult<(), storage::Error>;

                fn highest_block_number(&self) -> BTResult<Option<u64>, storage::Error>;
            }

            impl Storage for CachedNodeManagerStorage {
                type Id = TestNodeId;
                type Item = TestNode;

                fn open(_path: &Path, db_mode: storage::DbMode) -> BTResult<Self, storage::Error>;

                fn get(
                    &self,
                    id: <Self as Storage>::Id,
                ) -> BTResult<<Self as Storage>::Item, storage::Error>;

                fn reserve(&self, _item: &<Self as Storage>::Item) -> <Self as Storage>::Id;

                fn set(
                    &self,
                    id: <Self as Storage>::Id,
                    item: &<Self as Storage>::Item,
                ) -> BTResult<(), storage::Error>;

                fn delete(&self, _id: <Self as Storage>::Id) -> BTResult<(), storage::Error>;

                fn close(self) -> BTResult<(), storage::Error>;
            }
        }
    }
    use mock::MockCachedNodeManagerStorage;

    /// Type alias for a closure that calls either `get_read_access` or `get_write_access`
    type GetMethod = fn(
        &CachedNodeManager<MockCachedNodeManagerStorage>,
        TestNodeId,
    ) -> BTResult<TestNode, Error>;

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
