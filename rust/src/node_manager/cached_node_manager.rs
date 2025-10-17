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
    sync::{Arc, RwLockReadGuard, RwLockWriteGuard},
};

use crate::{
    error::Error,
    node_manager::{
        NodeManager,
        lock_cache::{LockCache, OnEvict},
    },
    storage::{Checkpointable, Storage},
};

/// A wrapper which dereferences to [`Node`] and additionally stores its dirty status,
/// indicating whether it needs to be flushed to storage.
/// The node status is set to dirty when a mutable reference is requested.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct NodeWithMetadata<Node> {
    node: Node,
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

/// A wrapper around a storage backend that implements the [`OnEvict`] trait
struct StorageEvictionHandler<S> {
    storage: S,
}

impl<S> OnEvict for StorageEvictionHandler<S>
where
    S: Storage + Send + Sync,
{
    type Key = S::Id;
    type Value = NodeWithMetadata<S::Item>;

    /// Stores the evicted node in the underlying storage if it is dirty.
    fn on_evict(&self, key: S::Id, node: NodeWithMetadata<S::Item>) -> Result<(), Error> {
        if node.is_dirty {
            return self.storage.set(key, &node).map_err(Error::Storage);
        }
        Ok(())
    }
}

impl<S> Deref for StorageEvictionHandler<S> {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.storage
    }
}

/// A node manager that caches nodes in memory, with a underlying storage backend.
/// Nodes are retrieved from the underlying storage if they are not present in the cache, and store
/// them back when evicted if they have been modified.
/// It only guarantees that no access to a deleted or evicted node is possible.
pub struct CachedNodeManager<K, N, S>
where
    S: Storage<Id = K, Item = N>,
{
    // Cache for storing nodes in memory
    nodes: LockCache<K, NodeWithMetadata<N>>,
    // Storage for managing IDs, fetching missing nodes, and saving evicted nodes to
    storage: Arc<StorageEvictionHandler<S>>,
}

impl<K: Eq + Hash + Copy + std::fmt::Debug, S, N> CachedNodeManager<K, N, S>
where
    S: Storage<Id = K, Item = N> + Send + Sync + 'static,
    N: Default + Clone,
{
    /// Creates a new [`CachedNodeManager`] with the given capacity and storage backend.
    pub fn new(capacity: usize, storage: S) -> Self {
        let storage = Arc::new(StorageEvictionHandler { storage });
        CachedNodeManager {
            nodes: LockCache::new(
                capacity,
                storage.clone() as Arc<dyn OnEvict<Key = K, Value = NodeWithMetadata<N>>>,
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
        let id = self.storage.reserve(&node);
        let _guard = self.nodes.get_read_access_or_insert(id, move || {
            Ok(NodeWithMetadata {
                node: node.clone(),
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

    /// Returns a write guard for a node in the node manager. If the node is not present in the
    /// cache, it is fetched from the underlying storage and cached.
    /// If the node does not exist in storage, returns [`crate::storage::Error::NotFound`].
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

    /// Deletes a node with the given ID from the node manager and the underlying storage.
    fn delete(&self, id: Self::Id) -> Result<(), Error> {
        self.nodes.remove(id)?;
        self.storage.delete(id)?;
        Ok(())
    }
}

impl<K, N, S> Checkpointable for CachedNodeManager<K, N, S>
where
    K: Eq + Hash + Copy + Send + Sync,
    N: Default + Send + Sync,
    S: Storage<Id = K, Item = N> + Checkpointable,
{
    fn checkpoint(&self) -> Result<(), crate::storage::Error> {
        for (id, mut guard) in self.nodes.iter_write() {
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
    use std::path::Path;

    use mockall::{
        mock,
        predicate::{always, eq},
    };

    use super::*;
    use crate::storage::{self};

    type NodeId = u32;
    type Node = i32;

    /// Helper function to return a [`storage::Error::NotFound`] wrapped in an [`Error`]
    fn not_found() -> Result<NodeWithMetadata<Node>, Error> {
        Err(Error::Storage(storage::Error::NotFound))
    }

    /// Helper function to retrieve or insert a node and discard the guard.
    fn get_read_access_or_insert_no_guard(
        manager: &CachedNodeManager<NodeId, Node, MockCachedNodeManagerStorage>,
        id: NodeId,
        node: Node,
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
        let manager = CachedNodeManager::new(10, storage);
        let id = manager.add(node).unwrap();
        assert_eq!(id, expected_id);
        let node_res = manager
            .nodes
            .get_read_access_or_insert(id, not_found)
            .unwrap();
        assert!(node_res.is_dirty);
        assert_eq!(node_res.node, node);
    }

    #[rstest_reuse::apply(get_method)]
    fn cached_node_manager_get_methods_return_cached_entry(#[case] get_method: GetMethod) {
        let id = 0;
        let expected_entry = 123;
        let mut storage = MockCachedNodeManagerStorage::new();
        storage.expect_get().never(); // Shouldn't query storage if entry is in cache
        let manager = CachedNodeManager::new(10, storage);

        get_read_access_or_insert_no_guard(&manager, id, expected_entry, true);
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
        let res = get_method(&manager, 0);
        assert!(res.is_err());
        assert!(matches!(
            res.err().unwrap(),
            Error::Storage(storage::Error::NotFound)
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
            let manager = CachedNodeManager::new(2, storage);

            get_read_access_or_insert_no_guard(&manager, 0, 123, true);
            get_read_access_or_insert_no_guard(&manager, 1, 456, true);
            // Trigger eviction with next insertion
            get_read_access_or_insert_no_guard(&manager, 2, 789, true);
        }
        // Clean entries are not stored
        {
            let mut storage = MockCachedNodeManagerStorage::new();
            storage.expect_set().never();
            let manager = CachedNodeManager::new(2, storage);

            get_read_access_or_insert_no_guard(&manager, 0, 123, false);
            get_read_access_or_insert_no_guard(&manager, 1, 456, false);
            // Trigger eviction with next insertion
            get_read_access_or_insert_no_guard(&manager, 2, 789, false);
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
        storage.expect_checkpoint().times(1).returning(|| Ok(()));

        let manager = CachedNodeManager::new(NUM_NODES as usize, storage);
        for i in 0..NUM_NODES {
            get_read_access_or_insert_no_guard(&manager, i, 123, true);
        }
        manager.checkpoint().expect("checkpoint should succeed");
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
        let manager = CachedNodeManager::new(2, storage);

        get_read_access_or_insert_no_guard(&manager, id, entry, true);
        // Check the element is in the manager
        drop(
            manager
                .nodes
                .get_read_access_or_insert(id, not_found)
                .unwrap(),
        );
        manager.delete(id).unwrap();
        assert!(
            manager
                .nodes
                .get_read_access_or_insert(id, not_found)
                .is_err(),
        );
    }

    #[test]
    fn cached_node_manager_delete_fails_on_storage_error() {
        let mut storage = MockCachedNodeManagerStorage::new();
        let id = 0;
        storage
            .expect_delete()
            .times(1)
            .with(eq(id))
            .returning(|_| Err(storage::Error::NotFound));

        let manager = CachedNodeManager::new(2, storage);
        get_read_access_or_insert_no_guard(&manager, id, 123, true);
        let res = manager.delete(id);
        assert!(res.is_err());
        assert!(matches!(
            res.unwrap_err(),
            Error::Storage(storage::Error::NotFound)
        ));
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
    fn storage_eviction_handler_on_evict_saves_dirty_nodes() {
        let mut storage = MockCachedNodeManagerStorage::new();
        storage
            .expect_set()
            .times(1)
            .with(eq(0), eq(123))
            .returning(|_, _| Ok(()));
        let handler = StorageEvictionHandler { storage };
        handler
            .on_evict(
                0,
                NodeWithMetadata {
                    node: 123,
                    is_dirty: true,
                },
            )
            .unwrap();
        // Clean elements don't trigger storage set
        handler
            .on_evict(
                1,
                NodeWithMetadata {
                    node: 456,
                    is_dirty: false,
                },
            )
            .unwrap();
    }

    #[test]
    fn storage_eviction_handler_on_evict_fails_on_storage_error() {
        let mut storage = MockCachedNodeManagerStorage::new();
        storage
            .expect_set()
            .returning(|_, _| Err(storage::Error::NotFound));
        let handler = StorageEvictionHandler { storage };
        let res = handler.on_evict(
            0,
            NodeWithMetadata {
                node: 123,
                is_dirty: true,
            },
        );
        assert!(res.is_err());
        assert!(matches!(
            res.unwrap_err(),
            Error::Storage(storage::Error::NotFound)
        ));
    }

    mock! {
        pub CachedNodeManagerStorage {}

        impl Checkpointable for CachedNodeManagerStorage {
            fn checkpoint(&self) -> Result<(), storage::Error>;
        }

        impl Storage for CachedNodeManagerStorage {
            type Id = u32;
            type Item = i32;

            fn open(_path: &Path) -> Result<Self, storage::Error>;

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
}
