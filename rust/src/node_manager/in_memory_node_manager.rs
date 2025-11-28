// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::ops::{Deref, DerefMut};

use crossbeam_queue::ArrayQueue;
use dashmap::DashMap;

use crate::{
    error::{BTResult, Error},
    node_manager::NodeManager,
    storage::{self, RootIdProvider},
    sync::{RwLock, RwLockReadGuard, RwLockWriteGuard},
    types::{HasEmptyId, HasEmptyNode, ToNodeKind, TreeId},
};

/// Dummy wrapper around a node `N`, implementing `Deref` and `DerefMut`.
/// This is required by the [`NodeManager`] trait.
struct NodeWrapper<N> {
    node: N,
}

impl<N> Deref for NodeWrapper<N> {
    type Target = N;

    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

impl<N> DerefMut for NodeWrapper<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.node
    }
}

/// An in-memory implementation of a [`NodeManager`] with fixed capacity.
///
/// After reaching capacity, attempts to add new nodes will fail.
pub struct InMemoryNodeManager<I, N>
where
    I: TreeId,
{
    nodes: Vec<RwLock<NodeWrapper<N>>>,
    empty_node: RwLock<NodeWrapper<N>>,
    free_slots: ArrayQueue<u64>,
    root_ids: DashMap<u64, I>,

    _phantom: std::marker::PhantomData<I>,
}

impl<I, N> InMemoryNodeManager<I, N>
where
    I: TreeId,
    N: HasEmptyNode + Default,
{
    pub fn new(capacity: usize) -> Self {
        let mut nodes = Vec::with_capacity(capacity);
        let free_slots = ArrayQueue::new(capacity);
        for i in 0..capacity {
            nodes.push(RwLock::new(NodeWrapper { node: N::default() }));
            free_slots.push(i as u64).unwrap(); // Safe to unwrap because we insert only up to capacity
        }
        InMemoryNodeManager {
            nodes,
            empty_node: RwLock::new(NodeWrapper {
                node: N::empty_node(),
            }),
            free_slots,
            root_ids: DashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<I, N> NodeManager for InMemoryNodeManager<I, N>
where
    I: TreeId + HasEmptyId + Copy,
    N: Default + HasEmptyNode + ToNodeKind<Target = I::Target>,
{
    type Id = I;
    type Node = N;

    fn add(&self, value: Self::Node) -> BTResult<Self::Id, Error> {
        if value.is_empty_node() {
            return Ok(I::empty_id());
        }
        let idx = self
            .free_slots
            .pop()
            .ok_or(Error::CorruptedState("no remaining free slots".to_owned()))?;
        let key = I::from_idx_and_node_kind(idx, value.to_node_kind().unwrap());
        self.nodes[idx as usize].write().unwrap().node = value;
        Ok(key)
    }

    fn get_read_access(
        &self,
        id: I,
    ) -> BTResult<RwLockReadGuard<'_, impl Deref<Target = Self::Node>>, Error> {
        if id.is_empty_id() {
            return Ok(self.empty_node.read().unwrap());
        }
        Ok(self.nodes[id.to_index() as usize].read().unwrap())
    }

    fn get_write_access(
        &self,
        id: Self::Id,
    ) -> BTResult<RwLockWriteGuard<'_, impl std::ops::DerefMut<Target = Self::Node>>, Error> {
        if id.is_empty_id() {
            return Ok(self.empty_node.write().unwrap());
        }
        Ok(self.nodes[id.to_index() as usize].write().unwrap())
    }

    fn delete(&self, id: I) -> BTResult<(), Error> {
        if id.is_empty_id() {
            return Ok(());
        }
        let mut guard = self.nodes[id.to_index() as usize].write().unwrap();
        guard.node = N::default();
        self.free_slots.push(id.to_index()).map_err(|_| {
            Error::CorruptedState("failed to push freed slot back to free slots queue".to_owned())
        })?;
        Ok(())
    }
}

impl<I, N> RootIdProvider for InMemoryNodeManager<I, N>
where
    I: TreeId + Copy,
{
    type Id = I;

    fn get_root_id(&self, block_number: u64) -> BTResult<Self::Id, crate::storage::Error> {
        match self.root_ids.get(&block_number) {
            Some(id) => Ok(*id),
            None => Err(storage::Error::NotFound.into()),
        }
    }

    fn set_root_id(&self, block_number: u64, id: Self::Id) -> BTResult<(), crate::storage::Error> {
        self.root_ids.insert(block_number, id);
        Ok(())
    }

    fn highest_block_number(&self) -> BTResult<Option<u64>, storage::Error> {
        Ok(self.root_ids.iter().map(|entry| *entry.key()).max())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        error::BTError,
        node_manager::test_utils::{TestNode, TestNodeId},
    };

    type TestInMemoryNodeManager = InMemoryNodeManager<TestNodeId, TestNode>;

    #[test]
    fn new_creates_manager_with_specified_capacity() {
        let capacity = 10;
        let manager = TestInMemoryNodeManager::new(capacity);
        assert_eq!(manager.nodes.len(), capacity);
        assert_eq!(manager.free_slots.len(), capacity);
        for i in 0..capacity as u64 {
            assert!(manager.free_slots.pop().unwrap() == i);
        }
    }

    #[test]
    fn add_stores_value_and_assigns_free_id() {
        let manager = TestInMemoryNodeManager::new(10);
        let node_value: TestNode = 42;
        let id = manager.add(node_value).unwrap();
        {
            let guard = manager.get_read_access(id).unwrap();
            assert_eq!(**guard, node_value);
        }
        assert!(
            !manager
                .free_slots
                .into_iter()
                .any(|idx| idx == id.to_index())
        );
    }

    #[test]
    fn add_ignores_empty_nodes() {
        let capacity = 10;
        let manager = TestInMemoryNodeManager::new(capacity);
        let id = manager.add(TestNode::empty_node()).unwrap();
        assert_eq!(id, TestNodeId::empty_id());
        assert_eq!(manager.free_slots.len(), capacity);
    }

    #[test]
    fn add_returns_error_when_capacity_exceeded() {
        let manager = TestInMemoryNodeManager::new(1);
        let _ = manager.add(42).unwrap();
        let result = manager.add(100);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::CorruptedState(e)) if e.contains("no remaining free slots"),
        ));
    }

    #[test]
    fn get_access_returns_lock_for_corresponding_slot() {
        let manager = TestInMemoryNodeManager::new(10);
        let id = TestNodeId::from_idx_and_node_kind(4, ());
        let index = id.to_index() as usize;
        {
            let _guard = manager.get_read_access(id).unwrap();
            assert!(manager.nodes[index].try_write().is_err());
        }
        {
            let _guard = manager.get_write_access(id).unwrap();
            assert!(manager.nodes[index].try_write().is_err());
        }
    }

    #[test]
    fn get_access_on_empty_id_returns_empty_node_lock() {
        let manager = TestInMemoryNodeManager::new(10);
        {
            let _guard = manager.get_read_access(TestNodeId::empty_id()).unwrap();
            assert!(manager.empty_node.try_write().is_err());
        }
        {
            let _guard = manager.get_write_access(TestNodeId::empty_id()).unwrap();
            assert!(manager.empty_node.try_write().is_err());
        }
    }

    #[test]
    fn delete_frees_slot_and_resets_node() {
        let manager = TestInMemoryNodeManager::new(10);
        let node_value: TestNode = 42;
        let id = manager.add(node_value).unwrap();
        manager.delete(id).unwrap();
        {
            let guard = manager.nodes[id.to_index() as usize].read().unwrap();
            assert_eq!(**guard, TestNode::default());
        }
        assert!(
            manager
                .free_slots
                .into_iter()
                .any(|idx| idx == id.to_index())
        );
    }

    #[test]
    fn delete_on_empty_id_is_noop() {
        let manager = TestInMemoryNodeManager::new(10);
        let result = manager.delete(TestNodeId::empty_id());
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn root_ids_can_be_set_and_retrieved() {
        let manager = TestInMemoryNodeManager::new(10);

        let result = manager.get_root_id(42);
        assert_eq!(result, Err(storage::Error::NotFound.into()));

        let id = TestNodeId::from_idx_and_node_kind(3, ());
        manager.set_root_id(42, id).unwrap();
        let received = manager.get_root_id(42).unwrap();
        assert_eq!(received, id);

        let other_id = TestNodeId::from_idx_and_node_kind(27, ());
        manager.set_root_id(33, other_id).unwrap();
        let received = manager.get_root_id(33).unwrap();
        assert_eq!(received, other_id);

        let received = manager.get_root_id(42).unwrap();
        assert_eq!(received, id);
    }

    #[test]
    fn get_root_id_retrieves_root_id_from_root_id_map() {
        let root_id0 = TestNodeId::from_idx_and_node_kind(0, ());
        let root_id1 = TestNodeId::from_idx_and_node_kind(1, ());

        let manager = InMemoryNodeManager {
            nodes: Vec::new(),
            empty_node: RwLock::new(NodeWrapper {
                node: TestNode::empty_node(),
            }),
            free_slots: ArrayQueue::new(1),
            root_ids: DashMap::new(),
            _phantom: std::marker::PhantomData,
        };

        assert_eq!(
            manager.get_root_id(0).map_err(BTError::into_inner),
            Err(storage::Error::NotFound)
        );
        assert_eq!(
            manager.get_root_id(1).map_err(BTError::into_inner),
            Err(storage::Error::NotFound)
        );

        manager.root_ids.insert(0, root_id0);

        assert_eq!(manager.get_root_id(0), Ok(root_id0));
        assert_eq!(
            manager.get_root_id(1).map_err(BTError::into_inner),
            Err(storage::Error::NotFound)
        );

        manager.root_ids.insert(1, root_id1);

        assert_eq!(manager.get_root_id(0), Ok(root_id0));
        assert_eq!(manager.get_root_id(1), Ok(root_id1));
    }

    #[test]
    fn set_root_id_stores_root_id_in_root_id_map() {
        let root_id = TestNodeId::from_idx_and_node_kind(1, ());

        let manager = InMemoryNodeManager {
            nodes: Vec::new(),
            empty_node: RwLock::new(NodeWrapper {
                node: TestNode::empty_node(),
            }),
            free_slots: ArrayQueue::new(1),
            root_ids: DashMap::new(),
            _phantom: std::marker::PhantomData,
        };

        assert_eq!(manager.root_ids.get(&0).as_deref(), None);
        assert_eq!(manager.root_ids.get(&1).as_deref(), None);

        manager.set_root_id(0, root_id).unwrap();
        assert_eq!(manager.root_ids.get(&0).as_deref(), Some(&root_id));
        assert_eq!(manager.root_ids.get(&1).as_deref(), None);

        manager.set_root_id(1, root_id).unwrap();
        assert_eq!(manager.root_ids.get(&0).as_deref(), Some(&root_id));
        assert_eq!(manager.root_ids.get(&1).as_deref(), Some(&root_id));
    }

    #[test]
    fn get_highest_block_number_gets_highest_block_number_from_root_id_map() {
        let root_id = TestNodeId::from_idx_and_node_kind(1, ());

        let manager = InMemoryNodeManager {
            nodes: Vec::new(),
            empty_node: RwLock::new(NodeWrapper {
                node: TestNode::empty_node(),
            }),
            free_slots: ArrayQueue::new(1),
            root_ids: DashMap::new(),
            _phantom: std::marker::PhantomData,
        };

        assert_eq!(manager.highest_block_number(), Ok(None));

        manager.set_root_id(0, root_id).unwrap();
        assert_eq!(manager.highest_block_number(), Ok(Some(0)));

        manager.set_root_id(1, root_id).unwrap();
        assert_eq!(manager.highest_block_number(), Ok(Some(1)));
    }
}
