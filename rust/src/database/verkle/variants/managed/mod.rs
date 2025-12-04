// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

#[allow(unused)]
pub use nodes::VerkleNodeFileStorageManager;
pub use nodes::{
    VerkleNode, empty::EmptyNode, id::VerkleNodeId, inner::InnerNode, leaf::FullLeafNode,
    sparse_leaf::SparseLeafNode,
};

use crate::{
    database::{
        managed_trie::{ManagedTrieNode, TrieUpdateLog, lookup, store},
        verkle::{
            crypto::Commitment,
            variants::managed::commitment::{
                update_commitments_concurrent, update_commitments_concurrent_recursive,
                update_commitments_concurrent_task_graph, update_commitments_sequential,
            },
            verkle_trie::VerkleTrie,
        },
        visitor::{AcceptVisitor, NodeVisitor},
    },
    error::{BTError, BTResult, Error},
    node_manager::NodeManager,
    storage::{self, RootIdProvider},
    sync::{Arc, RwLock},
    types::{Key, Value},
};

mod commitment;
mod nodes;

pub struct ManagedVerkleTrie<M>
where
    M: NodeManager<Id = VerkleNodeId, Node = VerkleNode> + Send + Sync,
{
    root: RwLock<VerkleNodeId>,
    manager: Arc<M>,
    update_log: TrieUpdateLog<VerkleNodeId>,
}

impl<M> ManagedVerkleTrie<M>
where
    M: NodeManager<Id = VerkleNodeId, Node = VerkleNode>
        + RootIdProvider<Id = VerkleNodeId>
        + Send
        + Sync,
{
    /// Creates a new [`ManagedVerkleTrie`] using the given node manager.
    ///
    /// If the node manager does not provide a root node ID for block height 0,
    /// a new empty trie is created.
    ///
    /// Returns an error if the root node ID cannot be obtained from the node manager.
    pub fn try_new(manager: Arc<M>) -> BTResult<Self, Error> {
        // We currently always pass block height 0 since we assume this to be a live DB.
        // TODO: Forward actual block height for archive DB
        let root = match manager.get_root_id(0).map_err(BTError::into_inner) {
            Ok(root_id) => root_id,
            Err(storage::Error::NotFound) => {
                let empty_node = VerkleNode::Empty(EmptyNode {});
                manager.add(empty_node)?
            }
            Err(e) => return Err(e.into()),
        };
        Ok(ManagedVerkleTrie {
            root: RwLock::new(root),
            manager,
            update_log: TrieUpdateLog::new(),
        })
    }
}

impl<M: NodeManager<Id = VerkleNodeId, Node = VerkleNode> + Send + Sync> AcceptVisitor
    for ManagedVerkleTrie<M>
{
    type Node = VerkleNode;

    fn accept(&self, visitor: &mut impl NodeVisitor<Self::Node>) -> BTResult<(), Error> {
        let root = self.manager.get_read_access(*self.root.read().unwrap())?;
        root.accept(visitor, &*self.manager, 0)
    }
}

impl<M> VerkleTrie for ManagedVerkleTrie<M>
where
    M: NodeManager<Id = VerkleNodeId, Node = VerkleNode>
        + RootIdProvider<Id = VerkleNodeId>
        + Send
        + Sync,
{
    fn lookup(&self, key: &Key) -> BTResult<Value, Error> {
        lookup(*self.root.read().unwrap(), key, &*self.manager)
    }

    fn store(&self, key: &Key, value: &Value) -> BTResult<(), Error> {
        let root_id_lock = self.root.write().unwrap();
        store(root_id_lock, key, value, &*self.manager, &self.update_log)
    }

    fn commit(&self) -> BTResult<Commitment, Error> {
        // update_commitments_sequential(&self.update_log, &*self.manager)?;
        // update_commitments_concurrent(&self.update_log, &*self.manager)?;
        // update_commitments_concurrent_recursive(
        //     *self.root.read().unwrap(),
        //     &self.update_log,
        //     &*self.manager,
        // )?;
        update_commitments_concurrent_task_graph(
            *self.root.read().unwrap(),
            &self.update_log,
            &*self.manager,
        )?;
        Ok(self
            .manager
            .get_read_access(*self.root.read().unwrap())?
            .get_commitment()
            .commitment())
    }

    fn after_update(&self, _block_height: u64) -> BTResult<(), Error> {
        let root_id = *self.root.read().unwrap();
        // We currently always pass block height 0 since we assume this to be a live DB.
        // TODO: Forward actual block height for archive DB
        self.manager.set_root_id(0, root_id)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ops::{Deref, DerefMut};

    use mockall::{mock, predicate::eq};

    use super::*;
    use crate::{
        database::{
            verkle::test_utils::{make_leaf_key, make_value},
            visitor::MockNodeVisitor,
        },
        node_manager::in_memory_node_manager::InMemoryNodeManager,
        sync::{RwLockReadGuard, RwLockWriteGuard},
    };

    // NOTE: Most tests are in verkle_trie.rs

    #[test]
    fn try_new_creates_empty_trie() {
        let manager = Arc::new(InMemoryNodeManager::<VerkleNodeId, VerkleNode>::new(10));
        let trie = ManagedVerkleTrie::try_new(manager.clone()).unwrap();

        let root_node = manager.get_read_access(*trie.root.read().unwrap()).unwrap();
        assert!(matches!(&**root_node, VerkleNode::Empty(_)));
    }

    #[test]
    fn try_new_gets_root_id_for_block_zero_from_node_manager() {
        let manager = Arc::new(InMemoryNodeManager::<VerkleNodeId, VerkleNode>::new(10));
        let expected_root_id = manager.add(VerkleNode::Inner(Box::default())).unwrap();
        manager.set_root_id(0, expected_root_id).unwrap();

        let trie = ManagedVerkleTrie::try_new(manager.clone()).unwrap();
        let received_root_id = *trie.root.read().unwrap();
        assert_eq!(received_root_id, expected_root_id);
    }

    #[test]
    fn try_new_propagates_error_from_node_manager() {
        let mut manager = MockTestNodeManager::new();
        manager
            .expect_get_root_id()
            .with(eq(0))
            .returning(|_| Err(storage::Error::Frozen.into()));
        let result = ManagedVerkleTrie::try_new(Arc::new(manager)).map_err(BTError::into_inner);
        assert!(matches!(
            result,
            Err(Error::Storage(storage::Error::Frozen))
        ));
    }

    #[test]
    fn trie_commitment_of_non_empty_trie_is_root_node_commitment() {
        let manager = Arc::new(InMemoryNodeManager::<VerkleNodeId, VerkleNode>::new(10));
        let trie = ManagedVerkleTrie::try_new(manager.clone()).unwrap();
        trie.store(&make_leaf_key(&[1], 1), &make_value(1)).unwrap();
        trie.store(&make_leaf_key(&[2], 2), &make_value(2)).unwrap();
        trie.store(&make_leaf_key(&[3], 3), &make_value(3)).unwrap();

        let received = trie.commit().unwrap();
        let expected = manager
            .get_read_access(*trie.root.read().unwrap())
            .unwrap()
            .get_commitment()
            .commitment();

        assert_eq!(received, expected);
    }

    #[test]
    fn after_update_updates_root_id_in_node_manager() {
        let manager = Arc::new(InMemoryNodeManager::<VerkleNodeId, VerkleNode>::new(10));
        let trie = ManagedVerkleTrie::try_new(manager.clone()).unwrap();
        trie.store(&make_leaf_key(&[1], 1), &make_value(1)).unwrap();
        let root_id = *trie.root.read().unwrap();

        trie.after_update(42).unwrap();
        // We currently always set the root id for block height 0
        let stored_root_id = manager.get_root_id(0).unwrap();
        assert_eq!(root_id, stored_root_id);
    }

    #[test]
    fn accept_traverses_all_nodes() {
        let node_manager = Arc::new(InMemoryNodeManager::new(10));

        let leaf_node_1_id = node_manager.add(VerkleNode::Leaf2(Box::default())).unwrap();
        let mut inner_node_child = InnerNode::default();
        inner_node_child.children[0] = leaf_node_1_id;
        let inner_node_child_id = node_manager
            .add(VerkleNode::Inner(Box::new(inner_node_child)))
            .unwrap();
        let leaf_node_2_id = node_manager
            .add(VerkleNode::Leaf256(Box::default()))
            .unwrap();
        let mut inner_node = InnerNode::default();
        inner_node.children[0] = inner_node_child_id;
        inner_node.children[1] = leaf_node_2_id;
        let inner_node_id = node_manager
            .add(VerkleNode::Inner(Box::new(inner_node)))
            .unwrap();

        // Register the root node
        node_manager.set_root_id(0, inner_node_id).unwrap();
        let trie = ManagedVerkleTrie::try_new(node_manager.clone()).unwrap();

        let mut mock_visitor = MockNodeVisitor::<VerkleNode>::new();
        mock_visitor
            .expect_visit()
            .withf(|node, level| matches!(node, VerkleNode::Inner(_)) && *level == 0)
            .times(1)
            .returning(|_, _| Ok(()));
        mock_visitor
            .expect_visit()
            .withf(|node, level| matches!(node, VerkleNode::Inner(_)) && *level == 1)
            .times(1)
            .returning(|_, _| Ok(()));
        mock_visitor
            .expect_visit()
            .withf(|node, level| matches!(node, VerkleNode::Leaf2(_)) && *level == 2)
            .times(1)
            .returning(|_, _| Ok(()));
        mock_visitor
            .expect_visit()
            .withf(|node, level| matches!(node, VerkleNode::Leaf256(_)) && *level == 1)
            .times(1)
            .returning(|_, _| Ok(()));
        mock_visitor
            .expect_visit()
            .withf(|node, _| matches!(node, VerkleNode::Empty(_)))
            .returning(|_, _| Ok(()));

        trie.accept(&mut mock_visitor).unwrap();
    }

    struct TestNodeWrapper {
        node: VerkleNode,
    }

    impl Deref for TestNodeWrapper {
        type Target = VerkleNode;

        fn deref(&self) -> &Self::Target {
            &self.node
        }
    }

    impl DerefMut for TestNodeWrapper {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.node
        }
    }

    #[allow(clippy::disallowed_types)]
    mod mock {
        use super::*;
        mock! {
            pub TestNodeManager {}

            impl RootIdProvider for TestNodeManager {
                type Id = VerkleNodeId;

                fn get_root_id(&self, block_height: u64) -> BTResult<<Self as RootIdProvider>::Id, storage::Error>;

                fn set_root_id(&self, block_height: u64, root_id: <Self as RootIdProvider>::Id) -> BTResult<(), storage::Error>;

                fn highest_block_number(&self) -> BTResult<Option<u64>, storage::Error>;
            }

            impl NodeManager for TestNodeManager {
                type Id = VerkleNodeId;
                type Node = VerkleNode;

                fn add(&self, node: <Self as NodeManager>::Node) -> BTResult<<Self as NodeManager>::Id, Error>;

                #[allow(refining_impl_trait)]
                fn get_read_access<'a>(
                    &'a self,
                    id: <Self as NodeManager>::Id,
                ) -> BTResult<RwLockReadGuard<'a, TestNodeWrapper>, Error>;

                #[allow(refining_impl_trait)]
                fn get_write_access<'a>(
                    &'a self,
                    id: <Self as NodeManager>::Id,
                ) -> BTResult<RwLockWriteGuard<'a, TestNodeWrapper>, Error>;

                fn delete(&self, id: <Self as NodeManager>::Id) -> BTResult<(), Error>;
            }
        }
    }
    use mock::MockTestNodeManager;
}
