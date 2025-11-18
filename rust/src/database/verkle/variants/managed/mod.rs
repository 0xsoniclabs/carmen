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
            crypto::Commitment, variants::managed::commitment::update_commitments,
            verkle_trie::VerkleTrie,
        },
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
        update_commitments(&self.update_log, &*self.manager)?;
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
    use super::*;
    use crate::{
        database::verkle::test_utils::{make_leaf_key, make_value},
        node_manager::in_memory_node_manager::InMemoryNodeManager,
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
}
