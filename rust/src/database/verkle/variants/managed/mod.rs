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
pub use nodes::{VerkleNode, empty::EmptyNode, id::VerkleNodeId, inner::InnerNode};

use crate::{
    database::{
        managed_trie::{ManagedTrieNode, TrieUpdateLog, lookup, store},
        verkle::{
            crypto::Commitment, variants::managed::commitment::update_commitments,
            verkle_trie::VerkleTrie,
        },
    },
    error::{BTResult, Error},
    node_manager::NodeManager,
    sync::{Arc, RwLock},
    types::{Key, Value},
};

mod commitment;
mod nodes;

pub struct ManagedVerkleTrie<M: NodeManager<Id = VerkleNodeId, Node = VerkleNode> + Send + Sync> {
    root: RwLock<VerkleNodeId>,
    manager: Arc<M>,
    update_log: TrieUpdateLog<VerkleNodeId>,
}

impl<M: NodeManager<Id = VerkleNodeId, Node = VerkleNode> + Send + Sync> ManagedVerkleTrie<M> {
    /// Creates a new empty [`ManagedVerkleTrie`] using the given node manager.
    /// Returns an error if the root node cannot be created.
    pub fn try_new(manager: Arc<M>) -> BTResult<Self, Error> {
        let root = manager.add(VerkleNode::Empty(EmptyNode))?;
        Ok(ManagedVerkleTrie {
            root: RwLock::new(root),
            manager,
            update_log: TrieUpdateLog::new(),
        })
    }
}

impl<M: NodeManager<Id = VerkleNodeId, Node = VerkleNode> + Send + Sync> VerkleTrie
    for ManagedVerkleTrie<M>
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
}
