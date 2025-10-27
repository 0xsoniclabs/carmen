// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::sync::{Arc, RwLock};

pub use fake_cache::FakeCache;
pub use nodes::{
    Node, NodeFileStorageManager, NodeType, empty::EmptyNode, id::NodeId, inner::InnerNode,
    leaf::FullLeafNode, sparse_leaf::SparseLeafNode,
};

use crate::{
    database::{
        managed_trie::{ManagedTrieNode, TrieUpdateLog, lookup, store},
        verkle::{
            crypto::Commitment, variants::managed::commitment::update_commitments,
            verkle_trie::VerkleTrie,
        },
    },
    error::Error,
    node_manager::NodeManager,
    statistics::{NodeStatisticVisitor, Statistics, TrieStatistics},
    types::{Key, Value},
};

mod commitment;
mod fake_cache;
mod nodes;

pub struct ManagedVerkleTrie<M: NodeManager<Id = NodeId, NodeType = Node> + Send + Sync> {
    root: RwLock<NodeId>,
    manager: Arc<M>,
    update_log: TrieUpdateLog<NodeId>,
}

impl<M: NodeManager<Id = NodeId, NodeType = Node> + Send + Sync> TrieStatistics
    for ManagedVerkleTrie<M>
{
    fn get_statistics(&self) -> Statistics {
        let mut visitor = NodeStatisticVisitor::default();
        let root = self
            .manager
            .get_read_access(*self.root.read().unwrap())
            .unwrap();
        root.accept(&mut visitor, &self.manager, 0);
        visitor.statistics
    }
}

impl<M: NodeManager<Id = NodeId, NodeType = Node> + Send + Sync> ManagedVerkleTrie<M> {
    pub fn new(manager: Arc<M>) -> Self {
        let root = manager.add(Node::Empty(EmptyNode)).unwrap(); // FIXME Unwrap
        ManagedVerkleTrie {
            root: RwLock::new(root),
            manager,
            update_log: TrieUpdateLog::new(),
        }
    }
}

impl<M: NodeManager<Id = NodeId, NodeType = Node> + Send + Sync> VerkleTrie
    for ManagedVerkleTrie<M>
{
    fn lookup(&self, key: &Key) -> Result<Value, Error> {
        lookup(*self.root.read().unwrap(), key, &*self.manager)
    }

    fn store(&self, key: &Key, value: &Value) -> Result<(), Error> {
        let root_id_lock = self.root.write().unwrap();
        store(root_id_lock, key, value, &*self.manager, &self.update_log)
    }

    fn commit(&self) -> Result<Commitment, Error> {
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
    use crate::database::verkle::test_utils::{make_leaf_key, make_value};

    // NOTE: Most tests are in verkle_trie.rs

    #[test]
    fn trie_commitment_of_non_empty_trie_is_root_node_commitment() {
        let manager = Arc::new(FakeCache::new());
        let trie = ManagedVerkleTrie::new(manager.clone());
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
