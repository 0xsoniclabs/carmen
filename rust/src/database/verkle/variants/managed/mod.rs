mod fake_cache;
mod id_trie_node;
mod nodes;

use std::sync::{Arc, Mutex, RwLock};

pub use fake_cache::FakeCache;
pub use id_trie_node::CachedCommitment;
use id_trie_node::{IdTrieNode, lookup, store};

use crate::{
    database::verkle::{
        crypto::Commitment,
        variants::managed::id_trie_node::{TrieUpdateLog, update_commitments},
        verkle_trie::VerkleTrie,
    },
    error::Error,
    node_manager::NodeManager,
    types::{EmptyNode, Key, Node, NodeId, Value},
};

pub struct ManagedVerkleTrie<M: NodeManager<Id = NodeId, NodeType = Node> + Send + Sync> {
    root: RwLock<NodeId>,
    manager: Arc<M>,
    // FIXME: This needs to have interior mutability
    update_log: Mutex<TrieUpdateLog<NodeId>>,
}

impl<M: NodeManager<Id = NodeId, NodeType = Node> + Send + Sync> ManagedVerkleTrie<M> {
    pub fn new(manager: Arc<M>) -> Self {
        let root = manager.add(Node::Empty(EmptyNode)).unwrap(); // FIXME Unwrap
        ManagedVerkleTrie {
            root: RwLock::new(root),
            manager,
            update_log: Mutex::new(TrieUpdateLog::new()),
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
        let mut update_log = self.update_log.lock().unwrap();
        store(root_id_lock, key, value, &*self.manager, &mut update_log)
    }

    fn commit(&self) -> Commitment {
        // FIXME Unwrap
        // compute_commitment_uncached_recursive(*self.root.read().unwrap(), &self.manager).unwrap()
        // FIXME Unwrap
        // compute_commitment_cached_recursive(*self.root.read().unwrap(), &self.manager).unwrap()

        let mut update_log = self.update_log.lock().unwrap();
        // FIXME Unwrap
        update_commitments(&mut update_log, &*self.manager).unwrap();
        self.manager
            .get_read_access(*self.root.read().unwrap())
            .unwrap()
            .get_cached_commitment()
            .commitment()
    }
}

// TODO: Test trait implementation on node types

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

        let received = trie.commit();
        let expected = manager
            .get_read_access(*trie.root.read().unwrap())
            .unwrap()
            .get_cached_commitment()
            .commitment();

        assert_eq!(received, expected);
    }
}
