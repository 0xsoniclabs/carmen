mod fake_cache;
mod id_trie_node;

use std::sync::{Arc, Mutex, RwLock};

pub use fake_cache::FakeCache;
pub use id_trie_node::CachedCommitment;
use id_trie_node::{CanStoreResult, CommitmentInput, IdTrieNode, LookupResult, lookup, store};

use crate::{
    database::verkle::{
        crypto::Commitment,
        variants::managed::id_trie_node::{TrieUpdateLog, UnionIdTrieNode, update_commitments},
        verkle_trie::VerkleTrie,
    },
    error::Error,
    node_manager::NodeManager,
    types::{
        EmptyNode, FullLeafNode, InnerNode, Key, Node, NodeId, SparseLeafNode, Value,
        ValueWithIndex,
    },
};

impl UnionIdTrieNode for Node {}

impl IdTrieNode for Node {
    type Union = Node;
    type Id = NodeId;

    fn lookup(&self, key: &Key, depth: u8) -> Result<LookupResult<Self::Id>, Error> {
        match self {
            Node::Empty(n) => n.lookup(key, depth),
            Node::Inner(n) => n.lookup(key, depth),
            Node::Leaf2(n) => n.lookup(key, depth),
            Node::Leaf256(n) => n.lookup(key, depth),
        }
    }

    fn can_store(&self, key: &Key, depth: u8) -> Result<CanStoreResult<Self::Id>, Error> {
        match self {
            Node::Empty(n) => n.can_store(key, depth),
            Node::Inner(n) => n.can_store(key, depth),
            Node::Leaf2(n) => n.can_store(key, depth),
            Node::Leaf256(n) => n.can_store(key, depth),
        }
    }

    fn transform(&self, key: &Key, depth: u8) -> Result<Self::Union, Error> {
        match self {
            Node::Empty(n) => n.transform(key, depth),
            Node::Inner(n) => n.transform(key, depth),
            Node::Leaf2(n) => n.transform(key, depth),
            Node::Leaf256(n) => n.transform(key, depth),
        }
    }

    fn reparent(&self, key: &Key, depth: u8, self_id: Self::Id) -> Result<Self::Union, Error> {
        match self {
            Node::Empty(n) => n.reparent(key, depth, self_id),
            Node::Inner(n) => n.reparent(key, depth, self_id),
            Node::Leaf2(n) => n.reparent(key, depth, self_id),
            Node::Leaf256(n) => n.reparent(key, depth, self_id),
        }
    }

    fn replace_child(&mut self, key: &Key, depth: u8, new: NodeId) -> Result<(), Error> {
        match self {
            Node::Empty(n) => n.replace_child(key, depth, new),
            Node::Inner(n) => n.replace_child(key, depth, new),
            Node::Leaf2(n) => n.replace_child(key, depth, new),
            Node::Leaf256(n) => n.replace_child(key, depth, new),
        }
    }

    fn store(&mut self, key: &Key, value: &Value) -> Result<(), Error> {
        match self {
            Node::Empty(n) => n.store(key, value),
            Node::Inner(n) => n.store(key, value),
            Node::Leaf2(n) => n.store(key, value),
            Node::Leaf256(n) => n.store(key, value),
        }
    }

    fn get_cached_commitment(&self) -> CachedCommitment {
        match self {
            Node::Empty(n) => n.get_cached_commitment(),
            Node::Inner(n) => n.get_cached_commitment(),
            Node::Leaf2(n) => n.get_cached_commitment(),
            Node::Leaf256(n) => n.get_cached_commitment(),
        }
    }

    fn set_cached_commitment(&mut self, cache: CachedCommitment) -> Result<(), Error> {
        match self {
            Node::Empty(n) => n.set_cached_commitment(cache),
            Node::Inner(n) => n.set_cached_commitment(cache),
            Node::Leaf2(n) => n.set_cached_commitment(cache),
            Node::Leaf256(n) => n.set_cached_commitment(cache),
        }
    }

    fn get_commitment_input(&self) -> CommitmentInput<Self::Id> {
        match self {
            Node::Empty(n) => n.get_commitment_input(),
            Node::Inner(n) => n.get_commitment_input(),
            Node::Leaf2(n) => n.get_commitment_input(),
            Node::Leaf256(n) => n.get_commitment_input(),
        }
    }
}

// TODO PROBLEM: There will be a lot of contention around the lock for empty nodes!!
impl IdTrieNode for EmptyNode {
    type Union = Node;

    type Id = NodeId;

    fn lookup(&self, _key: &Key, _depth: u8) -> Result<LookupResult<Self::Id>, Error> {
        Ok(LookupResult::Value(Value::default()))
    }

    fn can_store(&self, _key: &Key, _depth: u8) -> Result<CanStoreResult<Self::Id>, Error> {
        Ok(CanStoreResult::Transform)
    }

    fn transform(&self, key: &Key, depth: u8) -> Result<Self::Union, Error> {
        if depth == 0 {
            // While conceptually it would suffice to create a leaf node here,
            // Geth always creates an inner node (and we want to stay compatible).
            let inner = InnerNode::default();
            Ok(Node::Inner(Box::new(inner)))
        } else {
            let new_leaf = SparseLeafNode::<2> {
                // Safe to unwrap: Slice is always 31 bytes
                stem: key[..31].try_into().unwrap(),
                ..Default::default()
            };
            Ok(Node::Leaf2(Box::new(new_leaf)))
        }
    }

    fn get_cached_commitment(&self) -> CachedCommitment {
        CachedCommitment::default()
    }

    fn get_commitment_input(&self) -> CommitmentInput<Self::Id> {
        CommitmentInput::Empty
    }
}

impl IdTrieNode for InnerNode {
    type Union = Node;
    type Id = NodeId;

    fn lookup(&self, key: &Key, depth: u8) -> Result<LookupResult<Self::Id>, Error> {
        Ok(LookupResult::Node(
            self.children[key[depth as usize] as usize],
        ))
    }

    fn can_store(&self, key: &Key, depth: u8) -> Result<CanStoreResult<Self::Id>, Error> {
        let pos = key[depth as usize];
        Ok(CanStoreResult::Descend(self.children[pos as usize]))
    }

    fn replace_child(&mut self, key: &Key, depth: u8, new: NodeId) -> Result<(), Error> {
        self.children[key[depth as usize] as usize] = new;
        Ok(())
    }

    fn get_cached_commitment(&self) -> CachedCommitment {
        self.commitment
    }

    fn set_cached_commitment(&mut self, cache: CachedCommitment) -> Result<(), Error> {
        self.commitment = cache;
        Ok(())
    }

    fn get_commitment_input(&self) -> CommitmentInput<Self::Id> {
        CommitmentInput::Inner(self.children)
    }
}

impl IdTrieNode for FullLeafNode {
    type Union = Node;
    type Id = NodeId;

    fn lookup(&self, key: &Key, _depth: u8) -> Result<LookupResult<Self::Id>, Error> {
        if key[..31] != self.stem[..] {
            Ok(LookupResult::Value(Value::default()))
        } else {
            Ok(LookupResult::Value(self.values[key[31] as usize]))
        }
    }

    fn can_store(&self, key: &Key, _depth: u8) -> Result<CanStoreResult<Self::Id>, Error> {
        if key[..31] == self.stem[..] {
            Ok(CanStoreResult::Yes)
        } else {
            Ok(CanStoreResult::Reparent)
        }
    }

    fn reparent(&self, key: &Key, depth: u8, self_id: NodeId) -> Result<Self::Union, Error> {
        assert!(matches!(
            self.can_store(key, depth)?,
            CanStoreResult::Reparent
        ));

        let pos = self.stem[depth as usize];
        // TODO: Need better ctor
        let mut inner = InnerNode::default();
        inner.children[pos as usize] = self_id;
        Ok(Node::Inner(Box::new(inner)))
    }

    // TODO: We could implement a conversion to SparseLeafNode if enough values are zero
    // => We would have to retain the used bits however!
    fn store(&mut self, key: &Key, value: &Value) -> Result<(), Error> {
        assert_eq!(self.stem[..], key[..31]);

        let suffix = key[31];
        self.values[suffix as usize] = *value;
        self.used_bits[(suffix / 8) as usize] |= 1 << (suffix % 8);
        Ok(())
    }

    fn get_cached_commitment(&self) -> CachedCommitment {
        self.commitment
    }

    fn set_cached_commitment(&mut self, cache: CachedCommitment) -> Result<(), Error> {
        self.commitment = cache;
        Ok(())
    }

    fn get_commitment_input(&self) -> CommitmentInput<Self::Id> {
        CommitmentInput::Leaf(self.values, self.used_bits, self.stem)
    }
}

// TODO: Implement for generic N?
// => Ensuring that entries are sorted could make things a lot easier
impl IdTrieNode for SparseLeafNode<2> {
    type Union = Node;
    type Id = NodeId;

    fn lookup(&self, key: &Key, _depth: u8) -> Result<LookupResult<Self::Id>, Error> {
        if key[..31] != self.stem[..] {
            return Ok(LookupResult::Value(Value::default()));
        }

        for ValueWithIndex { index, value } in &self.values {
            if *index == key[31] {
                return Ok(LookupResult::Value(*value));
            }
        }
        Ok(LookupResult::Value(Value::default()))
    }

    fn can_store(&self, key: &Key, _depth: u8) -> Result<CanStoreResult<Self::Id>, Error> {
        if key[..31] != self.stem[..] {
            return Ok(CanStoreResult::Reparent);
        }

        // TODO: Need to thoroughly test this behavior
        for ValueWithIndex { index, value } in &self.values {
            if *index == key[31] || *value == Value::default() {
                return Ok(CanStoreResult::Yes);
            }
        }
        Ok(CanStoreResult::Transform)
    }

    fn transform(&self, key: &Key, depth: u8) -> Result<Self::Union, Error> {
        assert!(matches!(
            self.can_store(key, depth)?,
            CanStoreResult::Transform
        ));

        assert_eq!(key[..31], self.stem[..]);
        // If the stems match, we have to convert to a full leaf.
        let new_leaf = FullLeafNode {
            stem: self.stem,
            values: {
                let mut values = [Value::default(); 256];
                for ValueWithIndex { index, value } in &self.values {
                    values[*index as usize] = *value;
                }
                values
            },
            used_bits: self.used_bits,
            commitment: CachedCommitment::default(),
        };
        Ok(Node::Leaf256(Box::new(new_leaf)))
    }

    fn reparent(&self, key: &Key, depth: u8, self_id: NodeId) -> Result<Self::Union, Error> {
        assert!(matches!(
            self.can_store(key, depth)?,
            CanStoreResult::Reparent
        ));

        // Otherwise, we have to re-parent.
        let pos = self.stem[depth as usize];
        // TODO: Need better ctor
        let mut inner = InnerNode::default();
        inner.children[pos as usize] = self_id;
        Ok(Node::Inner(Box::new(inner)))
    }

    fn store(&mut self, key: &Key, value: &Value) -> Result<(), Error> {
        assert_eq!(self.stem[..], key[..31]);

        let mut slot = None;
        // TODO: Need to thoroughly test this behavior
        for (i, ValueWithIndex { index, value: v }) in self.values.iter().enumerate() {
            if *index == key[31] || *v == Value::default() {
                slot = Some(i);
                break;
            }
        }
        self.values[slot.unwrap()] = ValueWithIndex {
            index: key[31],
            value: *value,
        };
        // NOTE: Used bits are NOT cleared! The whole point is to remember which slots were
        // modified, even if they are set back to zero.
        // TODO: Test
        self.used_bits[(key[31] / 8) as usize] |= 1 << (key[31] % 8);

        Ok(())
    }

    fn get_cached_commitment(&self) -> CachedCommitment {
        self.commitment
    }

    fn set_cached_commitment(&mut self, cache: CachedCommitment) -> Result<(), Error> {
        self.commitment = cache;
        Ok(())
    }

    // FIXME: This should not have to pass 256 values!
    fn get_commitment_input(&self) -> CommitmentInput<Self::Id> {
        let mut values = [Value::default(); 256];
        for ValueWithIndex { index, value } in &self.values {
            values[*index as usize] = *value;
        }
        CommitmentInput::Leaf(values, self.used_bits, self.stem)
    }
}

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
