use crate::{
    database::verkle::{
        CachedCommitment,
        variants::managed::managed_trie_node::{
            CanStoreResult, CommitmentInput, LookupResult, ManagedTrieNode, UnionManagedTrieNode,
        },
    },
    error::Error,
    types::{Key, Node, NodeId, Value},
};

mod empty;
mod inner;
mod leaf;
mod sparse_leaf;

impl UnionManagedTrieNode for Node {}

impl ManagedTrieNode for Node {
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
