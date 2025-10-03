use crate::{
    database::verkle::{
        CachedCommitment,
        variants::managed::managed_trie_node::{
            CanStoreResult, CommitmentInput, LookupResult, ManagedTrieNode,
        },
    },
    error::Error,
    types::{EmptyNode, InnerNode, Key, Node, NodeId, SparseLeafNode, Value},
};

// TODO PROBLEM: There will be a lot of contention around the lock for empty nodes!!
impl ManagedTrieNode for EmptyNode {
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
