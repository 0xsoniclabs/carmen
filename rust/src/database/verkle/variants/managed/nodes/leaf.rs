use crate::{
    database::verkle::{
        CachedCommitment,
        crypto::Commitment,
        variants::managed::managed_trie_node::{
            CanStoreResult, LookupResult, ManagedTrieNode, VerkleCommitmentInput,
        },
    },
    error::Error,
    types::{FullLeafNode, InnerNode, Key, Node, NodeId, Value},
};

impl ManagedTrieNode for FullLeafNode {
    type Union = Node;
    type Id = NodeId;
    type Commitment = Commitment;
    type CommitmentInput = VerkleCommitmentInput;

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

    fn get_cached_commitment(&self) -> CachedCommitment<Self::Commitment> {
        self.commitment
    }

    fn set_cached_commitment(
        &mut self,
        cache: CachedCommitment<Self::Commitment>,
    ) -> Result<(), Error> {
        self.commitment = cache;
        Ok(())
    }

    fn get_commitment_input(&self) -> Self::CommitmentInput {
        VerkleCommitmentInput::Leaf(self.values, self.used_bits, self.stem)
    }
}
