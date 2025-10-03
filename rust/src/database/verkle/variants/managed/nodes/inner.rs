use crate::{
    database::verkle::{
        CachedCommitment,
        crypto::Commitment,
        variants::managed::managed_trie_node::{
            CanStoreResult, LookupResult, ManagedTrieNode, VerkleCommitmentInput,
        },
    },
    error::Error,
    types::{InnerNode, Key, Node, NodeId},
};

impl ManagedTrieNode for InnerNode {
    type Union = Node;
    type Id = NodeId;
    type Commitment = Commitment;
    type CommitmentInput = VerkleCommitmentInput;

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
        VerkleCommitmentInput::Inner(self.children)
    }
}
