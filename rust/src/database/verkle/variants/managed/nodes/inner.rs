use crate::{
    database::verkle::{
        CachedCommitment,
        variants::managed::id_trie_node::{
            CanStoreResult, CommitmentInput, IdTrieNode, LookupResult,
        },
    },
    error::Error,
    types::{InnerNode, Key, Node, NodeId},
};

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
