use std::sync::RwLock;

use ipa_multipoint::committer::DefaultCommitter;
use verkle_trie::{DefaultConfig, Trie, TrieTrait, database::memory_db::MemoryDb};

use crate::{
    database::verkle::{crypto::Commitment, verkle_trie::VerkleTrie},
    error::Error,
    storage,
    types::{Key, Value},
};

pub struct CrateCryptoInMemoryVerkleTrie {
    trie: RwLock<Trie<MemoryDb, DefaultCommitter>>,
}

impl CrateCryptoInMemoryVerkleTrie {
    pub fn new() -> Self {
        let db = MemoryDb::new();
        let config = DefaultConfig::new(db);
        Self {
            trie: RwLock::new(Trie::new(config)),
        }
    }
}

impl VerkleTrie for CrateCryptoInMemoryVerkleTrie {
    fn lookup(&self, key: &Key) -> Result<Value, Error> {
        Ok(self.trie.read().unwrap().get(*key).unwrap_or_default())
    }

    fn store(
        &self,
        key: &crate::types::Key,
        value: &crate::types::Value,
    ) -> Result<(), crate::error::Error> {
        // TODO: There is also a batched method
        self.trie.write().unwrap().insert_single(*key, *value);
        Ok(())
    }

    fn commit(&self) -> Result<Commitment, crate::error::Error> {
        Ok(Commitment::from(
            self.trie.read().unwrap().root_commitment(),
        ))
    }

    fn depth(&self) -> usize {
        0
    }

    fn node_count(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    // #[test]
    // fn foo() {
    //     // use crate::database::ReadOnlyHigherDb;
    //     let db = MemoryDb::new();
    //     let mut trie = Trie::new(DefaultConfig::new(db));
    // }
}
