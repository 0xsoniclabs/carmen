use std::sync::RwLock;

use ipa_multipoint::committer::DefaultCommitter;
use verkle_trie::{DefaultConfig, Trie, TrieTrait, database::memory_db::MemoryDb};

use crate::{
    database::verkle::{crypto::Commitment, verkle_trie::VerkleTrie},
    error::{BTResult, Error},
    types::{Key, Value},
};

/// A in-memory implementation of the Verkle trie adapting the third-party [`verkle_trie::Trie`].
///
/// This implementation is not meant for production use and has no concurrency support.
pub struct CrateCryptoInMemoryVerkleTrie {
    trie: RwLock<Trie<MemoryDb, DefaultCommitter>>,
}

impl CrateCryptoInMemoryVerkleTrie {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let db = MemoryDb::new();
        let config = DefaultConfig::new(db);
        Self {
            trie: RwLock::new(Trie::new(config)),
        }
    }
}

impl VerkleTrie for CrateCryptoInMemoryVerkleTrie {
    fn lookup(&self, key: &Key) -> BTResult<Value, Error> {
        Ok(self.trie.read().unwrap().get(*key).unwrap_or_default())
    }

    fn store(
        &self,
        key: &crate::types::Key,
        value: &crate::types::Value,
    ) -> BTResult<(), crate::error::Error> {
        self.trie.write().unwrap().insert_single(*key, *value);
        Ok(())
    }

    fn commit(&self) -> BTResult<Commitment, crate::error::Error> {
        Ok(Commitment::from(
            self.trie.read().unwrap().root_commitment(),
        ))
    }
}

// NOTE: Tests are in verkle_trie.rs
