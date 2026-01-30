// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::path::Path;

use ipa_multipoint::committer::DefaultCommitter;
use verkle_db::{BareMetalDiskDb, RocksDb};
use verkle_trie::{DefaultConfig, Trie, TrieTrait, database::VerkleDb};

use crate::{
    database::verkle::{
        crypto::Commitment,
        keyed_update::{KeyedUpdate, KeyedUpdateBatch},
        verkle_trie::VerkleTrie,
    },
    error::{BTResult, Error},
    sync::RwLock,
    types::{Key, Value},
};

/// A in-memory implementation of the Verkle trie adapting the third-party [`verkle_trie::Trie`].
///
/// This implementation is not meant for production use and has no concurrency support.
pub struct CrateCryptoInMemoryVerkleTrie {
    trie: RwLock<Trie<VerkleDb<RocksDb>, DefaultCommitter>>,
}

impl CrateCryptoInMemoryVerkleTrie {
    #[allow(clippy::new_without_default)]
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            trie: RwLock::new(Trie::new(DefaultConfig::new(VerkleDb::from_path(path)))),
        }
    }
    pub fn flush(&self) -> BTResult<(), Error> {
        self.trie.write().unwrap().flush_database();
        Ok(())
    }
}

impl VerkleTrie for CrateCryptoInMemoryVerkleTrie {
    fn lookup(&self, key: &Key) -> BTResult<Value, Error> {
        Ok(self.trie.read().unwrap().get(*key).unwrap_or_default())
    }

    fn store(
        &self,
        updates: &KeyedUpdateBatch,
        is_archive: bool,
    ) -> BTResult<(), crate::error::Error> {
        if is_archive {
            return Err(Error::UnsupportedImplementation(
                "CrateCryptoInMemoryVerkleTrie does not support archive mode".to_owned(),
            )
            .into());
        }
        let mut trie = self.trie.write().unwrap();
        for update in updates.iter() {
            match update {
                KeyedUpdate::FullSlot { key, value } => {
                    trie.insert_single(*key, *value);
                }
                KeyedUpdate::PartialSlot { key, .. } => {
                    let mut existing_value = trie.get(*key).unwrap_or_default();
                    update.apply_to_value(&mut existing_value);
                    trie.insert_single(*key, existing_value);
                }
            }
        }

        Ok(())
    }

    fn commit(&self) -> BTResult<Commitment, crate::error::Error> {
        Ok(Commitment::from(
            self.trie.read().unwrap().root_commitment(),
        ))
    }

    fn after_update(&self, _block_height: u64) -> BTResult<(), Error> {
        // No-op for this implementation
        Ok(())
    }
}

// NOTE: Tests are in verkle_trie.rs
