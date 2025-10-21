// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

mod node;

use std::sync::Mutex;

use crate::{
    database::verkle::{crypto::Commitment, variants::simple::node::Node, verkle_trie::VerkleTrie},
    error::Error,
    types::{Key, Value},
};

/// A simple in-memory implementation of the Verkle trie.
///
/// This implementation is not meant for production use, but rather as a reference,
/// and for validation, prototyping and testing.
///
/// The implementation has several shortcomings:
/// - Total tree size is limited by available memory.
/// - No concurrency support (single lock on root node).
/// - Not optimized for memory usage (all nodes store 256 children / values).
pub struct SimpleInMemoryVerkleTrie {
    root: Mutex<Node>,
}

impl SimpleInMemoryVerkleTrie {
    pub fn new() -> Self {
        SimpleInMemoryVerkleTrie {
            root: Mutex::new(Node::Empty),
        }
    }
}

impl VerkleTrie for SimpleInMemoryVerkleTrie {
    fn lookup(&self, key: &Key) -> Result<Value, Error> {
        Ok(self.root.lock().unwrap().lookup(key, 0))
    }

    fn store(&self, key: &Key, value: &Value) -> Result<(), Error> {
        let mut root_lock = self.root.lock().unwrap();
        let root = std::mem::replace(&mut *root_lock, Node::Empty);
        *root_lock = root.store(key, 0, value);
        Ok(())
    }

    fn commit(&self) -> Result<Commitment, Error> {
        let _span = tracy_client::span!("SimpleInMemoryVerkleTrie::commit");
        Ok(self.root.lock().unwrap().commit())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::verkle::test_utils::{make_leaf_key, make_value};

    // NOTE: Most tests are in verkle_trie.rs

    #[test]
    fn commitment_of_non_empty_trie_is_root_node_commitment() {
        let trie = SimpleInMemoryVerkleTrie::new();
        trie.store(&make_leaf_key(&[1], 1), &make_value(1)).unwrap();
        trie.store(&make_leaf_key(&[2], 2), &make_value(2)).unwrap();
        trie.store(&make_leaf_key(&[3], 3), &make_value(3)).unwrap();

        let have = trie.commit().unwrap();
        let want = trie.root.lock().unwrap().commit();

        assert_eq!(have, want);
    }
}
