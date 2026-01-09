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

use crate::{
    database::{
        verkle::{
            crypto::Commitment, keyed_update::KeyedUpdateBatch, variants::simple::node::Node,
            verkle_trie::VerkleTrie,
        },
        visitor::{AcceptVisitor, NodeVisitor},
    },
    error::{BTResult, Error},
    sync::Mutex,
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
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        SimpleInMemoryVerkleTrie {
            root: Mutex::new(Node::Empty),
        }
    }
}

impl VerkleTrie for SimpleInMemoryVerkleTrie {
    fn lookup(&self, key: &Key) -> BTResult<Value, Error> {
        let _span = tracy_client::span!("SimpleInMemoryVerkleTrie::lookup");
        Ok(self.root.lock().unwrap().lookup(key, 0))
    }

    fn store(&self, updates: &KeyedUpdateBatch, is_archive: bool) -> BTResult<(), Error> {
        if is_archive {
            return Err(Error::UnsupportedImplementation(
                "SimpleInMemoryVerkleTrie does not support archive mode".to_owned(),
            )
            .into());
        }
        let mut root_lock = self.root.lock().unwrap();
        let root = std::mem::replace(&mut *root_lock, Node::Empty);
        *root_lock = root.store(updates.borrowed(), 0);
        Ok(())
    }

    fn commit(&self) -> BTResult<Commitment, Error> {
        let _span = tracy_client::span!("SimpleInMemoryVerkleTrie::commit");
        Ok(self.root.lock().unwrap().commit())
    }

    fn after_update(&self, _block_height: u64) -> BTResult<(), Error> {
        // No-op for this implementation
        Ok(())
    }
}

impl AcceptVisitor for SimpleInMemoryVerkleTrie {
    type Node = Node;

    fn accept(&self, visitor: &mut impl NodeVisitor<Self::Node>) -> BTResult<(), Error> {
        self.root.lock().unwrap().accept(visitor, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::verkle::{
        KeyedUpdateBatch,
        test_utils::{make_leaf_key, make_value},
    };

    // NOTE: Most tests are in verkle_trie.rs

    #[test]
    fn commitment_of_non_empty_trie_is_root_node_commitment() {
        let trie = SimpleInMemoryVerkleTrie::new();
        trie.store(
            &KeyedUpdateBatch::from_key_value_pairs(&[
                (make_leaf_key(&[1], 1), make_value(1)),
                (make_leaf_key(&[2], 2), make_value(2)),
                (make_leaf_key(&[3], 3), make_value(3)),
            ]),
            false,
        )
        .unwrap();

        let have = trie.commit().unwrap();
        let want = trie.root.lock().unwrap().commit();

        assert_eq!(have, want);
    }

    #[test]
    fn accept_traverses_all_nodes() {
        struct TestVisitor {
            values: Vec<Value>,
        }

        impl NodeVisitor<Node> for TestVisitor {
            fn visit(&mut self, node: &Node, _level: u64) -> BTResult<(), Error> {
                match node {
                    Node::Leaf(leaf_node) => {
                        for key in [
                            make_leaf_key(&[1], 1),
                            make_leaf_key(&[2], 2),
                            make_leaf_key(&[3], 3),
                        ] {
                            let value = leaf_node.lookup(&key);
                            if value != Value::default() {
                                self.values.push(value);
                            }
                        }
                    }
                    Node::Empty | Node::Inner(_) => {}
                }
                Ok(())
            }
        }
        let trie = SimpleInMemoryVerkleTrie::new();
        trie.store(
            &KeyedUpdateBatch::from_key_value_pairs(&[
                (make_leaf_key(&[1], 1), make_value(1)),
                (make_leaf_key(&[2], 2), make_value(2)),
                (make_leaf_key(&[3], 3), make_value(3)),
            ]),
            false,
        )
        .unwrap();

        let mut visitor = TestVisitor { values: Vec::new() };
        trie.accept(&mut visitor).unwrap();

        visitor.values.sort();
        assert_eq!(
            visitor.values,
            vec![make_value(1), make_value(2), make_value(3)]
        );
    }
}
