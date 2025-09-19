mod node;
#[cfg(test)]
mod test_utils;

use std::sync::Mutex;

use crate::database::verkle::variants::simple::node::Node;
#[cfg(test)]
use crate::{
    database::verkle::{crypto::Commitment, verkle_trie::VerkleTrie},
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
    #[cfg_attr(not(test), expect(unused))]
    root: Mutex<Node>,
}

#[cfg_attr(not(test), expect(unused))]
impl SimpleInMemoryVerkleTrie {
    pub fn new() -> Self {
        SimpleInMemoryVerkleTrie {
            root: Mutex::new(Node::Empty),
        }
    }
}

#[cfg(test)]
impl VerkleTrie for SimpleInMemoryVerkleTrie {
    fn get(&self, key: &Key) -> Result<Value, Error> {
        Ok(self.root.lock().unwrap().get(key, 0))
    }

    fn set(&self, key: &Key, value: &Value) -> Result<(), Error> {
        let mut root_lock = self.root.lock().unwrap();
        let root = std::mem::replace(&mut *root_lock, Node::Empty);
        *root_lock = root.set(key, 0, value);
        Ok(())
    }

    fn commit(&self) -> Commitment {
        self.root.lock().unwrap().commit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::verkle::variants::simple::test_utils::{
        make_key, make_leaf_key, make_value,
    };

    #[test]
    fn new_creates_empty_trie() {
        let trie = SimpleInMemoryVerkleTrie::new();
        assert_eq!(trie.get(&make_key(&[1])).unwrap(), Value::default());
        assert_eq!(trie.get(&make_key(&[2])).unwrap(), Value::default());
        assert_eq!(trie.get(&make_key(&[3])).unwrap(), Value::default());
    }

    #[test]
    fn values_can_be_set_and_retrieved() {
        let trie = SimpleInMemoryVerkleTrie::new();

        assert_eq!(trie.get(&make_key(&[1])).unwrap(), Value::default());
        assert_eq!(trie.get(&make_key(&[2])).unwrap(), Value::default());
        assert_eq!(trie.get(&make_leaf_key(&[0], 1)).unwrap(), Value::default());
        assert_eq!(trie.get(&make_leaf_key(&[0], 2)).unwrap(), Value::default());

        trie.set(&make_key(&[1]), &make_value(1)).unwrap();

        assert_eq!(trie.get(&make_key(&[1])).unwrap(), make_value(1));
        assert_eq!(trie.get(&make_key(&[2])).unwrap(), Value::default());
        assert_eq!(trie.get(&make_leaf_key(&[0], 1)).unwrap(), Value::default());
        assert_eq!(trie.get(&make_leaf_key(&[0], 2)).unwrap(), Value::default());

        trie.set(&make_key(&[2]), &make_value(2)).unwrap();

        assert_eq!(trie.get(&make_key(&[1])).unwrap(), make_value(1));
        assert_eq!(trie.get(&make_key(&[2])).unwrap(), make_value(2));
        assert_eq!(trie.get(&make_leaf_key(&[0], 1)).unwrap(), Value::default());
        assert_eq!(trie.get(&make_leaf_key(&[0], 2)).unwrap(), Value::default());

        trie.set(&make_leaf_key(&[0], 1), &make_value(3)).unwrap();

        assert_eq!(trie.get(&make_key(&[1])).unwrap(), make_value(1));
        assert_eq!(trie.get(&make_key(&[2])).unwrap(), make_value(2));
        assert_eq!(trie.get(&make_leaf_key(&[0], 1)).unwrap(), make_value(3));
        assert_eq!(trie.get(&make_leaf_key(&[0], 2)).unwrap(), Value::default());

        trie.set(&make_leaf_key(&[0], 2), &make_value(4)).unwrap();

        assert_eq!(trie.get(&make_key(&[1])).unwrap(), make_value(1));
        assert_eq!(trie.get(&make_key(&[2])).unwrap(), make_value(2));
        assert_eq!(trie.get(&make_leaf_key(&[0], 1)).unwrap(), make_value(3));
        assert_eq!(trie.get(&make_leaf_key(&[0], 2)).unwrap(), make_value(4));
    }

    #[test]
    fn values_can_be_updated() {
        let trie = SimpleInMemoryVerkleTrie::new();

        let key = make_key(&[1]);
        assert_eq!(trie.get(&key).unwrap(), Value::default());
        trie.set(&key, &make_value(1)).unwrap();
        assert_eq!(trie.get(&key).unwrap(), make_value(1));
        trie.set(&key, &make_value(2)).unwrap();
        assert_eq!(trie.get(&key).unwrap(), make_value(2));
        trie.set(&key, &make_value(3)).unwrap();
        assert_eq!(trie.get(&key).unwrap(), make_value(3));
    }

    #[test]
    fn many_values_can_be_set_and_retrieved() {
        const N: u32 = 1000;

        let to_key = |i: u32| {
            make_leaf_key(
                &[(i >> 8 & 0x0F) as u8, (i >> 4 & 0x0F) as u8],
                (i & 0x0F) as u8,
            )
        };

        let trie = SimpleInMemoryVerkleTrie::new();

        for i in 0..N {
            for j in 0..N {
                let want = if j < i {
                    make_value(j as u64)
                } else {
                    Value::default()
                };
                let got = trie.get(&to_key(j)).unwrap();
                assert_eq!(got, want, "mismatch for key: {:?}", to_key(j));
            }
            trie.set(&to_key(i), &make_value(i as u64)).unwrap();
        }
    }

    #[test]
    fn commitment_of_empty_trie_is_default_commitment() {
        let trie = SimpleInMemoryVerkleTrie::new();
        let have = trie.commit();
        let want = Commitment::default();
        assert_eq!(have, want);
    }

    #[test]
    fn commitment_of_non_empty_trie_is_root_node_commitment() {
        let trie = SimpleInMemoryVerkleTrie::new();
        trie.set(&make_leaf_key(&[1], 1), &make_value(1)).unwrap();
        trie.set(&make_leaf_key(&[2], 2), &make_value(2)).unwrap();
        trie.set(&make_leaf_key(&[3], 3), &make_value(3)).unwrap();

        let have = trie.commit();
        let want = trie.root.lock().unwrap().commit();

        assert_eq!(have, want);
    }
}
