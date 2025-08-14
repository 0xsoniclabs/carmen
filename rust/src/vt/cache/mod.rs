use std::sync::{Arc, RwLockReadGuard};

use crate::{
    cache::{Cache, NodeCache, NodeCacheEntry, NodeCacheEntryImpl},
    error::{Error, ErrorState},
    storage::Storage,
    types::NodeId,
    vt::commit::{Commitment, Value},
};

// TODO: Naming (currently collides with crypto primitives - Value)
pub type TrieKey = [u8; 32];
pub type TrieValue = [u8; 32];

pub struct Trie<S: Storage<Id = NodeId, Item = Arc<NodeCacheEntryImpl>>> {
    root_id: NodeId,
    cache: NodeCache<S>,
}

pub fn get(
    key: TrieKey,
    depth: usize,
    node: RwLockReadGuard<NodeCachePayload<Node>>,
    parent_lock: RwLockReadGuard<NodeCachePayload<Node>>,
    cache: &NodeCache,
) -> Result<TrieValue, Error> {
    match &node.value {
        Node::Empty => Ok(TrieValue::default()),
        Node::Inner(inner) => {
            if let Some(child_id) = &inner.children[key[depth as usize] as usize] {
                let child_entry = cache.get(*child_id)?;
                let child_node = child_entry.read()?;
                drop(parent_lock); // drop the parent lock before recursing
                get(key, depth + 1, child_node, node, cache)
            } else {
                Ok(TrieValue::default())
            }
        }
        Node::Leaf(leaf) => {
            if key[..31] != leaf.stem[..] {
                Ok(TrieValue::default())
            } else {
                Ok(leaf.values[key[31] as usize])
            }
        }
    }
}

impl<S> Trie<S>
where
    S: Storage<Id = NodeId, Item = Arc<NodeCacheEntryImpl>>,
{
    pub fn new(storage: S, error: Arc<ErrorState>) -> Self {
        let root_id = storage.reserve(node::Node::Empty);
        Trie {
            cache: NodeCache::try_new(storage, 100, error).unwrap(),
            root_id,
        }
    }

    pub fn get(&self, key: &TrieKey) -> Result<TrieValue, Error> {
        let root_node = self.cache.get(self.root_id)?;
        get(
            *key,
            0,
            root_node.read()?,
            NodeCacheEntry::<Node>::default().read()?,
            &self.cache,
        )
    }

    pub fn set(&mut self, key: &TrieKey, value: &TrieValue) -> Result<(), Error> {
        let root_node = self.cache.get(self.root_id)?;

        self.root_id = root_node
            .write()?
            .value
            .set(key, value, 0, self.root_id, &self.cache)?;
        Ok(())
    }

    pub fn commit(&mut self) -> Result<Commitment, Error> {
        let root_node = self.cache.get(self.root_id)?;
        root_node.write()?.value.commit(&self.cache)
    }
}

#[derive(Debug, Clone, Default)]
pub enum Node {
    #[default]
    Empty,
    Inner(InnerNode),
    Leaf(LeafNode),
}

impl Node {
    pub fn set(
        &mut self,
        key: &TrieKey,
        value: &TrieValue,
        depth: usize,
        id: Id,
        cache: &NodeCache,
    ) -> Result<Id, Error> {
        match self {
            Node::Empty => {
                //TODO: Needs to be simplified
                let leaf_node = Node::Leaf(LeafNode::new(key));
                let leaf_id = cache.reserve(&leaf_node);
                let leaf = cache.get(leaf_id)?;
                let mut leaf_guard = leaf.write()?;
                leaf_guard.value = leaf_node;
                leaf_guard.value.set(key, value, depth, leaf_id, cache)
            }
            Node::Inner(inner) => inner.set(key, value, depth, id, cache),
            Node::Leaf(leaf) => leaf.set(key, value, depth, id, cache),
        }
    }

    // TODO: Return reference?
    // TODO: Can we make this use internal mutability to not require &mut self?
    pub fn commit(&mut self, cache: &NodeCache) -> Result<Commitment, Error> {
        match self {
            Node::Empty => Ok(Commitment::default()),
            Node::Inner(inner) => inner.commit(cache),
            Node::Leaf(leaf) => leaf.commit(cache),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InnerNode {
    children: Vec<Option<Id>>,

    /// Cached commitment, only valid if `commitment_clean` is true.S
    /// TODO: Rename to dirty?
    commitment: Commitment,
    dirty: bool,
}

impl InnerNode {
    // pub fn new() -> Self {
    //     let inner = Self::new_empty();
    // }

    fn new() -> Self {
        let mut children = Vec::with_capacity(256);
        for _ in 0..256 {
            children.push(None);
        }
        InnerNode {
            children,
            commitment: Commitment::default(),
            dirty: true,
        }
    }

    pub fn set(
        &mut self,
        key: &TrieKey,
        value: &TrieValue,
        depth: usize,
        id: Id,
        cache: &NodeCache,
    ) -> Result<Id, Error> {
        // TODO: This may need to be split in case the number of children augments
        self.dirty = false;
        let pos = key[depth as usize];
        let (child_id, child_entry) = match self.children[pos as usize] {
            Some(child_id) => (child_id, cache.get(child_id)?),
            None => {
                // Create a new leaf node if there is no child at this position
                let leaf_node = Node::Leaf(LeafNode::new(key));
                let child_id = cache.reserve(&leaf_node);
                (child_id, cache.get(child_id)?)
            }
        };

        self.children[pos as usize] = Some(child_entry.write()?.value.set(
            key,
            value,
            depth + 1,
            child_id,
            cache,
        )?);
        // NOTE: No changes now but the inner node may change its id if it was split
        Ok(id)
    }

    pub fn commit(&mut self, cache: &NodeCache) -> Result<Commitment, Error> {
        if self.dirty {
            return Ok(self.commitment.clone());
        }

        let mut child_commitments = vec![Commitment::default().to_value(); 256];
        for (i, child) in self.children.iter_mut().enumerate() {
            if let Some(node_id) = child {
                let child_entry = cache.get(*node_id)?;
                child_commitments[i] = child_entry.write()?.value.commit(cache)?.to_value();
            }
        }

        self.commitment = Commitment::new(&child_commitments);
        self.dirty = false;
        Ok(self.commitment.clone())
    }
}

// TODO: geth calls this an "extension node"
#[derive(Debug, Clone)]
pub struct LeafNode {
    stem: [u8; 31],
    values: Vec<TrieValue>,
    used: [u8; 256 / 8],

    commitment: Commitment,
    dirty: bool,
}

impl LeafNode {
    pub fn new(key: &TrieKey) -> Self {
        LeafNode {
            stem: key[..31].try_into().expect("Key must be 32 bytes long"),
            values: vec![TrieValue::default(); 256],
            used: [0; 256 / 8],
            commitment: Commitment::default(),
            dirty: true, // TODO: commitment of empty node is the default?
        }
    }

    pub fn set(
        &mut self,
        key: &TrieKey,
        value: &TrieValue,
        depth: usize,
        id: Id,
        cache: &NodeCache,
    ) -> Result<Id, Error> {
        if key[..31] == self.stem[..] {
            let suffix = key[31];
            self.values[suffix as usize] = *value;
            self.used[(suffix / 8) as usize] |= 1 << (suffix % 8);
            self.dirty = true;
            return Ok(id);
        }

        // This leaf needs to be split
        let pos = self.stem[depth as usize];
        let mut inner = InnerNode::new();
        inner.children[pos as usize] = Some(id.clone());
        // Construct a `Node` instance
        let inner = Node::Inner(inner);
        let id = cache.reserve(&inner);
        let inner_node = cache.get(id)?;
        let mut inner_node = inner_node.write()?;
        inner_node.value = inner;
        inner_node.value.set(key, value, depth, id, cache)
    }

    pub fn commit(&mut self, _cache: &NodeCache) -> Result<Commitment, Error> {
        if self.dirty {
            return Ok(self.commitment.clone());
        }

        // The commitment of a leaf node is computed as a Pedersen commitment
        // as follows:
        //
        //    C = Commit([1,stem, C1, C2])
        //
        // where C1 and C2 are the Pedersen commitments of the interleaved modified
        // lower and upper halves of the values stored in the leaf node, computed
        // by:
        //
        //   C1 = Commit([v[0][:16]), v[0][16:]), v[1][:16]), v[1][16:]), ...])
        //   C2 = Commit([v[128][:16]), v[128][16:]), v[129][:16]), v[129][16:]), ...])
        //
        // For details on the commitment procedure, see
        // https://blog.ethereum.org/2021/12/02/verkle-tree-structure#commitment-to-the-values-leaf-nodes

        // Compute the commitment for this leaf node.
        let mut values = vec![vec![Commitment::default().to_value(); 256]; 2];
        for (i, value) in self.values.iter().enumerate() {
            let mut lower = Value::from_le_bytes(&value[..16]);
            let upper = Value::from_le_bytes(&value[16..]);

            if self.is_used(i as u8) {
                lower.set_bit128();
            }

            values[i / 128][(2 * i + 0) % 256] = lower;
            values[i / 128][(2 * i + 1) % 256] = upper;
        }

        let c1 = Commitment::new(&values[0]);
        let c2 = Commitment::new(&values[1]);

        let mut combined = vec![Commitment::default().to_value(); 256];
        combined[0] = Value::new(1);
        combined[1] = Value::from_le_bytes(&self.stem);
        combined[2] = c1.to_value();
        combined[3] = c2.to_value();
        self.commitment = Commitment::new(&combined);
        self.dirty = false;
        Ok(self.commitment.clone())
    }

    fn is_used(&self, suffix: u8) -> bool {
        return self.used[(suffix / 8) as usize] & (1 << (suffix % 8)) != 0;
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::StubStorage;

    use super::*;

    fn make_key(prefix: &[u8]) -> TrieKey {
        let mut key = [0; 32];
        key[..prefix.len()].copy_from_slice(prefix);
        key
    }

    fn make_leaf_key(prefix: &[u8], suffix: u8) -> TrieKey {
        let mut key = make_key(prefix);
        key[31] = suffix;
        key
    }

    fn make_value(value: u64) -> TrieValue {
        let mut val = [0; 32];
        val[0..8].copy_from_slice(&value.to_le_bytes());
        val
    }

    // #[test]
    // fn inner_node_get_returns_zero_if_there_is_no_next_node() {
    //     let inner = InnerNode::new();
    //     let key = [0; 32];
    //     let value = inner.get(&key, 0);
    //     assert_eq!(value, TrieValue::default());
    // }

    // #[test]
    // fn inner_node_get_returns_value_from_next_node() {
    //     let key1 = make_key(&[1, 2, 3]);
    //     let key2 = make_key(&[1, 2, 4]);

    //     let root = Node::Leaf(LeafNode::new(&key1));
    //     let root = root.set(&key1, 2, &make_value(42));
    //     let root = root.set(&key2, 2, &make_value(84));

    //     assert!(
    //         matches!(root, Node::Inner(_)),
    //         "Root should be an InnerNode"
    //     );

    //     assert_eq!(root.get(&key1, 2), make_value(42));
    //     assert_eq!(root.get(&key2, 2), make_value(84));
    // }

    // #[test]
    // fn inner_node_set_creates_new_leaf_if_there_is_no_next_node() {
    //     let key = make_key(&[1, 2, 3]);
    //     let inner = InnerNode::new_empty();
    //     assert!(matches!(inner.children[key[2] as usize], None));

    //     let inner = inner.set(&key, 2, &make_value(42));
    //     let Node::Inner(inner) = inner else {
    //         panic!("Expected InnerNode after set");
    //     };
    //     assert!(matches!(
    //         inner.children[key[2] as usize],
    //         Some(Node::Leaf(_))
    //     ));
    // }

    // #[test]
    // fn inner_node_commit_clean_state_is_tracked() {
    //     let inner = InnerNode::new_empty();
    //     assert!(!inner.commitment_clean);

    //     // Setting a value should mark the commitment as dirty.
    //     let key = make_key(&[1, 2, 3]);
    //     let inner = inner.set(&key, 2, &make_value(42));
    //     let Node::Inner(mut inner) = inner else {
    //         panic!("Expected InnerNode after set");
    //     };
    //     assert!(!inner.commitment_clean);

    //     // Committing should clean the state.
    //     let fist_commitment = inner.commit();
    //     assert!(inner.commitment_clean);

    //     // Committing again should return the same commitment.
    //     let second_commitment = inner.commit();
    //     assert!(inner.commitment_clean);
    //     assert_eq!(fist_commitment, second_commitment);

    //     // Setting another value should mark the commitment as dirty again.
    //     let inner = inner.set(&make_key(&[1, 2, 4]), 2, &make_value(84));
    //     let Node::Inner(inner) = inner else {
    //         panic!("Expected InnerNode after set");
    //     };
    //     assert!(!inner.commitment_clean);
    // }

    // #[test]
    // fn inner_node_commit_computes_commitment_from_children() {
    //     let inner = InnerNode::new_empty();
    //     let key1 = make_key(&[1, 2, 3]);
    //     let key2 = make_key(&[1, 2, 4]);

    //     let inner = inner.set(&key1, 2, &make_value(42));
    //     let inner = inner.set(&key2, 2, &make_value(84));
    //     let Node::Inner(mut inner) = inner else {
    //         panic!("Expected InnerNode after set");
    //     };

    //     let commitment = inner.commit();
    //     // TODO: Implement and check for commitment.is_valid()

    //     let mut child_commitments = vec![Commitment::default().to_value(); 256];
    //     child_commitments[key1[2] as usize] = inner.children[key1[2] as usize]
    //         .as_mut()
    //         .map(|node| node.commit().to_value())
    //         .unwrap();
    //     child_commitments[key2[2] as usize] = inner.children[key2[2] as usize]
    //         .as_mut()
    //         .map(|node| node.commit().to_value())
    //         .unwrap();
    //     let expected_commitment = Commitment::new(&child_commitments);
    //     assert_eq!(commitment, expected_commitment);
    // }

    // #[test]
    // fn leaf_node_new_produces_empty_leaf_with_stem() {
    //     let key = make_key(&[1, 2, 3, 4, 5]);
    //     let leaf = LeafNode::new(&key);

    //     assert_eq!(
    //         &leaf.stem[..],
    //         &key[..31],
    //         "Stem should match the first 31 bytes of the key"
    //     );
    //     assert_eq!(
    //         leaf.values,
    //         vec![TrieValue::default(); 256],
    //         "All values should be initialized to zero"
    //     );
    //     assert_eq!(leaf.used, [0; 256 / 8], "Used bitmap should be empty");
    // }

    // #[test]
    // fn leaf_node_get_returns_value_for_matching_stem() {
    //     let key = make_leaf_key(&[1, 2, 3, 4, 5], 1);
    //     let leaf = LeafNode::new(&key);

    //     // Initially, the value for the key should be zero.
    //     assert_eq!(leaf.get(&key, 0), TrieValue::default());

    //     let leaf = leaf.set(&key, 0, &make_value(42));
    //     let Node::Leaf(leaf) = leaf else {
    //         panic!("Expected LeafNode after set");
    //     };
    //     assert_eq!(leaf.get(&key, 0), make_value(42),);
    // }

    // #[test]
    // fn leaf_node_get_returns_zero_for_non_matching_stem() {
    //     let key1 = make_key(&[1, 2, 3]);
    //     let key2 = make_key(&[4, 5, 6]);
    //     let leaf = LeafNode::new(&key1);
    //     let leaf = leaf.set(&key1, 0, &make_value(42));

    //     assert_eq!(
    //         leaf.get(&key2, 0),
    //         TrieValue::default(),
    //         "Value for non-matching key should be zero"
    //     );
    // }

    // #[test]
    // fn leaf_node_set_splits_leaf_if_steam_does_not_match() {
    //     let key1 = make_key(&[1, 2, 3]);
    //     let key2 = make_key(&[1, 2, 4]);

    //     let leaf = LeafNode::new(&key1);
    //     let leaf = leaf.set(&key1, 0, &make_value(42));

    //     let new_node = leaf.set(&key2, 2, &make_value(84));
    //     let Node::Inner(mut inner) = new_node else {
    //         panic!("Expected InnerNode after set");
    //     };

    //     // Original leaf is now a child of the inner node.
    //     let value = inner.children[key1[2] as usize]
    //         .as_mut()
    //         .unwrap()
    //         .get(&key1, 2);
    //     assert_eq!(value, make_value(42));
    // }

    // #[test]
    // fn leaf_node_can_set_and_get_values() {
    //     let key1 = make_leaf_key(&[1, 2, 3], 1);
    //     let key2 = make_leaf_key(&[1, 2, 3], 2);
    //     let key3 = make_leaf_key(&[1, 2, 3], 3);

    //     let leaf = LeafNode::new(&key1);

    //     assert!(!leaf.is_used(key1[31]));
    //     assert!(!leaf.is_used(key2[31]));
    //     assert!(!leaf.is_used(key3[31]));

    //     assert_eq!(leaf.get(&key1, 0), TrieValue::default());
    //     assert_eq!(leaf.get(&key2, 0), TrieValue::default());
    //     assert_eq!(leaf.get(&key3, 0), TrieValue::default());

    //     // Setting a value for key 1 makes the value retrievable and marks the suffix as used.
    //     let leaf = leaf.set(&key1, 0, &make_value(10));
    //     let Node::Leaf(leaf) = leaf else {
    //         panic!("Expected LeafNode after set");
    //     };

    //     assert!(leaf.is_used(key1[31]));
    //     assert!(!leaf.is_used(key2[31]));
    //     assert!(!leaf.is_used(key3[31]));

    //     assert_eq!(leaf.get(&key1, 0), make_value(10));
    //     assert_eq!(leaf.get(&key2, 0), TrieValue::default());
    //     assert_eq!(leaf.get(&key3, 0), TrieValue::default());

    //     // Setting the value for key 2 to zero does not change the value but marks the suffix as used.
    //     let leaf = leaf.set(&key2, 0, &TrieValue::default());
    //     let Node::Leaf(leaf) = leaf else {
    //         panic!("Expected LeafNode after set");
    //     };

    //     assert!(leaf.is_used(key1[31]));
    //     assert!(leaf.is_used(key2[31]));
    //     assert!(!leaf.is_used(key3[31]));

    //     assert_eq!(leaf.get(&key1, 0), make_value(10));
    //     assert_eq!(leaf.get(&key2, 0), TrieValue::default());
    //     assert_eq!(leaf.get(&key3, 0), TrieValue::default());

    //     // Resetting the value for key 1 to zero does not change the used bitmap.
    //     let leaf = leaf.set(&key1, 0, &TrieValue::default());
    //     let Node::Leaf(leaf) = leaf else {
    //         panic!("Expected LeafNode after set");
    //     };

    //     assert!(leaf.is_used(key1[31]));
    //     assert!(leaf.is_used(key2[31]));
    //     assert!(!leaf.is_used(key3[31]));

    //     assert_eq!(leaf.get(&key1, 0), TrieValue::default());
    //     assert_eq!(leaf.get(&key2, 0), TrieValue::default());
    //     assert_eq!(leaf.get(&key3, 0), TrieValue::default());
    // }

    // #[test]
    // fn leaf_node_can_compute_commitment() {
    //     let key1 = make_leaf_key(&[1, 2, 3], 1);
    //     let key2 = make_leaf_key(&[1, 2, 3], 130);

    //     let mut val1 = [0; 32];
    //     val1[8..16].copy_from_slice(&42u64.to_be_bytes());
    //     let mut val2 = [0; 32];
    //     val2[8..16].copy_from_slice(&84u64.to_be_bytes());

    //     let leaf = LeafNode::new(&key1);
    //     let leaf = leaf.set(&key1, 0, &val1);
    //     let leaf = leaf.set(&key2, 0, &val2);
    //     let Node::Leaf(mut leaf) = leaf else {
    //         panic!("Expected LeafNode after set");
    //     };

    //     let have = leaf.commit();

    //     // TODO: Implement and check for commitment.is_valid()

    //     let mut low1 = Value::from_le_bytes(&val1[..16]);
    //     let mut low2 = Value::from_le_bytes(&val2[..16]);
    //     let high1 = Value::from_le_bytes(&val1[16..]);
    //     let high2 = Value::from_le_bytes(&val2[16..]);
    //     low1.set_bit128();
    //     low2.set_bit128();

    //     let mut c1_values = vec![Commitment::default().to_value(); 256];
    //     let mut c2_values = vec![Commitment::default().to_value(); 256];
    //     c1_values[2] = low1;
    //     c1_values[3] = high1;
    //     c2_values[4] = low2;
    //     c2_values[5] = high2;

    //     let c1 = Commitment::new(&c1_values);
    //     let c2 = Commitment::new(&c2_values);
    //     let mut combined = vec![Commitment::default().to_value(); 256];
    //     combined[0] = Value::new(1);
    //     combined[1] = Value::from_le_bytes(&key1[..31]);
    //     combined[2] = c1.to_value();
    //     combined[3] = c2.to_value();
    //     let want = Commitment::new(&combined);

    //     assert_eq!(have, want);
    // }

    // #[test]
    // fn leaf_node_commitment_clean_state_is_tracked() {
    //     let key1 = make_leaf_key(&[1, 2, 3], 1);
    //     let key2 = make_leaf_key(&[1, 2, 3], 130);

    //     let leaf = LeafNode::new(&key1);
    //     assert!(!leaf.commitment_clean);

    //     let leaf = leaf.set(&key1, 0, &make_value(10));
    //     let Node::Leaf(leaf) = leaf else {
    //         panic!("Expected LeafNode after set");
    //     };
    //     assert!(!leaf.commitment_clean);

    //     let leaf = leaf.set(&key2, 0, &make_value(20));
    //     let Node::Leaf(mut leaf) = leaf else {
    //         panic!("Expected LeafNode after set");
    //     };
    //     assert!(!leaf.commitment_clean);

    //     let first = leaf.commit();
    //     assert!(leaf.commitment_clean);

    //     let second = leaf.commit();
    //     assert!(leaf.commitment_clean);
    //     assert_eq!(first, second);

    //     let leaf = leaf.set(&key1, 0, &make_value(30));
    //     let Node::Leaf(mut leaf) = leaf else {
    //         panic!("Expected LeafNode after set");
    //     };
    //     assert!(!leaf.commitment_clean);

    //     let third = leaf.commit();
    //     assert!(leaf.commitment_clean);

    //     assert_ne!(first, third);
    // }

    #[test]
    fn trie_new_creates_empty_trie() {
        let storage = Box::new(StubStorage::<Node, 100>::new());
        let trie = Trie::new(storage, Arc::new(ErrorState::default()));
        assert_eq!(trie.get(&make_key(&[1])).unwrap(), TrieValue::default());
        assert_eq!(trie.get(&make_key(&[2])).unwrap(), TrieValue::default());
        assert_eq!(trie.get(&make_key(&[3])).unwrap(), TrieValue::default());
    }

    #[test]
    fn trie_values_can_be_set_and_retrieved() {
        let storage = Box::new(StubStorage::<Node, 100>::new());

        let mut trie = Trie::new(storage, Arc::new(ErrorState::default()));

        assert_eq!(trie.get(&make_key(&[1])).unwrap(), TrieValue::default());
        assert_eq!(trie.get(&make_key(&[2])).unwrap(), TrieValue::default());
        assert_eq!(
            trie.get(&make_leaf_key(&[0], 1)).unwrap(),
            TrieValue::default()
        );
        assert_eq!(
            trie.get(&make_leaf_key(&[0], 2)).unwrap(),
            TrieValue::default()
        );

        trie.set(&make_key(&[1]), &make_value(1)).unwrap();

        assert_eq!(trie.get(&make_key(&[1])).unwrap(), make_value(1));
        assert_eq!(trie.get(&make_key(&[2])).unwrap(), TrieValue::default());
        assert_eq!(
            trie.get(&make_leaf_key(&[0], 1)).unwrap(),
            TrieValue::default()
        );
        assert_eq!(
            trie.get(&make_leaf_key(&[0], 2)).unwrap(),
            TrieValue::default()
        );

        trie.set(&make_key(&[2]), &make_value(2));

        assert_eq!(trie.get(&make_key(&[1])).unwrap(), make_value(1));
        assert_eq!(trie.get(&make_key(&[2])).unwrap(), make_value(2));
        assert_eq!(
            trie.get(&make_leaf_key(&[0], 1)).unwrap(),
            TrieValue::default()
        );
        assert_eq!(
            trie.get(&make_leaf_key(&[0], 2)).unwrap(),
            TrieValue::default()
        );

        trie.set(&make_leaf_key(&[0], 1), &make_value(3));

        assert_eq!(trie.get(&make_key(&[1])).unwrap(), make_value(1));
        assert_eq!(trie.get(&make_key(&[2])).unwrap(), make_value(2));
        assert_eq!(trie.get(&make_leaf_key(&[0], 1)).unwrap(), make_value(3));
        assert_eq!(
            trie.get(&make_leaf_key(&[0], 2)).unwrap(),
            TrieValue::default()
        );

        trie.set(&make_leaf_key(&[0], 2), &make_value(4));

        assert_eq!(trie.get(&make_key(&[1])).unwrap(), make_value(1));
        assert_eq!(trie.get(&make_key(&[2])).unwrap(), make_value(2));
        assert_eq!(trie.get(&make_leaf_key(&[0], 1)).unwrap(), make_value(3));
        assert_eq!(trie.get(&make_leaf_key(&[0], 2)).unwrap(), make_value(4));
    }

    // #[test]
    // fn trie_values_can_be_updated() {
    //     let mut trie = Trie::new();

    //     let key = make_key(&[1]);
    //     assert_eq!(trie.get(&key), TrieValue::default());
    //     trie.set(&key, &make_value(1));
    //     assert_eq!(trie.get(&key), make_value(1));
    //     trie.set(&key, &make_value(2));
    //     assert_eq!(trie.get(&key), make_value(2));
    //     trie.set(&key, &make_value(3));
    //     assert_eq!(trie.get(&key), make_value(3));
    // }

    // #[test]
    // fn trie_many_values_can_be_set_and_retrieved() {
    //     const N: u32 = 1000;

    //     let to_key = |i: u32| {
    //         make_leaf_key(
    //             &[(i >> 8 & 0x0F) as u8, (i >> 4 & 0x0F) as u8],
    //             (i & 0x0F) as u8,
    //         )
    //     };

    //     let mut trie = Trie::new();

    //     for i in 0..N {
    //         for j in 0..N {
    //             let want = if j < i {
    //                 make_value(j as u64)
    //             } else {
    //                 TrieValue::default()
    //             };
    //             let got = trie.get(&to_key(j));
    //             assert_eq!(got, want, "Mismatch for key: {:?}", to_key(j));
    //         }
    //         trie.set(&to_key(i), &make_value(i as u64));
    //     }
    // }
}

// pub fn set(
//     key: &TrieKey,
//     value: &TrieValue,
//     depth: usize,
//     id: &Id,
//     node: &mut RwLockWriteGuard<CacheEntryPayload<Node>>,
//     cache: &NodeCache<Node>,
// ) -> Result<Option<Id>, Error> {
//     match &mut node.value {
//         Node::Empty => {
//             let leaf_node = Node::Leaf(LeafNode::new(key));
//             let leaf_id = cache.reserve(&leaf_node);
//             let leaf = cache.get(leaf_id)?;
//             let mut leaf_guard = leaf.write()?;
//             leaf_guard.value = leaf_node;
//             set(key, value, depth, &leaf_id, &mut leaf_guard, cache)
//         }
//         Node::Inner(inner_node) => {
//             // TODO: This may need to be split in case the number of children augments
//             inner_node.commitment_clean = false;
//             let pos = key[depth as usize];
//             let (child_id, child_entry) = match inner_node.children[pos as usize] {
//                 Some(child_id) => (child_id, cache.get(child_id)?),
//                 None => {
//                     // Create a new leaf node if there is no child at this position
//                     let leaf_node = Node::Leaf(LeafNode::new(key));
//                     let child_id = cache.reserve(&leaf_node);
//                     (child_id, cache.get(child_id)?)
//                 }
//             };

//             if let Some(new_child_id) = set(
//                 key,
//                 value,
//                 depth + 1,
//                 &child_id,
//                 &mut child_entry.write()?,
//                 cache,
//             )? {
//                 inner_node.children[pos as usize] = Some(new_child_id);
//             }
//             // NOTE: No changes now but the inner node may change its id if it was split
//             Ok(None)
//         }
//         Node::Leaf(leaf_node) => {
//             if key[..31] == leaf_node.stem[..] {
//                 let suffix = key[31];
//                 leaf_node.values[suffix as usize] = *value;
//                 leaf_node.used[(suffix / 8) as usize] |= 1 << (suffix % 8);
//                 leaf_node.commitment_clean = false;
//                 return Ok(None);
//             }

//             // This leaf needs to be split
//             let pos = leaf_node.stem[depth as usize];
//             let mut inner = InnerNode::new();
//             inner.children[pos as usize] = Some(id.clone());
//             // Construct a `Node` instance
//             let inner = Node::Inner(inner);
//             let id = cache.reserve(&inner);
//             set(key, value, depth, &id, &mut cache.get(id)?.write()?, cache)
//         }
//     }
// }

// pub fn commit(
//     node: &mut RwLockWriteGuard<CacheEntryPayload<Node>>,
//     cache: &NodeCache<Node>,
// ) -> Result<Commitment, Error> {
//     match &mut node.value {
//         Node::Empty => Ok(Commitment::default()),
//         Node::Inner(inner_node) => {
//             if inner_node.commitment_clean {
//                 return Ok(inner_node.commitment.clone());
//             }

//             let mut child_commitments = vec![Commitment::default().to_value(); 256];
//             for (i, child) in inner_node.children.iter_mut().enumerate() {
//                 if let Some(node_id) = child {
//                     let child_entry = cache.get(*node_id)?;
//                     child_commitments[i] = commit(&mut child_entry.write()?, cache)?.to_value();
//                     // child_commitments[i] = child_entry.write()?.value.commit().to_value();
//                 }
//             }

//             inner_node.commitment = Commitment::new(&child_commitments);
//             inner_node.commitment_clean = true;
//             Ok(inner_node.commitment.clone())
//         }
//         Node::Leaf(leaf_node) => {
//             if leaf_node.commitment_clean {
//                 return Ok(leaf_node.commitment.clone());
//             }

//             // The commitment of a leaf node is computed as a Pedersen commitment
//             // as follows:
//             //
//             //    C = Commit([1,stem, C1, C2])
//             //
//             // where C1 and C2 are the Pedersen commitments of the interleaved modified
//             // lower and upper halves of the values stored in the leaf node, computed
//             // by:
//             //
//             //   C1 = Commit([v[0][:16]), v[0][16:]), v[1][:16]), v[1][16:]), ...])
//             //   C2 = Commit([v[128][:16]), v[128][16:]), v[129][:16]), v[129][16:]), ...])
//             //
//             // For details on the commitment procedure, see
//             // https://blog.ethereum.org/2021/12/02/verkle-tree-structure#commitment-to-the-values-leaf-nodes

//             // Compute the commitment for this leaf node.
//             let mut values = vec![vec![Commitment::default().to_value(); 256]; 2];
//             for (i, value) in leaf_node.values.iter().enumerate() {
//                 let mut lower = Value::from_le_bytes(&value[..16]);
//                 let upper = Value::from_le_bytes(&value[16..]);

//                 if leaf_node.is_used(i as u8) {
//                     lower.set_bit128();
//                 }

//                 values[i / 128][(2 * i + 0) % 256] = lower;
//                 values[i / 128][(2 * i + 1) % 256] = upper;
//             }

//             let c1 = Commitment::new(&values[0]);
//             let c2 = Commitment::new(&values[1]);

//             let mut combined = vec![Commitment::default().to_value(); 256];
//             combined[0] = Value::new(1);
//             combined[1] = Value::from_le_bytes(&leaf_node.stem);
//             combined[2] = c1.to_value();
//             combined[3] = c2.to_value();
//             leaf_node.commitment = Commitment::new(&combined);
//             leaf_node.commitment_clean = true;
//             Ok(leaf_node.commitment.clone())
//         }
//     }
// }
