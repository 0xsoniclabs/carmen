use crate::vt::commit::{Commitment, Value};

// TODO: Naming (currently collides with crypto primitives - Value)
pub type TrieKey = [u8; 32];
pub type TrieValue = [u8; 32];

pub struct Trie {
    root: Node,
}

impl Trie {
    pub fn new() -> Self {
        Trie { root: Node::Empty }
    }

    pub fn get(&self, key: &TrieKey) -> TrieValue {
        self.root.get(key, 0)
    }

    pub fn set(&mut self, key: &TrieKey, value: &TrieValue) {
        let root = std::mem::replace(&mut self.root, Node::Empty);
        self.root = root.set(key, 0, value);
    }

    pub fn commit(&mut self) -> Commitment {
        self.root.commit()
    }
}

#[derive(Debug)]
pub enum Node {
    Empty,
    Inner(InnerNode),
    Leaf(LeafNode),
}

impl Node {
    pub fn get(&self, key: &TrieKey, depth: u8) -> TrieValue {
        match self {
            Node::Empty => TrieValue::default(),
            Node::Inner(inner) => inner.get(key, depth),
            Node::Leaf(leaf) => leaf.get(key, depth),
        }
    }

    pub fn set(self, key: &TrieKey, depth: u8, value: &TrieValue) -> Node {
        match self {
            Node::Empty => {
                let leaf = LeafNode::new(key);
                leaf.set(key, depth, value)
            }
            Node::Inner(inner) => inner.set(key, depth, value),
            Node::Leaf(leaf) => leaf.set(key, depth, value),
        }
    }

    // TODO: Return reference?
    // TODO: Can we make this use internal mutability to not require &mut self?
    pub fn commit(&mut self) -> Commitment {
        match self {
            Node::Empty => Commitment::default(),
            Node::Inner(inner) => inner.commit(),
            Node::Leaf(leaf) => leaf.commit(),
        }
    }
}

#[derive(Debug)]
pub struct InnerNode {
    children: Vec<Option<Node>>,

    /// Cached commitment, only valid if `commitment_clean` is true.S
    /// TODO: Rename to dirty?
    commitment: Commitment,
    commitment_clean: bool,
}

impl InnerNode {
    pub fn new(leaf: LeafNode, position: u8) -> Self {
        let mut inner = Self::new_empty();
        inner.children[position as usize] = Some(Node::Leaf(leaf));
        inner
    }

    /// Creates an empty inner node.
    /// Used in [InnerNode::new] and for testing.
    fn new_empty() -> Self {
        let mut children = Vec::with_capacity(256);
        for _ in 0..256 {
            children.push(None);
        }
        InnerNode {
            children,
            commitment: Commitment::default(),
            commitment_clean: false,
        }
    }

    pub fn get(&self, key: &TrieKey, depth: u8) -> TrieValue {
        if let Some(child) = &self.children[key[depth as usize] as usize] {
            child.get(key, depth + 1)
        } else {
            TrieValue::default()
        }
    }

    pub fn set(mut self, key: &TrieKey, depth: u8, value: &TrieValue) -> Node {
        self.commitment_clean = false;

        let pos = key[depth as usize];
        let next = self.children[pos as usize]
            .take()
            .unwrap_or(Node::Leaf(LeafNode::new(key)));
        self.children[pos as usize] = Some(next.set(key, depth + 1, value));
        return Node::Inner(self);
    }

    pub fn commit(&mut self) -> Commitment {
        if self.commitment_clean {
            return self.commitment.clone();
        }

        let mut child_commitments = vec![Commitment::default().to_value(); 256];
        for (i, child) in self.children.iter_mut().enumerate() {
            if let Some(node) = child {
                child_commitments[i] = node.commit().to_value();
            }
        }

        self.commitment = Commitment::new(&child_commitments);
        self.commitment_clean = true;
        self.commitment.clone()
    }
}

// TODO: geth calls this an "extension node"
#[derive(Debug)]
pub struct LeafNode {
    stem: [u8; 31],
    values: Vec<TrieValue>,
    used: [u8; 256 / 8],

    commitment: Commitment,
    commitment_clean: bool,
}

impl LeafNode {
    pub fn new(key: &TrieKey) -> Self {
        LeafNode {
            stem: key[..31].try_into().expect("Key must be 32 bytes long"),
            values: vec![TrieValue::default(); 256],
            used: [0; 256 / 8],
            commitment: Commitment::default(),
            commitment_clean: false,
        }
    }

    // TODO: What about depth parameter? Get rid of?
    pub fn get(&self, key: &TrieKey, _depth: u8) -> TrieValue {
        if key[..31] != self.stem[..] {
            TrieValue::default()
        } else {
            self.values[key[31] as usize]
        }
    }

    pub fn set(mut self, key: &TrieKey, depth: u8, value: &TrieValue) -> Node {
        if key[..31] == self.stem[..] {
            let suffix = key[31];
            self.values[suffix as usize] = *value;
            self.used[(suffix / 8) as usize] |= 1 << (suffix % 8);
            self.commitment_clean = false;
            return Node::Leaf(self);
        }

        // This leaf needs to be split
        let pos = self.stem[depth as usize];
        let inner = InnerNode::new(self, pos);
        return inner.set(key, depth, value);
    }

    pub fn commit(&mut self) -> Commitment {
        if self.commitment_clean {
            return self.commitment.clone();
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
        self.commitment_clean = true;
        self.commitment.clone()
    }

    fn is_used(&self, suffix: u8) -> bool {
        return self.used[(suffix / 8) as usize] & (1 << (suffix % 8)) != 0;
    }
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn inner_node_get_returns_zero_if_there_is_no_next_node() {
        let inner = InnerNode::new_empty();
        let key = [0; 32];
        let value = inner.get(&key, 0);
        assert_eq!(value, TrieValue::default());
    }

    #[test]
    fn inner_node_get_returns_value_from_next_node() {
        let key1 = make_key(&[1, 2, 3]);
        let key2 = make_key(&[1, 2, 4]);

        let root = Node::Leaf(LeafNode::new(&key1));
        let root = root.set(&key1, 2, &make_value(42));
        let root = root.set(&key2, 2, &make_value(84));

        assert!(
            matches!(root, Node::Inner(_)),
            "Root should be an InnerNode"
        );

        assert_eq!(root.get(&key1, 2), make_value(42));
        assert_eq!(root.get(&key2, 2), make_value(84));
    }

    #[test]
    fn inner_node_set_creates_new_leaf_if_there_is_no_next_node() {
        let key = make_key(&[1, 2, 3]);
        let inner = InnerNode::new_empty();
        assert!(matches!(inner.children[key[2] as usize], None));

        let inner = inner.set(&key, 2, &make_value(42));
        let Node::Inner(inner) = inner else {
            panic!("Expected InnerNode after set");
        };
        assert!(matches!(
            inner.children[key[2] as usize],
            Some(Node::Leaf(_))
        ));
    }

    #[test]
    fn inner_node_commit_clean_state_is_tracked() {
        let inner = InnerNode::new_empty();
        assert!(!inner.commitment_clean);

        // Setting a value should mark the commitment as dirty.
        let key = make_key(&[1, 2, 3]);
        let inner = inner.set(&key, 2, &make_value(42));
        let Node::Inner(mut inner) = inner else {
            panic!("Expected InnerNode after set");
        };
        assert!(!inner.commitment_clean);

        // Committing should clean the state.
        let fist_commitment = inner.commit();
        assert!(inner.commitment_clean);

        // Committing again should return the same commitment.
        let second_commitment = inner.commit();
        assert!(inner.commitment_clean);
        assert_eq!(fist_commitment, second_commitment);

        // Setting another value should mark the commitment as dirty again.
        let inner = inner.set(&make_key(&[1, 2, 4]), 2, &make_value(84));
        let Node::Inner(inner) = inner else {
            panic!("Expected InnerNode after set");
        };
        assert!(!inner.commitment_clean);
    }

    #[test]
    fn inner_node_commit_computes_commitment_from_children() {
        let inner = InnerNode::new_empty();
        let key1 = make_key(&[1, 2, 3]);
        let key2 = make_key(&[1, 2, 4]);

        let inner = inner.set(&key1, 2, &make_value(42));
        let inner = inner.set(&key2, 2, &make_value(84));
        let Node::Inner(mut inner) = inner else {
            panic!("Expected InnerNode after set");
        };

        let commitment = inner.commit();
        // TODO: Implement and check for commitment.is_valid()

        let mut child_commitments = vec![Commitment::default().to_value(); 256];
        child_commitments[key1[2] as usize] = inner.children[key1[2] as usize]
            .as_mut()
            .map(|node| node.commit().to_value())
            .unwrap();
        child_commitments[key2[2] as usize] = inner.children[key2[2] as usize]
            .as_mut()
            .map(|node| node.commit().to_value())
            .unwrap();
        let expected_commitment = Commitment::new(&child_commitments);
        assert_eq!(commitment, expected_commitment);
    }

    #[test]
    fn leaf_node_new_produces_empty_leaf_with_stem() {
        let key = make_key(&[1, 2, 3, 4, 5]);
        let leaf = LeafNode::new(&key);

        assert_eq!(
            &leaf.stem[..],
            &key[..31],
            "Stem should match the first 31 bytes of the key"
        );
        assert_eq!(
            leaf.values,
            vec![TrieValue::default(); 256],
            "All values should be initialized to zero"
        );
        assert_eq!(leaf.used, [0; 256 / 8], "Used bitmap should be empty");
    }

    #[test]
    fn leaf_node_get_returns_value_for_matching_stem() {
        let key = make_leaf_key(&[1, 2, 3, 4, 5], 1);
        let leaf = LeafNode::new(&key);

        // Initially, the value for the key should be zero.
        assert_eq!(leaf.get(&key, 0), TrieValue::default());

        let leaf = leaf.set(&key, 0, &make_value(42));
        let Node::Leaf(leaf) = leaf else {
            panic!("Expected LeafNode after set");
        };
        assert_eq!(leaf.get(&key, 0), make_value(42),);
    }

    #[test]
    fn leaf_node_get_returns_zero_for_non_matching_stem() {
        let key1 = make_key(&[1, 2, 3]);
        let key2 = make_key(&[4, 5, 6]);
        let leaf = LeafNode::new(&key1);
        let leaf = leaf.set(&key1, 0, &make_value(42));

        assert_eq!(
            leaf.get(&key2, 0),
            TrieValue::default(),
            "Value for non-matching key should be zero"
        );
    }

    #[test]
    fn leaf_node_set_splits_leaf_if_steam_does_not_match() {
        let key1 = make_key(&[1, 2, 3]);
        let key2 = make_key(&[1, 2, 4]);

        let leaf = LeafNode::new(&key1);
        let leaf = leaf.set(&key1, 0, &make_value(42));

        let new_node = leaf.set(&key2, 2, &make_value(84));
        let Node::Inner(mut inner) = new_node else {
            panic!("Expected InnerNode after set");
        };

        // Original leaf is now a child of the inner node.
        let value = inner.children[key1[2] as usize]
            .as_mut()
            .unwrap()
            .get(&key1, 2);
        assert_eq!(value, make_value(42));
    }

    #[test]
    fn leaf_node_can_set_and_get_values() {
        let key1 = make_leaf_key(&[1, 2, 3], 1);
        let key2 = make_leaf_key(&[1, 2, 3], 2);
        let key3 = make_leaf_key(&[1, 2, 3], 3);

        let leaf = LeafNode::new(&key1);

        assert!(!leaf.is_used(key1[31]));
        assert!(!leaf.is_used(key2[31]));
        assert!(!leaf.is_used(key3[31]));

        assert_eq!(leaf.get(&key1, 0), TrieValue::default());
        assert_eq!(leaf.get(&key2, 0), TrieValue::default());
        assert_eq!(leaf.get(&key3, 0), TrieValue::default());

        // Setting a value for key 1 makes the value retrievable and marks the suffix as used.
        let leaf = leaf.set(&key1, 0, &make_value(10));
        let Node::Leaf(leaf) = leaf else {
            panic!("Expected LeafNode after set");
        };

        assert!(leaf.is_used(key1[31]));
        assert!(!leaf.is_used(key2[31]));
        assert!(!leaf.is_used(key3[31]));

        assert_eq!(leaf.get(&key1, 0), make_value(10));
        assert_eq!(leaf.get(&key2, 0), TrieValue::default());
        assert_eq!(leaf.get(&key3, 0), TrieValue::default());

        // Setting the value for key 2 to zero does not change the value but marks the suffix as used.
        let leaf = leaf.set(&key2, 0, &TrieValue::default());
        let Node::Leaf(leaf) = leaf else {
            panic!("Expected LeafNode after set");
        };

        assert!(leaf.is_used(key1[31]));
        assert!(leaf.is_used(key2[31]));
        assert!(!leaf.is_used(key3[31]));

        assert_eq!(leaf.get(&key1, 0), make_value(10));
        assert_eq!(leaf.get(&key2, 0), TrieValue::default());
        assert_eq!(leaf.get(&key3, 0), TrieValue::default());

        // Resetting the value for key 1 to zero does not change the used bitmap.
        let leaf = leaf.set(&key1, 0, &TrieValue::default());
        let Node::Leaf(leaf) = leaf else {
            panic!("Expected LeafNode after set");
        };

        assert!(leaf.is_used(key1[31]));
        assert!(leaf.is_used(key2[31]));
        assert!(!leaf.is_used(key3[31]));

        assert_eq!(leaf.get(&key1, 0), TrieValue::default());
        assert_eq!(leaf.get(&key2, 0), TrieValue::default());
        assert_eq!(leaf.get(&key3, 0), TrieValue::default());
    }

    #[test]
    fn leaf_node_can_compute_commitment() {
        let key1 = make_leaf_key(&[1, 2, 3], 1);
        let key2 = make_leaf_key(&[1, 2, 3], 130);

        let mut val1 = [0; 32];
        val1[8..16].copy_from_slice(&42u64.to_be_bytes());
        let mut val2 = [0; 32];
        val2[8..16].copy_from_slice(&84u64.to_be_bytes());

        let leaf = LeafNode::new(&key1);
        let leaf = leaf.set(&key1, 0, &val1);
        let leaf = leaf.set(&key2, 0, &val2);
        let Node::Leaf(mut leaf) = leaf else {
            panic!("Expected LeafNode after set");
        };

        let have = leaf.commit();

        // TODO: Implement and check for commitment.is_valid()

        let mut low1 = Value::from_le_bytes(&val1[..16]);
        let mut low2 = Value::from_le_bytes(&val2[..16]);
        let high1 = Value::from_le_bytes(&val1[16..]);
        let high2 = Value::from_le_bytes(&val2[16..]);
        low1.set_bit128();
        low2.set_bit128();

        let mut c1_values = vec![Commitment::default().to_value(); 256];
        let mut c2_values = vec![Commitment::default().to_value(); 256];
        c1_values[2] = low1;
        c1_values[3] = high1;
        c2_values[4] = low2;
        c2_values[5] = high2;

        let c1 = Commitment::new(&c1_values);
        let c2 = Commitment::new(&c2_values);
        let mut combined = vec![Commitment::default().to_value(); 256];
        combined[0] = Value::new(1);
        combined[1] = Value::from_le_bytes(&key1[..31]);
        combined[2] = c1.to_value();
        combined[3] = c2.to_value();
        let want = Commitment::new(&combined);

        assert_eq!(have, want);
    }

    #[test]
    fn leaf_node_commitment_clean_state_is_tracked() {
        let key1 = make_leaf_key(&[1, 2, 3], 1);
        let key2 = make_leaf_key(&[1, 2, 3], 130);

        let leaf = LeafNode::new(&key1);
        assert!(!leaf.commitment_clean);

        let leaf = leaf.set(&key1, 0, &make_value(10));
        let Node::Leaf(leaf) = leaf else {
            panic!("Expected LeafNode after set");
        };
        assert!(!leaf.commitment_clean);

        let leaf = leaf.set(&key2, 0, &make_value(20));
        let Node::Leaf(mut leaf) = leaf else {
            panic!("Expected LeafNode after set");
        };
        assert!(!leaf.commitment_clean);

        let first = leaf.commit();
        assert!(leaf.commitment_clean);

        let second = leaf.commit();
        assert!(leaf.commitment_clean);
        assert_eq!(first, second);

        let leaf = leaf.set(&key1, 0, &make_value(30));
        let Node::Leaf(mut leaf) = leaf else {
            panic!("Expected LeafNode after set");
        };
        assert!(!leaf.commitment_clean);

        let third = leaf.commit();
        assert!(leaf.commitment_clean);

        assert_ne!(first, third);
    }

    #[test]
    fn trie_new_creates_empty_trie() {
        let trie = Trie::new();
        assert_eq!(trie.get(&make_key(&[1])), TrieValue::default());
        assert_eq!(trie.get(&make_key(&[2])), TrieValue::default());
        assert_eq!(trie.get(&make_key(&[3])), TrieValue::default());
    }

    #[test]
    fn trie_values_can_be_set_and_retrieved() {
        let mut trie = Trie::new();

        assert_eq!(trie.get(&make_key(&[1])), TrieValue::default());
        assert_eq!(trie.get(&make_key(&[2])), TrieValue::default());
        assert_eq!(trie.get(&make_leaf_key(&[0], 1)), TrieValue::default());
        assert_eq!(trie.get(&make_leaf_key(&[0], 2)), TrieValue::default());

        trie.set(&make_key(&[1]), &make_value(1));

        assert_eq!(trie.get(&make_key(&[1])), make_value(1));
        assert_eq!(trie.get(&make_key(&[2])), TrieValue::default());
        assert_eq!(trie.get(&make_leaf_key(&[0], 1)), TrieValue::default());
        assert_eq!(trie.get(&make_leaf_key(&[0], 2)), TrieValue::default());

        trie.set(&make_key(&[2]), &make_value(2));

        assert_eq!(trie.get(&make_key(&[1])), make_value(1));
        assert_eq!(trie.get(&make_key(&[2])), make_value(2));
        assert_eq!(trie.get(&make_leaf_key(&[0], 1)), TrieValue::default());
        assert_eq!(trie.get(&make_leaf_key(&[0], 2)), TrieValue::default());

        trie.set(&make_leaf_key(&[0], 1), &make_value(3));

        assert_eq!(trie.get(&make_key(&[1])), make_value(1));
        assert_eq!(trie.get(&make_key(&[2])), make_value(2));
        assert_eq!(trie.get(&make_leaf_key(&[0], 1)), make_value(3));
        assert_eq!(trie.get(&make_leaf_key(&[0], 2)), TrieValue::default());

        trie.set(&make_leaf_key(&[0], 2), &make_value(4));

        assert_eq!(trie.get(&make_key(&[1])), make_value(1));
        assert_eq!(trie.get(&make_key(&[2])), make_value(2));
        assert_eq!(trie.get(&make_leaf_key(&[0], 1)), make_value(3));
        assert_eq!(trie.get(&make_leaf_key(&[0], 2)), make_value(4));
    }

    #[test]
    fn trie_values_can_be_updated() {
        let mut trie = Trie::new();

        let key = make_key(&[1]);
        assert_eq!(trie.get(&key), TrieValue::default());
        trie.set(&key, &make_value(1));
        assert_eq!(trie.get(&key), make_value(1));
        trie.set(&key, &make_value(2));
        assert_eq!(trie.get(&key), make_value(2));
        trie.set(&key, &make_value(3));
        assert_eq!(trie.get(&key), make_value(3));
    }

    #[test]
    fn trie_many_values_can_be_set_and_retrieved() {
        const N: u32 = 1000;

        let to_key = |i: u32| {
            make_leaf_key(
                &[(i >> 8 & 0x0F) as u8, (i >> 4 & 0x0F) as u8],
                (i & 0x0F) as u8,
            )
        };

        let mut trie = Trie::new();

        for i in 0..N {
            for j in 0..N {
                let want = if j < i {
                    make_value(j as u64)
                } else {
                    TrieValue::default()
                };
                let got = trie.get(&to_key(j));
                assert_eq!(got, want, "Mismatch for key: {:?}", to_key(j));
            }
            trie.set(&to_key(i), &make_value(i as u64));
        }
    }
}
