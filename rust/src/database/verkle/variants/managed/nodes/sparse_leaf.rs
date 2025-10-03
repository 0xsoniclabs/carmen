use crate::{
    database::verkle::{
        CachedCommitment,
        variants::managed::managed_trie_node::{
            CanStoreResult, CommitmentInput, LookupResult, ManagedTrieNode,
        },
    },
    error::Error,
    types::{FullLeafNode, InnerNode, Key, Node, NodeId, SparseLeafNode, Value, ValueWithIndex},
};

// TODO: Implement for generic N?
// => Ensuring that entries are sorted could make things a lot easier
impl ManagedTrieNode for SparseLeafNode<2> {
    type Union = Node;
    type Id = NodeId;

    fn lookup(&self, key: &Key, _depth: u8) -> Result<LookupResult<Self::Id>, Error> {
        if key[..31] != self.stem[..] {
            return Ok(LookupResult::Value(Value::default()));
        }

        for ValueWithIndex { index, value } in &self.values {
            if *index == key[31] {
                return Ok(LookupResult::Value(*value));
            }
        }
        Ok(LookupResult::Value(Value::default()))
    }

    fn can_store(&self, key: &Key, _depth: u8) -> Result<CanStoreResult<Self::Id>, Error> {
        if key[..31] != self.stem[..] {
            return Ok(CanStoreResult::Reparent);
        }

        // TODO: Need to thoroughly test this behavior
        for ValueWithIndex { index, value } in &self.values {
            if *index == key[31] || *value == Value::default() {
                return Ok(CanStoreResult::Yes);
            }
        }
        Ok(CanStoreResult::Transform)
    }

    fn transform(&self, key: &Key, depth: u8) -> Result<Self::Union, Error> {
        assert!(matches!(
            self.can_store(key, depth)?,
            CanStoreResult::Transform
        ));

        assert_eq!(key[..31], self.stem[..]);
        // If the stems match, we have to convert to a full leaf.
        let new_leaf = FullLeafNode {
            stem: self.stem,
            values: {
                let mut values = [Value::default(); 256];
                for ValueWithIndex { index, value } in &self.values {
                    values[*index as usize] = *value;
                }
                values
            },
            used_bits: self.used_bits,
            commitment: CachedCommitment::default(),
        };
        Ok(Node::Leaf256(Box::new(new_leaf)))
    }

    fn reparent(&self, key: &Key, depth: u8, self_id: NodeId) -> Result<Self::Union, Error> {
        assert!(matches!(
            self.can_store(key, depth)?,
            CanStoreResult::Reparent
        ));

        // Otherwise, we have to re-parent.
        let pos = self.stem[depth as usize];
        // TODO: Need better ctor
        let mut inner = InnerNode::default();
        inner.children[pos as usize] = self_id;
        Ok(Node::Inner(Box::new(inner)))
    }

    fn store(&mut self, key: &Key, value: &Value) -> Result<(), Error> {
        assert_eq!(self.stem[..], key[..31]);

        let mut slot = None;
        // TODO: Need to thoroughly test this behavior
        for (i, ValueWithIndex { index, value: v }) in self.values.iter().enumerate() {
            if *index == key[31] || *v == Value::default() {
                slot = Some(i);
                break;
            }
        }
        self.values[slot.unwrap()] = ValueWithIndex {
            index: key[31],
            value: *value,
        };
        // NOTE: Used bits are NOT cleared! The whole point is to remember which slots were
        // modified, even if they are set back to zero.
        // TODO: Test
        self.used_bits[(key[31] / 8) as usize] |= 1 << (key[31] % 8);

        Ok(())
    }

    fn get_cached_commitment(&self) -> CachedCommitment {
        self.commitment
    }

    fn set_cached_commitment(&mut self, cache: CachedCommitment) -> Result<(), Error> {
        self.commitment = cache;
        Ok(())
    }

    // FIXME: This should not have to pass 256 values!
    fn get_commitment_input(&self) -> CommitmentInput<Self::Id> {
        let mut values = [Value::default(); 256];
        for ValueWithIndex { index, value } in &self.values {
            values[*index as usize] = *value;
        }
        CommitmentInput::Leaf(values, self.used_bits, self.stem)
    }
}
