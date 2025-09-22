// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use zerocopy::{FromBytes, Immutable, IntoBytes, Unaligned};

use crate::{
    database::{
        managed_trie::{CanStoreResult, LookupResult, ManagedTrieNode},
        verkle::variants::managed::{
            FullLeafNode, InnerNode, Node, NodeId,
            commitment::{VerkleCommitment, VerkleCommitmentInput},
        },
    },
    error::Error,
    types::{Key, Value},
};

/// A value of a leaf node in a managed Verkle trie, together with its index.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned,
)]
#[repr(C)]
pub struct ValueWithIndex {
    /// The index of the value in the leaf node.
    pub index: u8,
    /// The value stored in the leaf node.
    pub value: Value,
}

/// A sparsely populated leaf node in a managed Verkle trie.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
#[repr(C)]
pub struct SparseLeafNode<const N: usize> {
    pub stem: [u8; 31],
    pub values: [ValueWithIndex; N],
    pub commitment: VerkleCommitment,
}

impl<const N: usize> SparseLeafNode<N> {
    // TODO: This should not have to pass 256 values: https://github.com/0xsoniclabs/sonic-admin/issues/384
    pub fn get_commitment_input(&self) -> Result<VerkleCommitmentInput, Error> {
        let mut values = [Value::default(); 256];
        for ValueWithIndex { index, value } in &self.values {
            values[*index as usize] = *value;
        }
        Ok(VerkleCommitmentInput::Leaf(values, self.stem))
    }
}

impl<const N: usize> Default for SparseLeafNode<N> {
    fn default() -> Self {
        let mut values = [ValueWithIndex::default(); N];
        values.iter_mut().enumerate().for_each(|(i, v)| {
            v.index = i as u8;
        });

        SparseLeafNode {
            stem: [0; 31],
            values,
            commitment: VerkleCommitment::default(),
        }
    }
}

// TODO: Implement for generic N?
// => Ensuring that entries are sorted could make things a lot easier
impl ManagedTrieNode for SparseLeafNode<2> {
    type Union = Node;
    type Id = NodeId;
    type Commitment = VerkleCommitment;

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
                return Ok(CanStoreResult::Yes(key[31] as usize));
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
            // TODO Test: Commitment is preserved
            commitment: self.commitment,
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

    fn store(&mut self, key: &Key, value: &Value) -> Result<Value, Error> {
        assert_eq!(self.stem[..], key[..31]);

        let mut slot = None;
        // TODO: Need to thoroughly test this behavior
        for (i, ValueWithIndex { index, value: v }) in self.values.iter().enumerate() {
            if *index == key[31] || *v == Value::default() {
                slot = Some(i);
                break;
            }
        }
        let prev_value = self.values[slot.unwrap()].value;
        self.values[slot.unwrap()] = ValueWithIndex {
            index: key[31],
            value: *value,
        };

        Ok(prev_value)
    }

    fn get_commitment(&self) -> Self::Commitment {
        self.commitment
    }

    fn set_commitment(&mut self, cache: Self::Commitment) -> Result<(), Error> {
        self.commitment = cache;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_leaf_node_default_returns_leaf_node_with_default_values_and_unique_indices() {
        const N: usize = 2;
        let node: SparseLeafNode<N> = SparseLeafNode::default();

        assert_eq!(node.stem, [0; 31]);
        assert_eq!(node.commitment, VerkleCommitment::default());

        for (i, value) in node.values.iter().enumerate() {
            assert_eq!(value.index, i as u8);
            assert_eq!(value.value, Value::default());
        }
    }
}
