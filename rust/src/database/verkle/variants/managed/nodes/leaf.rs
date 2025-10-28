// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::{
    database::{
        managed_trie::{CanStoreResult, LookupResult, ManagedTrieNode},
        verkle::variants::managed::{
            InnerNode, Node, NodeId,
            commitment::{VerkleCommitment, VerkleCommitmentInput},
        },
    },
    error::Error,
    types::{Key, Value},
};

/// A leaf node with 256 children in a managed Verkle trie.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
#[repr(C)]
pub struct FullLeafNode {
    pub stem: [u8; 31],
    pub values: [Value; 256],
    pub commitment: VerkleCommitment,
}

impl FullLeafNode {
    pub fn get_commitment_input(&self) -> Result<VerkleCommitmentInput, Error> {
        Ok(VerkleCommitmentInput::Leaf(self.values, self.stem))
    }
}

impl Default for FullLeafNode {
    fn default() -> Self {
        FullLeafNode {
            stem: [0; 31],
            values: [Value::default(); 256],
            commitment: VerkleCommitment::default(),
        }
    }
}

impl ManagedTrieNode for FullLeafNode {
    type Union = Node;
    type Id = NodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, key: &Key, _depth: u8) -> Result<LookupResult<Self::Id>, Error> {
        if key[..31] != self.stem[..] {
            Ok(LookupResult::Value(Value::default()))
        } else {
            Ok(LookupResult::Value(self.values[key[31] as usize]))
        }
    }

    fn can_store(&self, key: &Key, _depth: u8) -> Result<CanStoreResult<Self::Id>, Error> {
        if key[..31] == self.stem[..] {
            Ok(CanStoreResult::Yes(key[31] as usize))
        } else {
            Ok(CanStoreResult::Reparent)
        }
    }

    fn reparent(&self, key: &Key, depth: u8, self_id: NodeId) -> Result<Self::Union, Error> {
        assert!(matches!(
            self.can_store(key, depth)?,
            CanStoreResult::Reparent
        ));

        let pos = self.stem[depth as usize];
        // TODO: Need better ctor
        let mut inner = InnerNode::default();
        inner.children[pos as usize] = self_id;
        Ok(Node::Inner(Box::new(inner)))
    }

    // TODO: We could implement a conversion to SparseLeafNode if enough values are zero
    // => We would have to retain the used bits however!
    fn store(&mut self, key: &Key, value: &Value) -> Result<Value, Error> {
        assert_eq!(self.stem[..], key[..31]);

        let suffix = key[31];
        let prev_value = self.values[suffix as usize];
        self.values[suffix as usize] = *value;
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
    fn full_leaf_node_default_returns_leaf_node_with_all_values_set_to_default() {
        let node: FullLeafNode = FullLeafNode::default();
        assert_eq!(node.stem, [0; 31]);
        assert_eq!(node.values, [Value::default(); 256]);
        assert_eq!(node.commitment, VerkleCommitment::default());
    }
}
