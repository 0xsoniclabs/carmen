// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::borrow::Cow;

use zerocopy::{FromBytes, Immutable, IntoBytes, Unaligned};

use crate::{
    database::{
        managed_trie::{LookupResult, ManagedTrieNode, StoreAction},
        verkle::variants::managed::{
            InnerNode, Node, NodeId,
            commitment::{OnDiskVerkleCommitment, VerkleCommitment, VerkleCommitmentInput},
        },
    },
    error::Error,
    types::{DiskRepresentable, Key, Value},
};

/// A leaf node with 256 children in a managed Verkle trie.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullLeafNode {
    pub stem: [u8; 31],
    pub values: [Value; 256],
    pub commitment: VerkleCommitment,
}

#[derive(Debug, Clone, PartialEq, Eq, Immutable, FromBytes, IntoBytes, Unaligned)]
#[repr(C)]
pub struct OnDiskFullLeafNode {
    pub stem: [u8; 31],
    pub values: [Value; 256],
    pub commitment: OnDiskVerkleCommitment,
}

impl From<&FullLeafNode> for OnDiskFullLeafNode {
    fn from(node: &FullLeafNode) -> Self {
        OnDiskFullLeafNode {
            stem: node.stem,
            values: node.values,
            commitment: OnDiskVerkleCommitment::from(&node.commitment),
        }
    }
}

impl From<OnDiskFullLeafNode> for FullLeafNode {
    fn from(node: OnDiskFullLeafNode) -> Self {
        FullLeafNode {
            stem: node.stem,
            values: node.values,
            commitment: VerkleCommitment::from(node.commitment),
        }
    }
}

impl DiskRepresentable for FullLeafNode {
    fn from_disk_repr<E>(
        read_into_buffer: impl FnOnce(&mut [u8]) -> Result<(), E>,
    ) -> Result<Self, E> {
        OnDiskFullLeafNode::from_disk_repr(read_into_buffer).map(Into::into)
    }

    fn to_disk_repr(&'_ self) -> Cow<'_, [u8]> {
        Cow::Owned(OnDiskFullLeafNode::from(self).to_disk_repr().into_owned())
    }

    fn size() -> usize {
        std::mem::size_of::<OnDiskFullLeafNode>()
    }
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

    fn next_store_action(
        &self,
        key: &Key,
        depth: u8,
        self_id: Self::Id,
    ) -> Result<StoreAction<Self::Id, Self::Union>, Error> {
        if key[..31] == self.stem[..] {
            Ok(StoreAction::Store(key[31] as usize))
        } else {
            let pos = self.stem[depth as usize];
            // TODO: Need better ctor
            let mut inner = InnerNode::default();
            inner.children[pos as usize] = self_id;
            Ok(StoreAction::HandleReparent(Node::Inner(Box::new(inner))))
        }
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
