// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use crate::{
    database::{
        managed_trie::{LookupResult, ManagedTrieNode, StoreAction},
        verkle::variants::managed::{
            InnerNode, Node, NodeId, SparseLeafNode,
            commitment::{VerkleCommitment, VerkleCommitmentInput},
        },
    },
    error::Error,
    types::{Key, Value},
};

/// An empty node in a managed Verkle trie.
/// This is a zero-sized type that only exists for implementing traits on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyNode;

impl EmptyNode {
    pub fn get_commitment_input(&self) -> Result<VerkleCommitmentInput, Error> {
        Err(Error::UnsupportedOperation(
            "EmptyNode does not support get_commitment_input".to_owned(),
        ))
    }
}

impl ManagedTrieNode for EmptyNode {
    type Union = Node;
    type Id = NodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, _key: &Key, _depth: u8) -> Result<LookupResult<Self::Id>, Error> {
        Ok(LookupResult::Value(Value::default()))
    }

    fn next_store_action(
        &self,
        key: &Key,
        depth: u8,
        _self_id: Self::Id,
    ) -> Result<StoreAction<Self::Id, Self::Union>, Error> {
        if depth == 0 {
            // While conceptually it would suffice to create a leaf node here,
            // Geth always creates an inner node (and we want to stay compatible).
            let inner = InnerNode::default();
            Ok(StoreAction::HandleTransform(Node::Inner(Box::new(inner))))
        } else {
            // TODO: Deleting empty node from NodeManager after transforming will lead to cache
            // misses https://github.com/0xsoniclabs/sonic-admin/issues/385
            let new_leaf = SparseLeafNode::<1> {
                // Safe to unwrap: Slice is always 31 bytes
                stem: key[..31].try_into().unwrap(),
                ..Default::default()
            };
            Ok(StoreAction::HandleTransform(Node::Leaf1(Box::new(
                new_leaf,
            ))))
        }
    }

    // TODO: Is this even needed?
    fn get_commitment(&self) -> Self::Commitment {
        VerkleCommitment::default()
    }

    fn set_commitment(&mut self, _cache: Self::Commitment) -> Result<(), Error> {
        Err(Error::UnsupportedOperation(std::format!(
            "{}::set_commitment",
            std::any::type_name::<Self>()
        )))
    }
}
