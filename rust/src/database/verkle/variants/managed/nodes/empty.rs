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
        managed_trie::{CachedCommitment, CanStoreResult, LookupResult, ManagedTrieNode},
        verkle::{
            crypto::Commitment,
            variants::managed::{
                InnerNode, Node, NodeId, SparseLeafNode, commitment::VerkleCommitmentInput,
            },
        },
    },
    error::Error,
    types::{Key, Value},
};

/// TODO Docblock
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyNode;

impl ManagedTrieNode for EmptyNode {
    type Union = Node;
    type Id = NodeId;
    type Commitment = Commitment;
    type CommitmentInput = VerkleCommitmentInput;

    fn lookup(&self, _key: &Key, _depth: u8) -> Result<LookupResult<Self::Id>, Error> {
        Ok(LookupResult::Value(Value::default()))
    }

    fn can_store(&self, _key: &Key, _depth: u8) -> Result<CanStoreResult<Self::Id>, Error> {
        Ok(CanStoreResult::Transform)
    }

    // TODO: Deleting empty node from NodeManager after transforming will lead to cache misses
    // https://github.com/0xsoniclabs/sonic-admin/issues/385
    fn transform(&self, key: &Key, depth: u8) -> Result<Self::Union, Error> {
        if depth == 0 {
            // While conceptually it would suffice to create a leaf node here,
            // Geth always creates an inner node (and we want to stay compatible).
            let inner = InnerNode::default();
            Ok(Node::Inner(Box::new(inner)))
        } else {
            let new_leaf = SparseLeafNode::<2> {
                // Safe to unwrap: Slice is always 31 bytes
                stem: key[..31].try_into().unwrap(),
                ..Default::default()
            };
            Ok(Node::Leaf2(Box::new(new_leaf)))
        }
    }

    fn get_cached_commitment(&self) -> CachedCommitment<Self::Commitment> {
        CachedCommitment::default()
    }
}
