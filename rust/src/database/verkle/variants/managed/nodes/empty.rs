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
        NodeVisitor,
        managed_trie::{LookupResult, ManagedTrieNode, StoreAction},
        verkle::variants::managed::{
            InnerNode, VerkleNode, VerkleNodeId,
            commitment::{VerkleCommitment, VerkleCommitmentInput},
            nodes::make_smallest_leaf_node_for,
        },
    },
    error::{BTResult, Error},
    statistics::trie_count::TrieCountVisitor,
    types::{Key, Value},
};

/// An empty node in a managed Verkle trie.
/// This is a zero-sized type that only exists for implementing traits on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyNode;

impl EmptyNode {
    /// Returns an error as EmptyNode does not support commitment inputs.
    /// This method only exists for implementation symmetry in [`VerkleNode::get_commitment_input`].
    #[cfg_attr(not(test), allow(unused))]
    #[allow(clippy::unused_self, clippy::trivially_copy_pass_by_ref)]
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
        Err(Error::UnsupportedOperation(
            "EmptyNode does not support get_commitment_input".to_owned(),
        )
        .into())
    }
}

impl ManagedTrieNode for EmptyNode {
    type Union = VerkleNode;
    type Id = VerkleNodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, _key: &Key, _depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {
        Ok(LookupResult::Value(Value::default()))
    }

    fn next_store_action(
        &self,
        key: &Key,
        depth: u8,
        _self_id: Self::Id,
    ) -> BTResult<StoreAction<Self::Id, Self::Union>, Error> {
        if depth == 0 {
            // While conceptually it would suffice to create a leaf node here,
            // Geth always creates an inner node (and we want to stay compatible).
            let inner = InnerNode::default();
            Ok(StoreAction::HandleTransform(VerkleNode::Inner(Box::new(
                inner,
            ))))
        } else {
            // Safe to unwrap: Slice is always 31 bytes
            let stem = key[..31].try_into().unwrap();
            let new_leaf = make_smallest_leaf_node_for(1, stem, &[], self.get_commitment())?;
            // TODO: Deleting empty node from NodeManager after transforming will lead to cache
            // misses https://github.com/0xsoniclabs/sonic-admin/issues/385
            Ok(StoreAction::HandleTransform(new_leaf))
        }
    }

    fn get_commitment(&self) -> Self::Commitment {
        VerkleCommitment::default()
    }
}

impl NodeVisitor<EmptyNode> for TrieCountVisitor {
    fn visit(&mut self, node: &EmptyNode, level: u64) -> BTResult<(), Error> {
        self.record_node_statistics(node, level, "Empty", None::<fn(&EmptyNode) -> u64>);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        database::verkle::{test_utils::FromIndexValues, variants::managed::nodes::VerkleNodeKind},
        error::BTError,
        types::TreeId,
    };

    #[test]
    fn get_commitment_input_returns_error() {
        let node = EmptyNode;
        let result = node.get_commitment_input();
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::UnsupportedOperation(e)) if e.contains(
                "EmptyNode does not support get_commitment_input",
            )
        ));
    }

    #[test]
    fn lookup_returns_default_value() {
        let node = EmptyNode;
        let key = Key::default();
        let result = node.lookup(&key, 0).unwrap();
        assert_eq!(result, LookupResult::Value(Value::default()));
    }

    #[test]
    fn next_store_action_at_depth_zero_transforms_to_inner_node() {
        let node = EmptyNode;
        let key = Key::default();
        let action = node
            .next_store_action(
                &key,
                0,
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty),
            )
            .unwrap();
        assert!(matches!(action, StoreAction::HandleTransform(_)));
    }

    #[test]
    fn next_store_action_non_zero_depth_transforms_to_leaf_node() {
        let node = EmptyNode;
        let key = Key::from_index_values(1, &[(31, 34)]);
        let action = node
            .next_store_action(
                &key,
                1,
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty),
            )
            .unwrap();
        match action {
            StoreAction::HandleTransform(leaf) => {
                // This new leaf is able to store the value
                assert_eq!(
                    leaf.next_store_action(
                        &key,
                        1,
                        VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty) // type does not matter
                    ),
                    Ok(StoreAction::Store { index: 34 })
                );
            }
            _ => panic!("expected HandleTransform to leaf node"),
        }
    }

    #[test]
    fn get_commitment_returns_default_commitment() {
        let node = EmptyNode;
        let commitment = node.get_commitment();
        assert_eq!(commitment, VerkleCommitment::default());
    }
}
