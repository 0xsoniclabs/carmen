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
        managed_trie::{LookupResult, ManagedTrieNode, StoreAction},
        verkle::variants::managed::{
            InnerNode, VerkleNode, VerkleNodeId,
            commitment::{VerkleCommitment, VerkleCommitmentInput},
        },
    },
    error::{BTResult, Error},
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
    /// Returns the values and stem of this leaf node as commitment input.
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
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
    type Union = VerkleNode;
    type Id = VerkleNodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, key: &Key, _depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {
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
    ) -> BTResult<StoreAction<Self::Id, Self::Union>, Error> {
        // If key does not match the stem, we have to introduce a new inner node.
        if key[..31] != self.stem[..] {
            let index = key[depth as usize];
            let mut inner = InnerNode::default();
            inner.children[index as usize] = self_id;
            return Ok(StoreAction::HandleReparent(VerkleNode::Inner(Box::new(
                inner,
            ))));
        }

        Ok(StoreAction::Store {
            index: key[31] as usize,
        })
    }

    fn store(&mut self, key: &Key, value: &Value) -> BTResult<Value, Error> {
        if self.stem[..] != key[..31] {
            return Err(Error::CorruptedState(
                "called store on a leaf with non-matching stem".to_owned(),
            )
            .into());
        }

        let suffix = key[31];
        let prev_value = self.values[suffix as usize];
        self.values[suffix as usize] = *value;
        Ok(prev_value)
    }

    fn get_commitment(&self) -> Self::Commitment {
        self.commitment
    }

    fn set_commitment(&mut self, cache: Self::Commitment) -> BTResult<(), Error> {
        self.commitment = cache;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        database::{
            managed_trie::TrieCommitment,
            verkle::{test_utils::FromIndexValues, variants::managed::nodes::VerkleNodeKind},
        },
        error::BTError,
        types::{TreeId, Value},
    };

    #[test]
    fn full_leaf_node_default_returns_leaf_node_with_all_values_set_to_default() {
        let node: FullLeafNode = FullLeafNode::default();
        assert_eq!(node.stem, [0; 31]);
        assert_eq!(node.values, [Value::default(); 256]);
        assert_eq!(node.commitment, VerkleCommitment::default());
    }

    #[test]
    fn get_commitment_input_returns_values_and_stem() {
        let node = FullLeafNode {
            stem: <[u8; 31]>::from_index_values(3, &[]),
            values: (0..=255)
                .map(|i| Value::from_index_values(i, &[]))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            ..Default::default()
        };
        let result = node.get_commitment_input().unwrap();
        assert_eq!(result, VerkleCommitmentInput::Leaf(node.values, node.stem));
    }

    #[test]
    fn lookup_with_matching_stem_returns_value_at_final_key_index() {
        let index = 78;
        let key = Key::from_index_values(1, &[(31, index)]);
        let mut node = FullLeafNode {
            stem: key[..31].try_into().unwrap(),
            ..Default::default()
        };
        let value = Value::from_index_values(42, &[]);
        node.values[index as usize] = value;

        let result = node.lookup(&key, 0).unwrap();
        assert_eq!(result, LookupResult::Value(value));

        // Depth is irrelevant
        let result = node.lookup(&key, 42).unwrap();
        assert_eq!(result, LookupResult::Value(value));

        // Mismatching stem returns default value
        let other_key = Key::from_index_values(7, &[]);
        let other_result = node.lookup(&other_key, 0).unwrap();
        assert_eq!(other_result, LookupResult::Value(Value::default()));

        // Other index has default value
        let other_key = Key::from_index_values(1, &[(31, index + 1)]);
        let other_result = node.lookup(&other_key, 0).unwrap();
        assert_eq!(other_result, LookupResult::Value(Value::default()));
    }

    #[test]
    fn next_store_action_with_non_matching_stem_is_reparent() {
        let node = FullLeafNode::default();
        let depth = 10;
        let index = 78;
        let key = Key::from_index_values(1, &[(depth, index)]);
        let self_id = VerkleNodeId::from_idx_and_node_kind(33, VerkleNodeKind::Leaf256);

        let result = node.next_store_action(&key, depth as u8, self_id).unwrap();
        match result {
            StoreAction::HandleReparent(VerkleNode::Inner(inner)) => {
                assert_eq!(inner.children[index as usize], self_id);
            }
            _ => panic!("expected HandleReparent with inner node"),
        }
    }

    #[test]
    fn next_store_action_with_matching_stem_is_store() {
        let index = 78;
        let key = Key::from_index_values(1, &[(31, index)]);
        let node = FullLeafNode {
            stem: key[..31].try_into().unwrap(),
            ..Default::default()
        };

        let result = node
            .next_store_action(
                &key,
                0,
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Leaf256),
            )
            .unwrap();
        assert_eq!(
            result,
            StoreAction::Store {
                index: index as usize
            }
        );
    }

    #[test]
    fn store_sets_value_at_final_key_index() {
        let index = 78;
        let key = Key::from_index_values(1, &[(31, index)]);
        let mut node = FullLeafNode {
            stem: key[..31].try_into().unwrap(),
            ..Default::default()
        };
        let value = Value::from_index_values(42, &[]);

        node.store(&key, &value).unwrap();
        assert_eq!(node.values[index as usize], value);
    }

    #[test]
    fn store_with_non_matching_stem_returns_error() {
        let key = Key::from_index_values(1, &[(31, 78)]);
        let mut node = FullLeafNode::default();
        let value = Value::from_index_values(42, &[]);

        let result = node.store(&key, &value);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::CorruptedState(_))
        ));
    }

    #[test]
    fn commitment_can_be_set_and_retrieved() {
        let mut node = FullLeafNode::default();
        assert_eq!(node.get_commitment(), VerkleCommitment::default());

        let mut new_commitment = VerkleCommitment::default();
        new_commitment.store(5, Value::from_index_values(4, &[]));

        node.set_commitment(new_commitment).unwrap();
        assert_eq!(node.get_commitment(), new_commitment);
    }
}
