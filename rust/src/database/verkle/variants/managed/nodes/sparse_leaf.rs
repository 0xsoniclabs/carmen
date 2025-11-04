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
            FullLeafNode, InnerNode, Node, NodeId,
            commitment::{OnDiskVerkleCommitment, VerkleCommitment, VerkleCommitmentInput},
            nodes::Leaf2Node,
        },
    },
    error::Error,
    types::{DiskRepresentable, Key, Value},
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseLeafNode<const N: usize> {
    pub stem: [u8; 31],
    pub values: [ValueWithIndex; N],
    pub commitment: VerkleCommitment,
}

#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned)]
#[repr(C)]
pub struct OnDiskSparseLeafNode<const N: usize> {
    pub stem: [u8; 31],
    pub values: [ValueWithIndex; N],
    pub commitment: OnDiskVerkleCommitment,
}

impl<const N: usize> From<OnDiskSparseLeafNode<N>> for SparseLeafNode<N> {
    fn from(on_disk: OnDiskSparseLeafNode<N>) -> Self {
        SparseLeafNode {
            stem: on_disk.stem,
            values: on_disk.values,
            commitment: VerkleCommitment::from(on_disk.commitment),
        }
    }
}

impl<const N: usize> From<&SparseLeafNode<N>> for OnDiskSparseLeafNode<N> {
    fn from(node: &SparseLeafNode<N>) -> Self {
        OnDiskSparseLeafNode {
            stem: node.stem,
            values: node.values,
            commitment: OnDiskVerkleCommitment::from(&node.commitment),
        }
    }
}

impl<const N: usize> DiskRepresentable for SparseLeafNode<N> {
    fn from_disk_repr<E>(
        read_into_buffer: impl FnOnce(&mut [u8]) -> Result<(), E>,
    ) -> Result<Self, E> {
        OnDiskSparseLeafNode::<N>::from_disk_repr(read_into_buffer).map(Into::into)
    }

    fn to_disk_repr(&'_ self) -> Cow<'_, [u8]> {
        Cow::Owned(OnDiskSparseLeafNode::from(self).to_disk_repr().into_owned())
    }

    fn size() -> usize {
        std::mem::size_of::<OnDiskSparseLeafNode<N>>()
    }
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

    fn transform(&self, key: &Key, depth: u8) -> Result<Node, Error> {
        assert_eq!(key[..31], self.stem[..]);
        match N {
            1 => {
                let mut values = [ValueWithIndex::default(); 2];
                values[..1].copy_from_slice(&self.values);
                let new_leaf = Leaf2Node {
                    stem: self.stem,
                    values,
                    commitment: self.commitment,
                };
                Ok(Node::Leaf2(Box::new(new_leaf)))
            }
            2 => {
                let mut values = [ValueWithIndex::default(); 21];
                values[..2].copy_from_slice(&self.values);
                let new_leaf = SparseLeafNode::<21> {
                    stem: self.stem,
                    values,
                    commitment: self.commitment,
                };
                Ok(Node::Leaf21(Box::new(new_leaf)))
            }
            21 => {
                let mut values = [ValueWithIndex::default(); 64];
                values[..21].copy_from_slice(&self.values);
                let new_leaf = SparseLeafNode::<64> {
                    stem: self.stem,
                    values,
                    commitment: self.commitment,
                };
                Ok(Node::Leaf64(Box::new(new_leaf)))
            }
            64 => {
                let mut values = [ValueWithIndex::default(); 141];
                values[..64].copy_from_slice(&self.values);
                let new_leaf = SparseLeafNode::<141> {
                    stem: self.stem,
                    values,
                    commitment: self.commitment,
                };
                Ok(Node::Leaf141(Box::new(new_leaf)))
            }
            _ => {
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
        }
    }

    fn reparent(&self, key: &Key, depth: u8, self_id: NodeId) -> Result<Node, Error> {
        // Otherwise, we have to re-parent.
        let pos = self.stem[depth as usize];
        // TODO: Need better ctor
        let mut inner = InnerNode::default();
        inner.children[pos as usize] = self_id;
        Ok(Node::Inner(Box::new(inner)))
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
impl<const N: usize> ManagedTrieNode for SparseLeafNode<N> {
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

    fn next_store_action(
        &self,
        key: &Key,
        _depth: u8,
        self_id: Self::Id,
    ) -> Result<StoreAction<Self::Id, Self::Union>, Error> {
        if key[..31] != self.stem[..] {
            return Ok(StoreAction::HandleReparent(
                self.reparent(key, _depth, self_id)?,
            ));
        }

        // TODO: Need to thoroughly test this behavior
        for ValueWithIndex { index, value } in &self.values {
            if *index == key[31] || *value == Value::default() {
                return Ok(StoreAction::Store(key[31] as usize));
            }
        }

        Ok(StoreAction::HandleTransform(self.transform(key, _depth)?))
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

// // TODO: This needs to be generic over N
// impl ManagedTrieNode for SparseLeafNode<1> {
//     type Union = Node;
//     type Id = NodeId;
//     type Commitment = VerkleCommitment;

//     fn lookup(&self, key: &Key, _depth: u8) -> Result<LookupResult<Self::Id>, Error> {
//         if key[..31] != self.stem[..] {
//             return Ok(LookupResult::Value(Value::default()));
//         }

//         for ValueWithIndex { index, value } in &self.values {
//             if *index == key[31] {
//                 return Ok(LookupResult::Value(*value));
//             }
//         }
//         Ok(LookupResult::Value(Value::default()))
//     }

//     fn can_store(&self, key: &Key, _depth: u8) -> Result<CanStoreResult<Self::Id>, Error> {
//         if key[..31] != self.stem[..] {
//             return Ok(CanStoreResult::Reparent);
//         }

//         // TODO: Need to thoroughly test this behavior
//         for ValueWithIndex { index, value } in &self.values {
//             if *index == key[31] || *value == Value::default() {
//                 return Ok(CanStoreResult::Yes(key[31] as usize));
//             }
//         }
//         Ok(CanStoreResult::Transform)
//     }

//     fn transform(&self, key: &Key, depth: u8) -> Result<Self::Union, Error> {
//         assert!(matches!(
//             self.can_store(key, depth)?,
//             CanStoreResult::Transform
//         ));

//         assert_eq!(key[..31], self.stem[..]);
//         let mut values = [ValueWithIndex::default(); 2];
//         values[..1].copy_from_slice(&self.values);
//         // If the stems match, we have to convert to a full leaf.
//         let new_leaf = Leaf2Node {
//             stem: self.stem,
//             values,
//             // TODO Test: Commitment is preserved
//             commitment: self.commitment,
//         };
//         Ok(Node::Leaf2(Box::new(new_leaf)))
//     }

//     fn reparent(&self, key: &Key, depth: u8, self_id: NodeId) -> Result<Self::Union, Error> {
//         assert!(matches!(
//             self.can_store(key, depth)?,
//             CanStoreResult::Reparent
//         ));

//         // Otherwise, we have to re-parent.
//         let pos = self.stem[depth as usize];
//         // TODO: Need better ctor
//         let mut inner = InnerNode::default();
//         inner.children[pos as usize] = self_id;
//         Ok(Node::Inner(Box::new(inner)))
//     }

//     fn store(&mut self, key: &Key, value: &Value) -> Result<Value, Error> {
//         assert_eq!(self.stem[..], key[..31]);

//         let mut slot = None;
//         // TODO: Need to thoroughly test this behavior
//         for (i, ValueWithIndex { index, value: v }) in self.values.iter().enumerate() {
//             if *index == key[31] || *v == Value::default() {
//                 slot = Some(i);
//                 break;
//             }
//         }
//         let prev_value = self.values[slot.unwrap()].value;
//         self.values[slot.unwrap()] = ValueWithIndex {
//             index: key[31],
//             value: *value,
//         };

//         Ok(prev_value)
//     }

//     fn get_commitment(&self) -> Self::Commitment {
//         self.commitment
//     }

//     fn set_commitment(&mut self, cache: Self::Commitment) -> Result<(), Error> {
//         self.commitment = cache;
//         Ok(())
//     }
// }

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
