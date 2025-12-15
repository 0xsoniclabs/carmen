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
        managed_trie::TrieCommitment,
        verkle::{KeyedUpdate, KeyedUpdateBatch},
    },
    error::{BTResult, Error},
    types::{Key, Value},
};

/// The result of a call to [`ManagedTrieNode::lookup`].
#[derive(Debug, PartialEq, Eq)]
pub enum LookupResult<ID> {
    /// Indicates that the value associated with the key was found in this node.
    Value(Value),
    /// Indicates that the child node with the given `ID` should be descended into.
    Node(ID),
}

/// The result of a call to [`ManagedTrieNode::next_store_action`].
#[derive(Debug, PartialEq, Eq)]
pub enum StoreAction<'a, ID, U> {
    /// Indicates that the updates can be stored directly in this node.
    Store(KeyedUpdateBatch<'a>),
    /// Indicates that the updates can be stored in child nodes of this node.
    Descend(Vec<DescendAction<'a, ID>>), // TODO: replace with iterator if possible
    /// Indicates that a new node had to be created at this node's depth, which is the
    /// new parent of this node. The contained `U` is the new node.
    HandleReparent(U),
    /// Indicates that this node had to be transformed before the updates could be stored
    /// in it or one of its children. The contained `U` is the transformed node.
    HandleTransform(U),
}

/// The updates to be applied to a child node during a descent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescendAction<'a, ID> {
    /// The ID of the child node to descend into.
    pub id: ID,
    /// The updates to apply in the child node.
    pub updates: KeyedUpdateBatch<'a>,
}

/// A helper trait to constrain a [`ManagedTrieNode`] to be its own union type.
pub trait UnionManagedTrieNode: ManagedTrieNode<Union = Self> {}

/// A generic interface for working with nodes in a managed (ID-based, as opposed to pointer-based)
/// trie (Verkle, Binary, Merkle-Patricia, ...).
///
/// Besides simple value lookup, the trait specifies a set of lifecycle operations that allow to
/// update/store values in the trie using an iterative algorithm, as well as to keep track of
/// a node's commitment dirty status.
///
/// The trait is designed with the following goals in mind:
/// - Decouple nodes from their storage mechanism.
/// - Make nodes agnostic to locking schemes required for concurrent access.
/// - Clearly distinguish between operations that modify the trie structure and those that modify
///   node contents, allowing for accurate tracking of dirty states.
/// - Move shared logic out of the individual node types, such as tree traversal and commitment
///   updates/caching.
///
/// Since not all lifecycle methods make sense for all node types, the trait provides default
/// implementations that return an [`Error::UnsupportedOperation`] for most methods.
pub trait ManagedTrieNode {
    /// The union type (enum) that encompasses all node types in the trie.
    type Union;

    /// The ID type used to identify nodes.
    type Id;

    /// The type used for cryptographic commitments.
    type Commitment: TrieCommitment;

    /// Looks up the value associated with the given key in this node.
    fn lookup(&self, _key: &Key, _depth: u8) -> BTResult<LookupResult<Self::Id>, Error>;

    /// Returns information about the next action required to store a value at the given key.
    fn next_store_action<'a>(
        &self,
        update: KeyedUpdateBatch<'a>,
        depth: u8,
        self_id: Self::Id,
    ) -> BTResult<StoreAction<'a, Self::Id, Self::Union>, Error>;

    /// Replaces the child node at the given key with a new node ID.
    fn replace_child(&mut self, _key: &Key, _depth: u8, _new: Self::Id) -> BTResult<(), Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::replace_child",
            std::any::type_name::<Self>()
        ))
        .into())
    }

    /// Stores the given update in this node.
    /// Returns the previously stored value.
    ///
    /// It is only valid to call this method if
    /// [`next_store_action`](ManagedTrieNode::next_store_action) returned
    /// `StoreAction::Store`.
    // NOTE: We cannot directly do this inside of `next_store_action` because that method
    //       takes `&self` instead of `&mut self`.
    fn store(&mut self, _update: &KeyedUpdate) -> BTResult<Value, Error> {
        Err(Error::UnsupportedOperation(format!("{}::store", std::any::type_name::<Self>())).into())
    }

    /// Returns the commitment associated with this node.
    fn get_commitment(&self) -> Self::Commitment;

    /// Sets the commitment associated with this node.
    fn set_commitment(&mut self, _commitment: Self::Commitment) -> BTResult<(), Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::set_commitment",
            std::any::type_name::<Self>()
        ))
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::BTError;

    struct TestCommitment {}
    impl TrieCommitment for TestCommitment {
        fn modify_child(&mut self, _index: usize) {}
        fn store(&mut self, _index: usize, _prev: Value) {}
    }

    struct TestNode;
    impl ManagedTrieNode for TestNode {
        type Union = TestNode;
        type Id = u32;
        type Commitment = TestCommitment;

        fn lookup(&self, _key: &Key, _depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {
            unimplemented!()
        }

        fn next_store_action<'a>(
            &self,
            _update: KeyedUpdateBatch<'a>,
            _depth: u8,
            _self_id: Self::Id,
        ) -> BTResult<StoreAction<'a, Self::Id, Self::Union>, Error> {
            unimplemented!()
        }

        fn get_commitment(&self) -> Self::Commitment {
            unimplemented!()
        }
    }

    #[test]
    fn default_implementations_return_error() {
        let mut node = TestNode;

        assert!(matches!(
            node.replace_child(&Key::default(), 0, 0).map_err(BTError::into_inner),
            Err(Error::UnsupportedOperation(e)) if e.contains("TestNode::replace_child")
        ));
        assert!(matches!(
            node.store(&KeyedUpdate::FullSlot { key: Key::default(), value: Value::default() }).map_err(BTError::into_inner),
            Err(Error::UnsupportedOperation(e)) if e.contains("TestNode::store")
        ));
        assert!(matches!(
            node.set_commitment(TestCommitment{}).map_err(BTError::into_inner),
            Err(Error::UnsupportedOperation(e)) if e.contains("TestNode::set_commitment")
        ));
    }
}
