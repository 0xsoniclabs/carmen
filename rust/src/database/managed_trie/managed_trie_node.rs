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
    database::managed_trie::TrieCommitment,
    error::Error,
    types::{Key, Value},
};

/// The result of a call to [`ManagedTrieNode::lookup`].
#[expect(unused)]
pub enum LookupResult<ID> {
    /// Indicates that the value associated with the key was found in this node.
    Value(Value),
    /// Indicates that the child node with the given `ID` should be descended into.
    Node(ID),
}

/// The result of a call to [`ManagedTrieNode::next_store_action`].
#[expect(unused)]
pub enum StoreAction<ID> {
    /// Indicates that the value can be stored directly in this node.
    /// The contained `usize` is the index of the slot in which the value will be stored.
    Store(usize),
    /// Indicates that the value cannot be stored in this node.
    /// The contained `(usize, ID)` is the index and ID of the child node that should be
    /// descended into.
    Descend(usize, ID),
    /// Indicates that a new node needs to be created at this node's depth, which becomes
    /// the parent of this node.
    Reparent,
    /// Indicates that this node needs to be transformed before the value can be stored
    /// in it or one of its children.
    Transform,
}

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
#[cfg_attr(not(test), expect(unused))]
pub trait ManagedTrieNode {
    /// The union type (enum) that encompasses all node types in the trie.
    type Union;

    /// The ID type used to identify nodes.
    type Id;

    /// The type used for cryptographic commitments.
    type Commitment: TrieCommitment;

    /// Looks up the value associated with the given key in the trie node.
    fn lookup(&self, _key: &Key, _depth: u8) -> Result<LookupResult<Self::Id>, Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::lookup",
            std::any::type_name::<Self>()
        )))
    }

    /// Returns information about the next action required to store a value at the given key.
    fn next_store_action(&self, _key: &Key, _depth: u8) -> Result<StoreAction<Self::Id>, Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::next_store_action",
            std::any::type_name::<Self>()
        )))
    }

    /// Transforms this node into a different node type to accommodate a new value at the given key.
    ///
    /// It is only valid to call this method if
    /// [`next_store_action`](ManagedTrieNode::next_store_action) returned
    /// [`StoreAction::Transform`].
    fn transform(&self, _key: &Key, _depth: u8) -> Result<Self::Union, Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::transform",
            std::any::type_name::<Self>()
        )))
    }

    /// Creates a new node that becomes the parent of this node.
    ///
    /// It is only valid to call this method if
    /// [`next_store_action`](ManagedTrieNode::next_store_action) returned
    /// [`StoreAction::Reparent`].
    fn reparent(&self, _key: &Key, _depth: u8, _self_id: Self::Id) -> Result<Self::Union, Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::reparent",
            std::any::type_name::<Self>()
        )))
    }

    /// Replaces the child node at the given key with a new node ID.
    fn replace_child(&mut self, _key: &Key, _depth: u8, _new: Self::Id) -> Result<(), Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::replace_child",
            std::any::type_name::<Self>()
        )))
    }

    /// Stores the given value at the specified key in this node.
    /// Returns the previous value stored at the key.
    ///
    /// It is only valid to call this method if
    /// [`next_store_action`](ManagedTrieNode::next_store_action) returned
    /// [`StoreAction::Store`].
    fn store(&mut self, _key: &Key, _value: &Value) -> Result<Value, Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::store",
            std::any::type_name::<Self>()
        )))
    }

    /// Returns the commitment associated with this node.
    fn get_commitment(&self) -> Self::Commitment;

    /// Sets the commitment associated with this node.
    fn set_commitment(&mut self, _cache: Self::Commitment) -> Result<(), Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::set_commitment",
            std::any::type_name::<Self>()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        fn get_commitment(&self) -> Self::Commitment {
            TestCommitment {}
        }
    }

    #[test]
    fn default_implementations_return_error() {
        let mut node = TestNode;

        assert!(matches!(
            node.lookup(&Key::default(), 0),
            Err(Error::UnsupportedOperation(e)) if e == "TestNode::lookup"
        ));
        assert!(matches!(
            node.next_store_action(&Key::default(), 0),
            Err(Error::UnsupportedOperation(e)) if e == "TestNode::next_store_action"
        ));
        assert!(matches!(
            node.transform(&Key::default(), 0),
            Err(Error::UnsupportedOperation(e)) if e == "TestNode::transform"
        ));
        assert!(matches!(
            node.reparent(&Key::default(), 0, 0),
            Err(Error::UnsupportedOperation(e)) if e == "TestNode::reparent"
        ));
        assert!(matches!(
            node.replace_child(&Key::default(), 0, 0),
            Err(Error::UnsupportedOperation(e)) if e == "TestNode::replace_child"
        ));
        assert!(matches!(
            node.store(&Key::default(), &Value::default()),
            Err(Error::UnsupportedOperation(e)) if e == "TestNode::store"
        ));
        let commitment = node.get_commitment();
        assert!(matches!(
            node.set_commitment(commitment),
            Err(Error::UnsupportedOperation(e)) if e == "TestNode::set_commitment"
        ));
    }
}
