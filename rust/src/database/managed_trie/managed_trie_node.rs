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

pub enum LookupResult<IdType> {
    Value(Value),
    Node(IdType),
}

pub enum CanStoreResult<IdType> {
    /// Indicates that the value can be stored directly in this node.
    /// The contained `usize` is the index where the value will be stored.
    Yes(usize),
    /// Indicates that the value cannot be stored in this node, and that we need to
    /// descend into the child node at the given index.
    Descend(usize, IdType),
    /// Indicates that a new node needs to be created at this node's depth, which becomes
    /// the parent of this node.
    Reparent,
    /// Indicates that the node needs to be transformed before the value can be stored
    /// in it or one of its children.
    Transform,
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
///
/// TODO Test default error?
pub trait ManagedTrieNode {
    /// The union type (enum) that encompasses all node types in the trie.
    type Union;

    /// The ID type used to identify nodes.
    type Id;

    /// The type used for cryptographic commitments.
    type Commitment: TrieCommitment;

    /// TODO: Docblock
    fn lookup(&self, _key: &Key, _depth: u8) -> Result<LookupResult<Self::Id>, Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::lookup",
            std::any::type_name::<Self>()
        )))
    }

    /// TODO: Docblock
    fn can_store(&self, _key: &Key, _depth: u8) -> Result<CanStoreResult<Self::Id>, Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::can_store",
            std::any::type_name::<Self>()
        )))
    }

    /// TODO: Docblock
    fn transform(&self, _key: &Key, _depth: u8) -> Result<Self::Union, Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::transform",
            std::any::type_name::<Self>()
        )))
    }

    /// TODO: Docblock
    fn reparent(&self, _key: &Key, _depth: u8, _self_id: Self::Id) -> Result<Self::Union, Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::reparent",
            std::any::type_name::<Self>()
        )))
    }

    /// TODO: Docblock
    fn replace_child(&mut self, _key: &Key, _depth: u8, _new: Self::Id) -> Result<(), Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::replace_child",
            std::any::type_name::<Self>()
        )))
    }

    /// TODO: Docblock
    /// Returns the previous value stored at the key.
    fn store(&mut self, _key: &Key, _value: &Value) -> Result<Value, Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::store",
            std::any::type_name::<Self>()
        )))
    }

    /// TODO: Docblock
    fn get_commitment(&self) -> Self::Commitment;

    /// TODO: Docblock
    fn set_commitment(&mut self, _cache: Self::Commitment) -> Result<(), Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::set_commitment",
            std::any::type_name::<Self>()
        )))
    }
}
