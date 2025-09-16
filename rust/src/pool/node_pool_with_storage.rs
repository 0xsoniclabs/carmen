// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::ops::{Deref, DerefMut};

use crate::types::Node;

/// A [`Node`] with additional metadata about the node lifecycle.
/// [`NodeWithMetadata`] automatically dereferences to `Node` via the [`Deref`] trait.
/// The node's status is set to [`NodeStatus::Dirty`] when a mutable reference is requested.
/// Accessing a deleted node will panic.
#[derive(Debug, PartialEq, Eq)]
pub struct NodeWithMetadata {
    value: Node,
    status: NodeStatus,
}

/// The status of a [`NodeWithMetadata`].
/// It can be:
/// - `Clean`: the node is in sync with the storage
/// - `Dirty`: the node has been modified and needs to be flushed to storage
/// - `Deleted`: the node has been deleted and should not be used anymore
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeStatus {
    Clean,
    Dirty,
    Deleted,
}

impl NodeWithMetadata {
    /// Creates a new [`NodeWithMetadata`].
    pub fn new(value: Node) -> Self {
        NodeWithMetadata {
            value,
            status: NodeStatus::Clean,
        }
    }
}

impl Deref for NodeWithMetadata {
    type Target = Node;

    fn deref(&self) -> &Self::Target {
        if self.status == NodeStatus::Deleted {
            panic!("Attempted to access a deleted node");
        }
        &self.value
    }
}

impl DerefMut for NodeWithMetadata {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.status == NodeStatus::Deleted {
            panic!("Attempted to access a deleted node");
        }
        self.status = NodeStatus::Dirty; // Mark as dirty on mutable borrow
        &mut self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_with_metadata_sets_dirty_flag_on_deref_mut() {
        let mut cached_node = NodeWithMetadata::new(Node::Empty);
        assert_eq!(cached_node.status, NodeStatus::Clean);
        let _ = cached_node.deref();
        assert_eq!(cached_node.status, NodeStatus::Clean);
        let _ = cached_node.deref_mut();
        assert_eq!(cached_node.status, NodeStatus::Dirty);
    }

    #[test]
    #[should_panic = "Attempted to access a deleted node"]
    fn node_with_metadata_deref_mut_panics_on_deleted_node() {
        let cached_node = NodeWithMetadata {
            value: Node::Empty,
            status: NodeStatus::Deleted,
        };
        let _ = cached_node.deref();
    }

    #[test]
    #[should_panic = "Attempted to access a deleted node"]
    fn node_with_metadata_deref_panics_on_deleted_node() {
        let mut cached_node = NodeWithMetadata {
            value: Node::Empty,
            status: NodeStatus::Deleted,
        };
        let _ = cached_node.deref_mut();
    }
}
