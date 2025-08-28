use std::ops::{Deref, DerefMut};

use crate::types::Node;

/// A [`Node`] with a **status** to store metadata about the node lifecycle.
/// The internal node is set as [`NodeStatus::Dirty`] when a mutable reference is requested.
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
pub enum NodeStatus {
    Clean,
    Dirty,
    Deleted,
}

impl NodeWithMetadata {
    /// Creates a new [`NodeWithMetadata`] with the given [`Node`] and status.
    pub fn new(value: Node, status: NodeStatus) -> Self {
        NodeWithMetadata { value, status }
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
    use std::panic::catch_unwind;

    use super::*;

    #[test]
    fn node_with_metadata_sets_dirty_flag_on_get_mut() {
        let mut cached_node = NodeWithMetadata::new(Node::Empty, NodeStatus::Clean);
        assert!(cached_node.status != NodeStatus::Dirty);
        let _ = cached_node.deref();
        assert!(cached_node.status == NodeStatus::Clean);
        let _ = cached_node.deref_mut();
        assert!(cached_node.status == NodeStatus::Dirty);
    }

    #[test]
    fn node_with_metadata_deref_panics_on_deleted_node() {
        let res = catch_unwind(|| {
            let cached_node = NodeWithMetadata::new(Node::Empty, NodeStatus::Deleted);
            let _ = cached_node.deref();
        });
        assert!(res.is_err());

        let res = catch_unwind(|| {
            let mut cached_node = NodeWithMetadata::new(Node::Empty, NodeStatus::Deleted);
            let _ = cached_node.deref_mut();
        });
        assert!(res.is_err());
    }
}
