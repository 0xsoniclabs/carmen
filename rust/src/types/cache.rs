use std::{
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock},
};

use crate::{storage::file::WriteCertificate, types::Node};

/// A [`Node`] with an associated **dirty** flag for cache storage.
/// The **dirty** flag indicates if the entry has been modified and needs to be flushed to the
/// storage when evicted.
#[derive(Debug, PartialEq, Eq)]
pub struct CachedNode {
    value: Node,
    dirty: bool,
}

impl CachedNode {
    /// Creates a new [`CachedNode`] with the given [`Node`] and sets the **dirty** flag to false.
    pub fn new_clean(value: Node) -> Self {
        CachedNode {
            value,
            dirty: false,
        }
    }
    /// Creates a new [`CachedNode`] with the given [`Node`] and sets the **dirty** flag to true.
    pub fn new_dirty(value: Node) -> Self {
        CachedNode { value, dirty: true }
    }

    /// Returns whether node is dirty and needs to be flushed to storage when evicted.
    pub fn dirty(&self) -> bool {
        self.dirty
    }

    /// Sets the **dirty** flag to `false`.
    /// To make sure this method can only be called from the EvictionCache, it takes a
    /// `WriteCertificate` which can only be instantiated there.
    pub fn set_clean(&mut self, _cert: WriteCertificate) {
        self.dirty = false;
    }
}

impl Deref for CachedNode {
    type Target = Node;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl DerefMut for CachedNode {
    /// Mutably dereferences to the inner [`Node`] and sets the **dirty** flag to true.
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.dirty = true;
        &mut self.value
    }
}

/// A node cache entry that can be safely shared across threads.
pub type CacheEntry = Arc<RwLock<CachedNode>>;

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn cached_node_new_clean_creates_cached_node_with_dirty_flag_to_false() {
        let cached_node = CachedNode::new_clean(Node::Empty);
        assert!(!cached_node.dirty());
    }

    #[test]
    fn cached_node_new_dirty_creates_cached_node_with_dirty_flag_to_true() {
        let cached_node = CachedNode::new_dirty(Node::Empty);
        assert!(cached_node.dirty());
    }

    #[test]
    fn cached_node_sets_dirty_flag_on_mutable_dereference() {
        let mut cached_node = CachedNode::new_clean(Node::Empty);
        assert!(!cached_node.dirty());
        let _ = cached_node.deref(); // Immutable dereference
        assert!(!cached_node.dirty());
        let _ = cached_node.deref_mut(); // Mutable dereference
        assert!(cached_node.dirty());
    }
}
