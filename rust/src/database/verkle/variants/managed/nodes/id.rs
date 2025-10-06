// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use zerocopy::{FromBytes, Immutable, IntoBytes, Unaligned};

use crate::{database::verkle::variants::managed::nodes::NodeType, types::NodeSize};

/// An identifier for a node in a (file-based) Verkle trie.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    FromBytes,
    IntoBytes,
    Immutable,
    Unaligned,
    PartialOrd,
    Ord,
)]
#[repr(transparent)]
pub struct NodeId([u8; 6]);

impl NodeId {
    // The upper 2 bits are used to encode the node type.
    const EMPTY_NODE_PREFIX: u64 = 0x0000_0000_0000_0000;
    const INNER_NODE_PREFIX: u64 = 0x0000_4000_0000_0000;
    const LEAF_NODE_2_PREFIX: u64 = 0x0000_8000_0000_0000;
    const LEAF_NODE_256_PREFIX: u64 = 0x0000_C000_0000_0000;

    const PREFIX_MASK: u64 = 0x0000_C000_0000_0000;
    const INDEX_MASK: u64 = 0x0000_3FFF_FFFF_FFFF;

    /// Creates a new [`NodeId`] from a [`u64`] index and a [`NodeType`].
    /// The index must be smaller than 2^46.
    pub fn from_idx_and_node_type(idx: u64, node_type: NodeType) -> Self {
        assert!(
            (idx & !Self::INDEX_MASK) == 0,
            "indices cannot get this large, unless we have a bug somewhere"
        );
        let prefix = match node_type {
            NodeType::Empty => Self::EMPTY_NODE_PREFIX,
            NodeType::Inner => Self::INNER_NODE_PREFIX,
            NodeType::Leaf2 => Self::LEAF_NODE_2_PREFIX,
            NodeType::Leaf256 => Self::LEAF_NODE_256_PREFIX,
        };
        NodeId::from_u64(idx | prefix)
    }

    /// Converts the [`NodeId`] to a [`u64`] index, stripping the prefix.
    /// The index is guaranteed to be smaller than 2^46.
    pub fn to_index(self) -> u64 {
        self.to_u64() & Self::INDEX_MASK
    }

    /// Converts the [`NodeId`] to a [`NodeType`], if the prefix is valid.
    pub fn to_node_type(self) -> Option<NodeType> {
        match self.to_u64() & Self::PREFIX_MASK {
            Self::EMPTY_NODE_PREFIX => Some(NodeType::Empty),
            Self::INNER_NODE_PREFIX => Some(NodeType::Inner),
            Self::LEAF_NODE_2_PREFIX => Some(NodeType::Leaf2),
            Self::LEAF_NODE_256_PREFIX => Some(NodeType::Leaf256),
            // There are only two ways to create a NodeId:
            // - Using `from_idx_and_node_type` with guarantees that the prefix is valid.
            // - Deserializing from a file which may hold invalid prefixes in case the data was
            //   corrupted.
            _ => None,
        }
    }

    fn from_u64(value: u64) -> Self {
        let mut bytes = [0; 6];
        bytes[0..6].copy_from_slice(&value.to_be_bytes()[2..8]);
        NodeId(bytes)
    }

    fn to_u64(self) -> u64 {
        let mut bytes = [0; 8];
        bytes[2..8].copy_from_slice(&self.0);
        u64::from_be_bytes(bytes)
    }
}

impl NodeSize for NodeId {
    /// Returns the byte size of the [`NodeType`] it refers to.
    /// Panics if the ID does not refer to a valid node type.
    fn node_byte_size(&self) -> usize {
        self.to_node_type().unwrap().node_byte_size()
    }

    /// Returns the minimum byte size of [`NodeType`].
    fn min_non_empty_node_size() -> usize {
        NodeType::min_non_empty_node_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_idx_and_node_type_creates_id_from_lower_6_bytes_logic_or_node_type_prefix() {
        let idx = 0x0000_1234_5678_9abc;
        let cases = [
            (NodeType::Empty, 0x0000_0000_0000_0000),
            (NodeType::Inner, 0x0000_4000_0000_0000),
            (NodeType::Leaf2, 0x0000_8000_0000_0000),
            (NodeType::Leaf256, 0x0000_C000_0000_0000),
        ];

        for (node_type, prefix) in cases {
            let id = NodeId::from_idx_and_node_type(idx, node_type);
            assert_eq!(id.to_u64(), idx | prefix);
        }
    }

    #[test]
    #[should_panic]
    fn from_idx_and_node_type_panics_if_index_too_large() {
        let idx = 0x0000_f000_0000_0000;

        NodeId::from_idx_and_node_type(idx, NodeType::Empty);
    }

    #[test]
    fn to_index_masks_out_node_type() {
        let id = NodeId([0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
        assert_eq!(id.to_index(), 0x3f_ff_ff_ff_ff_ff);
    }

    #[test]
    fn to_node_type_returns_node_type_for_valid_prefixes() {
        let cases = [
            (
                NodeId([0x00, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(NodeType::Empty),
            ),
            (
                NodeId([0x40, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(NodeType::Inner),
            ),
            (
                NodeId([0x80, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(NodeType::Leaf2),
            ),
            (
                NodeId([0xc0, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(NodeType::Leaf256),
            ),
        ];
        for (node_id, node_type) in cases {
            assert_eq!(node_id.to_node_type(), node_type);
        }
    }

    // Note: this is currently impossible because we have 4 node types and use 2 bits for the
    // prefix. If we add more node types, and there are in valid prefixes we should implement this
    // test.
    // #[test] fn node_id_to_node_type_returns_node_for_invalid_prefixes() {}

    #[test]
    fn from_u64_constructs_integer_from_lower_6_bytes() {
        let id = NodeId::from_u64(0x1234_5678_90ab_cdef);
        assert_eq!(id.0, [0x56, 0x78, 0x90, 0xab, 0xcd, 0xef]);
    }

    #[test]
    fn to_u64_converts_node_id_to_integer_with_lower_6_bytes() {
        let id = NodeId([0x12, 0x34, 0x56, 0x78, 0x90, 0xab]);
        assert_eq!(id.to_u64(), 0x1234_5678_90ab);
    }

    #[test]
    fn node_id_byte_size_returns_byte_size_of_encoded_node_type() {
        let cases = [
            (
                NodeId::from_idx_and_node_type(0, NodeType::Empty),
                NodeType::Empty,
            ),
            (
                NodeId::from_idx_and_node_type(0, NodeType::Inner),
                NodeType::Inner,
            ),
            (
                NodeId::from_idx_and_node_type(0, NodeType::Leaf2),
                NodeType::Leaf2,
            ),
            (
                NodeId::from_idx_and_node_type(0, NodeType::Leaf256),
                NodeType::Leaf256,
            ),
        ];
        for (node_id, node_type) in cases {
            assert_eq!(node_id.node_byte_size(), node_type.node_byte_size());
        }
    }

    #[test]
    fn node_id_min_non_empty_node_size_returns_min_byte_size_of_node_type() {
        assert_eq!(
            NodeId::min_non_empty_node_size(),
            NodeType::min_non_empty_node_size()
        );
    }
}
