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

use crate::{
    database::verkle::variants::managed::nodes::VerkleNodeKind,
    types::{HasEmptyId, NodeSize, ToNodeKind, TreeId},
};

/// An identifier for a node in a managed Verkle trie.
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
pub struct VerkleNodeId([u8; 6]);

impl VerkleNodeId {
    // The upper 3 bits are used to encode the node type.
    const EMPTY_NODE_PREFIX: u64 = 0x0000_0000_0000_0000;
    const INNER_NODE_PREFIX: u64 = 0x0000_2000_0000_0000;
    const LEAF_NODE_1_PREFIX: u64 = 0x0000_4000_0000_0000;
    const LEAF_NODE_2_PREFIX: u64 = 0x0000_6000_0000_0000;
    const LEAF_NODE_21_PREFIX: u64 = 0x0000_8000_0000_0000;
    const LEAF_NODE_64_PREFIX: u64 = 0x0000_A000_0000_0000;
    const LEAF_NODE_141_PREFIX: u64 = 0x0000_C000_0000_0000;
    const LEAF_NODE_256_PREFIX: u64 = 0x0000_E000_0000_0000;

    const PREFIX_MASK: u64 = 0x0000_E000_0000_0000;
    const INDEX_MASK: u64 = 0x0000_1FFF_FFFF_FFFF;

    fn from_u64(value: u64) -> Self {
        let mut bytes = [0; 6];
        bytes[0..6].copy_from_slice(&value.to_be_bytes()[2..8]);
        VerkleNodeId(bytes)
    }

    fn to_u64(self) -> u64 {
        let mut bytes = [0; 8];
        bytes[2..8].copy_from_slice(&self.0);
        u64::from_be_bytes(bytes)
    }
}

impl ToNodeKind for VerkleNodeId {
    type Target = VerkleNodeKind;

    fn to_node_kind(&self) -> Option<VerkleNodeKind> {
        match self.to_u64() & Self::PREFIX_MASK {
            Self::EMPTY_NODE_PREFIX => Some(VerkleNodeKind::Empty),
            Self::INNER_NODE_PREFIX => Some(VerkleNodeKind::Inner),
            Self::LEAF_NODE_1_PREFIX => Some(VerkleNodeKind::Leaf1),
            Self::LEAF_NODE_2_PREFIX => Some(VerkleNodeKind::Leaf2),
            Self::LEAF_NODE_21_PREFIX => Some(VerkleNodeKind::Leaf21),
            Self::LEAF_NODE_64_PREFIX => Some(VerkleNodeKind::Leaf64),
            Self::LEAF_NODE_141_PREFIX => Some(VerkleNodeKind::Leaf141),
            Self::LEAF_NODE_256_PREFIX => Some(VerkleNodeKind::Leaf256),
            // There are only two ways to create a NodeId:
            // - Using `from_idx_and_node_type` with guarantees that the prefix is valid.
            // - Deserializing from a file which may hold invalid prefixes in case the data was
            //   corrupted.
            _ => None,
        }
    }
}

impl TreeId for VerkleNodeId {
    fn from_idx_and_node_kind(idx: u64, node_type: VerkleNodeKind) -> Self {
        assert!(
            (idx & !Self::INDEX_MASK) == 0,
            "indices cannot get this large, unless we have a bug somewhere"
        );
        let prefix = match node_type {
            VerkleNodeKind::Empty => Self::EMPTY_NODE_PREFIX,
            VerkleNodeKind::Inner => Self::INNER_NODE_PREFIX,
            VerkleNodeKind::Leaf1 => Self::LEAF_NODE_1_PREFIX,
            VerkleNodeKind::Leaf2 => Self::LEAF_NODE_2_PREFIX,
            VerkleNodeKind::Leaf256 => Self::LEAF_NODE_256_PREFIX,
            VerkleNodeKind::Leaf21 => Self::LEAF_NODE_21_PREFIX,
            VerkleNodeKind::Leaf64 => Self::LEAF_NODE_64_PREFIX,
            VerkleNodeKind::Leaf141 => Self::LEAF_NODE_141_PREFIX,
        };
        VerkleNodeId::from_u64(idx | prefix)
    }

    fn to_index(self) -> u64 {
        self.to_u64() & Self::INDEX_MASK
    }
}

impl NodeSize for VerkleNodeId {
    /// Returns the byte size of the node variant it refers to.
    /// Panics if the ID does not refer to a valid node type.
    fn node_byte_size(&self) -> usize {
        self.to_node_kind().unwrap().node_byte_size()
    }

    /// Returns the size of the smallest non-empty node variant.
    fn min_non_empty_node_size() -> usize {
        VerkleNodeKind::min_non_empty_node_size()
    }
}

impl HasEmptyId for VerkleNodeId {
    fn is_empty_id(&self) -> bool {
        self.to_node_kind() == Some(VerkleNodeKind::Empty)
    }

    fn empty_id() -> Self {
        VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_idx_and_node_type_creates_id_from_lower_6_bytes_logic_or_node_type_prefix() {
        let idx = 0x0000_1234_5678_9abc;
        let cases = [
            (VerkleNodeKind::Empty, 0x0000_0000_0000_0000),
            (VerkleNodeKind::Inner, 0x0000_2000_0000_0000),
            (VerkleNodeKind::Leaf1, 0x0000_4000_0000_0000),
            (VerkleNodeKind::Leaf2, 0x0000_6000_0000_0000),
            (VerkleNodeKind::Leaf21, 0x0000_8000_0000_0000),
            (VerkleNodeKind::Leaf64, 0x0000_A000_0000_0000),
            (VerkleNodeKind::Leaf141, 0x0000_C000_0000_0000),
            (VerkleNodeKind::Leaf256, 0x0000_E000_0000_0000),
        ];

        for (node_type, prefix) in cases {
            let id = VerkleNodeId::from_idx_and_node_kind(idx, node_type);
            assert_eq!(id.to_u64(), idx | prefix);
        }
    }

    #[test]
    #[should_panic]
    fn from_idx_and_node_type_panics_if_index_too_large() {
        let idx = 0x0000_f000_0000_0000;

        VerkleNodeId::from_idx_and_node_kind(idx, VerkleNodeKind::Empty);
    }

    #[test]
    fn to_index_masks_out_node_type() {
        let id = VerkleNodeId([0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
        assert_eq!(id.to_index(), 0x1f_ff_ff_ff_ff_ff);
    }

    #[test]
    fn to_node_type_returns_node_type_for_valid_prefixes() {
        let cases = [
            (
                VerkleNodeId([0x00, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Empty),
            ),
            (
                VerkleNodeId([0x20, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Inner),
            ),
            (
                VerkleNodeId([0x40, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Leaf1),
            ),
            (
                VerkleNodeId([0x60, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Leaf2),
            ),
            (
                VerkleNodeId([0x80, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Leaf21),
            ),
            (
                VerkleNodeId([0xa0, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Leaf64),
            ),
            (
                VerkleNodeId([0xc0, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Leaf141),
            ),
            (
                VerkleNodeId([0xe0, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Leaf256),
            ),
        ];
        for (node_id, node_type) in cases {
            assert_eq!(node_id.to_node_kind(), node_type);
        }
    }

    // Note: this is currently impossible because we have 4 node types and use 2 bits for the
    // prefix. If we add more node types, and there are in valid prefixes we should implement this
    // test.
    // #[test] fn node_id_to_node_type_returns_node_for_invalid_prefixes() {}

    #[test]
    fn from_u64_constructs_integer_from_lower_6_bytes() {
        let id = VerkleNodeId::from_u64(0x1234_5678_90ab_cdef);
        assert_eq!(id.0, [0x56, 0x78, 0x90, 0xab, 0xcd, 0xef]);
    }

    #[test]
    fn to_u64_converts_node_id_to_integer_with_lower_6_bytes() {
        let id = VerkleNodeId([0x12, 0x34, 0x56, 0x78, 0x90, 0xab]);
        assert_eq!(id.to_u64(), 0x1234_5678_90ab);
    }

    #[test]
    fn node_id_byte_size_returns_byte_size_of_encoded_node_type() {
        let cases = [
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty),
                VerkleNodeKind::Empty,
            ),
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Inner),
                VerkleNodeKind::Inner,
            ),
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Leaf2),
                VerkleNodeKind::Leaf2,
            ),
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Leaf256),
                VerkleNodeKind::Leaf256,
            ),
        ];
        for (node_id, node_type) in cases {
            assert_eq!(node_id.node_byte_size(), node_type.node_byte_size());
        }
    }

    #[test]
    fn node_id_min_non_empty_node_size_returns_min_byte_size_of_node_type() {
        assert_eq!(
            VerkleNodeId::min_non_empty_node_size(),
            VerkleNodeKind::min_non_empty_node_size()
        );
    }
}
