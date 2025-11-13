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
    database::verkle::variants::managed::{
        commitment::VerkleCommitment,
        nodes::{VerkleNodeKind, id::VerkleNodeId},
    },
    types::TreeId,
};

/// An inner node in a managed Verkle trie.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned)]
#[repr(C)]
pub struct InnerNode {
    pub children: [VerkleNodeId; 256],
    pub commitment: VerkleCommitment,
}

impl Default for InnerNode {
    fn default() -> Self {
        InnerNode {
            children: [VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty); 256],
            commitment: VerkleCommitment::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inner_node_default_returns_inner_node_with_all_children_set_to_empty_node_id() {
        let node: InnerNode = InnerNode::default();
        assert_eq!(node.commitment, VerkleCommitment::default());
        assert_eq!(
            node.children,
            [VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty); 256]
        );
    }
}
