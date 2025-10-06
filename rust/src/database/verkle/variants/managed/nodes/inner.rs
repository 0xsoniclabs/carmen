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

use crate::database::verkle::{
    crypto::Commitment,
    variants::managed::nodes::{NodeType, id::NodeId},
};

/// An inner node in a (file-based) Verkle trie.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned)]
#[repr(C)]
pub struct InnerNode {
    pub commitment: Commitment,
    pub values: [NodeId; 256],
}

impl Default for InnerNode {
    fn default() -> Self {
        InnerNode {
            commitment: Commitment::default(),
            values: [NodeId::from_idx_and_node_type(0, NodeType::Empty); 256],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::database::verkle::{
        crypto::Commitment,
        variants::managed::nodes::{NodeType, id::NodeId, inner::InnerNode},
    };

    #[test]
    fn inner_node_default_returns_inner_node_with_all_values_set_to_empty_node_id() {
        let node: InnerNode = InnerNode::default();
        assert_eq!(node.commitment, Commitment::default());
        assert_eq!(
            node.values,
            [NodeId::from_idx_and_node_type(0, NodeType::Empty); 256]
        );
    }
}
