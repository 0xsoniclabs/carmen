// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::{database::verkle::crypto::Commitment, types::Value};

/// A leaf node with 256 children in a (file-based) Verkle trie.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
#[repr(C)]
pub struct FullLeafNode {
    pub commitment: Commitment,
    pub stem: [u8; 31],
    pub values: [Value; 256],
}

impl Default for FullLeafNode {
    fn default() -> Self {
        FullLeafNode {
            commitment: Commitment::default(),
            stem: [0; 31],
            values: [Value::default(); 256],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_leaf_node_default_returns_leaf_node_with_all_values_set_to_default() {
        let node: FullLeafNode = FullLeafNode::default();
        assert_eq!(node.commitment, Commitment::default());
        assert_eq!(node.stem, [0; 31]);
        assert_eq!(node.values, [Value::default(); 256]);
    }
}
