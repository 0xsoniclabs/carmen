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
    error::{BTResult, Error},
    types::{HasDeltaVariant, HasEmptyId, HasEmptyNode, ToNodeKind, TreeId},
};

pub type TestNodeId = u32;
pub type TestNode = i32;

pub const EMPTY_TEST_NODE: TestNode = i32::MAX;
pub const EMPTY_TEST_NODE_ID: TestNodeId = u32::MAX;
pub const DELTA_TEST_NODE: TestNode = i32::MAX - 1;
pub const FULL_TEST_NODE_ID: TestNodeId = u32::MAX - 2;

impl ToNodeKind for TestNodeId {
    type Target = ();

    fn to_node_kind(&self) -> Option<Self::Target> {
        Some(())
    }
}

impl TreeId for TestNodeId {
    fn from_idx_and_node_kind(idx: u64, _node_type: Self::Target) -> Self {
        idx as u32
    }

    fn to_index(self) -> u64 {
        self as u64
    }
}

impl HasEmptyId for TestNodeId {
    fn is_empty_id(&self) -> bool {
        *self == EMPTY_TEST_NODE_ID
    }

    fn empty_id() -> Self {
        EMPTY_TEST_NODE_ID
    }
}

impl ToNodeKind for TestNode {
    type Target = ();

    fn to_node_kind(&self) -> Option<Self::Target> {
        Some(())
    }
}

impl HasEmptyNode for TestNode {
    fn is_empty_node(&self) -> bool {
        *self == EMPTY_TEST_NODE
    }

    fn empty_node() -> Self {
        EMPTY_TEST_NODE
    }
}

impl HasDeltaVariant for TestNode {
    type Id = TestNodeId;

    fn needs_full(&self) -> Option<Self::Id> {
        if *self == DELTA_TEST_NODE {
            Some(FULL_TEST_NODE_ID)
        } else {
            None
        }
    }

    fn copy_from_base(&mut self, _full: &Self) -> BTResult<(), Error> {
        Ok(())
    }
}
