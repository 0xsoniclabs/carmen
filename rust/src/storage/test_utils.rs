use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::types::{
    AllVariants, DiskRepresentable, DiskRepresentableByType, NodeSize, ToNodeType, TreeId,
};

pub type TestNodeId = [u8; 9];

impl TreeId for TestNodeId {
    type NodeType = TestNodeType;

    fn from_idx_and_node_type(idx: u64, node_type: Self::NodeType) -> Self {
        let upper = match node_type {
            TestNodeType::Empty => 0x1,
            TestNodeType::NonEmpty => 0x00u8,
        };
        let mut id = [0; 9];
        id[0] = upper;
        id[1..].copy_from_slice(&idx.to_be_bytes());
        id
    }

    fn to_index(self) -> u64 {
        let mut idx = [0; 8];
        idx.copy_from_slice(&self[1..]);
        u64::from_be_bytes(idx)
    }

    fn to_node_type(self) -> Option<Self::NodeType> {
        match self[0] {
            0x01 => Some(TestNodeType::Empty),
            0x00 => Some(TestNodeType::NonEmpty),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestNode {
    Empty(EmptyTestNode),
    NonEmpty(NonEmptyTestNode),
}

impl ToNodeType for TestNode {
    type NodeType = TestNodeType;

    fn to_node_type(&self) -> Self::NodeType {
        match self {
            TestNode::Empty(_) => TestNodeType::Empty,
            TestNode::NonEmpty(_) => TestNodeType::NonEmpty,
        }
    }
}

impl DiskRepresentableByType for TestNode {
    type EnumType = TestNodeType;

    fn from_disk_repr<E>(
        et: &Self::EnumType,
        read_into_buffer: impl FnOnce(&mut [u8]) -> Result<(), E>,
    ) -> Result<Self, E> {
        match et {
            TestNodeType::Empty => Ok(TestNode::Empty(EmptyTestNode)),
            TestNodeType::NonEmpty => Ok(TestNode::NonEmpty(NonEmptyTestNode(
                <[u8; 32]>::from_disk_repr(read_into_buffer)?,
            ))),
        }
    }

    fn to_disk_repr(&self) -> &[u8] {
        match self {
            TestNode::Empty(_) => &[],
            TestNode::NonEmpty(x_node) => &x_node.0,
        }
    }

    fn disk_size(et: &Self::EnumType) -> usize {
        match et {
            TestNodeType::Empty => size_of::<EmptyTestNode>(),
            TestNodeType::NonEmpty => size_of::<NonEmptyTestNode>(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestNodeType {
    Empty,
    NonEmpty,
}

impl AllVariants for TestNodeType {
    fn all_variants() -> &'static [(Self, &'static str)] {
        &[
            (TestNodeType::Empty, "empty_test_node"),
            (TestNodeType::NonEmpty, "non_empty_test_node"),
        ]
    }
}

impl NodeSize for TestNodeType {
    fn node_byte_size(&self) -> usize {
        match self {
            TestNodeType::Empty => size_of::<EmptyTestNode>(),
            TestNodeType::NonEmpty => size_of::<NonEmptyTestNode>(),
        }
    }

    fn min_non_empty_node_size() -> usize {
        size_of::<NonEmptyTestNode>()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmptyTestNode;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
pub struct NonEmptyTestNode(pub [u8; 32]);
