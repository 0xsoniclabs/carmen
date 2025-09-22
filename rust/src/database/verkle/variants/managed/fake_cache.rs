// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{
    ops::{Deref, DerefMut},
    sync::{RwLock, RwLockReadGuard, RwLockWriteGuard, atomic::AtomicU64},
};

use crate::{
    database::verkle::variants::managed::{Node, NodeId, nodes::empty::EmptyNode},
    error::{BTResult, Error},
    node_manager::NodeManager,
    types::TreeId,
};

struct NodeWrapper {
    node: Node,
}

impl Deref for NodeWrapper {
    type Target = Node;

    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

impl DerefMut for NodeWrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.node
    }
}

#[allow(dead_code)] // TODO: Remove
pub struct FakeCache {
    nodes: Vec<RwLock<NodeWrapper>>,
    next_id: AtomicU64,
}

impl FakeCache {
    #[allow(dead_code)] // TODO: Remove
    pub fn new() -> Self {
        let mut nodes = Vec::with_capacity(1000);
        for _ in 0..1000 {
            nodes.push(RwLock::new(NodeWrapper {
                node: Node::Empty(EmptyNode),
            }));
        }

        FakeCache {
            nodes,
            next_id: AtomicU64::new(0),
        }
    }
}

impl NodeManager for FakeCache {
    type Id = NodeId;
    type NodeType = Node;

    fn get_read_access(
        &self,
        id: NodeId,
    ) -> BTResult<RwLockReadGuard<'_, impl Deref<Target = Self::NodeType>>, Error> {
        Ok(self.nodes[id.to_index() as usize].read().unwrap())
    }

    fn get_write_access(
        &self,
        id: Self::Id,
    ) -> BTResult<RwLockWriteGuard<'_, impl std::ops::DerefMut<Target = Self::NodeType>>, Error>
    {
        Ok(self.nodes[id.to_index() as usize].write().unwrap())
    }

    fn add(&self, value: Node) -> BTResult<NodeId, Error> {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let key = NodeId::from_idx_and_node_type(id, value.to_node_type());
        self.nodes[id as usize].write().unwrap().node = value;
        Ok(key)
    }

    fn delete(&self, id: NodeId) -> BTResult<(), Error> {
        let _lock = self.nodes[id.to_index() as usize].write().unwrap(); // Dummy access just to properly represent locking behavior
        Ok(())
    }
}
