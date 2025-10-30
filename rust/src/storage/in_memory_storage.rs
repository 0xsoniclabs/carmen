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
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use dashmap::DashMap;

use crate::{
    error::BTResult,
    storage::{Checkpointable, Error, RootIdProvider, Storage},
    types::{ToNodeType, TreeId},
};

/// A storage backend that holds all nodes in memory.
/// It does not persist data to disk and consequently also does not load any data on startup.
pub struct InMemoryStorage<ID, N> {
    storage: DashMap<ID, N>,
    frozen_storage: DashMap<ID, N>,
    root_ids: DashMap<u64, ID>,
    next_id: Arc<AtomicU64>,
    checkpoint: AtomicU64,
}

impl<ID, N> Storage for InMemoryStorage<ID, N>
where
    ID: TreeId + Copy + Eq + std::hash::Hash + Send + Sync,
    N: ToNodeType<NodeType = ID::NodeType> + Clone + Send + Sync,
{
    type Id = ID;
    type Item = N;

    fn open(_path: &Path) -> BTResult<Self, Error> {
        Ok(Self {
            storage: DashMap::new(),
            frozen_storage: DashMap::new(),
            root_ids: DashMap::new(),
            next_id: Arc::new(AtomicU64::new(0)),
            checkpoint: AtomicU64::new(0),
        })
    }

    fn get(&self, id: Self::Id) -> BTResult<Self::Item, Error> {
        self.storage
            .get(&id)
            .map(|entry| entry.value().clone())
            .ok_or(Error::NotFound)
            .map_err(Into::into)
    }

    fn reserve(&self, node: &Self::Item) -> Self::Id {
        Self::Id::from_idx_and_node_type(
            self.next_id.fetch_add(1, Ordering::Relaxed),
            node.to_node_type().unwrap(),
        )
    }

    fn set(&self, id: Self::Id, node: &Self::Item) -> BTResult<(), Error> {
        self.storage.insert(id, node.clone());
        Ok(())
    }

    fn delete(&self, id: Self::Id) -> BTResult<(), Error> {
        self.storage.remove(&id);
        Ok(())
    }

    fn close(self) -> BTResult<(), Error> {
        Ok(())
    }
}

impl<ID, N> Checkpointable for InMemoryStorage<ID, N>
where
    ID: Copy + std::hash::Hash + Eq + Send + Sync,
    N: Clone + Send + Sync,
{
    fn checkpoint(&self) -> BTResult<u64, Error> {
        for entry in self.storage.iter() {
            self.frozen_storage
                .insert(*entry.key(), entry.value().clone());
        }
        self.storage.clear();
        Ok(0)
    }

    fn restore(_path: &Path, _checkpoint: u64) -> BTResult<(), Error> {
        unimplemented!()
    }
}

impl<ID, N> RootIdProvider for InMemoryStorage<ID, N>
where
    ID: Copy,
{
    type Id = ID;

    fn get_root_id(&self, block_number: u64) -> BTResult<Self::Id, Error> {
        self.root_ids
            .get(&block_number)
            .map(|entry| *entry.value())
            .ok_or(Error::NotFound)
            .map_err(Into::into)
    }

    fn set_root_id(&self, block_number: u64, id: Self::Id) -> BTResult<(), Error> {
        self.root_ids.insert(block_number, id);
        Ok(())
    }
}
