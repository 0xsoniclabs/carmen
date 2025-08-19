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
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
    },
};

use dashmap::DashMap;

use crate::{
    storage::{Error, Storage},
    types::{CacheEntry, CachedNode, Node, NodeId},
};

/// A storage backend that uses an eviction cache to hold updates and deletions while they get
/// written to the underlying storage layer in background threads.
///
/// The eviction workers (the background threads that write the updates and deletions to the
/// underlying storage layer) lock nodes while they are being written out to ensure that they are
/// not concurrently modified.
///
/// Deletions only involve the id, not the node itself, so they are not locked. The id is deleted in
/// the underlying storage layer first and then removed from the eviction cache. In case it was
/// deleted in the underlying storage layer and reassigned by a concurrent task before the eviction
/// worker deleted it from the eviction cache, the `StorageWithEvictionCache::reserve` method also
/// deletes the id from the eviction cache. This way it is guaranteed that a reassigned id is not
/// found in the eviction cache.
///
/// Queries always check the eviction cache first. If the id is found there and it is an update, the
/// node is removed from the eviction cache and returned. The node is then still considered dirty.
/// If the id is found in the eviction cache and it is a delete operation, a not found error is
/// returned. Only if the id is not found in the eviction cache, the underlying storage layer is
/// queried.
pub struct StorageWithEvictionCache<S>
where
    S: Storage<Item = Node>,
{
    cache: Arc<EvictionCache>, // Arc for shared ownership with eviction worker threads
    storage: Arc<S>,           // Arc for shared ownership with eviction worker threads
    eviction_workers: EvictionWorkers,
}

impl<S> StorageWithEvictionCache<S>
where
    S: Storage<Id = NodeId, Item = Node> + Send + Sync + 'static,
{
    pub fn shutdown_eviction_workers(self) -> Result<(), Error> {
        self.eviction_workers.shutdown()
    }
}

#[cfg_attr(test, mockall::automock)]
impl<S> Storage for StorageWithEvictionCache<S>
where
    S: Storage<Id = NodeId, Item = Node> + Send + Sync + 'static,
{
    type Id = NodeId;
    type Item = CacheEntry;

    fn open(path: &Path) -> Result<Self, Error> {
        let storage = Arc::new(S::open(path)?);
        let eviction_cache = Arc::new(DashMap::new());
        let workers = EvictionWorkers::new(&eviction_cache, &storage);
        Ok(StorageWithEvictionCache {
            cache: eviction_cache,
            eviction_workers: workers,
            storage,
        })
    }

    fn get(&self, id: NodeId) -> Result<Self::Item, Error> {
        match self.cache.remove(&id) {
            Some((_, Op::Set(node))) => Ok(node),
            Some((_, Op::Delete)) => Err(Error::NotFound),
            None => Ok(Arc::new(RwLock::new(CachedNode::new_clean(
                self.storage.get(id)?,
            )))),
        }
    }

    fn reserve(&self, node: &Self::Item) -> Self::Id {
        let id = self.storage.reserve(&node.read().unwrap());
        // The id may have been deleted in the underlying storage layer and reassigned here, but not
        // yet removed from the eviction cache. In this case, we remove it from the eviction
        // cache to ensure that it is no longer in the cache.
        self.cache.remove(&id);
        id
    }

    fn set(&self, id: NodeId, node: &Self::Item) -> Result<(), Error> {
        self.cache.insert(id, Op::Set(node.clone()));
        Ok(())
    }

    fn delete(&self, id: NodeId) -> Result<(), Error> {
        self.cache.insert(id, Op::Delete);
        Ok(())
    }

    fn flush(&self) -> Result<(), Error> {
        // Busy loop until all eviction workers are done.
        // Because there are no concurrent inserts, len() might only return a number that is higher
        // that the actual number of items. This is however not a problem because we will wait a
        // little bit longer.
        while !self.cache.is_empty() {}
        self.storage.flush()
    }
}

/// A wrapper around a set of eviction worker threads that allows to shut them down gracefully.
struct EvictionWorkers {
    workers: Vec<std::thread::JoinHandle<Result<(), Error>>>,
    shutdown: Arc<AtomicBool>, // Arc for shared ownership with eviction worker threads
}

impl EvictionWorkers {
    const WORKER_COUNT: usize = 10; // TODO the optimal number needs to be determined based on benchmarks

    /// Creates a new set of eviction workers that will process items from the eviction cache and
    /// write them to the underlying storage layer.
    pub fn new<S>(eviction_cache: &Arc<EvictionCache>, storage: &Arc<S>) -> Self
    where
        S: Storage<Id = NodeId, Item = Node> + Send + Sync + 'static,
    {
        let shutdown = Arc::new(AtomicBool::new(false));
        let workers = (0..Self::WORKER_COUNT)
            .map(|_| {
                let eviction_cache = eviction_cache.clone();
                let storage = storage.clone();
                let shutdown = shutdown.clone();
                std::thread::spawn(move || {
                    EvictionWorkers::task(&eviction_cache, &*storage, &shutdown)
                })
            })
            .collect();

        EvictionWorkers { workers, shutdown }
    }

    /// The task that each eviction worker runs. It processes items from the eviction cache
    /// and writes them to the underlying storage layer.
    fn task<S>(
        eviction_cache: &EvictionCache,
        storage: &S,
        shutdown: &Arc<AtomicBool>,
    ) -> Result<(), Error>
    where
        S: Storage<Id = NodeId, Item = Node>,
    {
        loop {
            let item = eviction_cache
                .iter()
                .next()
                .map(|entry| (*entry.key(), entry.value().clone()));

            if let Some((id, op)) = item {
                match op {
                    Op::Set(node) => {
                        // Acquire a write lock on the node to prevent concurrent accesses,
                        // write it to the underlying storage layer and then remove it from the
                        // eviction cache. If the node was queried during
                        // this time, it may be again in the node cache, but
                        // it cannot be modified until the lock is released.
                        let mut node_guard = node.write().unwrap();
                        storage.set(id, &node_guard)?;
                        eviction_cache.remove(&id);
                        node_guard.set_clean(WriteCertificate(()));
                    }
                    Op::Delete => {
                        // Delete the id in the underlying storage first, then remove it from the
                        // eviction cache.
                        // Once the id was deleted in the underlying storage layer, it may get
                        // reused in a call to `reserve`.
                        // For the case that the id was deleted in the storage layer and reassigned
                        // by a concurrent task before the eviction worker deleted it from the
                        // eviction cache, `StorageWithEvictionCache::reserve` also deletes the id
                        // from the eviction cache. This way it is guaranteed, that a reassigned id
                        // is not found in the eviction cache.
                        storage.delete(id)?;
                        eviction_cache.remove(&id);
                    }
                }
            } else {
                // the cache is currently empty
                if shutdown.load(Ordering::SeqCst) {
                    return Ok(());
                }
                // avoid busy looping
                std::thread::yield_now();
            }
        }
    }

    pub fn shutdown(self) -> Result<(), Error> {
        self.shutdown.store(true, Ordering::SeqCst);
        for worker in self.workers {
            worker.join().unwrap()?;
        }
        Ok(())
    }
}

type EvictionCache = DashMap<NodeId, Op>;

/// An element in the eviction cache that can either be a set operation or a delete operation.
#[derive(Debug, Clone)]
enum Op {
    Set(CacheEntry),
    Delete,
}

/// A zero-sized type used as a certificate that a node was written to disk and can therefore be
/// treated as clean again.
// Note: The struct has to have a private field so that it cannot be instantiated outside of this
// module.
pub struct WriteCertificate(());

#[cfg(test)]
mod tests {
    use std::{fs::File, sync::atomic::AtomicUsize, time::Duration};

    use super::*;
    use crate::{
        storage::file::{
            FileStorageManager, MockFileBackend, MockFileStorageManager, NoSeekFile,
            node_file_storage::NodeFileStorage,
        },
        types::NodeType,
    };

    #[test]
    fn open_all_nested_layers() {
        // The purpose of this test is to ensure, that `StorageWithEvictionCache` can be used with
        // the lower layers of the storage system (that the types and interfaces line up).
        let dir = tempfile::tempdir().unwrap();

        // this opens:
        // StorageWithEvictionCache
        //   -> FileStoreManager
        //     -> NodeFileStorage for InnerNode
        //       -> NoSeekFile
        //     -> NodeFileStorage for LeafNode2
        //       -> NoSeekFile
        //     -> NodeFileStorage for LeafNode256
        //       -> NoSeekFile
        StorageWithEvictionCache::<FileStorageManager<NoSeekFile>>::open(dir.path()).unwrap();
    }

    #[test]
    fn open_opens_underlying_storage_and_starts_eviction_workers() {
        let dir = tempfile::tempdir().unwrap();
        // Using mocks for `FileStorageManager` and `NoSeekFile` is not possible, because `open`
        // creates the mocks using calls to `open` on the mock type, but the mocks have no
        // expectations set up.
        let storage =
            StorageWithEvictionCache::<FileStorageManager<NoSeekFile>>::open(dir.path()).unwrap();

        // The node store files should be locked while opened
        let file = File::open(
            dir.path()
                .join(FileStorageManager::<NoSeekFile>::INNER_NODE_DIR)
                .join(NodeFileStorage::<u8, NoSeekFile>::NODE_STORE_FILE),
        )
        .unwrap();
        assert!(file.try_lock().is_err());

        assert_eq!(
            storage.eviction_workers.workers.len(),
            EvictionWorkers::WORKER_COUNT
        );
        for worker in &storage.eviction_workers.workers {
            assert!(!worker.is_finished()); // Ensure the worker is running
        }
    }

    #[test]
    fn get_removes_and_returns_node_from_cache_if_present_as_set_op() {
        let storage = StorageWithEvictionCache {
            cache: Arc::new(DashMap::new()),
            storage: Arc::new(MockFileStorageManager::<MockFileBackend>::new()),
            eviction_workers: EvictionWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);
        let node = CacheEntry::new(RwLock::new(CachedNode::new_clean(Node::Inner(
            Box::default(),
        ))));

        storage.cache.insert(id, Op::Set(node.clone()));

        let result = storage.get(id).unwrap();
        assert_eq!(**result.read().unwrap(), **node.read().unwrap());
        assert!(storage.cache.get(&id).is_none());
    }

    #[test]
    fn get_returns_not_found_error_if_id_is_present_as_delete_op() {
        let storage = StorageWithEvictionCache {
            cache: Arc::new(DashMap::new()),
            storage: Arc::new(MockFileStorageManager::<MockFileBackend>::new()),
            eviction_workers: EvictionWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);
        storage.cache.insert(id, Op::Delete);

        let result = storage.get(id);
        assert!(matches!(result, Err(Error::NotFound)));
    }

    #[test]
    fn get_returns_node_from_storage_if_not_in_cache() {
        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);
        let node = Node::Inner(Box::default());

        let mut mock_storage = MockFileStorageManager::<MockFileBackend>::new();
        mock_storage
            .expect_get()
            .withf(move |arg| *arg == id)
            .returning({
                let node = node.clone();
                move |_| Ok(node.clone())
            });

        let storage_with_eviction_cache = StorageWithEvictionCache {
            cache: Arc::new(DashMap::new()),
            storage: Arc::new(mock_storage),
            eviction_workers: EvictionWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        let result = storage_with_eviction_cache.get(id).unwrap();
        assert_eq!(**result.read().unwrap(), node);
    }

    #[test]
    fn reserve_retrieves_id_from_underlying_storage_layer_and_removes_from_cache() {
        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);
        let node = Node::Inner(Box::default());

        let mut mock_storage = MockFileStorageManager::<MockFileBackend>::new();
        mock_storage
            .expect_reserve()
            .withf({
                let node = node.clone();
                move |arg| arg == &node
            })
            .returning(move |_| id);

        let storage_with_eviction_cache = StorageWithEvictionCache {
            cache: Arc::new(DashMap::new()),
            storage: Arc::new(mock_storage),
            eviction_workers: EvictionWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        let reserved_id = storage_with_eviction_cache
            .reserve(&Arc::new(RwLock::new(CachedNode::new_clean(node))));
        assert_eq!(reserved_id, id);
        assert!(storage_with_eviction_cache.cache.get(&id).is_none());
    }

    #[test]
    fn set_inserts_set_op_into_cache() {
        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);
        let node = CacheEntry::new(RwLock::new(CachedNode::new_clean(Node::Inner(
            Box::default(),
        ))));

        let storage_with_eviction_cache = StorageWithEvictionCache {
            cache: Arc::new(DashMap::new()),
            storage: Arc::new(MockFileStorageManager::<MockFileBackend>::new()),
            eviction_workers: EvictionWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        storage_with_eviction_cache.set(id, &node).unwrap();

        let entry = storage_with_eviction_cache.cache.get(&id);
        assert!(entry.is_some());
        let entry = entry.unwrap();
        let value = entry.value();
        assert!(matches!(value, Op::Set(n) if Arc::ptr_eq(n, &node)));
    }

    #[test]
    fn delete_inserts_delete_op_into_cache() {
        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);

        let storage_with_eviction_cache = StorageWithEvictionCache {
            cache: Arc::new(DashMap::new()),
            storage: Arc::new(MockFileStorageManager::<MockFileBackend>::new()),
            eviction_workers: EvictionWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        storage_with_eviction_cache.delete(id).unwrap();

        let entry = storage_with_eviction_cache.cache.get(&id);
        assert!(entry.is_some());
        let entry = entry.unwrap();
        let value = entry.value();
        assert!(matches!(value, Op::Delete));
    }

    #[test]
    fn flush_waits_until_cache_is_empty_then_calls_flush_on_underlying_storage_layer() {
        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);
        let node = Node::Inner(Box::default());

        let mut mock_storage = MockFileStorageManager::<MockFileBackend>::new();
        mock_storage.expect_flush().returning(|| Ok(()));

        let storage_with_eviction_cache = StorageWithEvictionCache {
            cache: Arc::new(DashMap::new()),
            storage: Arc::new(mock_storage),
            eviction_workers: EvictionWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        storage_with_eviction_cache.cache.insert(
            id,
            Op::Set(CacheEntry::new(RwLock::new(CachedNode::new_clean(node)))),
        );

        let storage_with_eviction_cache = Arc::new(storage_with_eviction_cache);

        let thread = std::thread::spawn({
            let storage_with_eviction_cache = storage_with_eviction_cache.clone();
            move || storage_with_eviction_cache.flush()
        });

        // flush is waiting
        assert!(!thread.is_finished());
        std::thread::sleep(Duration::from_millis(100));
        // flush is still waiting
        assert!(!thread.is_finished());

        // remove the item from the cache to allow flush to complete
        storage_with_eviction_cache.cache.remove(&id);

        std::thread::sleep(Duration::from_millis(100));
        // flush should not call flush on the underlying storage layer and return
        assert!(thread.is_finished());
        assert!(thread.join().is_ok());
    }

    #[test]
    fn shutdown_eviction_workers_calls_shutdown_on_eviction_workers() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_received = Arc::new(AtomicUsize::new(0));
        let workers = vec![{
            let shutdown = shutdown.clone();
            let shutdown_received = shutdown_received.clone();
            std::thread::spawn(move || {
                while !shutdown.load(Ordering::SeqCst) {
                    std::thread::yield_now(); // Simulate worker doing work
                }
                shutdown_received.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        }];

        let storage_with_eviction_cache = StorageWithEvictionCache {
            cache: Arc::new(DashMap::new()),
            storage: Arc::new(MockFileStorageManager::<MockFileBackend>::new()),
            eviction_workers: EvictionWorkers { workers, shutdown },
        };

        assert_eq!(shutdown_received.load(Ordering::SeqCst), 0);
        storage_with_eviction_cache
            .shutdown_eviction_workers()
            .unwrap();
        assert_eq!(shutdown_received.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn eviction_workers_new_spawns_threads() {
        let eviction_cache = Arc::new(DashMap::new());
        let storage = Arc::new(MockFileStorageManager::<MockFileBackend>::new());

        let workers = EvictionWorkers::new(&eviction_cache, &storage);
        assert_eq!(workers.workers.len(), EvictionWorkers::WORKER_COUNT);
        for worker in &workers.workers {
            assert!(!worker.is_finished()); // Ensure the worker is running
        }
    }

    #[test]
    fn eviction_workers_task_calls_underlying_storage_layer_and_removes_elements_once_processed() {
        let eviction_cache = Arc::new(DashMap::new());
        let shutdown = Arc::new(AtomicBool::new(false));

        let mut storage = MockFileStorageManager::<MockFileBackend>::new();
        storage.expect_set().returning(|_, _| Ok(()));
        storage.expect_delete().returning(|_| Ok(()));
        let storage = Arc::new(storage);

        let workers = vec![{
            let eviction_cache = eviction_cache.clone();
            let shutdown = shutdown.clone();

            std::thread::spawn(move || EvictionWorkers::task(&eviction_cache, &*storage, &shutdown))
        }];

        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);
        let node = Arc::new(RwLock::new(CachedNode::new_clean(Node::Inner(
            Box::default(),
        ))));

        eviction_cache.insert(id, Op::Set(node.clone()));

        // Allow the worker to process the set operation
        std::thread::sleep(Duration::from_millis(100));
        assert!(eviction_cache.is_empty());

        eviction_cache.insert(id, Op::Delete);

        // Allow the worker to process the delete operation
        std::thread::sleep(Duration::from_millis(100));
        assert!(eviction_cache.is_empty());

        shutdown.store(true, Ordering::SeqCst);
        for worker in workers {
            assert!(worker.join().is_ok());
        }
    }

    #[test]
    fn eviction_workers_shutdowns_signals_threads_to_stop_and_waits_until_they_return() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_received = Arc::new(AtomicUsize::new(0));
        let workers = (0..EvictionWorkers::WORKER_COUNT)
            .map(|_| {
                let shutdown = shutdown.clone();
                let shutdown_received = shutdown_received.clone();
                std::thread::spawn(move || {
                    while !shutdown.load(Ordering::SeqCst) {
                        std::thread::yield_now(); // Simulate worker doing work
                    }
                    shutdown_received.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
            })
            .collect();
        let eviction_workers = EvictionWorkers { workers, shutdown };

        assert_eq!(shutdown_received.load(Ordering::SeqCst), 0);
        eviction_workers.shutdown().unwrap();
        assert_eq!(
            shutdown_received.load(Ordering::SeqCst),
            EvictionWorkers::WORKER_COUNT
        );
    }
}
