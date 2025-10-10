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
        atomic::{AtomicBool, Ordering},
    },
};

use dashmap::DashMap;

use crate::storage::{Checkpointable, Error, RootIdProvider, Storage};

/// A storage backend that uses a flush buffer to hold updates and deletions while they get
/// written to the underlying storage layer in background threads.
///
/// Upon deletion, the id is deleted in the underlying storage layer first and then removed from the
/// flush buffer. In case it was deleted in the underlying storage layer and reassigned by a
/// concurrent task before the flush worker deleted it from the flush buffer, the
/// `StorageWithFlushBuffer:reserve` method also deletes the id from the flush buffer. This way it
/// is guaranteed that a reassigned id is not found in the flush buffer.
///
/// Queries always check the flush buffer first. If the id is found there and it is an update, a
/// copy of the node is returned, but the node is kept in the flush buffer. This ensures that
/// once changes to a node enter the flush buffer, they will eventually be persisted. If a node
/// is updated repeatedly before a flush worker manages to flush the changes to the underlying
/// storage layer, later updates replace earlier ones and only the latest changes are persisted.
/// If the id is found in the flush buffer and it is a delete operation, a not found error is
/// returned. Only if the id is not found in the flush buffer, the underlying storage layer is
/// queried.
pub struct StorageWithFlushBuffer<S>
where
    S: Storage,
{
    // Arc for shared ownership with flush worker threads
    flush_buffer: Arc<FlushBuffer<S::Id, S::Item>>,
    storage: Arc<S>, // Arc for shared ownership with flush worker threads
    flush_workers: FlushWorkers,
}

impl<S> StorageWithFlushBuffer<S>
where
    S: Storage,
{
    pub fn shutdown_flush_workers(self) -> Result<(), Error> {
        self.flush_workers.shutdown()
    }
}

impl<S> Storage for StorageWithFlushBuffer<S>
where
    S: Storage + 'static,
    S::Id: std::hash::Hash + Eq + Send + Sync,
    S::Item: Clone + Send + Sync,
{
    type Id = S::Id;
    type Item = S::Item;

    fn open(path: &Path) -> Result<Self, Error> {
        let storage = Arc::new(S::open(path)?);
        let flush_buffer = Arc::new(DashMap::new());
        let workers = FlushWorkers::new(&flush_buffer, &storage);
        Ok(StorageWithFlushBuffer {
            flush_buffer,
            flush_workers: workers,
            storage,
        })
    }

    fn get(&self, id: Self::Id) -> Result<Self::Item, Error> {
        match self.flush_buffer.get(&id) {
            Some(value) => match value.value() {
                Op::Set(node) => Ok(node.clone()),
                Op::Delete => Err(Error::NotFound),
            },
            None => Ok(self.storage.get(id)?),
        }
    }

    fn reserve(&self, node: &Self::Item) -> Self::Id {
        let id = self.storage.reserve(node);
        // The id may have been deleted in the underlying storage layer and reassigned here, but not
        // yet removed from the flush buffer (racing against flush workers). In this case, we remove
        // it from the flush buffer to ensure that the id no longer returns an
        // [`Error::NotFound`].
        self.flush_buffer.remove(&id);
        id
    }

    fn set(&self, id: Self::Id, node: &Self::Item) -> Result<(), Error> {
        self.flush_buffer.insert(id, Op::Set(node.clone()));
        Ok(())
    }

    fn delete(&self, id: Self::Id) -> Result<(), Error> {
        self.flush_buffer.insert(id, Op::Delete);
        Ok(())
    }
}

impl<S> Checkpointable for StorageWithFlushBuffer<S>
where
    S: Storage + Checkpointable,
    S::Id: std::hash::Hash + Eq + Send + Sync,
    S::Item: Send + Sync,
{
    fn checkpoint(&self) -> Result<(), Error> {
        // Busy loop until all flush workers are done.
        // Because there are no concurrent operations, len() might only return a number that is
        // higher that the actual number of items (in case an element of the flush buffer
        // was removed by a flush worker while iterating over the shards). This is however not a
        // problem because we will wait a little bit longer.
        while !self.flush_buffer.is_empty() {}
        self.storage.checkpoint()
    }
}

impl<S> RootIdProvider for StorageWithFlushBuffer<S>
where
    S: Storage + RootIdProvider,
{
    type Id = <S as RootIdProvider>::Id;

    fn get_root_id(&self, block_number: u64) -> Result<Self::Id, Error> {
        self.storage.get_root_id(block_number)
    }

    fn set_root_id(&self, block_number: u64, id: Self::Id) -> Result<(), Error> {
        self.storage.set_root_id(block_number, id)
    }
}

/// A wrapper around a set of flush worker threads that allows to shut them down gracefully.
struct FlushWorkers {
    workers: Vec<std::thread::JoinHandle<Result<(), Error>>>,
    shutdown: Arc<AtomicBool>, // Arc for shared ownership with flush worker threads
}

impl FlushWorkers {
    const WORKER_COUNT: usize = 10; // TODO the optimal number needs to be determined based on benchmarks

    /// Creates a new set of flush workers that will process items from the flush buffer and
    /// write them to the underlying storage layer.
    pub fn new<S>(flush_buffer: &Arc<FlushBuffer<S::Id, S::Item>>, storage: &Arc<S>) -> Self
    where
        S: Storage + Send + Sync + 'static,
        S::Id: Eq + std::hash::Hash + Send + Sync,
        S::Item: Clone + Send + Sync,
    {
        let shutdown = Arc::new(AtomicBool::new(false));
        let workers = (0..Self::WORKER_COUNT)
            .map(|_| {
                let flush_buffer = flush_buffer.clone();
                let storage = storage.clone();
                let shutdown = shutdown.clone();
                std::thread::spawn(move || FlushWorkers::task(&flush_buffer, &*storage, &shutdown))
            })
            .collect();

        FlushWorkers { workers, shutdown }
    }

    /// The task that each flush worker runs. It processes items from the flush buffer
    /// and writes them to the underlying storage layer.
    fn task<S>(
        flush_buffer: &FlushBuffer<S::Id, S::Item>,
        storage: &S,
        shutdown: &Arc<AtomicBool>,
    ) -> Result<(), Error>
    where
        S: Storage,
        S::Id: Eq + std::hash::Hash + Send + Sync,
        S::Item: Clone + Send + Sync,
    {
        loop {
            let item = flush_buffer
                .iter()
                .next()
                .map(|entry| (*entry.key(), entry.value().clone()));

            if let Some((id, op)) = item {
                match op {
                    Op::Set(node) => {
                        storage.set(id, &node)?;
                    }
                    Op::Delete => {
                        // Delete the id in the underlying storage first, then remove it from the
                        // flush buffer.
                        // Once the id was deleted in the underlying storage layer, it may get
                        // reused in a call to `reserve`.
                        // For the case that the id was deleted in the storage layer and reassigned
                        // by a concurrent task before the flush worker deleted it from the
                        // flush buffer, `StorageWithFlushBuffer::reserve` also deletes the id
                        // from the flush buffer. This way it is guaranteed, that a reassigned id
                        // is not found in the flush buffer.
                        storage.delete(id)?;
                    }
                }
                flush_buffer.remove(&id);
            } else {
                // the buffer is currently empty
                if shutdown.load(Ordering::SeqCst) {
                    return Ok(());
                }
                // avoid busy looping
                // TODO: use a condvar or similar
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

type FlushBuffer<ID, N> = DashMap<ID, Op<N>>;

/// An element in the flush buffer that can either be a set operation or a delete operation.
#[derive(Debug, Clone)]
enum Op<N> {
    Set(N),
    Delete,
}

#[cfg(test)]
mod tests {
    use std::{fs::File, sync::atomic::AtomicUsize, time::Duration};

    use mockall::predicate::eq;

    use super::*;
    use crate::{
        database::verkle::variants::managed::{Node, NodeId, NodeType},
        storage::file::{FileStorageManager, NodeFileStorage, SeekFile},
        utils::test_dir::{Permissions, TestDir},
    };

    #[test]
    fn open_all_nested_layers() {
        // The purpose of this test is to ensure that `StorageWithFlushBuffer` can be used with
        // the lower layers of the storage system (that the types and interfaces line up).
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        // this opens:
        // StorageWithFlushBuffer
        //   -> FileStorageManager
        //     -> A NodeFileStorage for each node type (InnerNode, SparseLeafNode<N>, ...)
        //       -> SeekFile
        StorageWithFlushBuffer::<
            FileStorageManager<
                NodeFileStorage<_, SeekFile>,
                NodeFileStorage<_, SeekFile>,
                NodeFileStorage<_, SeekFile>,
            >,
        >::open(&dir)
        .unwrap();
    }

    #[test]
    fn open_opens_underlying_storage_and_starts_flush_workers() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        // Using mocks for `FileStorageManager` and `NoSeekFile` is not possible, because `open`
        // creates the mocks using calls to `open` on the mock type, but the mocks have no
        // expectations set up.
        let storage = StorageWithFlushBuffer::<
            FileStorageManager<
                NodeFileStorage<_, SeekFile>,
                NodeFileStorage<_, SeekFile>,
                NodeFileStorage<_, SeekFile>,
            >,
        >::open(&dir)
        .unwrap();

        // The node store files should be locked while opened
        let file = File::open(
            dir.join(
                FileStorageManager::<
                    NodeFileStorage<_, SeekFile>,
                    NodeFileStorage<_, SeekFile>,
                    NodeFileStorage<_, SeekFile>,
                >::INNER_NODE_DIR,
            )
            .join(NodeFileStorage::<u8, SeekFile>::NODE_STORE_FILE),
        )
        .unwrap();
        assert!(file.try_lock().is_err());

        assert_eq!(
            storage.flush_workers.workers.len(),
            FlushWorkers::WORKER_COUNT
        );
        for worker in &storage.flush_workers.workers {
            assert!(!worker.is_finished()); // Ensure the worker is running
        }
    }

    #[test]
    fn get_returns_copy_of_node_if_present_as_set_op() {
        let storage = StorageWithFlushBuffer {
            flush_buffer: Arc::new(DashMap::new()),
            storage: Arc::new(MockStorage::new()),
            flush_workers: FlushWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);
        let node = Node::Inner(Box::default());

        storage.flush_buffer.insert(id, Op::Set(node.clone()));

        let result = storage.get(id).unwrap();
        assert_eq!(result, node);
        // Node is kept for eventual flush.
        assert!(storage.flush_buffer.get(&id).is_some());
    }

    #[test]
    fn get_returns_not_found_error_if_id_is_present_as_delete_op() {
        let storage = StorageWithFlushBuffer {
            flush_buffer: Arc::new(DashMap::new()),
            storage: Arc::new(MockStorage::new()),
            flush_workers: FlushWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);
        storage.flush_buffer.insert(id, Op::Delete);

        let result = storage.get(id);
        assert!(matches!(result, Err(Error::NotFound)));
    }

    #[test]
    fn get_returns_node_from_storage_if_not_in_buffer() {
        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);
        let node = Node::Inner(Box::default());

        let mut mock_storage = MockStorage::new();
        mock_storage.expect_get().with(eq(id)).returning({
            let node = node.clone();
            move |_| Ok(node.clone())
        });

        let storage_with_flush_buffer = StorageWithFlushBuffer {
            flush_buffer: Arc::new(DashMap::new()),
            storage: Arc::new(mock_storage),
            flush_workers: FlushWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        let result = storage_with_flush_buffer.get(id).unwrap();
        assert_eq!(result, node);
    }

    #[test]
    fn reserve_retrieves_id_from_underlying_storage_layer_and_removes_from_buffer() {
        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);
        let node = Node::Inner(Box::default());

        let mut mock_storage = MockStorage::new();
        mock_storage
            .expect_reserve()
            .with(eq(node.clone()))
            .returning(move |_| id);

        let storage_with_flush_buffer = StorageWithFlushBuffer {
            flush_buffer: Arc::new(DashMap::new()),
            storage: Arc::new(mock_storage),
            flush_workers: FlushWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        let reserved_id = storage_with_flush_buffer.reserve(&node);
        assert_eq!(reserved_id, id);
        assert!(storage_with_flush_buffer.flush_buffer.get(&id).is_none());
    }

    #[test]
    fn set_inserts_set_op_into_buffer() {
        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);
        let node = Node::Inner(Box::default());

        let storage_with_flush_buffer = StorageWithFlushBuffer {
            flush_buffer: Arc::new(DashMap::new()),
            storage: Arc::new(MockStorage::new()),
            flush_workers: FlushWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        storage_with_flush_buffer.set(id, &node).unwrap();

        let entry = storage_with_flush_buffer.flush_buffer.get(&id);
        assert!(entry.is_some());
        let entry = entry.unwrap();
        let value = entry.value();
        assert!(matches!(value, Op::Set(n) if n == &node));
    }

    #[test]
    fn delete_inserts_delete_op_into_buffer() {
        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);

        let storage_with_flush_buffer = StorageWithFlushBuffer {
            flush_buffer: Arc::new(DashMap::new()),
            storage: Arc::new(MockStorage::new()),
            flush_workers: FlushWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        storage_with_flush_buffer.delete(id).unwrap();

        let entry = storage_with_flush_buffer.flush_buffer.get(&id);
        assert!(entry.is_some());
        let entry = entry.unwrap();
        let value = entry.value();
        assert!(matches!(value, Op::Delete));
    }

    #[test]
    fn checkpoint_waits_until_buffer_is_empty_then_calls_checkpoint_on_underlying_storage_layer() {
        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);
        let node = Node::Inner(Box::default());

        let mut mock_storage = MockStorage::new();
        mock_storage
            .expect_checkpoint()
            .times(1)
            .returning(|| Ok(()));

        let storage_with_flush_buffer = StorageWithFlushBuffer {
            flush_buffer: Arc::new(DashMap::new()),
            storage: Arc::new(mock_storage),
            flush_workers: FlushWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        storage_with_flush_buffer
            .flush_buffer
            .insert(id, Op::Set(node.clone()));

        let storage_with_flush_buffer = Arc::new(storage_with_flush_buffer);

        let thread = std::thread::spawn({
            let storage_with_flush_buffer = storage_with_flush_buffer.clone();
            move || storage_with_flush_buffer.checkpoint()
        });

        // flush is waiting
        assert!(!thread.is_finished());
        std::thread::sleep(Duration::from_millis(100));
        // flush is still waiting
        assert!(!thread.is_finished());

        // remove the item from the buffer to allow flush to complete
        storage_with_flush_buffer.flush_buffer.remove(&id);

        std::thread::sleep(Duration::from_millis(100));
        // flush should call flush on the underlying storage layer and return
        assert!(thread.is_finished());
        assert!(thread.join().is_ok());
    }

    #[test]
    fn shutdown_flush_workers_calls_shutdown_on_flush_workers() {
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

        let storage_with_flush_buffer = StorageWithFlushBuffer {
            flush_buffer: Arc::new(DashMap::new()),
            storage: Arc::new(MockStorage::new()),
            flush_workers: FlushWorkers { workers, shutdown },
        };

        assert_eq!(shutdown_received.load(Ordering::SeqCst), 0);
        storage_with_flush_buffer.shutdown_flush_workers().unwrap();
        assert_eq!(shutdown_received.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn flush_workers_new_spawns_threads() {
        let flush_buffer = Arc::new(DashMap::new());
        let storage = Arc::new(MockStorage::new());

        let workers = FlushWorkers::new(&flush_buffer, &storage);
        assert_eq!(workers.workers.len(), FlushWorkers::WORKER_COUNT);
        for worker in &workers.workers {
            assert!(!worker.is_finished()); // Ensure the worker is running
        }
    }

    #[test]
    fn flush_workers_task_calls_underlying_storage_layer_and_removes_elements_once_processed() {
        let flush_buffer = Arc::new(DashMap::new());
        let shutdown = Arc::new(AtomicBool::new(false));

        let mut storage = MockStorage::new();
        storage.expect_set().returning(|_, _| Ok(()));
        storage.expect_delete().returning(|_| Ok(()));
        let storage = Arc::new(storage);

        let workers = vec![{
            let flush_buffer = flush_buffer.clone();
            let shutdown = shutdown.clone();

            std::thread::spawn(move || FlushWorkers::task(&flush_buffer, &*storage, &shutdown))
        }];

        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);
        let node = Node::Inner(Box::default());

        flush_buffer.insert(id, Op::Set(node.clone()));

        // Allow the worker to process the set operation
        std::thread::sleep(Duration::from_millis(100));
        assert!(flush_buffer.is_empty());

        flush_buffer.insert(id, Op::Delete);

        // Allow the worker to process the delete operation
        std::thread::sleep(Duration::from_millis(100));
        assert!(flush_buffer.is_empty());

        shutdown.store(true, Ordering::SeqCst);
        for worker in workers {
            assert!(worker.join().is_ok());
        }
    }

    #[test]
    fn flush_workers_shutdowns_signals_threads_to_stop_and_waits_until_they_return() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_received = Arc::new(AtomicUsize::new(0));
        let workers = (0..FlushWorkers::WORKER_COUNT)
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
        let flush_workers = FlushWorkers { workers, shutdown };

        assert_eq!(shutdown_received.load(Ordering::SeqCst), 0);
        flush_workers.shutdown().unwrap();
        assert_eq!(
            shutdown_received.load(Ordering::SeqCst),
            FlushWorkers::WORKER_COUNT
        );
    }

    mockall::mock! {
        pub Storage {}

        impl Checkpointable for Storage {
            fn checkpoint(&self) -> Result<(), Error>;
        }

        impl Storage for Storage {
            type Id = NodeId;
            type Item = Node;

            fn open(_path: &Path) -> Result<Self, Error>;

            fn get(&self, id: <Self as Storage>::Id) -> Result<<Self as Storage>::Item, Error>;

            fn reserve(&self, item: &<Self as Storage>::Item) -> <Self as Storage>::Id;

            fn set(&self, id: <Self as Storage>::Id, item: &<Self as Storage>::Item) -> Result<(), Error>;

            fn delete(&self, id: <Self as Storage>::Id) -> Result<(), Error>;
        }
    }
}
