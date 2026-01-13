// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{cmp, path::Path, time::Duration};

use crossbeam_skiplist::SkipMap;

use crate::{
    error::BTResult,
    storage::{Checkpointable, DbMode, Error, RootIdProvider, Storage},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        hint, thread,
    },
};

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

impl<S> Storage for StorageWithFlushBuffer<S>
where
    S: Storage + 'static,
    S::Id: std::hash::Hash + cmp::Ord + Send + Sync,
    S::Item: Clone + Send + Sync,
{
    type Id = S::Id;
    type Item = S::Item;

    fn open(path: &Path, db_mode: DbMode) -> BTResult<Self, Error> {
        let storage = Arc::new(S::open(path, db_mode)?);
        let flush_buffer = Arc::new(SkipMap::new());
        let workers = FlushWorkers::new(&flush_buffer, &storage);
        Ok(StorageWithFlushBuffer {
            flush_buffer,
            flush_workers: workers,
            storage,
        })
    }

    fn get(&self, id: Self::Id) -> BTResult<Self::Item, Error> {
        match self.flush_buffer.get(&id) {
            Some(value) => match value.value() {
                Op::Set(node) => Ok(node.clone()),
                Op::Delete => Err(Error::NotFound.into()),
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

    fn set(&self, id: Self::Id, node: &Self::Item) -> BTResult<(), Error> {
        self.flush_buffer.insert(id, Op::Set(node.clone()));
        Ok(())
    }

    fn delete(&self, id: Self::Id) -> BTResult<(), Error> {
        self.flush_buffer.insert(id, Op::Delete);
        Ok(())
    }

    fn close(self) -> BTResult<(), Error> {
        // Busy loop until all flush workers are done.
        // Because there are no concurrent operations, len() might only return a number that is
        // higher that the actual number of items (in case an element of the flush buffer
        // was removed by a flush worker while iterating over the shards). This is however not a
        // problem because we will wait a little bit longer.
        while !self.flush_buffer.is_empty() {
            hint::spin_loop();
        }
        self.flush_workers.shutdown()?;
        let storage = Arc::into_inner(self.storage).ok_or_else(|| {
            Error::Internal(
                "storage reference count is not 1 although flush workers are shut down".into(),
            )
        })?;
        storage.close()?;
        Ok(())
    }
}

impl<S> Checkpointable for StorageWithFlushBuffer<S>
where
    S: Storage + Checkpointable,
    S::Id: std::hash::Hash + Eq + Send + Sync,
    S::Item: Send + Sync,
{
    fn checkpoint(&self) -> BTResult<u64, Error> {
        // Busy loop until all flush workers are done.
        // Because there are no concurrent operations, len() might only return a number that is
        // higher that the actual number of items (in case an element of the flush buffer
        // was removed by a flush worker while iterating over the shards). This is however not a
        // problem because we will wait a little bit longer.
        while !self.flush_buffer.is_empty() {}
        self.storage.checkpoint()
    }

    fn restore(path: &Path, checkpoint: u64) -> BTResult<(), Error> {
        S::restore(path, checkpoint)
    }
}

impl<S> RootIdProvider for StorageWithFlushBuffer<S>
where
    S: Storage + RootIdProvider,
{
    type Id = <S as RootIdProvider>::Id;

    fn get_root_id(&self, block_number: u64) -> BTResult<Self::Id, Error> {
        self.storage.get_root_id(block_number)
    }

    fn set_root_id(&self, block_number: u64, id: Self::Id) -> BTResult<(), Error> {
        self.storage.set_root_id(block_number, id)
    }

    fn highest_block_number(&self) -> BTResult<Option<u64>, Error> {
        self.storage.highest_block_number()
    }
}

/// A wrapper around a set of flush worker threads that allows to shut them down gracefully.
struct FlushWorkers {
    workers: Vec<thread::JoinHandle<BTResult<(), Error>>>,
    shutdown: Arc<AtomicBool>, // Arc for shared ownership with flush worker threads
}

impl FlushWorkers {
    const WORKER_COUNT: usize = 2; // TODO the optimal number needs to be determined based on benchmarks

    /// Creates a new set of flush workers that will process items from the flush buffer and
    /// write them to the underlying storage layer.
    fn new<S>(flush_buffer: &Arc<FlushBuffer<S::Id, S::Item>>, storage: &Arc<S>) -> Self
    where
        S: Storage + Send + Sync + 'static,
        S::Id: Eq + std::hash::Hash + Send + Sync + cmp::Ord,
        S::Item: Clone + Send + Sync,
    {
        let shutdown = Arc::new(AtomicBool::new(false));
        // TODO: Run this in a worker pool instead
        // https://github.com/0xsoniclabs/sonic-admin/issues/486
        let workers = (0..Self::WORKER_COUNT)
            .map(|_| {
                let flush_buffer = flush_buffer.clone();
                let storage = storage.clone();
                let shutdown = shutdown.clone();
                thread::spawn(move || FlushWorkers::task(&flush_buffer, &*storage, &shutdown))
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
    ) -> BTResult<(), Error>
    where
        S: Storage,
        S::Id: std::hash::Hash + Send + Sync + cmp::Ord + 'static,
        S::Item: Clone + Send + Sync + 'static,
    {
        let min_sleep_time = Duration::from_millis(10);
        let max_sleep_time = Duration::from_secs(1);
        let mut sleep_time = min_sleep_time;

        loop {
            if let Some((id, op)) = flush_buffer
                .pop_back()
                .map(|entry| (*entry.key(), entry.value().clone()))
            {
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
                sleep_time = min_sleep_time;
            } else {
                // the buffer is currently empty
                if shutdown.load(Ordering::SeqCst) {
                    return Ok(());
                }
                thread::sleep(sleep_time);
                sleep_time = cmp::min(sleep_time * 2, max_sleep_time);
            }
        }
    }

    fn shutdown(self) -> BTResult<(), Error> {
        self.shutdown.store(true, Ordering::SeqCst);
        for worker in self.workers {
            worker.join().unwrap()?;
        }
        Ok(())
    }
}

type FlushBuffer<ID, N> = SkipMap<ID, Op<N>>;

/// An element in the flush buffer that can either be a set operation or a delete operation.
#[derive(Debug, Clone)]
enum Op<N> {
    Set(N),
    Delete,
}

#[cfg(test)]
mod tests {
    use std::{fs::File, time::Duration};

    use mockall::predicate::eq;

    use super::*;
    use crate::{
        error::BTError,
        storage::file::{
            NodeFileStorage, SeekFile, TestNode, TestNodeFileStorageManager, TestNodeId,
            TestNodeKind, file_storage_manager::NonEmpty1TestNode,
        },
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
            is_finished, thread,
        },
        types::TreeId,
        utils::{
            shuttle::run_shuttle_check,
            test_dir::{Permissions, TestDir},
        },
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
            TestNodeFileStorageManager<NodeFileStorage<_, SeekFile>, NodeFileStorage<_, SeekFile>>,
        >::open(&dir, DbMode::ReadWrite)
        .unwrap();
    }

    #[test]
    fn open_opens_underlying_storage_and_starts_flush_workers() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        // Using mocks for `FileStorageManager` and `NoSeekFile` is not possible, because `open`
        // creates the mocks using calls to `open` on the mock type, but the mocks have no
        // expectations set up.
        let storage = StorageWithFlushBuffer::<
            TestNodeFileStorageManager<NodeFileStorage<_, SeekFile>, NodeFileStorage<_, SeekFile>>,
        >::open(&dir, DbMode::ReadWrite)
        .unwrap();

        // The node store files should be locked while opened
        let file = File::open(
            dir.join(
                TestNodeFileStorageManager::<
                    NodeFileStorage<_, SeekFile>,
                    NodeFileStorage<_, SeekFile>,
                >::NON_EMPTY1_DIR,
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
            assert!(!is_finished(worker)); // Ensure the worker is running
        }
    }

    #[test]
    fn get_returns_copy_of_node_if_present_as_set_op() {
        let storage = StorageWithFlushBuffer {
            flush_buffer: Arc::new(SkipMap::new()),
            storage: Arc::new(MockStorage::new()),
            flush_workers: FlushWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        let id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::NonEmpty1);
        let node = TestNode::NonEmpty1(Box::default());

        storage.flush_buffer.insert(id, Op::Set(node.clone()));

        let result = storage.get(id).unwrap();
        assert_eq!(result, node);
        // TestNode is kept for eventual flush.
        assert!(storage.flush_buffer.get(&id).is_some());
    }

    #[test]
    fn get_returns_not_found_error_if_id_is_present_as_delete_op() {
        let storage = StorageWithFlushBuffer {
            flush_buffer: Arc::new(SkipMap::new()),
            storage: Arc::new(MockStorage::new()),
            flush_workers: FlushWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        let id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::NonEmpty1);
        storage.flush_buffer.insert(id, Op::Delete);

        let result = storage.get(id);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::NotFound)
        ));
    }

    #[test]
    fn get_returns_node_from_storage_if_not_in_buffer() {
        let id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::NonEmpty1);
        let node = TestNode::NonEmpty1(Box::default());

        let mut mock_storage = MockStorage::new();
        mock_storage.expect_get().with(eq(id)).returning({
            let node = node.clone();
            move |_| Ok(node.clone())
        });

        let storage_with_flush_buffer = StorageWithFlushBuffer {
            flush_buffer: Arc::new(SkipMap::new()),
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
        let id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::NonEmpty1);
        let node = TestNode::NonEmpty1(Box::default());

        let mut mock_storage = MockStorage::new();
        mock_storage
            .expect_reserve()
            .with(eq(node.clone()))
            .returning(move |_| id);

        let storage_with_flush_buffer = StorageWithFlushBuffer {
            flush_buffer: Arc::new(SkipMap::new()),
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
        let id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::NonEmpty1);
        let node = TestNode::NonEmpty1(Box::default());

        let storage_with_flush_buffer = StorageWithFlushBuffer {
            flush_buffer: Arc::new(SkipMap::new()),
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
        let id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::NonEmpty1);

        let storage_with_flush_buffer = StorageWithFlushBuffer {
            flush_buffer: Arc::new(SkipMap::new()),
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
    fn close_calls_close_on_underlying_storage_layer() {
        let mut mock_storage = MockStorage::new();
        mock_storage.expect_close().times(1).returning(|| Ok(()));

        let storage_with_flush_buffer = StorageWithFlushBuffer {
            flush_buffer: Arc::new(SkipMap::new()),
            storage: Arc::new(mock_storage),
            flush_workers: FlushWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        storage_with_flush_buffer.close().unwrap();
    }

    #[test]
    fn checkpoint_waits_until_buffer_is_empty_then_calls_checkpoint_on_underlying_storage_layer() {
        let id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::NonEmpty1);
        let node = TestNode::NonEmpty1(Box::default());

        let mut mock_storage = MockStorage::new();
        mock_storage
            .expect_checkpoint()
            .times(1)
            .returning(|| Ok(1));

        let storage_with_flush_buffer = StorageWithFlushBuffer {
            flush_buffer: Arc::new(SkipMap::new()),
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

        let thread = thread::spawn({
            let storage_with_flush_buffer = storage_with_flush_buffer.clone();
            move || storage_with_flush_buffer.checkpoint()
        });

        // flush is waiting
        assert!(!is_finished(&thread));
        thread::sleep(Duration::from_millis(100));
        // flush is still waiting
        assert!(!is_finished(&thread));

        // remove the item from the buffer to allow flush to complete
        storage_with_flush_buffer.flush_buffer.remove(&id);

        thread::sleep(Duration::from_millis(100));
        // flush should call flush on the underlying storage layer and return
        assert!(is_finished(&thread));
        assert!(thread.join().is_ok());
    }

    #[test]
    fn restore_calls_restore_on_underlying_storage() {
        let ctx = MockStorage::restore_context();
        ctx.expect()
            .with(eq(Path::new("/path_of_restore_test")), eq(1))
            .returning(|_, _| Ok(()))
            .times(1);

        StorageWithFlushBuffer::<MockStorage>::restore(Path::new("/path_of_restore_test"), 1)
            .unwrap();
    }

    #[test]
    fn get_root_id_calls_get_root_id_on_underlying_storage_layer() {
        let block_number = 1;
        let root_id = TestNodeId::from_idx_and_node_kind(1, TestNodeKind::NonEmpty1);

        let mut mock_storage = MockStorage::new();
        mock_storage
            .expect_get_root_id()
            .with(eq(block_number))
            .returning(move |_| Ok(root_id))
            .times(1);

        let storage_with_flush_buffer = StorageWithFlushBuffer {
            flush_buffer: Arc::new(SkipMap::new()),
            storage: Arc::new(mock_storage),
            flush_workers: FlushWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        assert_eq!(
            storage_with_flush_buffer.get_root_id(block_number),
            Ok(root_id)
        );
    }

    #[test]
    fn set_root_id_calls_set_root_id_on_underlying_storage_layer() {
        let block_number = 1;
        let root_id = TestNodeId::from_idx_and_node_kind(1, TestNodeKind::NonEmpty1);

        let mut mock_storage = MockStorage::new();
        mock_storage
            .expect_set_root_id()
            .with(eq(block_number), eq(root_id))
            .returning(|_, _| Ok(()))
            .times(1);

        let storage_with_flush_buffer = StorageWithFlushBuffer {
            flush_buffer: Arc::new(SkipMap::new()),
            storage: Arc::new(mock_storage),
            flush_workers: FlushWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        assert!(
            storage_with_flush_buffer
                .set_root_id(block_number, root_id)
                .is_ok(),
        );
    }

    #[test]
    fn get_highest_block_number_calls_get_highest_block_number_on_underlying_storage_layer() {
        let highest_block_number = Some(1);

        let mut mock_storage = MockStorage::new();
        mock_storage
            .expect_highest_block_number()
            .returning(move || Ok(highest_block_number))
            .times(1);

        let storage_with_flush_buffer = StorageWithFlushBuffer {
            flush_buffer: Arc::new(SkipMap::new()),
            storage: Arc::new(mock_storage),
            flush_workers: FlushWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        assert_eq!(
            storage_with_flush_buffer.highest_block_number(),
            Ok(highest_block_number)
        );
    }

    #[test]
    fn flush_workers_new_spawns_threads() {
        let flush_buffer = Arc::new(SkipMap::new());
        let storage = Arc::new(MockStorage::new());

        let workers = FlushWorkers::new(&flush_buffer, &storage);
        assert_eq!(workers.workers.len(), FlushWorkers::WORKER_COUNT);
        for worker in &workers.workers {
            assert!(!is_finished(worker)); // Ensure the worker is running
        }
    }

    #[test]
    fn flush_workers_task_calls_underlying_storage_layer_and_removes_elements_once_processed() {
        let flush_buffer = Arc::new(SkipMap::new());
        let shutdown = Arc::new(AtomicBool::new(false));

        let mut storage = MockStorage::new();
        storage.expect_set().returning(|_, _| Ok(()));
        storage.expect_delete().returning(|_| Ok(()));
        let storage = Arc::new(storage);

        let workers = vec![{
            let flush_buffer = flush_buffer.clone();
            let shutdown = shutdown.clone();

            thread::spawn(move || FlushWorkers::task(&flush_buffer, &*storage, &shutdown))
        }];

        let id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::NonEmpty1);
        let node = TestNode::NonEmpty1(Box::default());

        flush_buffer.insert(id, Op::Set(node.clone()));

        // Allow the worker to process the set operation
        thread::sleep(Duration::from_millis(100));
        assert!(flush_buffer.is_empty());

        flush_buffer.insert(id, Op::Delete);

        // Allow the worker to process the delete operation
        thread::sleep(Duration::from_millis(100));
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
                thread::spawn(move || {
                    while !shutdown.load(Ordering::SeqCst) {
                        thread::yield_now(); // Simulate worker doing work
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

    #[test]
    fn shuttletest_flush_workers_executes_a_task_only_once() {
        run_shuttle_check(
            || {
                let testdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
                // Use an actual storage as mockall does not use shuttle sync primitives, which we
                // need to ensure context switches between shuttle threads.
                let storage = Arc::new(
                    NodeFileStorage::<_, SeekFile>::open(&testdir, DbMode::ReadWrite).unwrap(),
                );
                let node = NonEmpty1TestNode::default();
                let id = storage.reserve(&node);
                storage.set(id, &node).unwrap();

                let flush_buffer = Arc::new(SkipMap::new());
                let workers = FlushWorkers::new(&flush_buffer, &storage.clone());
                // This should call delete only once, and panic if two workers delete the same ID
                flush_buffer.insert(id, Op::Delete);

                workers.shutdown().unwrap();
            },
            100,
        );
    }

    #[test]
    fn shuttletest_deleted_nodes_are_not_found_when_queried() {
        run_shuttle_check(
            || {
                let testdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
                // Use an actual storage as mockall does not use shuttle sync primitives, which we
                // need to ensure context switches between shuttle threads.
                let storage = Arc::new(
                    NodeFileStorage::<_, SeekFile>::open(&testdir, DbMode::ReadWrite).unwrap(),
                );
                let node = NonEmpty1TestNode::default();
                let id = storage.reserve(&node);
                storage.set(id, &node).unwrap();

                let flush_buffer = Arc::new(SkipMap::new());
                let flush_workers = FlushWorkers::new(&flush_buffer, &storage);
                let storage_with_flush_buffer = StorageWithFlushBuffer {
                    flush_buffer,
                    storage,
                    flush_workers,
                };

                // Delete the `node` with `id`, which adds a delete op to the flush buffer.
                // The actual deleting happens asynchronously in the flush worker.
                storage_with_flush_buffer.delete(id).unwrap();

                // Either the delete op is still in the flush buffer or the node has been deleted in
                // the underlying storage layer.
                thread::yield_now();
                assert_eq!(
                    storage_with_flush_buffer
                        .get(id)
                        .map_err(BTError::into_inner),
                    Err(Error::NotFound)
                );

                storage_with_flush_buffer.flush_workers.shutdown().unwrap();
            },
            1000,
        );
    }

    #[test]
    fn shuttletest_newest_set_is_seen_when_queried() {
        run_shuttle_check(
            || {
                let testdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
                // Use an actual storage as mockall does not use shuttle sync primitives, which we
                // need to ensure context switches between shuttle threads.
                let storage = Arc::new(
                    NodeFileStorage::<_, SeekFile>::open(&testdir, DbMode::ReadWrite).unwrap(),
                );
                let node = NonEmpty1TestNode::default();
                let id = storage.reserve(&node);
                storage.set(id, &node).unwrap();

                let flush_buffer = Arc::new(SkipMap::new());
                let flush_workers = FlushWorkers::new(&flush_buffer, &storage);
                let storage_with_flush_buffer = StorageWithFlushBuffer {
                    flush_buffer,
                    storage,
                    flush_workers,
                };

                let mut node2 = node.clone();
                node2.0 = [1; _];

                // Update the `node` with `id` to `node2`, which adds a set op to the flush buffer.
                // The actual write out happens asynchronously in the flush worker.
                storage_with_flush_buffer.set(id, &node2).unwrap();

                // Either the set op is still in the flush buffer or the node has been updated in
                // the underlying storage layer.
                thread::yield_now();
                assert_eq!(storage_with_flush_buffer.get(id), Ok(node2));

                storage_with_flush_buffer.flush_workers.shutdown().unwrap();
            },
            1000,
        );
    }

    #[allow(clippy::disallowed_types)]
    mod mock {
        use super::*;

        mockall::mock! {
            pub Storage {}

            impl Checkpointable for Storage {
                fn checkpoint(&self) -> BTResult<u64, Error>;

                fn restore(path: &Path, checkpoint: u64) -> BTResult<(), Error>;
            }

            impl Storage for Storage {
                type Id = TestNodeId;
                type Item = TestNode;

                fn open(_path: &Path, db_mode: DbMode) -> BTResult<Self, Error>;

                fn get(&self, id: <Self as Storage>::Id) -> BTResult<<Self as Storage>::Item, Error>;

                fn reserve(&self, item: &<Self as Storage>::Item) -> <Self as Storage>::Id;

                fn set(&self, id: <Self as Storage>::Id, item: &<Self as Storage>::Item) -> BTResult<(), Error>;

                fn delete(&self, id: <Self as Storage>::Id) -> BTResult<(), Error>;

                fn close(self) -> BTResult<(), Error>;
            }

            impl RootIdProvider for Storage {
                type Id = TestNodeId;

                fn get_root_id(&self, block_number: u64) -> BTResult<<Self as RootIdProvider>::Id, Error> {
                    self.storage.get_root_id(block_number)
                }

                fn set_root_id(&self, block_number: u64, id: <Self as RootIdProvider>::Id) -> BTResult<(), Error> {
                    self.storage.set_root_id(block_number, id)
                }

                fn highest_block_number(&self) -> BTResult<Option<u64>, Error> {
                    self.storage.highest_block_number()
                }
            }
        }
    }

    use mock::MockStorage;
}
