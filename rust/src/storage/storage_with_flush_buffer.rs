// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

#[allow(clippy::disallowed_types)]
use std::{cmp, path::Path, sync::atomic::AtomicU8 as StdAtomicU8, time::Duration};

use dashmap::DashMap;

use crate::{
    error::BTResult,
    storage::{Checkpointable, DbOpenMode, Error, RootIdProvider, Storage},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        hint, thread,
    },
};

/// A storage backend that uses a flush buffer to hold updates and deletions while they get
/// written to the underlying storage layer in background threads.
///
/// Both updates and deletions are held in the buffer until until processing is complete.
/// This ensures that queries can still find the operation in the flush buffer while it is being
/// processed.
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
    flush_buffer: Arc<FlushBuffer<S::Id, S::Item>>, // Arc for shared ownership with flush workers
    storage: Arc<S>,                                // Arc for shared ownership with flush workers
    flush_workers: FlushWorkers,
}

impl<S> Storage for StorageWithFlushBuffer<S>
where
    S: Storage + 'static,
    S::Id: std::hash::Hash + Eq + Send + Sync,
    S::Item: Clone + Send + Sync,
{
    type Id = S::Id;
    type Item = S::Item;

    fn open(path: &Path, db_open_mode: DbOpenMode) -> BTResult<Self, Error> {
        let storage = Arc::new(S::open(path, db_open_mode)?);
        let flush_buffer = Arc::new(DashMap::new());
        let workers = FlushWorkers::new(&flush_buffer, &storage);
        Ok(StorageWithFlushBuffer {
            flush_buffer,
            flush_workers: workers,
            storage,
        })
    }

    fn get(&self, id: Self::Id) -> BTResult<Self::Item, Error> {
        match self.flush_buffer.get(&id) {
            Some(value) => match &value.value().op {
                Op::Set(node) => Ok(node.clone()),
                Op::Delete => Err(Error::NotFound.into()),
            },
            None => Ok(self.storage.get(id)?),
        }
    }

    fn reserve(&self, node: &Self::Item) -> Self::Id {
        let id = self.storage.reserve(node);
        // The id may have been deleted in the underlying storage layer and reassigned here, but not
        // yet removed from the flush buffer (racing against flush workers). In this case, which is
        // very rare, spin until the flush worker removed the id from the flush buffer to ensure
        // that a query for the id no longer returns an [`Error::NotFound`].
        while self.flush_buffer.contains_key(&id) {
            hint::spin_loop();
        }
        id
    }

    fn set(&self, id: Self::Id, node: &Self::Item) -> BTResult<(), Error> {
        self.flush_buffer
            .entry(id)
            .and_modify(|entry| {
                entry.op = Op::Set(node.clone());
                // Dirty stays Dirty, InProgress becomes InProgressDirty, InProgressDirty stays
                // InProgressDirty
                let _ = entry.status.compare_exchange(
                    Status::InProgress as u8,
                    Status::InProgressDirty as u8,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                );
            })
            .or_insert(OpWithStatus::new(Op::Set(node.clone())));
        Ok(())
    }

    fn delete(&self, id: Self::Id) -> BTResult<(), Error> {
        self.flush_buffer
            .entry(id)
            .and_modify(|entry| {
                entry.op = Op::Delete;
                // Dirty stays Dirty, InProgress becomes InProgressDirty, InProgressDirty stays
                // InProgressDirty
                let _ = entry.status.compare_exchange(
                    Status::InProgress as u8,
                    Status::InProgressDirty as u8,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                );
            })
            .or_insert(OpWithStatus::new(Op::Delete));
        Ok(())
    }

    fn close(self) -> BTResult<(), Error> {
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
        // `is_empty` internally calls `len` which iterates over all shards and sums up their
        // lengths. Because the shards are only locked one at a time, this number can be outdated as
        // soon as a lock on a previous shared is released. But because there are no concurrent
        // (insert) operations, `len()` can only return a number that is higher or equal to
        // the actual number of items (in case an element of the flush buffer was removed by
        // a flush worker while iterating over the shards) but not a number that is lower.
        // This is not a problem because we will just wait a little bit longer.
        while !self.flush_buffer.is_empty() {
            hint::spin_loop();
        }
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
        S::Id: Eq + std::hash::Hash + Send + Sync,
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
        S::Id: Eq + std::hash::Hash + Send + Sync,
        S::Item: Clone + Send + Sync,
    {
        let min_sleep_time = Duration::from_millis(10);
        let max_sleep_time = Duration::from_secs(1);
        let mut sleep_time = min_sleep_time;

        loop {
            // Find an operation that is not already being processed, but leave it in the flush
            // buffer until processing is complete. This ensures that queries can still
            // find the operation in the flush buffer while it is being processed.
            let item = flush_buffer
                .iter()
                .find(|entry| {
                    entry
                        .status
                        .compare_exchange(
                            Status::Dirty as u8,
                            Status::InProgress as u8,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        )
                        .is_ok()
                })
                .map(|entry| (*entry.key(), entry.value().op.clone()));

            if let Some((id, op)) = item {
                // Process the item.
                match op {
                    Op::Set(node) => storage.set(id, &node)?,
                    Op::Delete => storage.delete(id)?,
                }
                if flush_buffer
                    .remove_if(&id, |_, entry| {
                        entry.status.load(Ordering::SeqCst) == Status::InProgress as u8
                    })
                    .is_none()
                {
                    // The status was not InProgress, meaning it was set to InProgressDirty while we
                    // were processing the operation. Now that we are done, we need to set it back
                    // to Dirty so that it gets processed again.
                    flush_buffer
                        .get_mut(&id)
                        .unwrap()
                        .status
                        .store(Status::Dirty as u8, Ordering::SeqCst);
                }
                sleep_time = min_sleep_time;
            } else {
                // There are no elements in the flush buffer that are not already being processed.
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

type FlushBuffer<ID, N> = DashMap<ID, OpWithStatus<N>>;

/// An element in the flush buffer that can either be a set operation or a delete operation.
#[derive(Debug, Clone)]
enum Op<N> {
    Set(N),
    Delete,
}

/// The status of an operation in the flush buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Status {
    // The operation is in the flush buffer and has not yet been processed by a flush worker.
    Dirty = 0,
    // The operation is currently being processed by a flush worker and since the flush worker
    // picked it up, it has not been modified.
    InProgress = 1,
    // The operation is currently being processed by a flush worker, but since the flush worker
    // picked it up, it has been modified.
    InProgressDirty = 2,
}

/// An operation in the flush buffer along with its in-progress status.
#[derive(Debug)]
struct OpWithStatus<N> {
    op: Op<N>,
    #[allow(clippy::disallowed_types)]
    status: StdAtomicU8,
}

impl<N> OpWithStatus<N> {
    /// Creates a new `OpWithStatus` with the given operation and in-progress status set to dirty.
    fn new(op: Op<N>) -> Self {
        OpWithStatus {
            op,
            #[allow(clippy::disallowed_types)]
            status: StdAtomicU8::new(Status::Dirty as u8),
        }
    }
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
        >::open(&dir, DbOpenMode::ReadWrite)
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
        >::open(&dir, DbOpenMode::ReadWrite)
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
            flush_buffer: Arc::new(DashMap::new()),
            storage: Arc::new(MockStorage::new()),
            flush_workers: FlushWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        let id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::NonEmpty1);
        let node = TestNode::NonEmpty1(Box::default());

        storage
            .flush_buffer
            .insert(id, OpWithStatus::new(Op::Set(node.clone())));

        let result = storage.get(id).unwrap();
        assert_eq!(result, node);
        // TestNode is kept for eventual flush.
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

        let id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::NonEmpty1);
        storage
            .flush_buffer
            .insert(id, OpWithStatus::new(Op::Delete));

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
        let id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::NonEmpty1);
        let node = TestNode::NonEmpty1(Box::default());

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
        let id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::NonEmpty1);
        let node = TestNode::NonEmpty1(Box::default());

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
        let value = &entry.value().op;
        assert!(matches!(value, Op::Set(n) if n == &node));
    }

    #[test]
    fn delete_inserts_delete_op_into_buffer() {
        let id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::NonEmpty1);

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
        let value = &entry.value().op;
        assert!(matches!(value, Op::Delete));
    }

    #[test]
    fn close_calls_close_on_underlying_storage_layer() {
        let mut mock_storage = MockStorage::new();
        mock_storage.expect_close().times(1).returning(|| Ok(()));

        let storage_with_flush_buffer = StorageWithFlushBuffer {
            flush_buffer: Arc::new(DashMap::new()),
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
            flush_buffer: Arc::new(DashMap::new()),
            storage: Arc::new(mock_storage),
            flush_workers: FlushWorkers {
                workers: Vec::new(),
                shutdown: Arc::new(AtomicBool::new(false)),
            },
        };

        storage_with_flush_buffer
            .flush_buffer
            .insert(id, OpWithStatus::new(Op::Set(node.clone())));

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
            flush_buffer: Arc::new(DashMap::new()),
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
            flush_buffer: Arc::new(DashMap::new()),
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
            flush_buffer: Arc::new(DashMap::new()),
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
        let flush_buffer = Arc::new(DashMap::new());
        let storage = Arc::new(MockStorage::new());

        let workers = FlushWorkers::new(&flush_buffer, &storage);
        assert_eq!(workers.workers.len(), FlushWorkers::WORKER_COUNT);
        for worker in &workers.workers {
            assert!(!is_finished(worker)); // Ensure the worker is running
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

            thread::spawn(move || FlushWorkers::task(&flush_buffer, &*storage, &shutdown))
        }];

        let id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::NonEmpty1);
        let node = TestNode::NonEmpty1(Box::default());

        flush_buffer.insert(id, OpWithStatus::new(Op::Set(node.clone())));

        // Allow the worker to process the set operation
        thread::sleep(Duration::from_millis(100));
        assert!(flush_buffer.is_empty());

        flush_buffer.insert(id, OpWithStatus::new(Op::Delete));

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
                    NodeFileStorage::<_, SeekFile>::open(&testdir, DbOpenMode::ReadWrite).unwrap(),
                );
                let node = NonEmpty1TestNode::default();
                let id = storage.reserve(&node);
                storage.set(id, &node).unwrap();

                let flush_buffer = Arc::new(DashMap::new());
                let workers = FlushWorkers::new(&flush_buffer, &storage.clone());
                // This should call delete only once, and panic if two workers delete the same ID
                flush_buffer.insert(id, OpWithStatus::new(Op::Delete));

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
                    NodeFileStorage::<_, SeekFile>::open(&testdir, DbOpenMode::ReadWrite).unwrap(),
                );
                let node = NonEmpty1TestNode::default();
                let id = storage.reserve(&node);
                storage.set(id, &node).unwrap();

                let flush_buffer = Arc::new(DashMap::new());
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
                    NodeFileStorage::<_, SeekFile>::open(&testdir, DbOpenMode::ReadWrite).unwrap(),
                );
                let node = NonEmpty1TestNode::default();
                let id = storage.reserve(&node);
                storage.set(id, &node).unwrap();

                let flush_buffer = Arc::new(DashMap::new());
                let flush_workers = FlushWorkers::new(&flush_buffer, &storage);
                let storage_with_flush_buffer = StorageWithFlushBuffer {
                    flush_buffer,
                    storage,
                    flush_workers,
                };

                let mut node2 = node.clone();
                node2.0 = [1; _];

                // Update the node with `id` to `node2`, which adds a set op to the flush buffer.
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

    #[test]
    fn shuttletest_operations_dont_get_lost_when_inserted_while_old_operation_is_processed() {
        run_shuttle_check(
            || {
                let testdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
                // Use an actual storage as mockall does not use shuttle sync primitives, which we
                // need to ensure context switches between shuttle threads.
                let storage = Arc::new(
                    NodeFileStorage::<_, SeekFile>::open(&testdir, DbOpenMode::ReadWrite).unwrap(),
                );
                let node = NonEmpty1TestNode::default();
                let id = storage.reserve(&node);
                storage.set(id, &node).unwrap();

                let flush_buffer = Arc::new(DashMap::new());
                let flush_workers = FlushWorkers::new(&flush_buffer, &storage);
                let storage_with_flush_buffer = StorageWithFlushBuffer {
                    flush_buffer,
                    storage,
                    flush_workers,
                };

                let mut node2 = node.clone();
                node2.0 = [1; _];

                // Update the node with `id` to `node2`, which adds a set op to the flush buffer.
                // The actual write out happens asynchronously in the flush worker.
                storage_with_flush_buffer.set(id, &node2).unwrap();

                let mut node3 = node.clone();
                node3.0 = [2; _];

                // Update the node with `id` to `node3`, which adds a set op to the flush buffer.
                // The actual write out happens asynchronously in the flush worker.
                thread::yield_now();
                storage_with_flush_buffer.set(id, &node3).unwrap();

                // Either the first set op was still in the flush buffer when the second set op was
                // inserted and then overwritten by it or first set op was processed and removed and
                // the second set op was inserted after that.
                // The second set op is then either still in the flush buffer or the node has been
                // updated to `node3` in the underlying storage layer.
                thread::yield_now();
                assert_eq!(storage_with_flush_buffer.get(id), Ok(node3));

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

                fn open(_path: &Path, db_open_mode: DbOpenMode) -> BTResult<Self, Error>;

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
