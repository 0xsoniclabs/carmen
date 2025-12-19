// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{fmt, fs::File, io::Write, path::Path};

use crate::{
    error::BTResult,
    storage::{self, Checkpointable, DbMode, RootIdProvider, Storage},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        thread,
    },
    types::{ToNodeKind, TreeId},
};

/// An operation that can be performed on the storage.
#[derive(Hash, Eq, PartialEq, Debug, Copy, Clone)]
enum StorageOperation {
    Get,
    Set,
    Reserve,
    Delete,
}

impl std::fmt::Display for StorageOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageOperation::Get => write!(f, "G"),
            StorageOperation::Set => write!(f, "S"),
            StorageOperation::Reserve => write!(f, "R"),
            StorageOperation::Delete => write!(f, "D"),
        }
    }
}

/// A storage operation with associated metadata for logging purposes.
#[derive(Hash, Eq, PartialEq, Debug, Clone)]
struct StorageOperationWithMetadata {
    op: StorageOperation,
    timestamp: std::time::Duration,
    node_kind: String,
    index: u64,
}

impl StorageOperationWithMetadata {
    /// Format the operation with metadata as a CSV line.
    fn as_csv_line(&self) -> String {
        format!(
            "{},{},{},{}\n",
            self.op,
            self.timestamp.as_micros(),
            &self.node_kind,
            self.index
        )
    }
}

/// A storage wrapper that logs the performed operations to a file.
/// It records the type of operation, current timestamp, node kind name and the index of
/// each operation.
/// A background worker periodically flushes the collected operations to disk.
pub struct StorageOperationLogger<S: Storage> {
    storage: S,
    operations: Arc<Mutex<Vec<StorageOperationWithMetadata>>>,
    start_timestamp: std::time::Instant,
    _worker: StorageOperationLoggerWorker,
}

impl<S: Storage> StorageOperationLogger<S> {
    /// Creates a new storage operation logger that wraps the given storage.
    pub fn try_new(storage: S, path: &Path) -> BTResult<Self, storage::Error> {
        let operations = Arc::new(Mutex::new(Vec::new()));
        let _worker = StorageOperationLoggerWorker::new(operations.clone(), path)?;
        Ok(Self {
            storage,
            operations,
            start_timestamp: std::time::Instant::now(),
            _worker,
        })
    }

    /// Log a storage operation with associated metadata.
    fn log<T: TreeId>(&self, op: StorageOperation, id: T)
    where
        <T as ToNodeKind>::Target: fmt::Debug,
    {
        let mut operations = self.operations.lock().unwrap();
        operations.push(StorageOperationWithMetadata {
            op,
            // Safe to unwrap as we set it above if None
            timestamp: std::time::Instant::now().duration_since(self.start_timestamp),
            // NOTE: we could return an error here, however some storage operations (e.g.
            // `Storage::reserve`) cannot fail, therefore we would need to unwrap there.
            // However, corrupted ids should not happen in practice and this should only be used for
            // collecting statistics, so we opt for panicking in this rare case.
            node_kind: format!("{:?}", id.to_node_kind().expect("corrupted id received")),
            index: id.to_index(),
        });
    }
}

impl<S: Storage> Storage for StorageOperationLogger<S>
where
    S::Id: TreeId,
    <S::Id as ToNodeKind>::Target: fmt::Debug,
{
    type Id = S::Id;

    type Item = S::Item;

    fn open(path: &std::path::Path, mode: DbMode) -> BTResult<Self, crate::storage::Error> {
        let storage = S::open(path, mode)?;
        Self::try_new(storage, path)
    }

    fn get(&self, id: Self::Id) -> BTResult<Self::Item, crate::storage::Error> {
        self.log(StorageOperation::Get, id);
        self.storage.get(id)
    }

    fn reserve(&self, item: &Self::Item) -> Self::Id {
        let id = self.storage.reserve(item);
        self.log(StorageOperation::Reserve, id);
        id
    }

    fn set(&self, id: Self::Id, item: &Self::Item) -> BTResult<(), crate::storage::Error> {
        self.log(StorageOperation::Set, id);
        self.storage.set(id, item)
    }

    fn delete(&self, id: Self::Id) -> BTResult<(), crate::storage::Error> {
        self.log(StorageOperation::Delete, id);
        self.storage.delete(id)
    }

    fn close(self) -> BTResult<(), crate::storage::Error> {
        self.storage.close()
    }
}

impl<S> Checkpointable for StorageOperationLogger<S>
where
    S: Storage + Checkpointable,
{
    fn checkpoint(&self) -> BTResult<u64, crate::storage::Error> {
        self.storage.checkpoint()
    }

    fn restore(path: &Path, checkpoint: u64) -> BTResult<(), crate::storage::Error> {
        S::restore(path, checkpoint)
    }
}

impl<S> RootIdProvider for StorageOperationLogger<S>
where
    S: Storage + RootIdProvider,
{
    type Id = <S as RootIdProvider>::Id;

    fn get_root_id(&self, block_number: u64) -> BTResult<Self::Id, crate::storage::Error> {
        self.storage.get_root_id(block_number)
    }

    fn set_root_id(&self, block_number: u64, id: Self::Id) -> BTResult<(), crate::storage::Error> {
        self.storage.set_root_id(block_number, id)
    }

    fn highest_block_number(&self) -> BTResult<Option<u64>, storage::Error> {
        self.storage.highest_block_number()
    }
}

/// Worker that periodically flushes the collected storage operations to disk.
pub struct StorageOperationLoggerWorker {
    stop_signal: Arc<AtomicBool>,
    worker_handle: Option<thread::JoinHandle<()>>,
}

impl StorageOperationLoggerWorker {
    const OPERATION_LOG_FILE: &str = "carmen_storage_op_with_timestamp.csv";

    /// Construct a new worker and runs it in background.
    fn new(
        operations: Arc<Mutex<Vec<StorageOperationWithMetadata>>>,
        path: &Path,
    ) -> BTResult<Self, storage::Error> {
        let file = Mutex::new(File::create(path.join(Self::OPERATION_LOG_FILE))?);
        file.lock()
            .unwrap()
            .write_all("Op,Timestamp,NodeKind,Offset\n".as_bytes())?;

        let stop_signal = Arc::new(AtomicBool::new(false));
        let has_worker_finished = Arc::new(AtomicBool::new(false));
        let worker_handle = {
            let stop_signal = stop_signal.clone();
            let has_worker_finished = has_worker_finished.clone();
            thread::spawn(move || {
                while !stop_signal.load(Ordering::Relaxed) {
                    thread::sleep(std::time::Duration::from_secs(1));
                    Self::flush(&operations, &file).unwrap();
                }
                Self::flush(&operations, &file).unwrap(); // Final flush on stop
                has_worker_finished.store(true, Ordering::Relaxed);
            })
        };

        Ok(Self {
            stop_signal,
            worker_handle: Some(worker_handle),
        })
    }

    /// Writes the collected operations to disk and clears the in-memory buffer.
    fn flush(
        operations: &Mutex<Vec<StorageOperationWithMetadata>>,
        file: &Mutex<std::fs::File>,
    ) -> BTResult<(), storage::Error> {
        let operations = std::mem::take(&mut *operations.lock().unwrap());
        Ok(file.lock().unwrap().write_all(
            operations
                .iter()
                .flat_map(|entry| entry.as_csv_line().as_bytes().to_owned())
                .collect::<Vec<_>>()
                .as_slice(),
        )?)
    }
}

impl Drop for StorageOperationLoggerWorker {
    fn drop(&mut self) {
        // Stops the background worker and writes the total operations executed to disk.
        self.stop_signal.store(true, Ordering::Relaxed);
        let worker_handle = self.worker_handle.take().unwrap();
        worker_handle
            .join()
            .expect("Failed to join storage operation logger worker thread");
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use mockall::{mock, predicate::eq};

    use super::*;
    use crate::{
        statistics::storage::tests::mock::MockStorage,
        sync::is_finished,
        utils::{
            test_dir::{Permissions, TestDir},
            test_nodes::TestNodeId,
        },
    };

    #[test]
    fn storage_operation_logger_try_new_starts_worker_and_creates_file() {
        let test_dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let storage: MockStorage<()> = MockStorage::new();
        let _storage_stats = StorageOperationLogger::try_new(storage, test_dir.path()).unwrap();
        assert!(
            test_dir
                .path()
                .join(StorageOperationLoggerWorker::OPERATION_LOG_FILE)
                .exists()
        );
    }

    #[test]
    fn storage_operation_logger_open_opens_underlying_storage() {
        let test_dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let context = MockStorage::<()>::open_context();
        context
            .expect()
            .times(1)
            .returning(|_, _| Ok(MockStorage::new()));

        let _storage_stats =
            StorageOperationLogger::<MockStorage<()>>::open(test_dir.path(), DbMode::ReadWrite)
                .unwrap();
    }

    #[rstest::rstest]
    #[case(StorageOperation::Get)]
    #[case(StorageOperation::Set)]
    #[case(StorageOperation::Reserve)]
    #[case(StorageOperation::Delete)]
    fn storage_operation_logger_add_logs_operations_correctly(#[case] op: StorageOperation) {
        let test_dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let storage: MockStorage<()> = MockStorage::new();
        let storage_stats = StorageOperationLogger::try_new(storage, test_dir.path()).unwrap();
        storage_stats.log(op, 42);
        assert_eq!(storage_stats.operations.lock().unwrap().len(), 1);
        let entry = storage_stats
            .operations
            .lock()
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        assert_eq!(entry.op, op);
        assert_eq!(
            entry.node_kind,
            format!("{:?}", 42u32.to_node_kind().unwrap())
        );
        assert_eq!(entry.index, 42);
    }

    #[test]
    fn storage_operation_logger_logs_storage_op_correctly() {
        let test_dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let mut storage: MockStorage<()> = MockStorage::new();
        storage.expect_reserve().times(1).returning(|_| 0);
        storage.expect_set().times(2).returning(|_, _| Ok(()));
        storage.expect_get().times(3).returning(|_| Ok(()));
        storage.expect_delete().times(4).returning(|_| Ok(()));

        let storage_stats = StorageOperationLogger::try_new(storage, test_dir.path()).unwrap();

        let _ = storage_stats.reserve(&());
        for i in 1..3 {
            let _ = storage_stats.set(i, &());
        }
        for i in 4..7 {
            let _ = storage_stats.get(i);
        }
        for i in 7..11 {
            let _ = storage_stats.delete(i);
        }

        drop(storage_stats);

        let content = std::fs::read_to_string(
            test_dir
                .path()
                .join(StorageOperationLoggerWorker::OPERATION_LOG_FILE),
        )
        .unwrap();
        let mut op_with_num: HashMap<StorageOperation, Vec<u64>> = HashMap::from_iter([
            (StorageOperation::Get, vec![]),
            (StorageOperation::Set, vec![]),
            (StorageOperation::Reserve, vec![]),
            (StorageOperation::Delete, vec![]),
        ]);
        for line in content.lines().skip(1) {
            let parts: Vec<&str> = line.split(',').collect();
            let op = match parts[0] {
                "G" => StorageOperation::Get,
                "S" => StorageOperation::Set,
                "R" => StorageOperation::Reserve,
                "D" => StorageOperation::Delete,
                _ => panic!("Unknown operation"),
            };
            op_with_num
                .get_mut(&op)
                .unwrap()
                .push(parts[3].parse::<u64>().unwrap());
        }
        assert_eq!(op_with_num.get(&StorageOperation::Reserve).unwrap()[0], 0);
        let set_offsets = op_with_num.get_mut(&StorageOperation::Set).unwrap();
        set_offsets.sort_unstable();
        assert_eq!(set_offsets, &vec![1, 2]);
        let get_offsets = op_with_num.get_mut(&StorageOperation::Get).unwrap();
        get_offsets.sort_unstable();
        assert_eq!(get_offsets, &vec![4, 5, 6]);
        let delete_offsets = op_with_num.get_mut(&StorageOperation::Delete).unwrap();
        delete_offsets.sort_unstable();
        assert_eq!(delete_offsets, &vec![7, 8, 9, 10]);
    }

    #[test]
    fn storage_operation_logger_close_forwards_call_to_underlying_storage() {
        let test_dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let mut storage: MockStorage<()> = MockStorage::new();
        storage.expect_close().times(1).returning(|| Ok(()));

        let storage_stats = StorageOperationLogger::try_new(storage, test_dir.path()).unwrap();
        storage_stats.close().unwrap();
    }

    #[test]
    fn storage_operation_logger_checkpoint_and_restore_forward_calls() {
        let test_dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let mut storage: MockStorage<()> = MockStorage::new();
        storage.expect_checkpoint().times(1).returning(|| Ok(42));
        let restore_ctx = MockStorage::<()>::restore_context();
        restore_ctx
            .expect()
            .with(eq(test_dir.to_path_buf()), eq(42))
            .times(1)
            .returning(|_, _| Ok(()));

        let storage_stats = StorageOperationLogger::try_new(storage, test_dir.path()).unwrap();
        let checkpoint = storage_stats.checkpoint().unwrap();
        assert_eq!(checkpoint, 42);
        StorageOperationLogger::<MockStorage<()>>::restore(test_dir.path(), 42).unwrap();
    }

    #[test]
    fn storage_operation_logger_root_id_provider_forwards_calls() {
        let test_dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let mut storage: MockStorage<()> = MockStorage::new();
        storage
            .expect_get_root_id()
            .with(eq(100))
            .times(1)
            .returning(|_| Ok(1234));
        storage
            .expect_set_root_id()
            .with(eq(100), eq(1234))
            .times(1)
            .returning(|_, _| Ok(()));

        let storage_stats = StorageOperationLogger::try_new(storage, test_dir.path()).unwrap();
        let root_id = storage_stats.get_root_id(100).unwrap();
        assert_eq!(root_id, 1234);
        storage_stats.set_root_id(100, 1234).unwrap();
    }

    #[test]
    fn storage_operation_logger_worker_new_initializes_correctly_and_creates_file() {
        let test_dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let operations = Arc::new(Mutex::new(Vec::new()));
        let worker = StorageOperationLoggerWorker::new(operations, test_dir.path())
            .expect("Failed to create worker");

        assert!(
            test_dir
                .path()
                .join(StorageOperationLoggerWorker::OPERATION_LOG_FILE)
                .exists()
        );
        assert!(!is_finished(worker.worker_handle.as_ref().unwrap()));
    }

    #[test]
    fn storage_operation_logger_worker_flush_writes_and_clears_buffer() {
        let test_dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let mut operations = Vec::new();
        let file_path = test_dir.path().join("test_flush.csv");
        let file = Mutex::new(std::fs::File::create(&file_path).unwrap());
        file.lock()
            .unwrap()
            .write_all("Op,Timestamp,Offset\n".as_bytes())
            .unwrap();

        // Add some entries
        operations.push(StorageOperationWithMetadata {
            op: StorageOperation::Get,
            timestamp: std::time::Duration::from_micros(1000),
            node_kind: "Inner".to_string(),
            index: 1,
        });
        operations.push(StorageOperationWithMetadata {
            op: StorageOperation::Set,
            timestamp: std::time::Duration::from_micros(2000),
            node_kind: "Inner".to_string(),
            index: 2,
        });
        operations.push(StorageOperationWithMetadata {
            op: StorageOperation::Delete,
            timestamp: std::time::Duration::from_micros(3000),
            node_kind: "Leaf".to_string(),
            index: 3,
        });

        let operations = Arc::new(Mutex::new(operations));
        StorageOperationLoggerWorker::flush(&operations, &file).unwrap();

        // Check that the file has the correct content
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("G,1000,Inner,1"));
        assert!(content.contains("S,2000,Inner,2"));
        assert!(content.contains("D,3000,Leaf,3"));

        // Check that the in-memory buffer is cleared
        assert!(operations.lock().unwrap().is_empty());
    }

    #[test]
    #[should_panic(expected = "Failed to join storage operation logger worker thread")]
    fn storage_operation_logger_worker_panics_if_worker_thread_fails_to_join() {
        let worker = StorageOperationLoggerWorker {
            stop_signal: Arc::new(AtomicBool::new(false)),
            worker_handle: Some(thread::spawn(|| {
                panic!("Simulated worker thread failure");
            })),
        };

        // Dropping the storage_stats should panic when trying to join the failed thread
        drop(worker);
    }

    #[allow(clippy::disallowed_types)]
    mod mock {
        use super::*;
        use crate::storage::{DbMode, Error};

        mock! {

                pub Storage<T: Send + Sync + 'static> {}

                impl<T: Send + Sync + 'static> Checkpointable for Storage<T> {
                    fn checkpoint(&self) -> BTResult<u64, Error>;

                    fn restore(path: &Path, checkpoint: u64) -> BTResult<(), Error>;
                }

                impl<T: Send + Sync + 'static> RootIdProvider for Storage<T> {
                    type Id = TestNodeId;

                    fn get_root_id(&self, block_number: u64) -> BTResult<<Self as RootIdProvider>::Id, Error>;

                    fn set_root_id(&self, block_number: u64, id: <Self as RootIdProvider>::Id) -> BTResult<(), Error>;

                    fn highest_block_number(&self) -> BTResult<Option<u64>, Error>;
                }

                impl<T: Send + Sync + 'static> Storage for Storage<T> {
                    type Id = TestNodeId;
                    type Item = T;

                    fn open(path: &std::path::Path, mode: DbMode) -> BTResult<Self, Error>
                    where
                        Self: Sized;

                    fn get(&self, id: <Self as Storage>::Id) -> BTResult<<Self as Storage>::Item, Error>;

                    fn reserve(&self, item: &<Self as Storage>::Item) -> <Self as Storage>::Id;

                    fn set(&self, id: <Self as Storage>::Id, item: &<Self as Storage>::Item) -> BTResult<(), Error>;

                    fn delete(&self, id: <Self as Storage>::Id) -> BTResult<(), Error>;

                    fn close(self) -> BTResult<(), Error>;
                }
        }
    }
}
