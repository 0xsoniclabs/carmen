// TODO: Add timestamp since last access to element
// TODO: Operations per seconds
use std::{io::Write, path::Path};

use dashmap::{DashMap, DashSet};

use crate::{storage::Storage, types::TreeId};

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

pub struct StorageStatistics<I: TreeId, N, S>
where
    S: Storage<Id = I, Item = N>,
{
    storage: S,
    op_executed: DashMap<StorageOperation, u64>,
    op_with_stats: StorageFileStatistics,
}

pub struct StorageFileStatistics {
    op_with_time_stats: DashSet<(StorageOperation, u128, u64)>,
    file: std::sync::Mutex<std::fs::File>,
}

impl StorageFileStatistics {
    const BUFFER_SIZE: usize = 100000;

    fn new(path: &Path) -> Self {
        let mut file =
            std::fs::File::create(path).expect("Failed to create storage operation stats file");
        file.write_all("Op,Timestamp,Offset\n".as_bytes())
            .expect("Failed to write header to statistics file");
        Self {
            op_with_time_stats: DashSet::with_capacity(Self::BUFFER_SIZE),
            file: std::sync::Mutex::new(file),
        }
    }

    fn add(&self, op: StorageOperation, offset: u64) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards")
            .as_micros();
        self.op_with_time_stats.insert((op, timestamp, offset));
        if self.op_with_time_stats.len() >= Self::BUFFER_SIZE {
            self.flush();
        }
    }

    fn flush(&self) {
        let mut file = self.file.lock().unwrap();
        for entry in self.op_with_time_stats.iter() {
            let (op, timestamp, offset) = *entry;
            writeln!(file, "{op},{timestamp},{offset}")
                .expect("Failed to write operation statistics to file");
            self.op_with_time_stats.remove(&entry);
        }
    }
}

impl Drop for StorageFileStatistics {
    fn drop(&mut self) {
        self.flush();
    }
}

impl<I: TreeId, N, S> StorageStatistics<I, N, S>
where
    S: Storage<Id = I, Item = N>,
{
    const OP_STATS_FILE: &str = "storage_op_stats.txt";
    const OP_WITH_TIME_STATS: &str = "storage_op_with_time_stats.csv";

    pub fn new(storage: S) -> Self {
        Self {
            storage,
            op_executed: DashMap::from_iter([
                (StorageOperation::Get, 0),
                (StorageOperation::Set, 0),
                (StorageOperation::Reserve, 0),
                (StorageOperation::Delete, 0),
            ]),
            op_with_stats: StorageFileStatistics::new(Self::OP_WITH_TIME_STATS.as_ref()),
        }
    }

    fn count_op(&self, op: StorageOperation) {
        self.op_executed.entry(op).and_modify(|count| *count += 1);
    }
}

impl<I: TreeId, N, S> Drop for StorageStatistics<I, N, S>
where
    S: Storage<Id = I, Item = N>,
{
    fn drop(&mut self) {
        let mut file = std::fs::File::create(Self::OP_STATS_FILE)
            .expect("Failed to create operation stats file");
        writeln!(
            file,
            "Get operations: {}\nSet operations: {}\nReserve operations: {}\nDelete operations: {}",
            *self.op_executed.get(&StorageOperation::Get).unwrap(),
            *self.op_executed.get(&StorageOperation::Set).unwrap(),
            *self.op_executed.get(&StorageOperation::Reserve).unwrap(),
            *self.op_executed.get(&StorageOperation::Delete).unwrap(),
        )
        .expect("Failed to write operation statistics");
    }
}

impl<I: TreeId + Copy, N, S> Storage for StorageStatistics<I, N, S>
where
    S: Storage<Id = I, Item = N>,
{
    type Id = I;

    type Item = N;

    fn open(path: &std::path::Path) -> Result<Self, crate::storage::Error>
    where
        Self: Sized,
    {
        let storage = S::open(path)?;
        Ok(Self::new(storage))
    }

    fn get(&self, id: Self::Id) -> Result<Self::Item, crate::storage::Error> {
        self.count_op(StorageOperation::Get);
        self.op_with_stats.add(StorageOperation::Get, id.to_index());
        let item = self.storage.get(id)?;
        Ok(item)
    }

    fn reserve(&self, item: &Self::Item) -> Self::Id {
        self.count_op(StorageOperation::Reserve);
        let id = self.storage.reserve(item);
        self.op_with_stats
            .add(StorageOperation::Reserve, id.to_index());
        id
    }

    fn set(&self, id: Self::Id, item: &Self::Item) -> Result<(), crate::storage::Error> {
        self.count_op(StorageOperation::Set);
        self.op_with_stats.add(StorageOperation::Set, id.to_index());
        self.storage.set(id, item)?;
        Ok(())
    }

    fn delete(&self, id: Self::Id) -> Result<(), crate::storage::Error> {
        self.count_op(StorageOperation::Delete);
        self.op_with_stats
            .add(StorageOperation::Delete, id.to_index());
        self.storage.delete(id)
    }
}
