use std::{collections::HashMap, io::Write, path::Path, sync::Mutex};

use dashmap::DashMap;

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
    file: std::sync::Mutex<std::fs::File>,
}

impl StorageFileStatistics {
    fn new(path: &Path) -> Self {
        let mut file =
            std::fs::File::create(path).expect("Failed to create storage operation stats file");
        file.write_all("Op,Timestamp,Offset\n".as_bytes())
            .expect("Failed to write header to statistics file");
        Self {
            file: std::sync::Mutex::new(file),
        }
    }

    fn write_with_timestamp(&self, op: StorageOperation, offset: u64) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards")
            .as_micros();
        let mut file = self.file.lock().unwrap();
        writeln!(file, "{op},{timestamp},{offset}")
            .expect("Failed to write operation with timestamp");
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
        self.op_with_stats
            .write_with_timestamp(StorageOperation::Get, id.to_index());
        let item = self.storage.get(id)?;
        Ok(item)
    }

    fn reserve(&self, item: &Self::Item) -> Self::Id {
        self.count_op(StorageOperation::Reserve);
        self.op_with_stats
            .write_with_timestamp(StorageOperation::Reserve, 0);
        self.storage.reserve(item)
    }

    fn set(&self, id: Self::Id, item: &Self::Item) -> Result<(), crate::storage::Error> {
        self.count_op(StorageOperation::Set);
        self.op_with_stats
            .write_with_timestamp(StorageOperation::Set, id.to_index());
        self.storage.set(id, item)?;
        Ok(())
    }

    fn delete(&self, id: Self::Id) -> Result<(), crate::storage::Error> {
        self.count_op(StorageOperation::Delete);
        self.op_with_stats
            .write_with_timestamp(StorageOperation::Delete, id.to_index());
        self.storage.delete(id)
    }
}
