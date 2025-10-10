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
    cmp,
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use quick_cache::{UnitWeighter, sync::Cache};
use zerocopy::{FromBytes, IntoBytes};

use crate::storage::Error;

/// A wrapper around a file storing root IDs, which caches recently used IDs in memory for faster
/// access.
#[derive(Debug)]
pub struct RootIdsFile<ID> {
    /// The underlying file storing the IDs.
    file: Mutex<File>,
    /// The number of IDs stored in the file.
    id_count: AtomicU64,
    /// A cache holding recently used IDs.
    cache: Cache<u64, ID>,
}

impl<ID> RootIdsFile<ID>
where
    ID: Copy + FromBytes + IntoBytes,
{
    const CACHED_IDS: usize = 1_000_000;

    /// Opens the file at `path` and fills the cache with the most recently added ids. If the file
    /// does not exist, it is created.
    pub fn open(path: impl AsRef<Path>, frozen_count: u64) -> Result<Self, Error> {
        let mut file_opts = OpenOptions::new();
        file_opts
            .create(true)
            .truncate(false)
            .read(true)
            .write(true);
        let mut file = file_opts.open(path)?;
        let len = file.metadata()?.len();
        if len < frozen_count * size_of::<ID>() as u64 {
            return Err(Error::DatabaseCorruption);
        }

        let cache = Cache::with_weighter(Self::CACHED_IDS, Self::CACHED_IDS as u64, UnitWeighter);

        let mut raw_ids = vec![ID::new_zeroed(); cmp::min(frozen_count as usize, Self::CACHED_IDS)];
        file.seek(SeekFrom::Start(
            frozen_count.saturating_sub(Self::CACHED_IDS as u64) * size_of::<ID>() as u64,
        ))?;
        file.read_exact(raw_ids.as_mut_bytes())?;

        for (block_number, id) in raw_ids.into_iter().enumerate() {
            cache.insert(block_number as u64, id);
        }

        Ok(Self {
            file: Mutex::new(file),
            id_count: AtomicU64::new(frozen_count),
            cache,
        })
    }

    /// Retrieves the root ID for `block_number`, either from the cache or by reading it from disk.
    pub fn get(&self, block_number: u64) -> Result<ID, Error> {
        if let Some(id) = self.cache.get(&block_number) {
            return Ok(id);
        }

        if block_number >= self.id_count.load(Ordering::Relaxed) {
            return Err(Error::NotFound);
        }

        let offset = block_number * size_of::<ID>() as u64;
        let mut id = ID::new_zeroed();
        let mut file = self.file.lock().unwrap();
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(id.as_mut_bytes())?;
        file.flush()?;

        self.cache.insert(block_number, id);

        Ok(id)
    }

    /// Adds root ID `id` for `block_number` to the cache and writes it out to disk.
    pub fn set(&self, block_number: u64, mut id: ID) -> Result<(), Error> {
        if block_number < self.id_count.load(Ordering::Relaxed) {
            return Err(Error::Frozen);
        }

        let mut file = self.file.lock().unwrap();

        file.seek(SeekFrom::Start(block_number * size_of::<ID>() as u64))
            .unwrap();
        file.write_all(id.as_mut_bytes())?;
        self.cache.insert(block_number, id);

        self.id_count.fetch_max(block_number + 1, Ordering::Relaxed);

        Ok(())
    }

    /// Returns the number of IDs stored in the file.
    pub fn count(&self) -> u64 {
        self.id_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use zerocopy::IntoBytes;

    use super::*;
    use crate::utils::test_dir::{Permissions, TestDir};

    type Id = [u8; 6];
    type RootIdsFile = super::RootIdsFile<Id>;

    #[test]
    fn open_reads_frozen_part_of_file() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let ids = [[1u8; 6], [2; 6]];
        fs::write(path.as_path(), ids.as_bytes()).unwrap();

        let frozen_count = 2;
        let cached_file = RootIdsFile::open(path, frozen_count).unwrap();
        assert_eq!(cached_file.id_count.load(Ordering::Relaxed), frozen_count);
        assert_eq!(cached_file.cache.len(), frozen_count as usize);
        let mut cached_ids = cached_file.cache.iter().map(|(_, v)| v).collect::<Vec<_>>();
        cached_ids.sort();
        assert_eq!(cached_ids, &ids[..frozen_count as usize]);
    }

    #[test]
    fn open_returns_error_for_invalid_file_size() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        fs::write(path.as_path(), [0; 10]).unwrap();

        let frozen_count = 2;
        let result = RootIdsFile::open(path, frozen_count);
        assert!(matches!(result, Err(Error::DatabaseCorruption)));
    }

    #[test]
    fn open_fails_if_file_can_not_be_written() {
        let dir = TestDir::try_new(Permissions::ReadOnly).unwrap();
        let path = dir.join("reuse_list");

        let result = RootIdsFile::open(path, 0);
        assert!(matches!(result, Err(Error::Io(_))));
    }

    #[test]
    fn get_retrieves_id_from_cache() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");
        let path = path.as_path();

        fs::write(path, []).unwrap();

        let cached_file = RootIdsFile {
            file: Mutex::new(File::open(path).unwrap()),
            id_count: AtomicU64::new(0),
            cache: Cache::with_weighter(
                RootIdsFile::CACHED_IDS,
                RootIdsFile::CACHED_IDS as u64,
                UnitWeighter,
            ),
        };

        let block_number = 0;
        let id = [1; 6];

        cached_file.cache.insert(block_number, id);
        cached_file.id_count.store(1, Ordering::Relaxed);

        // The id was not written to the file, so this can only succeed if it is in cache.
        assert_eq!(cached_file.get(block_number).unwrap(), id);
    }

    #[test]
    fn get_retrieves_id_from_file_and_inserts_in_cache() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");
        let path = path.as_path();

        let id = [1; 6];
        fs::write(path, id.as_bytes()).unwrap();

        let cached_file = RootIdsFile {
            file: Mutex::new(File::open(path).unwrap()),
            id_count: AtomicU64::new(1),
            cache: Cache::with_weighter(
                RootIdsFile::CACHED_IDS,
                RootIdsFile::CACHED_IDS as u64,
                UnitWeighter,
            ),
        };

        // The cache is empty, so this must read from the file.
        assert_eq!(cached_file.get(0).unwrap(), id);
        // The id should now be in the cache.
        assert_eq!(cached_file.cache.get(&0), Some(id));
    }

    #[test]
    fn get_fails_if_file_cannot_be_read() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let cached_file = RootIdsFile {
            file: Mutex::new(File::create(path).unwrap()),
            id_count: AtomicU64::new(1),
            cache: Cache::with_weighter(
                RootIdsFile::CACHED_IDS,
                RootIdsFile::CACHED_IDS as u64,
                UnitWeighter,
            ),
        };

        let result = cached_file.get(0); // file is opened read-only
        assert!(matches!(result, Err(Error::Io(_))));
    }

    #[test]
    fn set_inserts_id_in_cache_and_writes_it_it_to_disk() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");
        let path = path.as_path();

        let cached_file = RootIdsFile {
            file: Mutex::new(File::create(path).unwrap()),
            id_count: AtomicU64::new(0),
            cache: Cache::with_weighter(
                RootIdsFile::CACHED_IDS,
                RootIdsFile::CACHED_IDS as u64,
                UnitWeighter,
            ),
        };

        let id = [1; 6];
        let block_number = 0;

        cached_file.set(block_number, id).unwrap();

        assert_eq!(cached_file.id_count.load(Ordering::Relaxed), 1);
        assert_eq!(cached_file.cache.get(&block_number), Some(id));

        assert_eq!(
            fs::metadata(path).unwrap().len(),
            size_of::<[u8; 6]>() as u64
        );
        let mut read_id = [0; 6];
        File::open(path)
            .unwrap()
            .read_exact(read_id.as_mut_bytes())
            .unwrap();
        assert_eq!(read_id, id);
    }

    #[test]
    fn set_fails_if_file_cannot_be_written() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");
        let path = path.as_path();

        fs::write(path, []).unwrap();

        let cached_file = RootIdsFile {
            file: Mutex::new(File::open(path).unwrap()),
            id_count: AtomicU64::new(0),
            cache: Cache::with_weighter(
                RootIdsFile::CACHED_IDS,
                RootIdsFile::CACHED_IDS as u64,
                UnitWeighter,
            ),
        };

        let id = [1; 6];
        let block_number = 0;

        let result = cached_file.set(block_number, id);
        assert!(matches!(result, Err(Error::Io(_))));
    }
}
