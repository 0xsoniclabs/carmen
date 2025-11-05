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
};

use quick_cache::sync::Cache;
use zerocopy::{FromBytes, IntoBytes};

use crate::{
    error::BTResult,
    storage::Error,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

/// A wrapper around a file storing root IDs, the ID of a tree's root node at a given block number.
/// Recently used IDs are cached in memory for faster access.
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
    const CACHE_SIZE: usize = 1_000_000;

    /// Opens the file at `path` and fills the cache with the ids corresponding to the highest block
    /// numbers. If the file does not exist, it is created.
    pub fn open(path: impl AsRef<Path>, frozen_count: u64) -> BTResult<Self, Error> {
        let mut file_opts = OpenOptions::new();
        file_opts
            .create(true)
            .truncate(false)
            .read(true)
            .write(true);
        let mut file = file_opts.open(path)?;
        let len = file.metadata()?.len();
        if len < frozen_count * size_of::<ID>() as u64 {
            return Err(Error::DatabaseCorruption.into());
        }

        let cache = Cache::new(Self::CACHE_SIZE);

        let mut raw_ids = vec![ID::new_zeroed(); cmp::min(frozen_count as usize, Self::CACHE_SIZE)];
        file.seek(SeekFrom::Start(
            frozen_count.saturating_sub(Self::CACHE_SIZE as u64) * size_of::<ID>() as u64,
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
    pub fn get(&self, block_number: u64) -> BTResult<ID, Error> {
        if let Some(id) = self.cache.get(&block_number) {
            return Ok(id);
        }

        if block_number >= self.id_count.load(Ordering::Relaxed) {
            return Err(Error::NotFound.into());
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
    pub fn set(&self, block_number: u64, mut id: ID) -> BTResult<(), Error> {
        if block_number < self.id_count.load(Ordering::Relaxed) {
            return Err(Error::Frozen.into());
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
    use crate::{
        error::BTError,
        utils::test_dir::{Permissions, TestDir},
    };

    type Id = [u8; 6];
    type RootIdsFile = super::RootIdsFile<Id>;

    #[test]
    fn open_reads_frozen_part_of_file() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("root_ids");

        let ids = [[1u8; 6], [2; 6]];
        fs::write(path.as_path(), ids.as_bytes()).unwrap();

        let frozen_count = 2;
        let root_ids_file = RootIdsFile::open(path, frozen_count).unwrap();
        assert_eq!(root_ids_file.id_count.load(Ordering::Relaxed), frozen_count);
        assert_eq!(root_ids_file.cache.len(), frozen_count as usize);
        let mut cached_ids = root_ids_file
            .cache
            .iter()
            .map(|(_, v)| v)
            .collect::<Vec<_>>();
        cached_ids.sort();
        assert_eq!(cached_ids, &ids[..frozen_count as usize]);
    }

    #[test]
    fn open_returns_error_for_invalid_file_size() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("root_ids");

        fs::write(path.as_path(), [0; 10]).unwrap();

        let frozen_count = 2;
        let result = RootIdsFile::open(path, frozen_count);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::DatabaseCorruption)
        ));
    }

    #[test]
    fn open_fails_if_file_can_not_be_created() {
        let dir = TestDir::try_new(Permissions::ReadOnly).unwrap();
        let path = dir.join("root_ids");

        let result = RootIdsFile::open(path, 0);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::Io(_))
        ));
    }

    #[test]
    fn get_retrieves_id_from_cache() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("root_ids");
        let path = path.as_path();

        fs::write(path, []).unwrap();

        let root_ids_file = RootIdsFile {
            file: Mutex::new(File::open(path).unwrap()),
            id_count: AtomicU64::new(0),
            cache: Cache::new(RootIdsFile::CACHE_SIZE),
        };

        let block_number = 0;
        let id = [1; 6];

        root_ids_file.cache.insert(block_number, id);
        root_ids_file.id_count.store(1, Ordering::Relaxed);

        // The id was not written to the file, so this can only succeed if it is in cache.
        assert_eq!(root_ids_file.get(block_number).unwrap(), id);
    }

    #[test]
    fn get_retrieves_id_from_file_and_inserts_in_cache() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("root_ids");
        let path = path.as_path();

        let id = [1; 6];
        fs::write(path, id.as_bytes()).unwrap();

        let root_ids_file = RootIdsFile {
            file: Mutex::new(File::open(path).unwrap()),
            id_count: AtomicU64::new(1),
            cache: Cache::new(RootIdsFile::CACHE_SIZE),
        };

        // The cache is empty, so this must read from the file.
        assert_eq!(root_ids_file.get(0).unwrap(), id);
        // The id should now be in the cache.
        assert_eq!(root_ids_file.cache.get(&0), Some(id));
    }

    #[test]
    fn get_fails_if_file_cannot_be_read() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("root_ids");

        let root_ids_file = RootIdsFile {
            file: Mutex::new(File::create(path).unwrap()),
            id_count: AtomicU64::new(1),
            cache: Cache::new(RootIdsFile::CACHE_SIZE),
        };

        let result = root_ids_file.get(0); // file is opened read-only
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::Io(_))
        ));
    }

    #[test]
    fn set_inserts_id_in_cache_and_writes_it_to_disk() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("root_ids");
        let path = path.as_path();

        let root_ids_file = RootIdsFile {
            file: Mutex::new(File::create(path).unwrap()),
            id_count: AtomicU64::new(0),
            cache: Cache::new(RootIdsFile::CACHE_SIZE),
        };

        let id = [1; 6];
        let block_number = 0;

        root_ids_file.set(block_number, id).unwrap();

        assert_eq!(root_ids_file.id_count.load(Ordering::Relaxed), 1);
        assert_eq!(root_ids_file.cache.get(&block_number), Some(id));

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
        let path = dir.join("root_ids");
        let path = path.as_path();

        fs::write(path, []).unwrap();

        let root_ids_file = RootIdsFile {
            file: Mutex::new(File::open(path).unwrap()),
            id_count: AtomicU64::new(0),
            cache: Cache::new(RootIdsFile::CACHE_SIZE),
        };

        let id = [1; 6];
        let block_number = 0;

        let result = root_ids_file.set(block_number, id);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::Io(_))
        ));
    }
}
