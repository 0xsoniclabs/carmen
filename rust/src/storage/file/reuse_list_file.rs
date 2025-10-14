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
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

use zerocopy::IntoBytes;

use crate::storage::Error;

/// A wrapper around a file storing reuse list indices, which caches the indices in memory for
/// faster access.
#[derive(Debug)]
pub struct ReuseListFile {
    /// The underlying file storing the indices.
    file: File,
    /// A cache holding all indices, including the frozen ones.
    cache: Vec<u64>,
    /// The number of frozen indices that can not be reused because they are part of a checkpoint.
    frozen_count: usize,
}

impl ReuseListFile {
    /// Opens the file at `path` and reads `frozen_count` indices from it. If the file does not
    /// exist, it is created. The `frozen_count` parameter specifies how many indices should be
    /// considered "frozen" and not available for reuse.
    pub fn open(path: impl AsRef<Path>, frozen_count: u64) -> Result<Self, Error> {
        let mut file_opts = OpenOptions::new();
        file_opts
            .create(true)
            .truncate(false)
            .read(true)
            .write(true);
        let mut file = file_opts.open(path)?;
        let len = file.metadata()?.len();
        if len < frozen_count * size_of::<u64>() as u64 {
            return Err(Error::DatabaseCorruption);
        }

        let mut reuse_idxs = vec![0u64; frozen_count as usize];
        file.read_exact(reuse_idxs.as_mut_bytes())?;

        Ok(Self {
            file,
            cache: reuse_idxs,
            frozen_count: frozen_count as usize,
        })
    }

    /// Writes the cached indices to the file.
    pub fn write(&self) -> Result<(), Error> {
        let data = self.cache.as_bytes();
        let mut file = &self.file;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(data)?;
        file.set_len(data.len() as u64)?;
        file.sync_all()?;

        Ok(())
    }

    /// Freezes all currently cached indices.
    pub fn freeze_all(&mut self) {
        self.frozen_count = self.cache.len();
    }

    /// Returns the number of frozen indices.
    #[cfg(test)]
    pub fn frozen_count(&self) -> usize {
        self.frozen_count
    }

    /// Sets the number of frozen indices.
    pub fn set_frozen_count(&mut self, frozen_count: usize) {
        self.frozen_count = frozen_count;
    }

    /// Pops an index from the cache, if there are non-frozen indices available.
    pub fn pop(&mut self) -> Option<u64> {
        if self.cache.len() <= self.frozen_count {
            return None;
        }
        self.cache.pop()
    }

    /// Pushes an index to the cache.
    pub fn push(&mut self, idx: u64) {
        self.cache.push(idx);
    }

    /// Returns a slice of the cached indices, including the frozen part.
    pub fn as_slice(&self) -> &[u64] {
        &self.cache
    }

    /// Returns the number of cached indices.
    pub fn count(&self) -> usize {
        self.cache.len()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use zerocopy::IntoBytes;

    use super::*;
    use crate::utils::test_dir::{Permissions, TestDir};

    #[test]
    fn open_reads_frozen_part_of_file() {
        use super::ReuseListFile;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let indices = [1u64, 2, 3];
        fs::write(path.as_path(), indices.as_bytes()).unwrap();

        let frozen_count = 2;
        let cached_file = ReuseListFile::open(path, frozen_count).unwrap();
        assert_eq!(cached_file.frozen_count, frozen_count as usize);
        assert_eq!(cached_file.cache, indices[..frozen_count as usize]);
    }

    #[test]
    fn open_returns_error_for_invalid_file_size() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        fs::write(path.as_path(), [0; 10]).unwrap();

        let frozen_count = 2;
        let result = ReuseListFile::open(path, frozen_count);
        assert!(matches!(result, Err(Error::DatabaseCorruption)));
    }

    #[test]
    fn open_fails_if_file_can_not_be_written() {
        let dir = TestDir::try_new(Permissions::ReadOnly).unwrap();
        let path = dir.join("reuse_list");

        let result = ReuseListFile::open(path, 0);
        println!("{result:?}");
        assert!(matches!(result, Err(Error::Io(_))));
    }

    #[test]
    fn write_writes_cache_to_file() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let cached_file = ReuseListFile {
            file: File::create(path.as_path()).unwrap(),
            cache: vec![1, 2, 3, 4, 5],
            frozen_count: 2,
        };

        cached_file.write().unwrap();

        let read_indices = fs::read(path.as_path()).unwrap();
        let read_indices: Vec<u64> = read_indices
            .chunks_exact(size_of::<u64>())
            .map(|chunk| {
                chunk.try_into().map(u64::from_le_bytes).unwrap() // slices are guaranteed to be of size 8
            })
            .collect();
        assert_eq!(read_indices, cached_file.cache);
    }

    #[test]
    fn write_fails_if_file_cannot_be_written() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        File::create(path.as_path()).unwrap();

        let cached_file = ReuseListFile {
            file: File::open(path.as_path()).unwrap(),
            cache: vec![1, 2, 3, 4, 5],
            frozen_count: 2,
        };

        let result = cached_file.write(); // file is opened read-only
        assert!(matches!(result, Err(Error::Io(_))));
    }

    #[test]
    fn freeze_all_sets_frozen_count_to_current_length() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.path().join("reuse_list");

        let mut cached_file = ReuseListFile {
            file: File::create(path).unwrap(),
            cache: vec![1, 2, 3, 4],
            frozen_count: 2,
        };

        assert_eq!(cached_file.frozen_count, 2);
        cached_file.freeze_all();
        assert_eq!(cached_file.frozen_count, 4);
    }

    #[test]
    fn frozen_count_returns_number_of_frozen_elements() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.path().join("reuse_list");

        let cached_file = ReuseListFile {
            file: File::create(path).unwrap(),
            cache: vec![1, 2, 3, 4],
            frozen_count: 2,
        };

        assert_eq!(cached_file.frozen_count(), 2);
    }

    #[test]
    fn set_frozen_count_updates_value() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let mut cached_file = ReuseListFile {
            file: File::create(path).unwrap(),
            cache: vec![1, 2, 3, 4],
            frozen_count: 2,
        };

        assert_eq!(cached_file.frozen_count, 2);
        cached_file.set_frozen_count(3);
        assert_eq!(cached_file.frozen_count, 3);
        cached_file.set_frozen_count(0);
        assert_eq!(cached_file.frozen_count, 0);
    }

    #[test]
    fn pop_returns_element_if_non_frozen_elements_exist() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let mut cached_file = ReuseListFile {
            file: File::create(path).unwrap(),
            cache: vec![1, 2, 3, 4],
            frozen_count: 2,
        };

        assert_eq!(cached_file.pop(), Some(4));
        assert_eq!(cached_file.cache, vec![1, 2, 3]);

        assert_eq!(cached_file.pop(), Some(3));
        assert_eq!(cached_file.cache, vec![1, 2]);

        assert_eq!(cached_file.pop(), None);
        assert_eq!(cached_file.cache, vec![1, 2]);
    }

    #[test]
    fn push_adds_element_to_cache() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let mut cached_file = ReuseListFile {
            file: File::create(path).unwrap(),
            cache: vec![1, 2],
            frozen_count: 2,
        };

        cached_file.push(3);
        assert_eq!(cached_file.cache, vec![1, 2, 3]);
        cached_file.push(4);
        assert_eq!(cached_file.cache, vec![1, 2, 3, 4]);
    }

    #[test]
    fn as_slice_returns_slice_of_all_elements() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let cached_file = ReuseListFile {
            file: File::create(path).unwrap(),
            cache: vec![1, 2, 3, 4],
            frozen_count: 2,
        };

        assert_eq!(cached_file.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn count_returns_number_of_cached_elements() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let cached_file = ReuseListFile {
            file: File::create(path).unwrap(),
            cache: vec![1, 2, 3, 4],
            frozen_count: 2,
        };

        assert_eq!(cached_file.count(), 4);
    }
}
