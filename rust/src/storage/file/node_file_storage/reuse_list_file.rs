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
    collections::HashSet,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

use zerocopy::IntoBytes;

use crate::{
    error::BTResult,
    storage::{DbMode, Error},
};

/// A wrapper around a file storing reuse list indices, which caches the indices in memory for
/// faster access.
#[derive(Debug)]
pub struct ReuseListFile {
    /// The underlying file storing the indices.
    file: File,
    /// A cache holding all frozen indices.
    // Frozen elements will accumulate over time, so using a HashSet speeds up lookups.
    // And because they are frozen, they don't need to be written back to the file.
    frozen_indices: HashSet<u64>,
    /// A cache for temporarily frozen indices that may be unfrozen or permanently frozen later.
    temp_frozen_indices: Vec<u64>,
    /// A cache holding all non-frozen indices.
    // There will typically be only a few non-frozen indices, so using a Vec simplifies write-back
    // without a big overhead for lookups.
    reusable_indices: Vec<u64>,
}

impl ReuseListFile {
    /// Opens the file at `path` and reads `count` indices from it. If the file does not exist, it
    /// is created. The `frozen_count` parameter specifies how many indices should be considered
    /// "frozen" and not available for reuse.
    pub fn open(
        path: impl AsRef<Path>,
        count: u64,
        frozen_count: u64,
        db_mode: DbMode,
    ) -> BTResult<Self, Error> {
        if frozen_count > count {
            return Err(Error::DatabaseCorruption(
                format!("frozen reuse index count {frozen_count} greater than total reuse index count {count}"),
            )
            .into());
        }
        let mut file = db_mode.to_open_options().open(path)?;
        let len = file.metadata()?.len();
        if len < count * size_of::<u64>() as u64 {
            return Err(Error::DatabaseCorruption(format!(
                "reuse index file too short: got {len}B, expected at least {}B",
                count * size_of::<u64>() as u64
            ))
            .into());
        }

        let mut all_indices = vec![0u64; count as usize];
        file.read_exact(all_indices.as_mut_bytes())?;
        let reusable_indices = all_indices.split_off(frozen_count as usize);
        let frozen_indices = all_indices.into_iter().collect();

        Ok(Self {
            file,
            frozen_indices,
            temp_frozen_indices: Vec::new(),
            reusable_indices,
        })
    }

    /// Writes all unfrozen indices to disk.
    pub fn write_to_disk(&mut self) -> BTResult<(), Error> {
        let data = self.reusable_indices.as_bytes();
        let old_size = (self.frozen_indices.len() * size_of::<u64>()) as u64;

        self.file.seek(SeekFrom::Start(old_size))?;
        self.file.write_all(data)?;
        self.file.sync_all()?;

        Ok(())
    }

    /// Temporarily freezes all unfrozen indices, in a way that they can be unfrozen again
    /// using [`Self::unfreeze_temp`] or be permanently frozen using [`Self::freeze_permanently`].
    pub fn freeze_temporarily(&mut self) {
        self.temp_frozen_indices.append(&mut self.reusable_indices);
    }

    /// Permanently freezes all temporarily frozen indices.
    pub fn freeze_permanently(&mut self) {
        self.frozen_indices
            .extend(self.temp_frozen_indices.drain(..));
    }

    /// Unfreezes all temporarily frozen indices, making them available for reuse again.
    pub fn unfreeze_temp(&mut self) {
        self.reusable_indices.append(&mut self.temp_frozen_indices);
    }

    /// Returns the number of all indices.
    pub fn count(&self) -> usize {
        self.frozen_indices.len() + self.temp_frozen_indices.len() + self.reusable_indices.len()
    }

    /// Returns the number of frozen indices.
    pub fn frozen_count(&self) -> usize {
        self.frozen_indices.len() + self.temp_frozen_indices.len()
    }

    /// Pops a non-frozen index, if any are available.
    pub fn pop(&mut self) -> Option<u64> {
        self.reusable_indices.pop()
    }

    /// Pushes an index to the cache.
    pub fn push(&mut self, idx: u64) {
        self.reusable_indices.push(idx);
    }

    /// Returns an iterator over all reusable indices.
    pub fn reusable_indices(&self) -> impl Iterator<Item = &u64> {
        self.reusable_indices.iter()
    }

    /// Returns an iterator over all frozen indices.
    pub fn frozen_indices(&self) -> impl Iterator<Item = &u64> {
        self.frozen_indices.iter().chain(&self.temp_frozen_indices)
    }

    pub fn contains(&self, idx: u64) -> bool {
        self.frozen_indices.contains(&idx)
            || self.temp_frozen_indices.contains(&idx)
            || self.reusable_indices.contains(&idx)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use zerocopy::IntoBytes;

    use super::*;
    use crate::{
        error::BTError,
        storage::tests::all_db_modes,
        utils::test_dir::{Permissions, TestDir},
    };

    #[rstest_reuse::apply(all_db_modes)]
    fn open_reads_count_number_of_indices_from_file(#[case] db_mode: DbMode) {
        use super::ReuseListFile;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let indices = [0u64, 1, 2, 3];
        fs::write(&path, indices.as_bytes()).unwrap();

        let frozen_count = 2;
        let count = 3;
        let reuse_list_file = ReuseListFile::open(path, count, frozen_count, db_mode).unwrap();
        assert_eq!(reuse_list_file.frozen_indices, [0, 1].into_iter().collect());
        assert!(reuse_list_file.temp_frozen_indices.is_empty());
        assert_eq!(reuse_list_file.reusable_indices, [2]);
    }

    #[rstest_reuse::apply(all_db_modes)]
    fn open_returns_error_when_frozen_count_larger_than_count(#[case] db_mode: DbMode) {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let count = 1;
        let frozen_count = 2;
        let result = ReuseListFile::open(path, count, frozen_count, db_mode);
        assert_eq!(
            result.unwrap_err().into_inner(),
            Error::DatabaseCorruption(
                "frozen reuse index count 2 greater than total reuse index count 1".to_owned()
            )
        );
    }

    #[rstest_reuse::apply(all_db_modes)]
    fn open_returns_error_for_invalid_file_size(#[case] db_mode: DbMode) {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        // smaller than `count * size_of::<u64>()`
        fs::write(&path, [0; 10]).unwrap();

        let count = 2;
        let frozen_count = 2;
        let result = ReuseListFile::open(path, count, frozen_count, db_mode);
        assert_eq!(
            result.unwrap_err().into_inner(),
            Error::DatabaseCorruption(
                "reuse index file shorter than expected: got 10B, expected at least 16B".to_owned()
            )
        );
    }

    #[test]
    fn open_fails_if_file_can_not_be_created() {
        let dir = TestDir::try_new(Permissions::ReadOnly).unwrap();
        let path = dir.join("reuse_list");

        let result = ReuseListFile::open(path, 0, 0, DbMode::ReadWrite);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::Io(_))
        ));
    }

    #[test]
    fn open_fails_if_file_does_not_exist_in_read_only_mode() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let result = ReuseListFile::open(path, 0, 0, DbMode::ReadOnly);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::Io(_))
        ));
    }

    #[test]
    fn write_to_disk_appends_unfrozen_indices_to_disk() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let mut reuse_list_file = ReuseListFile {
            file: File::create(&path).unwrap(),
            frozen_indices: vec![0, 1].into_iter().collect(),
            temp_frozen_indices: vec![],
            reusable_indices: vec![2, 3],
        };
        reuse_list_file
            .file
            .write_all([0u64, 1].as_bytes())
            .unwrap();

        reuse_list_file.write_to_disk().unwrap();

        let read_indices = fs::read(&path).unwrap();
        assert_eq!(read_indices, [0u64, 1, 2, 3].as_bytes());
    }

    #[test]
    fn write_to_disk_fails_if_file_cannot_be_written() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        File::create(&path).unwrap();

        let mut reuse_list_file = ReuseListFile {
            file: File::open(&path).unwrap(),
            frozen_indices: vec![0, 1].into_iter().collect(),
            temp_frozen_indices: vec![2, 3],
            reusable_indices: vec![4, 5],
        };

        let result = reuse_list_file.write_to_disk(); // file is opened read-only
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::Io(_))
        ));
    }

    #[test]
    fn freeze_temporarily_moves_unfrozen_indices_into_temporarily_frozen_indices() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let mut reuse_list_file = ReuseListFile {
            file: File::create(&path).unwrap(),
            frozen_indices: vec![0, 1].into_iter().collect(),
            temp_frozen_indices: vec![2, 3],
            reusable_indices: vec![4, 5],
        };

        reuse_list_file.freeze_temporarily();

        assert_eq!(
            reuse_list_file.frozen_indices,
            [0u64, 1].into_iter().collect()
        );
        assert_eq!(reuse_list_file.temp_frozen_indices, [2, 3, 4, 5]);
        assert!(reuse_list_file.reusable_indices.is_empty());
    }

    #[test]
    fn freeze_permanently_moves_temporarily_frozen_indices_into_permanently_frozen_indices() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.path().join("reuse_list");

        let mut reuse_list_file = ReuseListFile {
            file: File::create(path).unwrap(),
            frozen_indices: vec![0, 1].into_iter().collect(),
            temp_frozen_indices: vec![2, 3],
            reusable_indices: vec![4, 5],
        };

        reuse_list_file.freeze_permanently();

        assert_eq!(
            reuse_list_file.frozen_indices,
            [0, 1, 2, 3].into_iter().collect()
        );
        assert!(reuse_list_file.temp_frozen_indices.is_empty());
    }

    #[test]
    fn unfreeze_temp_moves_temporarily_frozen_indices_into_reusable_indices() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let mut reuse_list_file = ReuseListFile {
            file: File::create(path).unwrap(),
            frozen_indices: vec![0, 1].into_iter().collect(),
            temp_frozen_indices: vec![2, 3],
            reusable_indices: vec![4, 5],
        };

        reuse_list_file.unfreeze_temp();

        assert_eq!(reuse_list_file.frozen_indices, [0, 1].into_iter().collect());
        assert!(reuse_list_file.temp_frozen_indices.is_empty());
        assert_eq!(reuse_list_file.reusable_indices, [4, 5, 2, 3]);
    }

    #[test]
    fn count_returns_number_of_all_indices() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.path().join("reuse_list");

        let reuse_list_file = ReuseListFile {
            file: File::create(path).unwrap(),
            frozen_indices: vec![0].into_iter().collect(),
            temp_frozen_indices: vec![1, 2],
            reusable_indices: vec![3, 4, 5],
        };

        assert_eq!(reuse_list_file.count(), 1 + 2 + 3);
    }
    #[test]
    fn frozen_count_returns_number_of_frozen_indices() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.path().join("reuse_list");

        let reuse_list_file = ReuseListFile {
            file: File::create(path).unwrap(),
            frozen_indices: vec![0].into_iter().collect(),
            temp_frozen_indices: vec![1, 2],
            reusable_indices: vec![3, 4, 5],
        };

        assert_eq!(reuse_list_file.frozen_count(), 1 + 2);
    }

    #[test]
    fn pop_returns_element_if_non_frozen_elements_exist() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let mut reuse_list_file = ReuseListFile {
            file: File::create(path).unwrap(),
            frozen_indices: vec![0, 1].into_iter().collect(),
            temp_frozen_indices: vec![2, 3],
            reusable_indices: vec![4, 5],
        };

        assert_eq!(reuse_list_file.pop(), Some(5));
        assert_eq!(reuse_list_file.reusable_indices, [4]);

        assert_eq!(reuse_list_file.pop(), Some(4));
        assert!(reuse_list_file.reusable_indices.is_empty());

        assert_eq!(reuse_list_file.pop(), None);
        assert!(reuse_list_file.reusable_indices.is_empty());
    }

    #[test]
    fn push_adds_element_to_reusable_indices() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let mut reuse_list_file = ReuseListFile {
            file: File::create(path).unwrap(),
            frozen_indices: HashSet::new(),
            temp_frozen_indices: Vec::new(),
            reusable_indices: vec![1, 2],
        };

        reuse_list_file.push(3);
        assert_eq!(reuse_list_file.reusable_indices, vec![1, 2, 3]);
        reuse_list_file.push(4);
        assert_eq!(reuse_list_file.reusable_indices, vec![1, 2, 3, 4]);
    }

    #[test]
    fn reusable_indices_returns_iterator_over_reusable_indices() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let reuse_list_file = ReuseListFile {
            file: File::create(path).unwrap(),
            frozen_indices: vec![0, 1].into_iter().collect(),
            temp_frozen_indices: vec![2, 3],
            reusable_indices: vec![4, 5],
        };

        let mut all_indices: Vec<_> = reuse_list_file.reusable_indices().copied().collect();
        all_indices.sort();
        assert_eq!(all_indices, [4, 5]);
    }

    #[test]
    fn frozen_indices_returns_iterator_over_frozen_indices() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let reuse_list_file = ReuseListFile {
            file: File::create(path).unwrap(),
            frozen_indices: vec![0, 1].into_iter().collect(),
            temp_frozen_indices: vec![2, 3],
            reusable_indices: vec![4, 5],
        };

        let mut all_indices: Vec<_> = reuse_list_file.frozen_indices().copied().collect();
        all_indices.sort();
        assert_eq!(all_indices, [0, 1, 2, 3,]);
    }

    #[test]
    fn contains_checks_if_index_exists_in_any_cache() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("reuse_list");

        let reuse_list_file = ReuseListFile {
            file: File::create(path).unwrap(),
            frozen_indices: vec![0, 1].into_iter().collect(),
            temp_frozen_indices: vec![2, 3],
            reusable_indices: vec![4, 5],
        };

        for idx in 0..6 {
            assert!(reuse_list_file.contains(idx));
        }
        for idx in 6..10 {
            assert!(!reuse_list_file.contains(idx));
        }
    }
}
