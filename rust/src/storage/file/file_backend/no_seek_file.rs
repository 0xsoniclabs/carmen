// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{array, fs::OpenOptions, os::unix::fs::FileExt, path::Path};

use crate::{error::BTResult, storage::file::FileBackend, sync::RwLock};

/// A wrapper around [`std::fs::File`] that implements [`FileBackend`] using the Unix-specific file
/// operations `pread` and `pwrite` which do not modify the file offset. This avoids the syscall for
/// seeking and allows for concurrent access without needing to manage a cursor.
pub struct NoSeekFile {
    file: std::fs::File,
    locks: Box<[RwLock<()>; Self::NUM_LOCKS]>,
    chunk_size: usize,
}

impl NoSeekFile {
    const NUM_LOCKS: usize = 1024;

    /// Returns a reference to the `RwLock` corresponding to the given buffer and offset.
    /// This function also checks that all preconditions are met (buffer length matches chunk size,
    /// offset is aligned to chunk size) and returns an error if not.
    fn lock(&self, buf: &[u8], offset: u64) -> BTResult<&RwLock<()>, std::io::Error> {
        if buf.len() != self.chunk_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "buffer length {} does not match chunk size {}",
                    buf.len(),
                    self.chunk_size
                ),
            )
            .into());
        }
        if !offset.is_multiple_of(self.chunk_size as u64) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "offset {} is not aligned to chunk size {}",
                    offset, self.chunk_size
                ),
            )
            .into());
        }
        Ok(&self.locks[((offset / self.chunk_size as u64) as usize) % Self::NUM_LOCKS])
    }
}

impl FileBackend for NoSeekFile {
    fn open(
        path: &Path,
        options: OpenOptions,
        chunk_size: usize,
    ) -> BTResult<Self, std::io::Error> {
        let file = options.open(path)?;
        file.try_lock()?;
        Ok(Self {
            file,
            locks: Box::new(array::from_fn(|_| RwLock::new(()))),
            chunk_size,
        })
    }

    fn write_all_at(&self, buf: &[u8], offset: u64) -> BTResult<(), std::io::Error> {
        let _lock = self.lock(buf, offset)?.write().unwrap();
        self.file.write_all_at(buf, offset).map_err(Into::into)
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> BTResult<(), std::io::Error> {
        let _lock = self.lock(buf, offset)?.read().unwrap();
        self.file.read_exact_at(buf, offset).map_err(Into::into)
    }

    fn flush(&self) -> BTResult<(), std::io::Error> {
        self.file.sync_all().map_err(Into::into)
    }

    fn len(&self) -> BTResult<u64, std::io::Error> {
        self.file.metadata().map(|m| m.len()).map_err(Into::into)
    }
}

// Note: The tests for `NoSeekFile as FileBackend` are in `file_backend.rs`.

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::{
        sync,
        utils::test_dir::{Permissions, TestDir},
    };

    const CHUNK_SIZE: u64 = 10;

    #[rstest::rstest]
    #[case(0, 0)]
    #[case(CHUNK_SIZE, 1)]
    #[case(CHUNK_SIZE * 2, 2)]
    #[case(CHUNK_SIZE * NoSeekFile::NUM_LOCKS as u64, 0)]
    #[case(CHUNK_SIZE * (NoSeekFile::NUM_LOCKS as u64 +1), 1)]
    fn lock_returns_rwlock_for_offset(#[case] offset: u64, #[case] expected_index: usize) {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let file = NoSeekFile {
            file: std::fs::File::open(&dir).unwrap(),
            locks: Box::new(array::from_fn(|_| RwLock::new(()))),
            chunk_size: CHUNK_SIZE as usize,
        };

        assert!(file.locks[expected_index].try_write().is_ok());
        let lock1 = file.lock(&[0; CHUNK_SIZE as usize], offset).unwrap();
        let _g = lock1.write().unwrap();
        assert!(file.locks[expected_index].try_write().is_err());
    }

    #[test]
    fn lock_returns_error_for_unaligned_offset() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let file = NoSeekFile {
            file: std::fs::File::open(&dir).unwrap(),
            locks: Box::new(array::from_fn(|_| RwLock::new(()))),
            chunk_size: CHUNK_SIZE as usize,
        };

        let err = file.lock(&[0; CHUNK_SIZE as usize], 1).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn lock_returns_error_for_invalid_buffer_length() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("some-file");

        let file = NoSeekFile {
            file: std::fs::File::create(&path).unwrap(),
            locks: Box::new(array::from_fn(|_| RwLock::new(()))),
            chunk_size: CHUNK_SIZE as usize,
        };

        let err = file.lock(&[0; (CHUNK_SIZE as usize) - 1], 0).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn read_and_write_acquire_locks_for_offset() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("some-file");

        let mut buf = [0u8; CHUNK_SIZE as usize];
        std::fs::write(&path, buf).unwrap();

        let mut opts = OpenOptions::new();
        opts.read(true).write(true);
        let file = NoSeekFile {
            file: opts.open(&path).unwrap(),
            locks: Box::new(array::from_fn(|_| RwLock::new(()))),
            chunk_size: CHUNK_SIZE as usize,
        };
        let file = &file;

        // holding a read lock allows reads from the corresponding offset
        sync::thread::scope(|s| {
            let _read_guard = file.locks[0].try_read().unwrap();
            let t = s.spawn(|| file.read_exact_at(&mut buf, 0).unwrap());
            sync::thread::sleep(Duration::from_millis(100));
            assert!(t.is_finished());
        });

        // holding a write lock does not allow reads from the corresponding offset
        sync::thread::scope(|s| {
            let write_guard = file.locks[0].try_write().unwrap();
            let t = s.spawn(|| file.read_exact_at(&mut buf, 0).unwrap());
            sync::thread::sleep(Duration::from_millis(100));
            assert!(!t.is_finished());
            drop(write_guard);
            sync::thread::sleep(Duration::from_millis(100));
            assert!(t.is_finished());
        });

        // holding a read lock does not allow writes to the corresponding offset
        sync::thread::scope(|s| {
            let read_guard = file.locks[0].try_read().unwrap();
            let t = s.spawn(|| file.write_all_at(&buf, 0).unwrap());
            sync::thread::sleep(Duration::from_millis(100));
            assert!(!t.is_finished());
            drop(read_guard);
            sync::thread::sleep(Duration::from_millis(100));
            assert!(t.is_finished());
        });

        // holding a write lock does not allow writes to the corresponding offset
        sync::thread::scope(|s| {
            let write_guard = file.locks[0].try_write().unwrap();
            let t = s.spawn(|| file.write_all_at(&buf, 0).unwrap());
            sync::thread::sleep(Duration::from_millis(100));
            assert!(!t.is_finished());
            drop(write_guard);
            sync::thread::sleep(Duration::from_millis(100));
            assert!(t.is_finished());
        });
    }
}
