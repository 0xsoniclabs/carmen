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
    fs::OpenOptions,
    io::{Read, Seek, SeekFrom, Write},
    os::unix::fs::FileExt,
    path::Path,
    sync::Mutex,
};

/// An abstraction for concurrent file operations.
///
/// Implementations of this trait are required to ensure that concurrent operations of the
/// implementing type are safe (in that there are no data races) as long as the operations operate
/// on non-overlapping regions of the file.
/// When called with overlapping regions, the behavior is undefined and may lead to data corruption.
#[cfg_attr(test, mockall::automock)]
pub trait FileBackend {
    /// Opens a file at the given path with the specified options and tries to acquire an advisory
    /// file lock.
    fn open(path: &Path, options: &OpenOptions) -> std::io::Result<Self>
    where
        Self: Sized;

    /// Writes the entire content of `buf` starting at the given `offset`.
    fn write_all_at(&self, buf: &[u8], offset: u64) -> std::io::Result<()>;

    /// Fills the entire `buf` with data read from the file starting at the given `offset`.
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()>;

    /// Flushes all changes to disk.
    fn flush(&self) -> std::io::Result<()>;

    /// Returns the size of this file in bytes.
    fn len(&self) -> Result<u64, std::io::Error>;

    /// Truncates or extends the underlying file, updating the size of this file to become `size`.
    fn set_len(&self, size: u64) -> std::io::Result<()>;
}

/// A wrapper around [`std::fs::File`] that implements [`FileBackend`] using a mutex to ensure
/// exclusive access to the file. This is suitable for platforms where `pread` and `pwrite` are not
/// available or where seeking is required for other reasons.
pub struct SeekFile(Mutex<std::fs::File>);

impl FileBackend for SeekFile {
    fn open(path: &Path, options: &OpenOptions) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        let file = options.open(path)?;
        file.try_lock()?;
        Ok(Self(Mutex::new(file)))
    }

    fn write_all_at(&self, buf: &[u8], offset: u64) -> std::io::Result<()> {
        let mut file = self.0.lock().unwrap();
        file.seek(SeekFrom::Start(offset))?;
        file.write_all(buf)
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        let mut file = self.0.lock().unwrap();
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(buf)
    }

    fn flush(&self) -> std::io::Result<()> {
        self.0.lock().unwrap().sync_all()
    }

    fn len(&self) -> std::io::Result<u64> {
        self.0.lock().unwrap().metadata().map(|m| m.len())
    }

    fn set_len(&self, len: u64) -> std::io::Result<()> {
        self.0.lock().unwrap().set_len(len)
    }
}

/// A wrapper around [`std::fs::File`] that implements [`FileBackend`] using the Unix-specific file
/// operations `pread` and `pwrite` which do not modify the file offset. This avoids the syscall for
/// seeking and allows for concurrent access without needing to manage a cursor.
#[cfg(unix)]
pub struct NoSeekFile(std::fs::File);

#[cfg(unix)]
impl FileBackend for NoSeekFile {
    fn open(path: &Path, options: &OpenOptions) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        let file = options.open(path)?;
        file.try_lock()?;
        Ok(Self(file))
    }

    fn write_all_at(&self, buf: &[u8], offset: u64) -> std::io::Result<()> {
        self.0.write_all_at(buf, offset)
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        self.0.read_exact_at(buf, offset)
    }

    fn flush(&self) -> std::io::Result<()> {
        self.0.sync_all()
    }

    fn len(&self) -> std::io::Result<u64> {
        self.0.metadata().map(|m| m.len())
    }

    fn set_len(&self, len: u64) -> std::io::Result<()> {
        self.0.set_len(len)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{File, OpenOptions, Permissions},
        io::{Read, Write},
        os::unix::fs::PermissionsExt,
        sync::atomic::{AtomicU64, Ordering},
    };

    use super::*;

    fn open_backends() -> impl Iterator<
        Item = fn(&Path, &OpenOptions) -> std::io::Result<Box<dyn FileBackend + Send + Sync>>,
    > {
        [
            (|path, options| {
                <SeekFile as FileBackend>::open(path, options)
                    .map(|f| Box::new(f) as Box<dyn FileBackend + Send + Sync>)
            })
                as fn(&Path, &OpenOptions) -> std::io::Result<Box<dyn FileBackend + Send + Sync>>,
            (|path, options| {
                <NoSeekFile as FileBackend>::open(path, options)
                    .map(|f| Box::new(f) as Box<dyn FileBackend + Send + Sync>)
            })
                as fn(&Path, &OpenOptions) -> std::io::Result<Box<dyn FileBackend + Send + Sync>>,
        ]
        .into_iter()
    }

    #[test]
    fn open_creates_and_opens_file() {
        let tempdir = tempfile::tempdir().unwrap();
        let dir = tempdir.path();

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        for (i, backend) in open_backends().enumerate() {
            let path = dir.join(format!("test_file_{i}.bin"));
            let file = backend(path.as_path(), &options);
            assert!(file.is_ok());
            assert!(file.unwrap().len().unwrap() == 0);
        }
    }

    #[test]
    fn open_opens_existing_file() {
        let tempdir = tempfile::tempdir().unwrap();
        let dir = tempdir.path();

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        for (i, backend) in open_backends().enumerate() {
            let path = dir.join(format!("test_file_{i}.bin"));

            File::create(path.as_path()).unwrap();

            let file = backend(path.as_path(), &options);
            assert!(file.is_ok());
            assert!(file.unwrap().len().unwrap() == 0);
        }
    }

    #[test]
    fn open_fails_if_file_is_locked() {
        let tempdir = tempfile::tempdir().unwrap();
        let dir = tempdir.path();

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        for (i, backend) in open_backends().enumerate() {
            let path = dir.join(format!("test_file_{i}.bin"));

            // open the file once and lock it
            let file = backend(path.as_path(), &options);
            assert!(file.is_ok());

            // try to open it again, should fail
            let file = backend(path.as_path(), &options);
            assert!(file.is_err());
            assert!(file.map(|_| ()).unwrap_err().kind() == std::io::ErrorKind::WouldBlock);
        }
    }

    #[test]
    fn open_fails_if_no_permissions() {
        let tempdir = tempfile::tempdir().unwrap();
        let dir = tempdir.path();

        let path = dir.join("test_file.bin");
        let file = File::create(path.as_path()).unwrap();
        file.set_permissions(Permissions::from_mode(0o000)).unwrap();

        let mut options = OpenOptions::new();
        options.read(true).write(true);

        for backend in open_backends() {
            let file = backend(path.as_path(), &options);
            assert!(file.is_err());
        }
    }

    #[test]
    fn write_all_at_writes_whole_buffer_at_offset() {
        let tempdir = tempfile::tempdir().unwrap();
        let dir = tempdir.path();

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        let data = [1; 10];
        let offset = 5;

        for (i, backend) in open_backends().enumerate() {
            let path = dir.join(format!("test_file_{i}.bin"));

            {
                let file = backend(path.as_path(), &options).unwrap();
                file.write_all_at(&data, offset).unwrap();
            }
            // file: [_, _, _, _, _, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

            let mut file = File::open(path).unwrap();
            assert_eq!(file.metadata().unwrap().len(), 15);
            let mut buf = vec![0; 15];
            file.read_exact(&mut buf).unwrap();
            assert_eq!(buf[5..], data);
        }
    }

    #[test]
    fn read_exact_at_fills_whole_buffer_by_reading_at_offset() {
        let tempdir = tempfile::tempdir().unwrap();
        let dir = tempdir.path();

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        for (i, backend) in open_backends().enumerate() {
            let path = dir.join(format!("test_file_{i}.bin"));

            {
                let mut file = File::create(path.as_path()).unwrap();
                let buf = vec![1; 10];
                file.write_all(&buf).unwrap();
                let buf = vec![0; 5];
                file.write_all(&buf).unwrap();
            }
            // file: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
            // read:                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            let file = backend(path.as_path(), &options).unwrap();
            let mut buf = [0; 10];
            file.read_exact_at(&mut buf, 5).unwrap();
            assert_eq!(buf[..5], [1; 5]);
            assert_eq!(buf[5..], [0; 5]);
        }
    }

    #[test]
    fn read_exact_at_fails_when_out_of_bounds() {
        let tempdir = tempfile::tempdir().unwrap();
        let dir = tempdir.path();

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        for (i, backend) in open_backends().enumerate() {
            let path = dir.join(format!("test_file_{i}.bin"));

            {
                File::create(path.as_path()).unwrap();
            }

            let file = backend(path.as_path(), &options).unwrap();
            let mut buf = [0; 5];
            let res = file.read_exact_at(&mut buf, 5);
            dbg!(&res);
            assert!(res.is_err());
            let err = res.unwrap_err();
            assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
        }
    }

    #[test]
    fn flush_flushes_file_and_sets_length() {
        let tempdir = tempfile::tempdir().unwrap();
        let dir = tempdir.path();

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        for (i, backend) in open_backends().enumerate() {
            let path = dir.join(format!("test_file_{i}.bin"));

            // flush with no changes
            {
                let file = backend(path.as_path(), &options).unwrap();
                file.flush().unwrap();

                let file = File::open(path.as_path()).unwrap();
                assert_eq!(file.metadata().unwrap().len(), 0);
            }

            // flush with changes
            {
                let file = backend(path.as_path(), &options).unwrap();
                file.write_all_at(&[1; 10], 0).unwrap();
                file.flush().unwrap();

                let mut file = File::open(path.as_path()).unwrap();
                assert_eq!(file.metadata().unwrap().len(), 10);
                let mut buf = [0; 10];
                file.read_exact(&mut buf).unwrap();
                assert_eq!(buf, [1; 10]);
            }
        }
    }

    #[test]
    fn drop_flushes_file_and_sets_length() {
        let tempdir = tempfile::tempdir().unwrap();
        let dir = tempdir.path();

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        for (i, backend) in open_backends().enumerate() {
            let path = dir.join(format!("test_file_{i}.bin"));

            {
                let file = backend(path.as_path(), &options).unwrap();
                file.write_all_at(&[1; 10], 0).unwrap();
            }

            let mut file = File::open(path.as_path()).unwrap();
            assert_eq!(file.metadata().unwrap().len(), 10);
            let mut buf = [0; 10];
            file.read_exact(&mut buf).unwrap();
            assert_eq!(buf, [1; 10]);
        }
    }

    #[test]
    fn len_returns_file_length() {
        let tempdir = tempfile::tempdir().unwrap();
        let dir = tempdir.path();

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        for (i, backend) in open_backends().enumerate() {
            let path = dir.join(format!("test_file_{i}.bin"));

            let file = backend(path.as_path(), &options).unwrap();
            file.write_all_at(&[1; 10], 0).unwrap();
            assert_eq!(file.len().unwrap(), 10);
        }
    }

    #[test]
    fn set_len_sets_length() {
        let tempdir = tempfile::tempdir().unwrap();
        let dir = tempdir.path();

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        for (i, backend) in open_backends().enumerate() {
            let path = dir.join(format!("test_file_{i}.bin"));

            let file = backend(path.as_path(), &options).unwrap();
            file.set_len(100).unwrap();

            let check_file = File::open(path.as_path()).unwrap();
            assert_eq!(check_file.metadata().unwrap().len(), 100);
        }
    }

    #[test]
    fn read_observes_writes_of_other_threads() {
        let tempdir = tempfile::tempdir().unwrap();
        let dir = tempdir.path();

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        for (i, backend) in open_backends().enumerate() {
            let path = dir.join(format!("test_file_{i}.bin"));

            let file = backend(path.as_path(), &options).unwrap();

            let iteration = AtomicU64::new(0);

            let max_iterations = 1_000_000;

            std::thread::scope(|s| {
                s.spawn(|| {
                    let mut buf = [0; 32];
                    loop {
                        if iteration.load(Ordering::Relaxed) > max_iterations {
                            break;
                        }

                        // wait until it is thread1's turn
                        while iteration.load(Ordering::Relaxed) % 2 != 0 {}

                        let offset = iteration.load(Ordering::Relaxed) * 32;
                        if offset >= 32 {
                            // check that the previous write of thread2 is visible
                            file.read_exact_at(&mut buf, offset - 32).unwrap();
                            assert_eq!(buf, [2; 32]);
                        }

                        // write data
                        file.write_all_at([1; 32].as_slice(), offset).unwrap();

                        // check that thread1 observes its own write
                        file.read_exact_at(&mut buf, offset).unwrap();
                        assert_eq!(buf, [1; 32]);

                        iteration.fetch_add(1, Ordering::Relaxed);
                    }
                });
                s.spawn(|| {
                    let mut buf = [0; 32];
                    loop {
                        if iteration.load(Ordering::Relaxed) > max_iterations {
                            break;
                        }

                        // wait until it is thread2's turn
                        while iteration.load(Ordering::Relaxed) % 2 == 0 {}

                        let offset = iteration.load(Ordering::Relaxed) * 32;
                        if offset >= 32 {
                            // check that the previous write of thread1 is visible
                            file.read_exact_at(&mut buf, offset - 32).unwrap();
                            assert_eq!(buf, [1; 32]);
                        }

                        // write data
                        file.write_all_at([2; 32].as_slice(), offset).unwrap();

                        // check that thread2 observes its own write
                        file.read_exact_at(&mut buf, offset).unwrap();
                        assert_eq!(buf, [2; 32]);

                        iteration.fetch_add(1, Ordering::Relaxed);
                    }
                });
            });
        }
    }
}
