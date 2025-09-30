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
/// Implementations of this trait are required to ensure that concurrent operations are safe (in
/// that there are no data races) as long as they operate on non-overlapping regions of the file.
/// When called with overlapping regions, the behavior is undefined and may lead to data corruption.
#[allow(clippy::len_without_is_empty)]
#[cfg_attr(test, mockall::automock)]
pub trait FileBackend: Send + Sync {
    /// Opens a file at the given path with the specified options and tries to acquire a file lock.
    fn open(path: &Path, options: OpenOptions) -> std::io::Result<Self>
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
    fn open(path: &Path, options: OpenOptions) -> std::io::Result<Self> {
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
    fn open(path: &Path, options: OpenOptions) -> std::io::Result<Self> {
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
        fs::{File, OpenOptions},
        io::{Read, Write},
        sync::{
            Arc, Barrier,
            atomic::{AtomicU64, Ordering},
        },
    };

    use super::*;
    use crate::{
        storage::file::{PageCachedFile, page_utils::Page},
        utils::test_dir::{Permissions, TestDir},
    };

    type OpenBackendFn = fn(&Path, OpenOptions) -> std::io::Result<Arc<dyn FileBackend>>;

    #[rstest_reuse::template]
    #[rstest::rstest]
    #[case::seek_file(
        (|path, options| {
            <SeekFile as FileBackend>::open(path, options)
                .map(|f| Arc::new(f) as Arc<dyn FileBackend>)
        }) as OpenBackendFn
    )]
    #[case::no_seek_file(
        (|path, options| {
            <NoSeekFile as FileBackend>::open(path, options)
                .map(|f| Arc::new(f) as Arc<dyn FileBackend>)
        }) as OpenBackendFn
    )]
    #[case::page_cached_file__seek_file__direct_io(
        (|path, options| {
            <PageCachedFile<SeekFile, true> as FileBackend>::open(path, options)
                .map(|f| Arc::new(f) as Arc<dyn FileBackend>)
        }) as OpenBackendFn
    )]
    #[case::page_cached_file__no_seek_file__direct_io(
        (|path, options| {
            <PageCachedFile<NoSeekFile, true> as FileBackend>::open(path, options)
                .map(|f| Arc::new(f) as Arc<dyn FileBackend>)
        }) as OpenBackendFn
    )]
    #[case::page_cached_file__seek_file__no_direct_io(
        (|path, options| {
            <PageCachedFile<SeekFile, false> as FileBackend>::open(path, options)
                .map(|f| Arc::new(f) as Arc<dyn FileBackend>)
        }) as OpenBackendFn
    )]
    #[case::page_cached_file__no_seek_file__no_direct_io(
        (|path, options| {
            <PageCachedFile<NoSeekFile, false> as FileBackend>::open(path, options)
                .map(|f| Arc::new(f) as Arc<dyn FileBackend>)
        }) as OpenBackendFn
    )]
    fn open_backend(#[case] f: OpenBackendFn) {}

    #[rstest_reuse::apply(open_backend)]
    fn open_creates_and_opens_file(#[case] open_backend_fn: OpenBackendFn) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("test_file.bin");

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        let backend = open_backend_fn(path.as_path(), options.clone()).unwrap();
        assert_eq!(backend.len().unwrap(), 0);
        assert!(std::fs::exists(path).unwrap());
    }

    #[rstest_reuse::apply(open_backend)]
    fn open_opens_existing_file(#[case] open_backend_fn: OpenBackendFn) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("test_file.bin");

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        {
            let mut file = File::create(path.as_path()).unwrap();
            file.write_all(&[0; 10]).unwrap();
        }

        let backend = open_backend_fn(path.as_path(), options.clone()).unwrap();
        assert_eq!(backend.len().unwrap(), 10);
    }

    #[rstest_reuse::apply(open_backend)]
    fn open_fails_if_file_is_locked(#[case] open_backend_fn: OpenBackendFn) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("test_file.bin");

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        // Open the file once and lock it.
        let file = open_backend_fn(path.as_path(), options.clone());
        assert!(file.is_ok());

        // Try to open it again, while the first on is still open. This should fail.
        let file = open_backend_fn(path.as_path(), options.clone());
        assert_eq!(
            file.map(|_| ()).unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );
    }

    #[rstest_reuse::apply(open_backend)]
    fn open_fails_if_no_permissions(#[case] open_backend_fn: OpenBackendFn) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("test_file.bin");

        let _ = File::create(path.as_path()).unwrap();
        tempdir.set_permissions(Permissions::ReadOnly).unwrap();

        let mut options = OpenOptions::new();
        options.read(true).write(true);

        assert_eq!(
            open_backend_fn(path.as_path(), options.clone())
                .map(|_| ())
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::PermissionDenied
        );
    }

    #[rstest_reuse::apply(open_backend)]
    fn write_all_at_writes_whole_buffer_at_offset(#[case] open_backend_fn: OpenBackendFn) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("test_file.bin");

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        let data = [1; 10];
        let offset = 5;

        {
            let backend = open_backend_fn(path.as_path(), options.clone()).unwrap();
            backend.write_all_at(&data, offset).unwrap();
        }
        // file: [_, _, _, _, _, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        let mut file = File::open(path).unwrap();
        assert_eq!(file.metadata().unwrap().len(), 15);
        let mut buf = vec![0; 15];
        file.read_exact(&mut buf).unwrap();
        assert_eq!(buf[5..], data);
    }

    #[rstest_reuse::apply(open_backend)]
    fn write_all_at_can_write_across_pages(#[case] open_backend_fn: OpenBackendFn) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("test_file.bin");

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        let data = [1; Page::SIZE * 3];
        let offset = 0;

        {
            let backend = open_backend_fn(path.as_path(), options.clone()).unwrap();
            backend.write_all_at(&data, offset).unwrap();
        }

        let file = File::open(path).unwrap();
        assert_eq!(file.metadata().unwrap().len(), data.len() as u64);
        let mut buf = vec![0; Page::SIZE * 3];
        file.read_exact_at(&mut buf, offset).unwrap();
        assert_eq!(buf, data);
    }

    #[rstest_reuse::apply(open_backend)]
    fn write_all_at_can_write_data_to_different_pages(#[case] open_backend_fn: OpenBackendFn) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("test_file.bin");

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        let data = [1];
        let offset1 = 0;
        let offset2 = 10000;

        {
            let backend = open_backend_fn(path.as_path(), options.clone()).unwrap();
            backend.write_all_at(&data, offset1).unwrap();
            backend.write_all_at(&data, offset2).unwrap();
        }

        let mut file = File::open(path).unwrap();
        assert_eq!(file.metadata().unwrap().len(), offset2 + 1);
        let mut buf = [0; 1];
        file.seek(SeekFrom::Start(offset1)).unwrap();
        file.read_exact(&mut buf).unwrap();
        assert_eq!(buf, data);
        buf = [0; 1];
        file.seek(SeekFrom::Start(offset2)).unwrap();
        file.read_exact(&mut buf).unwrap();
        assert_eq!(buf, data);
    }

    #[rstest_reuse::apply(open_backend)]
    fn read_exact_at_fills_whole_buffer_by_reading_at_offset(
        #[case] open_backend_fn: OpenBackendFn,
    ) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("test_file.bin");

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        {
            let mut file = File::create(path.as_path()).unwrap();
            let buf = vec![1; 10];
            file.write_all(&buf).unwrap();
            let buf = vec![0; 5];
            file.write_all(&buf).unwrap();
        }
        // file: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        // read:                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        let backend = open_backend_fn(path.as_path(), options.clone()).unwrap();
        let mut buf = [0; 10];
        backend.read_exact_at(&mut buf, 5).unwrap();
        assert_eq!(buf[..5], [1; 5]);
        assert_eq!(buf[5..], [0; 5]);
    }

    #[rstest_reuse::apply(open_backend)]
    fn read_exact_at_can_read_across_pages(#[case] open_backend_fn: OpenBackendFn) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("test_file.bin");

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        let data = [1; Page::SIZE * 3];
        let offset = 0;

        {
            let mut file = File::create(path.as_path()).unwrap();
            file.write_all(&data).unwrap();
        }

        let backend = open_backend_fn(path.as_path(), options.clone()).unwrap();
        let mut buf = [0; Page::SIZE * 3];
        backend.read_exact_at(&mut buf, offset).unwrap();
        assert_eq!(buf, data);
    }

    #[rstest_reuse::apply(open_backend)]
    fn read_exact_at_can_read_data_from_different_pages(#[case] open_backend_fn: OpenBackendFn) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("test_file.bin");

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        let data = [1];
        let offset1 = 0;
        let offset2 = 10000;

        {
            let mut file = File::create(path.as_path()).unwrap();
            file.seek(SeekFrom::Start(offset1)).unwrap();
            file.write_all(&data).unwrap();
            file.seek(SeekFrom::Start(offset2)).unwrap();
            file.write_all(&data).unwrap();
        }

        let backend = open_backend_fn(path.as_path(), options.clone()).unwrap();
        let mut buf = [1];
        backend.read_exact_at(&mut buf, offset1).unwrap();
        assert_eq!(buf, data);
        backend.read_exact_at(&mut buf, offset2).unwrap();
        assert_eq!(buf, data);
    }

    #[rstest_reuse::apply(open_backend)]
    fn read_exact_at_fails_when_out_of_bounds(#[case] open_backend_fn: OpenBackendFn) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("test_file.bin");

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        {
            File::create(path.as_path()).unwrap();
        }
        // The file exists but is empty.

        let backend = open_backend_fn(path.as_path(), options.clone()).unwrap();
        let mut buf = [0; 5];
        let res = backend.read_exact_at(&mut buf, 5);
        assert_eq!(res.unwrap_err().kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[rstest_reuse::apply(open_backend)]
    fn access_same_page_in_parallel_does_not_deadlock(#[case] open_backend_fn: OpenBackendFn) {
        const THREADS: usize = 128;
        const PAGES: usize = 100;

        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("test_file.bin");

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        let data = vec![1; Page::SIZE * PAGES];

        {
            let mut file = File::create(path.as_path()).unwrap();
            file.write_all(&data).unwrap();
        }

        let backend = open_backend_fn(path.as_path(), options.clone()).unwrap();

        let barrier = Barrier::new(THREADS);

        std::thread::scope(|s| {
            for t in 0..THREADS {
                let barrier = &barrier;
                let backend = Arc::clone(&backend);
                s.spawn(move || {
                    const BUF_LEN: usize = Page::SIZE / THREADS;
                    barrier.wait(); // ensure that all threads have at least been spawned before any starts performing I/O
                    for page in 0..PAGES {
                        let mut buf = [0; BUF_LEN];
                        backend
                            .read_exact_at(&mut buf, (page * Page::SIZE + t * BUF_LEN) as u64)
                            .unwrap();
                        assert_eq!(buf, [1; BUF_LEN]);
                    }
                });
            }
        });
    }

    #[rstest_reuse::apply(open_backend)]
    fn flush_flushes_file_and_sets_length(#[case] open_backend_fn: OpenBackendFn) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("test_file.bin");

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        // flush with no changes
        {
            let backend = open_backend_fn(path.as_path(), options.clone()).unwrap();
            backend.flush().unwrap();

            let file = File::open(path.as_path()).unwrap();
            assert_eq!(file.metadata().unwrap().len(), 0);
        }

        // flush with changes
        {
            let backend = open_backend_fn(path.as_path(), options.clone()).unwrap();
            backend.write_all_at(&[1; 10], 0).unwrap();
            backend.flush().unwrap();

            let mut file = File::open(path.as_path()).unwrap();
            assert_eq!(file.metadata().unwrap().len(), 10);
            let mut buf = [0; 10];
            file.read_exact(&mut buf).unwrap();
            assert_eq!(buf, [1; 10]);
        }
    }

    #[rstest_reuse::apply(open_backend)]
    fn drop_flushes_file_and_sets_length(#[case] open_backend_fn: OpenBackendFn) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("test_file.bin");

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        {
            let backend = open_backend_fn(path.as_path(), options.clone()).unwrap();
            backend.write_all_at(&[1; 10], 0).unwrap();
        }

        let mut file = File::open(path.as_path()).unwrap();
        assert_eq!(file.metadata().unwrap().len(), 10);
        let mut buf = [0; 10];
        file.read_exact(&mut buf).unwrap();
        assert_eq!(buf, [1; 10]);
    }

    #[rstest_reuse::apply(open_backend)]
    fn len_returns_file_length(#[case] open_backend_fn: OpenBackendFn) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("test_file.bin");

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        let backend = open_backend_fn(path.as_path(), options.clone()).unwrap();
        backend.write_all_at(&[1; 10], 0).unwrap();
        assert_eq!(backend.len().unwrap(), 10);
    }

    #[rstest_reuse::apply(open_backend)]
    fn set_len_sets_length(#[case] open_backend_fn: OpenBackendFn) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("test_file.bin");

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        let backend = open_backend_fn(path.as_path(), options.clone()).unwrap();
        backend.write_all_at(&[1; 200], 0).unwrap();
        backend.set_len(100).unwrap();

        let check_file = File::open(path.as_path()).unwrap();
        assert_eq!(check_file.metadata().unwrap().len(), 100);
    }

    #[rstest_reuse::apply(open_backend)]
    fn read_observes_writes_of_other_threads(#[case] open_backend_fn: OpenBackendFn) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("test_file.bin");

        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);

        let backend = open_backend_fn(path.as_path(), options.clone()).unwrap();

        let iteration = AtomicU64::new(0);

        let max_iterations = 1_000;

        std::thread::scope(|s| {
            s.spawn(|| {
                let mut buf = [0; 32];
                loop {
                    if iteration.load(Ordering::Relaxed) > max_iterations {
                        break;
                    }

                    // wait until it is thread1's turn
                    while !iteration.load(Ordering::Relaxed).is_multiple_of(2) {}

                    let offset = iteration.load(Ordering::Relaxed) * 32;
                    if offset >= 32 {
                        // check that the previous write of thread2 is visible
                        backend.read_exact_at(&mut buf, offset - 32).unwrap();
                        assert_eq!(buf, [2; 32]);
                    }

                    // write data
                    backend.write_all_at([1; 32].as_slice(), offset).unwrap();

                    // check that thread1 observes its own write
                    backend.read_exact_at(&mut buf, offset).unwrap();
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
                    while iteration.load(Ordering::Relaxed).is_multiple_of(2) {}

                    let offset = iteration.load(Ordering::Relaxed) * 32;
                    if offset >= 32 {
                        // check that the previous write of thread1 is visible
                        backend.read_exact_at(&mut buf, offset - 32).unwrap();
                        assert_eq!(buf, [1; 32]);
                    }

                    // write data
                    backend.write_all_at([2; 32].as_slice(), offset).unwrap();

                    // check that thread2 observes its own write
                    backend.read_exact_at(&mut buf, offset).unwrap();
                    assert_eq!(buf, [2; 32]);

                    iteration.fetch_add(1, Ordering::Relaxed);
                }
            });
        });
    }
}
