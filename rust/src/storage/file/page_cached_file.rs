// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{cmp, fs::OpenOptions, os::unix::fs::OpenOptionsExt, path::Path, sync::Mutex};

use crate::storage::file::{
    FileBackend,
    page_utils::{O_DIRECT, O_SYNC, Page},
};

/// The actual implementation of [`PageCachedFile<F>`], but without concurrency control.
#[derive(Debug)]
struct InnerPageCachedFile<F: FileBackend> {
    file: F,
    /// The logical file size, which may be smaller than the actual file size due to padding.
    file_len: u64,
    page: Box<Page>,
    page_offset: u64,
    page_dirty: bool,
}

// All methods in this impl expect for `load_page_at_offset` correspond to the methods in
// `FileBackend`, except that they take mutable references since [`PageCachedFile`] adds the
// synchronization on top using a mutex.
impl<F: FileBackend> InnerPageCachedFile<F> {
    /// See [`FileBackend::open`].
    fn open(path: &Path, mut options: OpenOptions) -> std::io::Result<Self> {
        let file = F::open(path, options.clone())?;
        let file_len = file.len()?;
        let padded_len = file_len.div_ceil(Page::SIZE as u64) * Page::SIZE as u64;
        file.set_len(padded_len)?;
        drop(file);

        options.custom_flags(O_DIRECT | O_SYNC);
        let file = F::open(path, options)?;

        let mut page = Box::new(Page::zeroed());
        file.read_exact_at(&mut page[..cmp::min(padded_len as usize, Page::SIZE)], 0)?;

        Ok(Self {
            file,
            file_len,
            page,
            page_offset: 0,
            page_dirty: false,
        })
    }

    /// See [`FileBackend::write_all_at`].
    fn write_all_at(&mut self, buf: &[u8], offset: u64) -> std::io::Result<()> {
        self.load_page_at_offset(offset)?;

        let page_start_idx = (offset - self.page_offset) as usize;
        let page_end_idx = cmp::min(page_start_idx + buf.len(), Page::SIZE);
        let len = page_end_idx - page_start_idx;

        self.page_dirty = true;
        self.page[page_start_idx..page_end_idx].copy_from_slice(&buf[..len]);

        self.file_len = cmp::max(self.file_len, offset + len as u64);

        if buf.len() > len {
            self.write_all_at(&buf[len..], offset + len as u64)?;
        }
        Ok(())
    }

    /// See [`FileBackend::read_exact_at`].
    fn read_exact_at(&mut self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        if offset + buf.len() as u64 > self.file_len {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
        }

        self.load_page_at_offset(offset)?;

        let page_start_idx = (offset - self.page_offset) as usize;
        let page_end_idx = cmp::min(page_start_idx + buf.len(), Page::SIZE);
        let len = page_end_idx - page_start_idx;

        buf[..len].copy_from_slice(&self.page[page_start_idx..page_end_idx]);

        if buf.len() > len {
            self.read_exact_at(&mut buf[len..], offset + len as u64)?;
        }
        Ok(())
    }

    /// See [`FileBackend::flush`].
    fn flush(&mut self) -> std::io::Result<()> {
        if self.page_dirty {
            self.file.write_all_at(&self.page, self.page_offset)?;
        }
        self.file.flush()?;
        self.set_len(self.file_len)
    }

    /// See [`FileBackend::len`].
    fn len(&self) -> Result<u64, std::io::Error> {
        Ok(self.file_len)
    }

    /// See [`FileBackend::set_len`].
    fn set_len(&mut self, len: u64) -> std::io::Result<()> {
        self.file_len = len;
        self.file.set_len(len)
    }

    /// Load the page containing the given offset into memory, flushing the current page if dirty.
    /// If the offset is already within the currently loaded page, this is a no-op.
    fn load_page_at_offset(&mut self, offset: u64) -> std::io::Result<()> {
        if offset < self.page_offset || offset >= self.page_offset + Page::SIZE as u64 {
            // O_DIRECT requires reads and writes to operate on page aligned chunks with sizes that
            // are multiples of the page size. So the file is padded to have a size of a multiple of
            // the page size.
            let padded_len = self.file_len.div_ceil(Page::SIZE as u64) * Page::SIZE as u64;
            if self.file_len < padded_len {
                self.file.set_len(padded_len)?;
            }

            if self.page_dirty {
                self.file.write_all_at(&self.page, self.page_offset)?;
            }

            self.page_offset = (offset / Page::SIZE as u64) * Page::SIZE as u64;
            let len = cmp::min(
                padded_len.saturating_sub(self.page_offset) as usize,
                Page::SIZE,
            );
            self.file
                .read_exact_at(&mut self.page[..len], self.page_offset)?;
            self.page[len..].fill(0);
            self.page_dirty = false;
        }
        Ok(())
    }
}

/// A wrapper around a [`FileBackend`] that caches a single page (4096 bytes) in memory.
/// All read and write operations are performed on this page, which is flushed to the underlying
/// file when it is dirty and a different page is accessed, or when the file is flushed or dropped.
/// All file operations use direct I/O to bypass the OS page cache.
#[derive(Debug)]
pub struct PageCachedFile<F: FileBackend>(Mutex<InnerPageCachedFile<F>>);

impl<F: FileBackend> FileBackend for PageCachedFile<F> {
    fn open(path: &Path, options: OpenOptions) -> std::io::Result<Self> {
        Ok(Self(Mutex::new(InnerPageCachedFile::open(path, options)?)))
    }

    fn write_all_at(&self, buf: &[u8], offset: u64) -> std::io::Result<()> {
        self.0.lock().unwrap().write_all_at(buf, offset)
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        self.0.lock().unwrap().read_exact_at(buf, offset)
    }

    fn flush(&self) -> std::io::Result<()> {
        self.0.lock().unwrap().flush()
    }

    fn len(&self) -> Result<u64, std::io::Error> {
        self.0.lock().unwrap().len()
    }

    fn set_len(&self, size: u64) -> std::io::Result<()> {
        self.0.lock().unwrap().set_len(size)
    }
}

impl<F: FileBackend> Drop for PageCachedFile<F> {
    fn drop(&mut self) {
        let _ = self.0.lock().unwrap().flush();
    }
}

// Note: The tests for `PageCachedFile<F> as FileBackend` are in `file_backend.rs`.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::file::MockFileBackend;

    #[test]
    fn access_of_cache_data_does_not_trigger_io_operations() {
        // no expectations on the mock because there should not be no I/O operations.
        let _file = MockFileBackend::new();

        let file = PageCachedFile(Mutex::new(InnerPageCachedFile {
            file: _file,
            file_len: 4096,
            page: Box::new(Page::zeroed()),
            page_offset: 0,
            page_dirty: false,
        }));

        let data = vec![1u8; 4096];
        file.write_all_at(&data, 0).unwrap();

        // Read the data back, which should hit the cache and not trigger any I/O operations.
        let mut read_data = vec![0u8; 4096];
        file.read_exact_at(&mut read_data, 0).unwrap();
        assert_eq!(data, read_data);

        // Prevent the destructor from running, which would trigger a flush.
        std::mem::forget(file);
    }

    #[test]
    fn access_non_cached_data_triggers_write_of_old_and_read_of_new_page() {
        let mut _file = MockFileBackend::new();
        _file
            .expect_write_all_at()
            .withf(|buf, offset| buf == [0; 4096] && *offset == 0)
            .times(1)
            .returning(|_, _| Ok(()));
        _file
            .expect_read_exact_at()
            .withf(|_, offset| *offset == 4096)
            .times(1)
            .returning(|buf, _| {
                buf.fill(1);
                Ok(())
            });

        let file = PageCachedFile(Mutex::new(InnerPageCachedFile {
            file: _file,
            file_len: 8192,
            page: Box::new(Page::zeroed()),
            page_offset: 0,
            page_dirty: false,
        }));

        // Access data outside of the cached page, which should trigger a write of the old page and
        // a read of the new page.
        let mut read_data = vec![0u8; 4096];
        file.read_exact_at(&mut read_data, 4096).unwrap();
        assert_eq!(read_data, vec![1u8; 4096]);

        // Prevent the destructor from running, which would trigger a flush.
        std::mem::forget(file);
    }
}
