// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{cmp, fs::OpenOptions, os::unix::fs::OpenOptionsExt, path::Path};

use crate::{
    error::BTResult,
    storage::file::{
        FileBackend,
        file_backend::page_utils::{O_DIRECT, O_SYNC, Page},
    },
    sync::Mutex,
};

/// The actual implementation of [`PageCachedFile<F>`], but without concurrency control.
/// The generic parameter `D` controls whether to use direct I/O (`true`) or not (`false`).
#[derive(Debug)]
struct InnerPageCachedFile<F, const D: bool> {
    file: F,
    /// The offset up to which read operations are guaranteed to not result in EOF.
    file_len: u64,
    page: Box<Page>,
    page_index: u64,
    page_dirty: bool,
}

// All methods in this impl except for `change_page` correspond to the methods in `FileBackend`,
// except that they take mutable references since [`PageCachedFile`] adds the synchronization on top
// using a mutex.
impl<F: FileBackend, const D: bool> InnerPageCachedFile<F, D> {
    /// See [`FileBackend::open`].
    fn open(path: &Path, mut options: OpenOptions) -> BTResult<Self, std::io::Error> {
        let file = options.open(path)?;
        let file_len = file.metadata()?.len();
        let padded_len = file_len.div_ceil(Page::SIZE as u64) * Page::SIZE as u64;
        file.set_len(padded_len)?;
        drop(file);

        if D {
            options.custom_flags(O_DIRECT | O_SYNC);
        }
        let file = F::open(path, options)?;

        let mut page = Page::zeroed();
        if padded_len != 0 {
            file.read_exact_at(&mut page[..Page::SIZE], 0)?;
        }

        Ok(Self {
            file,
            file_len: padded_len,
            page,
            page_index: 0,
            page_dirty: false,
        })
    }

    /// See [`FileBackend::write_all_at`].
    fn write_all_at(&mut self, buf: &[u8], offset: u64) -> BTResult<(), std::io::Error> {
        self.change_page(offset)?;

        let start_in_page = offset as usize - self.page_index as usize * Page::SIZE;
        let end_in_page = cmp::min(start_in_page + buf.len(), Page::SIZE);
        let len = end_in_page - start_in_page;

        self.page_dirty = true;
        self.page[start_in_page..end_in_page].copy_from_slice(&buf[..len]);

        self.file_len = cmp::max(self.file_len, offset + len as u64);

        if buf.len() > len {
            self.write_all_at(&buf[len..], offset + len as u64)?;
        }
        Ok(())
    }

    /// See [`FileBackend::read_exact_at`].
    fn read_exact_at(&mut self, buf: &mut [u8], offset: u64) -> BTResult<(), std::io::Error> {
        if offset + buf.len() as u64 > self.file_len {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
        }

        self.change_page(offset)?;

        let start_in_page = offset as usize - self.page_index as usize * Page::SIZE;
        let end_in_page = cmp::min(start_in_page + buf.len(), Page::SIZE);
        let len = end_in_page - start_in_page;

        buf[..len].copy_from_slice(&self.page[start_in_page..end_in_page]);

        if buf.len() > len {
            self.read_exact_at(&mut buf[len..], offset + len as u64)?;
        }
        Ok(())
    }

    /// See [`FileBackend::flush`].
    fn flush(&mut self) -> BTResult<(), std::io::Error> {
        if self.page_dirty {
            self.file
                .write_all_at(&self.page, self.page_index * Page::SIZE as u64)?;
        }
        self.file.flush()
    }

    /// See [`FileBackend::len`].
    fn len(&self) -> BTResult<u64, std::io::Error> {
        Ok(self.file_len)
    }

    /// Load the page containing the given offset into memory, flushing the current page if dirty.
    /// If the offset is already within the currently loaded page, this is a no-op.
    fn change_page(&mut self, offset: u64) -> BTResult<(), std::io::Error> {
        if (self.page_index * Page::SIZE as u64..(self.page_index + 1) * Page::SIZE as u64)
            .contains(&offset)
        {
            // Page is already loaded.
            return Ok(());
        }

        if self.page_dirty {
            self.file
                .write_all_at(&self.page, self.page_index * Page::SIZE as u64)?;
            self.file_len = cmp::max(self.file_len, (self.page_index + 1) * Page::SIZE as u64);
        }

        self.page_index = offset / Page::SIZE as u64;
        if D {
            if self.file_len < (self.page_index + 1) * Page::SIZE as u64 {
                self.page.fill(0);
            } else {
                self.file
                    .read_exact_at(&mut self.page, self.page_index * Page::SIZE as u64)?;
            }
        } else {
            // Without O_DIRECT, the file size is not padded and we may read a partial page.
            let len = cmp::min(
                self.file_len
                    .saturating_sub(self.page_index * Page::SIZE as u64) as usize,
                Page::SIZE,
            );
            self.file
                .read_exact_at(&mut self.page[..len], self.page_index * Page::SIZE as u64)?;
            // In case we read a partial page, set the remainder to zero.
            self.page[len..].fill(0);
        }

        self.page_dirty = false;

        Ok(())
    }
}

/// A wrapper around a [`FileBackend`] that caches a single page (4096 bytes) in memory.
/// All read and write operations are performed on this page, which is flushed to the underlying
/// file when it is dirty and a different page is accessed, or when the file is flushed.
/// The generic parameter `D` controls whether to use direct I/O (`true`) or not (`false`).
#[derive(Debug)]
pub struct PageCachedFile<F, const D: bool>(Mutex<InnerPageCachedFile<F, D>>);

impl<F: FileBackend, const D: bool> FileBackend for PageCachedFile<F, D> {
    fn open(path: &Path, options: OpenOptions) -> BTResult<Self, std::io::Error> {
        Ok(Self(Mutex::new(InnerPageCachedFile::open(path, options)?)))
    }

    fn write_all_at(&self, buf: &[u8], offset: u64) -> BTResult<(), std::io::Error> {
        self.0.lock().unwrap().write_all_at(buf, offset)
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> BTResult<(), std::io::Error> {
        self.0.lock().unwrap().read_exact_at(buf, offset)
    }

    fn flush(&self) -> BTResult<(), std::io::Error> {
        self.0.lock().unwrap().flush()
    }

    fn len(&self) -> BTResult<u64, std::io::Error> {
        self.0.lock().unwrap().len()
    }
}

// Note: The tests for `PageCachedFile<F> as FileBackend` are in `file_backend.rs`.

#[cfg(test)]
mod tests {
    use mockall::predicate::{always, eq};

    use super::*;
    use crate::storage::file::MockFileBackend;

    #[test]
    fn access_of_cache_data_does_not_trigger_io_operations() {
        // no expectations on the mock because there should not be no I/O operations.
        let file = MockFileBackend::new();

        let file = PageCachedFile::<_, true>(Mutex::new(InnerPageCachedFile {
            file,
            file_len: Page::SIZE as u64,
            page: Page::zeroed(),
            page_index: 0,
            page_dirty: false,
        }));

        let data = vec![1u8; Page::SIZE];
        file.write_all_at(&data, 0).unwrap();

        // Read the data back, which should hit the cache and not trigger any I/O operations.
        let mut read_data = vec![0u8; Page::SIZE];
        file.read_exact_at(&mut read_data, 0).unwrap();
        assert_eq!(data, read_data);
    }

    #[test]
    fn access_non_cached_data_triggers_write_of_old_and_read_of_new_page() {
        let mut file = MockFileBackend::new();
        file.expect_write_all_at()
            .with(eq([0; Page::SIZE]), eq(0))
            .times(1)
            .returning(|_, _| Ok(()));
        file.expect_read_exact_at()
            .with(always(), eq(Page::SIZE as u64))
            .times(1)
            .returning(|buf, _| {
                buf.fill(1);
                Ok(())
            });

        let file = PageCachedFile::<_, true>(Mutex::new(InnerPageCachedFile {
            file,
            file_len: 8192,
            page: Page::zeroed(),
            page_index: 0,
            page_dirty: true,
        }));

        // Access data outside of the cached page, which should trigger a write of the old page and
        // a read of the new page.
        let mut read_data = vec![0u8; Page::SIZE];
        file.read_exact_at(&mut read_data, Page::SIZE as u64)
            .unwrap();
        assert_eq!(read_data, vec![1u8; Page::SIZE]);
    }
}
