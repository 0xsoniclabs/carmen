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
    fs::OpenOptions,
    os::unix::fs::OpenOptionsExt,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{
    error::BTResult,
    storage::file::{
        FileBackend,
        file_backend::page_utils::{O_DIRECT, O_SYNC, Page, PageGuard, Pages},
    },
};

/// A wrapper around a [`FileBackend`] that caches multiple pages (4096 bytes) in memory.
/// All read and write operations are performed on these pages, which are flushed to the underlying
/// file when they are dirty and a different page needs to be accessed, or when the file is flushed.
/// All file operations use direct I/O to bypass the OS page cache.
#[derive(Debug)]
pub struct MultiPageCachedFile<const P: usize, F: FileBackend, const D: bool> {
    file: F,
    /// The offset up to which read operations are guaranteed to not result in EOF.
    file_len: AtomicU64,
    pages: Pages<P>,
}

impl<const P: usize, F: FileBackend, const D: bool> FileBackend for MultiPageCachedFile<P, F, D> {
    fn open(path: &Path, mut options: OpenOptions) -> BTResult<Self, std::io::Error> {
        let file = options.clone().open(path)?;
        let file_len = file.metadata()?.len();
        let padded_len = file_len.div_ceil(Page::SIZE as u64) * Page::SIZE as u64;
        file.set_len(padded_len)?;
        drop(file);

        if D {
            options.custom_flags(O_DIRECT | O_SYNC);
        }
        let file = F::open(path, options)?;

        let pages = (0..P)
            .map(|page_index| {
                let mut page = Page::zeroed();
                let offset = (page_index * Page::SIZE) as u64;
                if offset >= padded_len {
                    Ok((page, page_index as u64))
                } else {
                    file.read_exact_at(&mut page, offset)
                        .map(|_| (page, page_index as u64))
                }
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .unwrap();

        Ok(Self {
            file,
            file_len: AtomicU64::new(padded_len),
            pages: Pages::new(pages),
        })
    }

    fn write_all_at(&self, buf: &[u8], offset: u64) -> BTResult<(), std::io::Error> {
        let mut locked_page = self.change_page(offset)?;

        let start_in_page = offset as usize - locked_page.page_index() as usize * Page::SIZE;
        let end_in_page = cmp::min(start_in_page + buf.len(), Page::SIZE);
        let len = end_in_page - start_in_page;

        locked_page[start_in_page..end_in_page].copy_from_slice(&buf[..len]);

        self.file_len
            .fetch_max(offset + len as u64, Ordering::Release);

        if buf.len() > len {
            self.write_all_at(&buf[len..], offset + len as u64)?;
        }
        Ok(())
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> BTResult<(), std::io::Error> {
        if offset + buf.len() as u64 > self.file_len.load(Ordering::Acquire) {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
        }

        let locked_page = self.change_page(offset)?;

        let start_in_page = offset as usize - locked_page.page_index() as usize * Page::SIZE;
        let end_in_page = cmp::min(start_in_page + buf.len(), Page::SIZE);
        let len = end_in_page - start_in_page;

        buf[..len].copy_from_slice(&locked_page[start_in_page..end_in_page]);

        if buf.len() > len {
            self.read_exact_at(&mut buf[len..], offset + len as u64)?;
        }
        Ok(())
    }

    fn flush(&self) -> BTResult<(), std::io::Error> {
        for mut locked_page in self.pages.iter_dirty_locked() {
            let page_index = locked_page.page_index();
            self.file
                .write_all_at(&locked_page, page_index * Page::SIZE as u64)?;
            locked_page.mark_clean();
        }
        self.file.flush()
    }

    fn len(&self) -> BTResult<u64, std::io::Error> {
        Ok(self.file_len.load(Ordering::Acquire))
    }
}

impl<const P: usize, F: FileBackend, const D: bool> MultiPageCachedFile<P, F, D> {
    /// Stores the given page at the given page index to the underlying file.
    fn store_page(&self, page: &Page, page_index: u64) -> BTResult<(), std::io::Error> {
        self.file
            .write_all_at(page, page_index * Page::SIZE as u64)?;

        Ok(())
    }

    /// Loads the given page at the given page index from the underlying file.
    fn load_page(&self, page: &mut Page, page_index: u64) -> BTResult<(), std::io::Error> {
        let file_len = self.file_len.load(Ordering::Relaxed);

        if D {
            if self.file_len.load(Ordering::Acquire) < (page_index + 1) * Page::SIZE as u64 {
                page.fill(0);
            } else {
                self.file
                    .read_exact_at(page, page_index * Page::SIZE as u64)?;
            }
        } else {
            // Without O_DIRECT, the file size is not padded and we may read a partial page.
            let len = cmp::min(
                file_len.saturating_sub(page_index * Page::SIZE as u64) as usize,
                Page::SIZE,
            );
            self.file
                .read_exact_at(&mut page[..len], page_index * Page::SIZE as u64)?;
            // In case we read a partial page, set the remainder to zero.
            page[len..].fill(0);
        }

        Ok(())
    }

    /// Returns a locked page that contains the data for the given offset.
    ///
    /// This function loads the page containing the given offset into memory, flushing the currently
    /// cached data if dirty. If the offset is already within the currently loaded page, this is
    /// a no-op.
    fn change_page(&'_ self, offset: u64) -> BTResult<PageGuard<'_>, std::io::Error> {
        self.pages.get_page_for_offset(
            offset,
            |page, page_index| self.load_page(page, page_index),
            |page, page_index| self.store_page(page, page_index),
        )
    }
}

// Note: The tests for `PageCachedFile<F> as FileBackend` are in `file_backend.rs`.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::storage::file::MockFileBackend;

    #[test]
    fn access_of_cache_data_does_not_trigger_io_operations() {
        // no expectations on the mock because there should not be no I/O operations.
        let file = MockFileBackend::new();
        let file = MultiPageCachedFile::<_, _, true> {
            file,
            file_len: AtomicU64::new(Page::SIZE as u64),
            pages: Pages::new([(Page::zeroed(), 0), (Page::zeroed(), 1)]),
        };

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
            .withf(|buf, offset| {
                buf == [0; Page::SIZE] && (*offset == 0 || *offset == Page::SIZE as u64)
            })
            .times(1)
            .returning(|_, _| Ok(()));
        file.expect_read_exact_at()
            .withf(|_, offset| *offset == 2 * Page::SIZE as u64)
            .times(1)
            .returning(|buf, _| {
                buf.fill(1);
                Ok(())
            });

        let file = MultiPageCachedFile::<_, _, true> {
            file,
            file_len: AtomicU64::new(3 * Page::SIZE as u64),
            pages: Pages::new([(Page::zeroed(), 0), (Page::zeroed(), Page::SIZE as u64)]),
        };
        // Mark first page as dirty by writing to it.
        file.write_all_at(&[0], 0).unwrap();

        // Access data outside of the cached pages, which should trigger a write of the old page and
        // a read of the new page.
        let mut read_data = vec![0u8; Page::SIZE];
        file.read_exact_at(&mut read_data, 2 * Page::SIZE as u64)
            .unwrap();
        assert_eq!(read_data, vec![1u8; Page::SIZE]);
    }

    #[test]
    fn access_cached_page_blocks_until_this_page_becomes_available() {
        let file = MockFileBackend::new();
        let file = MultiPageCachedFile::<_, _, true> {
            file,
            file_len: AtomicU64::new(2 * Page::SIZE as u64),
            pages: Pages::new([(Page::zeroed(), 0), (Page::zeroed(), 1)]),
        };
        let file = Arc::new(file);

        let _locked_page = file.change_page(0).unwrap();

        // Try to access the same page from another thread. This should block until the first lock
        // is dropped.
        let handle = std::thread::spawn({
            let file = Arc::clone(&file);
            move || {
                let _locked_page = file.change_page(0).unwrap();
            }
        });

        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!handle.is_finished());

        drop(_locked_page);

        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(handle.is_finished());
    }

    #[test]
    fn access_non_cached_page_blocks_until_any_page_becomes_available() {
        let file = MockFileBackend::new();
        let file = MultiPageCachedFile::<_, _, true> {
            file,
            file_len: AtomicU64::new(3 * Page::SIZE as u64),
            pages: Pages::new([(Page::zeroed(), 0), (Page::zeroed(), 1)]),
        };
        let file = Arc::new(file);

        let _locked_page1 = file.change_page(0).unwrap();
        let _locked_page2 = file.change_page(Page::SIZE as u64).unwrap();

        // Try to access a non-cached page from another thread. This should block until one of the
        // first two locks is dropped.
        let handle = std::thread::spawn({
            let file = Arc::clone(&file);
            move || {
                let _locked_page = file.change_page(2 * Page::SIZE as u64).unwrap();
            }
        });

        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!handle.is_finished());

        drop(_locked_page1);

        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(handle.is_finished());

        drop(_locked_page2);
    }
}
