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
        file_backend::page_utils::{O_DIRECT, O_SYNC, Page, PageGuard, Pages},
    },
    sync::{
        atomic::{AtomicU64, Ordering},
        thread,
    },
};

/// A wrapper around a [`FileBackend`] that caches multiple pages (4096 bytes) in memory.
/// All read and write operations are performed on these pages, which are flushed to the underlying
/// file when they are dirty and a different page needs to be accessed, or when the file is flushed.
/// If `D` is true, file operations will use direct I/O to bypass the OS page cache.
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
                if offset < padded_len {
                    file.read_exact_at(&mut page, offset)?;
                }
                BTResult::Ok((page, page_index as u64))
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

    fn write_all_at(&self, mut buf: &[u8], mut offset: u64) -> BTResult<(), std::io::Error> {
        let mut locked_pages: [Option<PageGuard>; P] = self.get_pages(buf, offset)?;

        for locked_page in locked_pages.iter_mut().flatten() {
            let start_in_page = offset as usize - locked_page.page_index() as usize * Page::SIZE;
            let end_in_page = cmp::min(start_in_page + buf.len(), Page::SIZE);
            let len = end_in_page - start_in_page;
            locked_page[start_in_page..end_in_page].copy_from_slice(&buf[..len]);
            buf = &buf[len..];
            offset += len as u64;
        }

        self.file_len
            .fetch_max(offset + buf.len() as u64, Ordering::Release);

        Ok(())
    }

    fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> BTResult<(), std::io::Error> {
        if offset + buf.len() as u64 > self.file_len.load(Ordering::Acquire) {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
        }

        let locked_pages: [Option<PageGuard>; P] = self.get_pages(buf, offset)?;

        for locked_page in locked_pages.iter().flatten() {
            let start_in_page = offset as usize - locked_page.page_index() as usize * Page::SIZE;
            let end_in_page = cmp::min(start_in_page + buf.len(), Page::SIZE);
            let len = end_in_page - start_in_page;

            buf[..len].copy_from_slice(&locked_page[start_in_page..end_in_page]);
            buf = &mut buf[len..];
            offset += len as u64;
        }

        Ok(())
    }

    fn flush(&self) -> BTResult<(), std::io::Error> {
        for mut locked_page in self.pages.iter_dirty_locked() {
            let page_index = locked_page.page_index();
            self.store_page(&locked_page, page_index)?;
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
        if self.file_len.load(Ordering::Acquire) < (page_index + 1) * Page::SIZE as u64 {
            page.fill(0);
        } else {
            self.file
                .read_exact_at(page, page_index * Page::SIZE as u64)?;
        }

        Ok(())
    }

    /// Returns a locked page that contains the data for the given offset, or `None` if this would
    /// block.
    ///
    /// This function loads the page containing the given offset into memory, flushing the currently
    /// cached data if dirty. If the offset is already within the currently loaded page, this is
    /// a no-op.
    fn change_page(&'_ self, offset: u64) -> BTResult<Option<PageGuard<'_>>, std::io::Error> {
        self.pages.get_page_for_offset(
            offset,
            |page, page_index| self.load_page(page, page_index),
            |page, page_index| self.store_page(page, page_index),
        )
    }

    /// Acquires locked pages for all pages needed to cover the given buffer at the given offset.
    /// This will spin until all pages are available.
    fn get_pages(
        &self,
        buf: &[u8],
        offset: u64,
    ) -> BTResult<[Option<PageGuard<'_>>; P], std::io::Error> {
        if buf.len() > P * Page::SIZE {
            panic!(
                "buffer size exceeds sum of all caches: {} > {}",
                buf.len(),
                P * Page::SIZE
            );
        }

        let mut locked_pages: [Option<PageGuard>; P] = std::array::from_fn(|_| None);
        'acquire_retry: loop {
            let mut offset = offset;
            let mut buf = buf;
            for i in 0..P {
                let Some(locked_page) = self.change_page(offset)? else {
                    locked_pages = std::array::from_fn(|_| None); // Release all previously acquired locks
                    thread::yield_now();
                    continue 'acquire_retry;
                };

                locked_pages[i] = Some(locked_page);
                let len = cmp::min(buf.len(), Page::SIZE - (offset as usize % Page::SIZE));

                if buf.len() == len {
                    break;
                }
                buf = &buf[len..];
                offset += len as u64;
            }
            break;
        }
        Ok(locked_pages)
    }
}

// Note: The tests for `MultiPageCachedFile<F> as FileBackend` are in `file_backend.rs`.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::storage::file::MockFileBackend;

    #[test]
    fn accessing_cache_data_does_not_trigger_io_operations() {
        let mut file = MockFileBackend::new();
        file.expect_read_exact_at().never();
        file.expect_write_all_at().never();
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
    fn accessing_non_cached_data_triggers_write_of_old_and_read_of_new_page() {
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
    fn accessing_cached_page_blocks_until_this_page_becomes_available() {
        let file = MockFileBackend::new();
        let file = MultiPageCachedFile::<_, _, true> {
            file,
            file_len: AtomicU64::new(2 * Page::SIZE as u64),
            pages: Pages::new([(Page::zeroed(), 0), (Page::zeroed(), 1)]),
        };
        let file = Arc::new(file);

        let _locked_page = file.get_pages(&[], 0).unwrap();

        // Try to access the same page from another thread. This should block until the first lock
        // is dropped.
        let handle = std::thread::spawn({
            let file = Arc::clone(&file);
            move || {
                let _locked_page = file.get_pages(&[], 0).unwrap();
            }
        });

        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!handle.is_finished());

        drop(_locked_page);

        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(handle.is_finished());
    }

    #[test]
    fn accessing_non_cached_page_blocks_until_page_becomes_available() {
        let file = MockFileBackend::new();
        let file = MultiPageCachedFile::<_, _, true> {
            file,
            file_len: AtomicU64::new(3 * Page::SIZE as u64),
            pages: Pages::new([(Page::zeroed(), 0), (Page::zeroed(), 1)]),
        };
        let file = Arc::new(file);

        // Lock the first cached page.
        let locked_page1 = file.get_pages(&[], 0).unwrap();

        // Locking the second cached page should succeed immediately.
        assert!(file.get_pages(&[], Page::SIZE as u64).is_ok());

        // Locking the third page should block until the guard to page one is dropped,
        // since it occupies the same cache slot.
        let handle = std::thread::spawn({
            let file = Arc::clone(&file);
            move || assert!(file.get_pages(&[], 2 * Page::SIZE as u64).is_ok())
        });

        std::thread::sleep(std::time::Duration::from_millis(100));
        // The guard to page one is not yet dropped, so trying to lock page three in the thread
        // should still block.
        assert!(!handle.is_finished());

        drop(locked_page1);

        std::thread::sleep(std::time::Duration::from_millis(100));
        // The guard to page one is dropped, so the thread should have been able to lock page three
        // and finish.
        assert!(handle.is_finished());
    }

    #[test]
    #[should_panic(expected = "buffer size exceeds sum of all caches")]
    fn acquiring_pages_for_buffer_which_is_larger_than_all_pages_panics() {
        let file = MockFileBackend::new();
        let file = MultiPageCachedFile::<_, _, true> {
            file,
            file_len: AtomicU64::new(0),
            pages: Pages::new([(Page::zeroed(), 0), (Page::zeroed(), 1)]),
        };

        file.get_pages(&[0u8; 3 * Page::SIZE], 0).unwrap();
    }
}
