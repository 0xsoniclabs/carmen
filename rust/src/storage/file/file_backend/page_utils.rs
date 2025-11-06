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
    ops::{Deref, DerefMut},
    sync::TryLockError,
};

use crate::{
    error::BTResult,
    sync::{Mutex, MutexGuard},
};

pub const O_DIRECT: i32 = 0x4000; // from libc::O_DIRECT
pub const O_SYNC: i32 = 1052672; // from libc::O_SYNC

/// A page aligned (4096 bytes) byte buffer.
#[derive(Debug)]
#[repr(align(4096))]
pub struct Page([u8; Self::SIZE]);

impl Page {
    /// The size of a page in bytes, which is typically 4 KiB on most SSDs.
    /// If this size is not equivalent to (a multiple of) the system page size,
    /// page read / writes on files opened with `O_DIRECT` will fail.
    pub const SIZE: usize = 4096;

    /// Creates a new page initialized to zero.
    pub fn zeroed() -> Box<Self> {
        Box::new(Self([0; Self::SIZE]))
    }
}

impl Deref for Page {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Page {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A [`Page`] with its offset and dirty flag.
#[derive(Debug)]
struct PageWithMetadata {
    page: Box<Page>,
    page_index: u64,
    dirty: bool,
}

/// A guard that holds a locked page and allows read and write access to the data. Acquiring write
/// access automatically marks it as dirty. It also allows manually marking the page as clean.
pub struct PageGuard<'a> {
    page_guard: MutexGuard<'a, PageWithMetadata>,
}

impl Deref for PageGuard<'_> {
    type Target = Page;

    fn deref(&self) -> &Self::Target {
        &self.page_guard.page
    }
}

impl DerefMut for PageGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.page_guard.dirty = true;
        &mut self.page_guard.page
    }
}

impl PageGuard<'_> {
    /// Returns the page index in the file.
    pub fn page_index(&self) -> u64 {
        self.page_guard.page_index
    }

    /// Marks the page as clean.
    pub fn mark_clean(&mut self) {
        self.page_guard.dirty = false;
    }
}

/// A collection of `P` page aligned byte buffers ([`Page`]s).
/// The pages are guaranteed to be mapped to non-overlapping regions of a file, and can be accessed
/// concurrently from multiple threads.
///
/// This type offers 2 main operations:
/// - Iterate over dirty pages using [`Self::iter_dirty_locked`], useful for flushing all updates to
///   disk.
/// - Get read and write access to a page for a specific file offset using
///   [`Self::get_page_for_offset`]. This page can then be used for reads and writes corresponding
///   to the offset of the page in the file.
#[derive(Debug)]
pub struct Pages<const P: usize> {
    pages: [Mutex<PageWithMetadata>; P],
}

impl<const P: usize> Pages<P> {
    /// Creates a new collection of `P` pages.
    pub fn new(pages: [(Box<Page>, u64); P]) -> Self {
        let pages: Vec<_> = pages
            .into_iter()
            .map(|(page, page_index)| {
                Mutex::new(PageWithMetadata {
                    page,
                    page_index,
                    dirty: false,
                })
            })
            .collect();
        Self {
            pages: pages.try_into().unwrap(), // the vector has length P
        }
    }

    /// Returns an iterator over all dirty pages, locking each page before yielding it.
    pub fn iter_dirty_locked<'a>(&'a self) -> impl Iterator<Item = PageGuard<'a>> + 'a {
        self.pages.iter().filter_map(|page_mutex| {
            let page_guard = page_mutex.lock().unwrap();
            if page_guard.dirty {
                Some(PageGuard { page_guard })
            } else {
                None
            }
        })
    }

    /// Returns a page which contains the data for the specified offset, or `None` if this would
    /// block.
    /// `load_page` is called to load the page data from disk if the requested page is
    /// not currently cached.
    /// `store_page` is called to write back the page data to disk if the cached page is dirty and
    /// needs to be remapped.
    pub fn get_page_for_offset(
        &self,
        offset: u64,
        load_page: impl Fn(&mut Page, u64) -> BTResult<(), std::io::Error>,
        store_page: impl Fn(&Page, u64) -> BTResult<(), std::io::Error>,
    ) -> BTResult<Option<PageGuard<'_>>, std::io::Error> {
        let requested_page_index = offset / Page::SIZE as u64;
        let index_of_cached_page = requested_page_index as usize % P;
        let mut page_guard = match self.pages[index_of_cached_page].try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(e) => panic!("{e:?}"),
        };

        if page_guard.page_index != requested_page_index {
            if page_guard.dirty {
                store_page(&page_guard.page, page_guard.page_index)?;
                page_guard.dirty = false;
            }
            load_page(&mut page_guard.page, requested_page_index)?;
            page_guard.page_index = requested_page_index;
        }

        Ok(Some(PageGuard { page_guard }))
    }
}
