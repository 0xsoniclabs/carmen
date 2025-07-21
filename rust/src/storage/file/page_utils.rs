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
    sync::{Condvar, Mutex, MutexGuard},
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

/// A [`Page`] with a dirty flag.
#[derive(Debug)]
struct PageWithDirtyFlag {
    page: Box<Page>,
    dirty: bool,
}

/// A guard that holds a locked page and allows read access to the data and offset and write access
/// to only the data which automatically marks it as dirty. It also allows marking the page as
/// clean. When dropped, it notifies all waiters on the assignment change condition variable.
pub struct LockedPage<'a> {
    page_guard: MutexGuard<'a, PageWithDirtyFlag>,
    page_index: u64,
    assignment_change: &'a Condvar,
}

impl Drop for LockedPage<'_> {
    fn drop(&mut self) {
        self.assignment_change.notify_all();
    }
}

impl Deref for LockedPage<'_> {
    type Target = Page;

    fn deref(&self) -> &Self::Target {
        &self.page_guard.page
    }
}

impl DerefMut for LockedPage<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.page_guard.dirty = true;
        &mut self.page_guard.page
    }
}

impl LockedPage<'_> {
    /// Returns the page index in the file.
    pub fn page_index(&self) -> u64 {
        self.page_index
    }

    /// Marks the page as clean.
    pub fn mark_clean(&mut self) {
        self.page_guard.dirty = false;
    }
}

/// A guard that provides access to all information to write back the old data of a page that was
/// remapped to a different offset in the file. It provides access to a guard that which must be
/// held while writing back the old data and the old page index in the file.
pub struct OldPageWriteBackGuard<'a, const P: usize> {
    pub page_write_back_guard: MutexGuard<'a, [u64; P]>,
    pub old_page_index: u64,
}

/// A type which is returned when a page is remapped to a different offset in the file.
/// If the old page was dirty, it holds a [`OldPageWriteBackGuard`].
pub struct PageRemap<'a, const P: usize> {
    pub old_page_write_back_guard: Option<OldPageWriteBackGuard<'a, P>>,
}

/// An iterator over all dirty pages, locking each page before yielding it.
/// This iterator holds the lock on the page assignment while iterating, so no page can be remapped
/// while iterating.
struct DirtyPageIter<'a, const P: usize> {
    pages: &'a [Mutex<PageWithDirtyFlag>; P],
    page_indices: MutexGuard<'a, [u64; P]>,
    assignment_change: &'a Condvar,
    index: usize,
}

impl<'a, const P: usize> Iterator for DirtyPageIter<'a, P> {
    type Item = LockedPage<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < P {
            let page = &self.pages[self.index];
            let page_index = self.page_indices[self.index];
            self.index += 1;
            let locked_page = page.lock().unwrap();
            if locked_page.dirty {
                return Some(LockedPage {
                    page_guard: locked_page,
                    page_index,
                    assignment_change: self.assignment_change,
                });
            }
        }
        None
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
    pages: [Mutex<PageWithDirtyFlag>; P],
    page_indices: Mutex<[u64; P]>,
    assignment_change: Condvar,
}

impl<const P: usize> Pages<P> {
    /// Creates a new collection of `P` pages.
    pub fn new(pages: [(Box<Page>, u64); P]) -> Self {
        let (pages, indices): (Vec<_>, Vec<_>) = pages
            .into_iter()
            .map(|(page, index)| (Mutex::new(PageWithDirtyFlag { page, dirty: false }), index))
            .unzip();
        Self {
            pages: pages.try_into().unwrap(), // the vector has length P
            page_indices: Mutex::new(indices.try_into().unwrap()), // the vector has length P
            assignment_change: Condvar::new(),
        }
    }

    /// Returns an iterator over all dirty pages, locking each page before yielding it.
    pub fn iter_dirty_locked<'a>(&'a self) -> impl Iterator<Item = LockedPage<'a>> + 'a {
        DirtyPageIter {
            pages: &self.pages,
            page_indices: self.page_indices.lock().unwrap(),
            assignment_change: &self.assignment_change,
            index: 0,
        }
    }

    /// Returns a page and optionally a [`PageRemapGuard`].
    /// The page can be used to access the data for the specified offset.
    /// If the page remap guard is [`Some`], the page was previously mapped to a different offset in
    /// the file and was remapped. The old data must be written back to the file before reading
    /// the new data into the page. The page remap guard must not be dropped before the data
    /// has been written back.
    pub fn get_page_for_offset(&self, offset: u64) -> (LockedPage<'_>, Option<PageRemap<'_, P>>) {
        let requested_page_index = offset / Page::SIZE as u64;
        let mut page_indices = self.page_indices.lock().unwrap();
        loop {
            // Try to find a page that is mapped to the requested offset.
            for (page_index, page) in page_indices.iter().zip(&self.pages) {
                if *page_index == requested_page_index {
                    let locked_page = page.lock().unwrap();
                    let locked_page = LockedPage {
                        page_guard: locked_page,
                        page_index: *page_index,
                        assignment_change: &self.assignment_change,
                    };
                    return (locked_page, None);
                }
            }

            // Try to find an unlocked page which can be remapped.
            for (page_index, page) in page_indices.iter_mut().zip(&self.pages) {
                if let Ok(locked_page) = page.try_lock() {
                    // We hold the page assignment guard, so we are allowed to change the offset
                    // of the page.
                    let old_page_index = *page_index;
                    *page_index = requested_page_index;
                    let page_remap = PageRemap {
                        old_page_write_back_guard: locked_page.dirty.then_some(
                            OldPageWriteBackGuard {
                                page_write_back_guard: page_indices,
                                old_page_index,
                            },
                        ),
                    };
                    let locked_page = LockedPage {
                        page_guard: locked_page,
                        page_index: requested_page_index,
                        assignment_change: &self.assignment_change,
                    };
                    return (locked_page, Some(page_remap));
                }
            }

            // Wait until a page becomes unlocked.
            page_indices = self.assignment_change.wait(page_indices).unwrap();
        }
    }
}
