// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::ops::{Deref, DerefMut};

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
