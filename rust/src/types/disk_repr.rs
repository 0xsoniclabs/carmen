// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use zerocopy::{FromBytes, Immutable, IntoBytes};

/// A trait for types that can be represented as raw bytes on disk.
/// The disk representation does not need to be the same as the in-memory representation.
/// There is a blanket implementation for types implementing [`FromBytes`], [`IntoBytes`] and
/// [`Immutable`].
pub trait DiskRepresentable {
    /// Constructs the value from its disk representation. `read_into_buffer` is expected to fill
    /// the provided buffer with the raw bytes read from disk. It is up to the implementation to
    /// convert the buffer into the value.
    fn from_disk_repr<E>(
        read_into_buffer: impl FnOnce(&mut [u8]) -> Result<(), E>,
    ) -> Result<Self, E>
    where
        Self: Sized;

    /// Returns the disk representation of the value as a byte slice.
    fn to_disk_repr(&self) -> &[u8];
}

impl<T: FromBytes + IntoBytes + Immutable> DiskRepresentable for T {
    fn from_disk_repr<E>(
        read_into_buffer: impl FnOnce(&mut [u8]) -> Result<(), E>,
    ) -> Result<Self, E> {
        let mut value = T::new_zeroed();
        read_into_buffer(value.as_mut_bytes())?;
        Ok(value)
    }

    fn to_disk_repr(&self) -> &[u8] {
        self.as_bytes()
    }
}
