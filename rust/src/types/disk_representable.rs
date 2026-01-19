// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{borrow::Cow, mem};

use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::{error::BTResult, storage};

/// A trait for types that can be represented as raw bytes on disk.
/// The disk representation does not need to be the same as the in-memory representation.
/// There is a blanket implementation for types implementing [`FromBytes`], [`IntoBytes`] and
/// [`Immutable`].
pub trait DiskRepresentable {
    /// The size of the disk representation in bytes.
    const DISK_REPR_SIZE: usize;

    /// Constructs the value from its disk representation. `read_into_buffer` is expected to fill
    /// the provided buffer with the raw bytes read from disk. It is up to the implementation to
    /// convert the buffer into the value.
    fn from_disk_repr(
        read_into_buffer: impl FnOnce(&mut [u8]) -> BTResult<(), storage::Error>,
    ) -> BTResult<Self, storage::Error>
    where
        Self: Sized;

    /// Returns the disk representation of the value as a byte slice (borrowed from self or owned
    /// depending on whether the disk representations is the same as the in-memory representation or
    /// needed to be serialized).
    fn to_disk_repr(&'_ self) -> Cow<'_, [u8]>;
}

impl<T: FromBytes + IntoBytes + Immutable> DiskRepresentable for T {
    const DISK_REPR_SIZE: usize = mem::size_of::<T>();

    fn from_disk_repr(
        read_into_buffer: impl FnOnce(&mut [u8]) -> BTResult<(), storage::Error>,
    ) -> BTResult<Self, storage::Error> {
        let mut value = T::new_zeroed();
        read_into_buffer(value.as_mut_bytes())?;
        Ok(value)
    }

    fn to_disk_repr(&'_ self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
    struct ZerocopyableType([u8; 32]);

    #[test]
    fn from_disk_repr_for_zerocopy_types_calls_read_into_buffer_with_bytes_representation_and_returns_reinterpreted_data()
     {
        let value = ZerocopyableType::from_disk_repr(|buf| {
            buf.fill(1u8);
            Ok(())
        })
        .unwrap();

        assert_eq!(value, ZerocopyableType([1u8; 32]));
    }

    #[test]
    fn to_disk_repr_for_zerocopy_types_returns_borrowed_bytes_representation() {
        let value = ZerocopyableType([1; 32]);
        let disk_repr = value.to_disk_repr();
        assert_eq!(disk_repr, Cow::Borrowed(&[1u8; 32]));
    }

    #[test]
    fn size_for_zerocopy_types_returns_size_of_type() {
        assert_eq!(ZerocopyableType::DISK_REPR_SIZE, 32);
    }
}
