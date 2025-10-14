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
    fs::{self, File},
    io::Read,
    path::Path,
};

use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::storage::Error;

/// An extension trait for types that can be read from and written to files as byte slices.
pub trait FromToFile: Sized + Default + FromBytes + IntoBytes + Immutable {
    /// Creates a new instance by reading from the file at the given path.
    /// If the file does not exist, it is created and initialized with the default value.
    fn read_or_init(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        if !fs::exists(path)? {
            fs::write(path, Self::default().as_bytes())?;
        }

        let mut file = File::open(path)?;
        let len = file.metadata()?.len();
        if len != size_of::<Self>() as u64 {
            return Err(Error::DatabaseCorruption);
        }

        let mut this = Self::default();
        file.read_exact(this.as_mut_bytes())?;

        Ok(this)
    }

    /// Writes self's byte representation to the file at the given path.
    fn write(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        fs::write(path.as_ref(), self.as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use zerocopy::IntoBytes;

    use super::*;
    use crate::utils::test_dir::{Permissions, TestDir};

    #[derive(Debug, Clone, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
    #[repr(C)]
    struct Dummy {
        pub a: u8,
        pub b: u8,
    }

    impl FromToFile for Dummy {}

    #[test]
    fn read_or_init_reads_data_from_file() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("data");

        let a = 1;
        let b = 2;

        let data = [a, b];
        fs::write(path.as_path(), data.as_bytes()).unwrap();

        let data = Dummy::read_or_init(path).unwrap();
        assert_eq!(data, Dummy { a, b });
    }

    #[test]
    fn read_or_init_returns_default_data_if_file_does_not_exist() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("data");

        let data = Dummy::read_or_init(path).unwrap();
        assert_eq!(data, Dummy::default());
    }

    #[test]
    fn read_or_init_returns_error_for_invalid_file_size() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("data");

        fs::write(&path, [0u8; 10]).unwrap();

        let result = Dummy::read_or_init(path);
        assert!(matches!(result, Err(Error::DatabaseCorruption)));
    }

    #[test]
    fn read_or_init_fails_if_file_cannot_be_read() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("data");

        fs::write(&path, []).unwrap();
        tempdir.set_permissions(Permissions::WriteOnly).unwrap();

        let result = Dummy::read_or_init(&path);
        assert!(matches!(result, Err(Error::Io(_))));
    }

    #[test]
    fn write_writes_data_to_file() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("data");

        let a = 1;
        let b = 2;

        let data = Dummy { a, b };
        data.write(&path).unwrap();

        let mut file = File::open(path).unwrap();
        let mut data = [0; 2];
        file.read_exact(data.as_mut_bytes()).unwrap();
        assert_eq!(data, [a, b]);
    }

    #[test]
    fn write_fails_if_file_cannot_be_written() {
        let tempdir = TestDir::try_new(Permissions::ReadOnly).unwrap();
        let path = tempdir.join("data");

        let result = Dummy::default().write(&path);
        assert!(matches!(result, Err(Error::Io(_))));
    }

    #[test]
    fn read_returns_what_write_wrote() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("data");

        let data = Dummy { a: 1, b: 2 };
        data.write(&path).unwrap();
        assert_eq!(Dummy::read_or_init(&path).unwrap(), data);
    }
}
