// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{fs, io::Read, path::Path};

use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::{
    error::BTResult,
    storage::{DbMode, Error},
};

/// An extension trait for types that can be read from and written to files as byte slices.
pub trait FromToFile: Sized + Default + FromBytes + IntoBytes + Immutable {
    /// Creates a new instance by reading from the file at the given path.
    /// If the file does not exist and the `db_mode` has write access, it is created and initialized
    /// with the default value.
    fn read_or_init(path: impl AsRef<Path>, db_mode: DbMode) -> BTResult<Self, Error> {
        let path = path.as_ref();
        let mut file = db_mode.to_open_options().open(path)?;
        let len = file.metadata()?.len();
        if len == 0 && db_mode.read_only() {
            return Err(Error::DatabaseCorruption.into());
        } else if len == 0 {
            // File was just created
            fs::write(path, Self::default().as_bytes())?;
        } else if len != std::mem::size_of::<Self>() as u64 {
            // File existed but has invalid content
            return Err(Error::DatabaseCorruption.into());
        }

        let mut this = Self::default();
        file.read_exact(this.as_mut_bytes())?;

        Ok(this)
    }

    /// Writes self's byte representation to the file at the given path.
    fn write(&self, path: impl AsRef<Path>) -> BTResult<(), Error> {
        fs::write(path.as_ref(), self.as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use zerocopy::IntoBytes;

    use super::*;
    use crate::{
        error::BTError,
        storage::all_db_modes,
        utils::test_dir::{Permissions, TestDir},
    };

    #[derive(Debug, Clone, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
    #[repr(C)]
    struct Dummy {
        pub a: u8,
        pub b: u8,
    }

    impl FromToFile for Dummy {}

    #[rstest_reuse::apply(all_db_modes)]
    fn read_or_init_reads_data_from_file(#[case] db_mode: DbMode) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("data");

        let a = 1;
        let b = 2;

        let data = [a, b];
        fs::write(path.as_path(), data.as_bytes()).unwrap();

        let data = Dummy::read_or_init(path, db_mode).unwrap();
        assert_eq!(data, Dummy { a, b });
    }

    #[test]
    fn read_or_init_returns_default_data_if_file_does_not_exist_if_db_mode_has_write_access() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("data");

        let data = Dummy::read_or_init(path, DbMode::ReadWrite).unwrap();
        assert_eq!(data, Dummy::default());
    }

    #[test]
    fn read_or_init_fails_if_file_does_not_exist_in_read_only_mode() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("data");

        let result = Dummy::read_or_init(path, DbMode::ReadOnly);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::Io(_))
        ));
    }

    #[rstest_reuse::apply(all_db_modes)]
    fn read_or_init_returns_error_for_invalid_file_size(#[case] db_mode: DbMode) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("data");

        fs::write(&path, [0u8; 10]).unwrap();

        let result = Dummy::read_or_init(path, db_mode);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::DatabaseCorruption)
        ));
    }

    #[rstest_reuse::apply(all_db_modes)]
    fn read_or_init_fails_if_file_cannot_be_read(#[case] db_mode: DbMode) {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("data");

        fs::write(&path, []).unwrap();
        tempdir.set_permissions(Permissions::WriteOnly).unwrap();

        let result = Dummy::read_or_init(&path, db_mode);
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::Io(_))
        ));
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
        assert!(matches!(
            result.map_err(BTError::into_inner),
            Err(Error::Io(_))
        ));
    }

    #[rstest_reuse::apply(all_db_modes)]
    fn read_returns_what_write_wrote(#[case] db_mode: DbMode) {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("data");

        let data = Dummy { a: 1, b: 2 };
        data.write(&path).unwrap();
        assert_eq!(Dummy::read_or_init(&path, db_mode).unwrap(), data);
    }
}
