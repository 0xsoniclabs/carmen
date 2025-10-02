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

/// Metadata stored in the metadata file.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
#[repr(C)]
pub struct Metadata {
    /// The checkpoint number.
    pub checkpoint: u64,
    /// The number of frozen nodes that can not be modified because they are part of a checkpoint.
    pub frozen_nodes: u64,
    /// The number of frozen reuse indices that can not be reused because they are part of a
    /// checkpoint.
    pub frozen_reuse_indices: u64,
}

impl Metadata {
    /// Reads the metadata from the file. It the file does not exist, it is initialized with
    /// [`Metadata::default`].
    pub fn read(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        if !fs::exists(path)? {
            fs::write(path, Self::default().as_bytes())?;
        }
        let mut metadata = Metadata::default();
        let mut file = File::open(path)?;
        match file.read_exact(metadata.as_mut_bytes()) {
            Ok(_) => Ok(metadata),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                Err(Error::DatabaseCorruption)
            }
            Err(e) => Err(Error::Io(e)),
        }
    }

    /// Writes the metadata to the file.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        fs::write(path.as_ref(), self.as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use zerocopy::IntoBytes;

    use super::*;
    use crate::utils::test_dir::{Permissions, TestDir};

    #[test]
    fn read_reads_metadata_from_file() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("metadata");

        let checkpoint: u64 = 1;
        let node_count: u64 = 2;
        let reuse_frozen_count: u64 = 3;

        let data = [checkpoint, node_count, reuse_frozen_count];
        fs::write(path.as_path(), data.as_bytes()).unwrap();

        let metadata = Metadata::read(path).unwrap();
        assert_eq!(metadata.frozen_nodes, node_count);
        assert_eq!(metadata.frozen_reuse_indices, reuse_frozen_count);
    }

    #[test]
    fn read_returns_default_metadata_if_file_does_not_exist() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("metadata");

        let metadata = Metadata::read(path).unwrap();
        assert_eq!(metadata, Metadata::default());
    }

    #[test]
    fn read_returns_error_for_invalid_file_size() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("metadata");

        fs::write(&path, [0u8; 10]).unwrap();

        let result = Metadata::read(path);
        assert!(matches!(result, Err(Error::DatabaseCorruption)));
    }

    #[test]
    fn read_fails_if_file_cannot_be_read() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("metadata");

        // Create the file so make sure Metadata::read tries to open it.
        fs::write(&path, []).unwrap();
        tempdir.set_permissions(Permissions::WriteOnly).unwrap();

        let result = Metadata::read(&path);
        assert!(matches!(result, Err(Error::Io(_))));
    }

    #[test]
    fn write_writes_metadata_to_file() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.join("metadata");

        let checkpoint: u64 = 1;
        let frozen_nodes: u64 = 2;
        let frozen_reuse_indices: u64 = 3;

        let metadata = Metadata {
            checkpoint,
            frozen_nodes,
            frozen_reuse_indices,
        };
        metadata.write(&path).unwrap();

        let mut file = File::open(path).unwrap();
        let mut data = [0; 3];
        file.read_exact(data.as_mut_bytes()).unwrap();
        assert_eq!(data, [checkpoint, frozen_nodes, frozen_reuse_indices]);
    }

    #[test]
    fn write_fails_if_file_cannot_be_written() {
        let tempdir = TestDir::try_new(Permissions::ReadOnly).unwrap();
        let path = tempdir.join("metadata");

        let result = Metadata::default().write(&path);
        assert!(matches!(result, Err(Error::Io(_))));
    }

    #[test]
    fn read_returns_what_write_wrote() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("metadata");

        let metadata = Metadata {
            checkpoint: 1,
            frozen_nodes: 2,
            frozen_reuse_indices: 3,
        };
        metadata.write(&path).unwrap();
        assert_eq!(Metadata::read(&path).unwrap(), metadata);
    }
}
