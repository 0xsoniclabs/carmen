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
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};

use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::storage::Error;

/// Metadata stored in the metadata file.
#[derive(Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
#[repr(C)]
pub struct Metadata {
    pub node_count: u64,
    pub reuse_frozen_count: u64,
}

/// A file that stores metadata about the data of the
/// [`NodeFileStorage`](crate::storage::file::NodeFileStorage) store for a certain node type.
#[derive(Debug)]
pub struct MetadataFile {
    file: File,
}

impl MetadataFile {
    /// Creates a new [`MetadataFile`] instance.
    pub fn new(file: File) -> Self {
        Self { file }
    }

    /// Reads the metadata from the file or returns [`Metadata::default`] if the file is empty.
    pub fn read(&self) -> Result<Metadata, Error> {
        let len = self.file.metadata().unwrap().len();
        if len == 0 {
            return Ok(Metadata::default());
        } else if len != size_of::<Metadata>() as u64 {
            return Err(Error::DatabaseCorruption);
        }
        let mut metadata = Metadata::default();
        (&self.file).seek(SeekFrom::Start(0)).unwrap();
        (&self.file).read_exact(metadata.as_mut_bytes())?;
        Ok(metadata)
    }

    /// Writes the metadata to the file.
    pub fn write(&self, metadata: &Metadata) -> Result<(), Error> {
        let mut file = &self.file;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(metadata.as_bytes())?;
        file.sync_all()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::{self, File, OpenOptions};

    use zerocopy::IntoBytes;

    use super::*;
    use crate::utils::test_dir::{Permissions, TestDir};

    #[test]
    fn read_reads_metadata_from_file() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.path().join("metadata");

        let node_count: u64 = 1;
        let reuse_frozen_count: u64 = 2;

        let data = [node_count, reuse_frozen_count];
        fs::write(path.as_path(), data.as_bytes()).unwrap();

        let metadata_file = MetadataFile::new(File::open(path).unwrap());
        let metadata = metadata_file.read().unwrap();
        assert_eq!(metadata.node_count, node_count);
        assert_eq!(metadata.reuse_frozen_count, reuse_frozen_count);
    }

    #[test]
    fn read_returns_zeroed_metadata_for_empty_file() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.path().join("metadata");

        let metadata_file = MetadataFile::new(File::create(path).unwrap());
        let metadata = metadata_file.read().unwrap();
        assert_eq!(metadata.node_count, 0);
        assert_eq!(metadata.reuse_frozen_count, 0);
    }

    #[test]
    fn read_returns_error_for_invalid_file_size() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.path().join("metadata");

        fs::write(&path, [0u8; 10]).unwrap();

        let metadata_file = MetadataFile::new(File::open(path).unwrap());
        let result = metadata_file.read();
        assert!(matches!(result, Err(Error::DatabaseCorruption)));
    }

    #[test]
    fn read_fails_if_file_cannot_be_read() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.path().join("metadata");

        // this is needed so that the file is not empty and actually has to be read
        fs::write(path.as_path(), [0u8; 16]).unwrap();

        let metadata_file = MetadataFile::new(OpenOptions::new().write(true).open(path).unwrap());
        let result = metadata_file.read();
        assert!(matches!(result, Err(Error::Io(_))));
    }

    #[test]
    fn write_writes_metadata_to_file() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.path().join("metadata");

        let node_count: u64 = 1;
        let reuse_frozen_count: u64 = 2;

        {
            let metadata_file = MetadataFile::new(File::create(path.as_path()).unwrap());
            let metadata = Metadata {
                node_count,
                reuse_frozen_count,
            };
            metadata_file.write(&metadata).unwrap();
        }

        let mut file = File::open(path).unwrap();
        let mut data = [0; 2];
        file.read_exact(data.as_mut_bytes()).unwrap();
        assert_eq!(data[0], node_count);
        assert_eq!(data[1], reuse_frozen_count);
    }

    #[test]
    fn write_fails_if_file_cannot_be_written() {
        let tempdir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = tempdir.path().join("metadata");

        File::create(path.as_path()).unwrap();

        let metadata_file = MetadataFile::new(File::open(path).unwrap());
        let metadata = Metadata {
            node_count: 1,
            reuse_frozen_count: 2,
        };
        let result = metadata_file.write(&metadata);
        assert!(matches!(result, Err(Error::Io(_))));
    }

    #[test]
    fn read_returns_what_write_wrote() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.path().join("metadata");

        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path.as_path())
            .unwrap();

        let metadata_file = MetadataFile::new(file);
        let metadata = Metadata {
            node_count: 1,
            reuse_frozen_count: 2,
        };
        metadata_file.write(&metadata).unwrap();
        assert_eq!(metadata_file.read().unwrap(), metadata);
    }
}
