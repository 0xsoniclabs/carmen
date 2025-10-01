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
    fs::{self, OpenOptions},
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{
        Mutex, RwLock,
        atomic::{AtomicU64, Ordering},
    },
};

use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::storage::{
    Error, Storage,
    file::{FileBackend, metadata_file::Metadata, reuse_list_file::ReuseListFile},
};

/// A file-based storage backend for elements of type `T`.
///
/// Concurrent operations on non-overlapping index ranges are thread safe. Concurrent access to
/// overlapping index ranges is undefined behavior.
#[derive(Debug)]
pub struct NodeFileStorage<T, F>
where
    T: FromBytes + IntoBytes + Immutable + 'static,
    F: FileBackend + 'static,
{
    checkpoint: AtomicU64,
    metadata: RwLock<Metadata>,
    metadata_file: PathBuf,

    node_file: F,
    next_idx: AtomicU64,

    reuse_list_file: Mutex<ReuseListFile>,

    _node_type: PhantomData<T>,
}

impl<T, F> NodeFileStorage<T, F>
where
    T: FromBytes + IntoBytes + Immutable + 'static,
    F: FileBackend + 'static,
{
    pub const NODE_STORE_FILE: &'static str = "node_store.bin";
    pub const REUSE_LIST_FILE: &'static str = "reuse_list.bin";
    pub const METADATA_FILE: &'static str = "metadata.bin";
}

impl<T, F> Storage for NodeFileStorage<T, F>
where
    T: FromBytes + IntoBytes + Immutable + 'static,
    F: FileBackend + 'static,
{
    type Id = u64;
    type Item = T;

    /// Creates all files for a file-based node storage in the specified directory.
    /// If the directory does not exist, it will be created.
    /// If the files do not exist, they will be created.
    /// If the files exist, they will be opened and their data verified.
    fn open(dir: &Path) -> Result<Self, Error> {
        fs::create_dir_all(dir)?;

        let metadata = Metadata::read(dir.join(Self::METADATA_FILE))?;

        let mut file_opts = OpenOptions::new();
        file_opts
            .create(true)
            .truncate(false)
            .read(true)
            .write(true);

        let reuse_file = file_opts.open(dir.join(Self::REUSE_LIST_FILE))?;
        let reuse_list_file = ReuseListFile::new(reuse_file, metadata.frozen_reuse_indices)?;
        if reuse_list_file
            .as_slice()
            .iter()
            .any(|&idx| idx >= metadata.frozen_nodes)
        {
            return Err(Error::DatabaseCorruption);
        }

        let node_file = F::open(dir.join(Self::NODE_STORE_FILE).as_path(), file_opts)?;
        let len = node_file.len()?;
        if len < metadata.frozen_nodes * size_of::<T>() as u64 {
            return Err(Error::DatabaseCorruption);
        }

        Ok(Self {
            checkpoint: AtomicU64::new(metadata.checkpoint),
            metadata: RwLock::new(metadata),
            metadata_file: dir.join(Self::METADATA_FILE),

            node_file,
            next_idx: AtomicU64::new(metadata.frozen_nodes),

            reuse_list_file: Mutex::new(reuse_list_file),

            _node_type: PhantomData,
        })
    }

    fn get(&self, idx: Self::Id) -> Result<Self::Item, Error> {
        let offset = idx * size_of::<Self::Item>() as u64;
        if self.node_file.len()? < offset + size_of::<T>() as u64 {
            return Err(Error::NotFound);
        }
        // this is hopefully optimized away
        let mut node = T::new_zeroed();
        self.node_file.read_exact_at(node.as_mut_bytes(), offset)?;
        Ok(node)
    }

    fn reserve(&self, _node: &Self::Item) -> Self::Id {
        self.reuse_list_file
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| self.next_idx.fetch_add(1, Ordering::Relaxed))
    }

    fn set(&self, idx: Self::Id, node: &Self::Item) -> Result<(), Error> {
        if idx >= self.next_idx.load(Ordering::Relaxed) {
            return Err(Error::NotFound);
        }
        let offset = idx * size_of::<Self::Item>() as u64;
        self.node_file.write_all_at(node.as_bytes(), offset)?;
        Ok(())
    }

    fn delete(&self, idx: Self::Id) -> Result<(), Error> {
        if idx >= self.next_idx.load(Ordering::Relaxed) {
            return Err(Error::NotFound);
        }
        self.reuse_list_file.lock().unwrap().push(idx);
        Ok(())
    }

    fn flush(&self) -> Result<(), Error> {
        self.node_file.flush()?;

        let mut reuse_file = self.reuse_list_file.lock().unwrap();
        reuse_file.write()?;
        let reuse_frozen_count = reuse_file.count();
        reuse_file.set_frozen_count(reuse_frozen_count);
        drop(reuse_file);

        let mut metadata = self.metadata.write().unwrap();
        *metadata = Metadata {
            checkpoint: self.checkpoint.load(Ordering::Acquire),
            frozen_nodes: self.next_idx.load(Ordering::Acquire),
            frozen_reuse_indices: reuse_frozen_count as u64,
        };
        metadata.write(&self.metadata_file)?;

        Ok(())
    }
}

impl<T, F> Drop for NodeFileStorage<T, F>
where
    T: FromBytes + IntoBytes + Immutable + 'static,
    F: FileBackend + 'static,
{
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, File},
        io::{Read, Seek, SeekFrom},
    };

    use super::*;
    use crate::{
        storage::{Error, file::SeekFile},
        utils::test_dir::{Permissions, TestDir},
    };

    type TestNode = [u8; 32];

    type NodeFileStorage = super::NodeFileStorage<TestNode, SeekFile>;

    #[test]
    fn open_creates_new_directory_and_files_for_non_existing_path() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let path = dir.join("non_existing_dir");
        let path = path.as_path();

        assert!(NodeFileStorage::open(path).is_ok());

        assert!(fs::exists(path.join(NodeFileStorage::NODE_STORE_FILE)).unwrap());
        assert!(fs::exists(path.join(NodeFileStorage::REUSE_LIST_FILE)).unwrap());
        assert!(fs::exists(path.join(NodeFileStorage::METADATA_FILE)).unwrap());
    }

    #[test]
    fn open_creates_new_files_in_empty_directory() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        assert!(NodeFileStorage::open(&dir).is_ok());

        assert!(fs::exists(dir.join(NodeFileStorage::NODE_STORE_FILE)).unwrap());
        assert!(fs::exists(dir.join(NodeFileStorage::REUSE_LIST_FILE)).unwrap());
        assert!(fs::exists(dir.join(NodeFileStorage::METADATA_FILE)).unwrap());
    }

    #[test]
    fn open_performs_consistency_checks_on_existing_files() {
        // files have valid sizes
        {
            let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
            write_metadata(&dir, 0, 1, 1);
            write_reuse_list(&dir, &[0]);
            write_nodes(&dir, &[[0; 32]]);

            assert!(NodeFileStorage::open(&dir).is_ok());
        }
        // metadata contains larger node count that node file sizes allows
        {
            let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
            write_metadata(&dir, 0, 2, 0);
            write_reuse_list(&dir, &[0]);
            write_nodes(&dir, &[[0; 32]]);

            assert!(matches!(
                NodeFileStorage::open(&dir),
                Err(Error::DatabaseCorruption)
            ));
        }
        // metadata contains larger frozen count that reuse list file sizes allows
        {
            let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
            write_metadata(&dir, 0, 0, 2);
            write_reuse_list(&dir, &[0]);
            write_nodes(&dir, &[[0; 32]]);

            assert!(matches!(
                NodeFileStorage::open(&dir),
                Err(Error::DatabaseCorruption)
            ));
        }
        // reuse list contains indices which are larger than node count in metadata
        {
            let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
            write_metadata(&dir, 0, 0, 0);
            write_reuse_list(&dir, &[1]);
            write_nodes(&dir, &[]);

            assert!(matches!(
                NodeFileStorage::open(&dir),
                Err(Error::DatabaseCorruption)
            ));
        }
    }

    #[test]
    fn open_forwards_io_errors() {
        let dir = TestDir::try_new(Permissions::ReadOnly).unwrap();

        assert!(matches!(NodeFileStorage::open(&dir), Err(Error::Io(_))));
    }

    #[test]
    fn get_reads_data_if_index_in_bounds() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        write_metadata(&dir, 0, 2, 0);
        write_reuse_list(&dir, &[]);
        write_nodes(&dir, &[[0; 32], [1; 32]]);

        let storage = NodeFileStorage::open(&dir).unwrap();

        assert_eq!(storage.get(0).unwrap(), [0; 32]);
        assert_eq!(storage.get(1).unwrap(), [1; 32]);
        assert!(matches!(storage.get(2).unwrap_err(), Error::NotFound));
    }

    #[test]
    fn reserve_returns_last_index_from_reuse_list() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        write_metadata(&dir, 0, 3, 0);
        write_reuse_list(&dir, &[0, 2]);
        write_nodes(&dir, &[[0; 32], [1; 32], [2; 32]]);

        let storage = NodeFileStorage::open(&dir).unwrap();

        assert_eq!(storage.reserve(&[0; 32]), 2); // last index in reuse list
        assert_eq!(storage.reserve(&[0; 32]), 0); // next index in reuse list
        assert_eq!(storage.reserve(&[0; 32]), 3); // new index
    }

    #[test]
    fn reserve_returns_new_index_if_no_reuse_available() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        // create a single node -> index 0 is used
        write_metadata(&dir, 0, 1, 0);
        write_reuse_list(&dir, &[]);
        write_nodes(&dir, &[[0; 32]]);

        let storage = NodeFileStorage::open(&dir).unwrap();

        assert_eq!(storage.reserve(&[0; 32]), 1);
    }

    #[test]
    fn set_writes_data_to_node_file_at_index() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        // prepare file: write some nodes into the file
        write_metadata(&dir, 0, 2, 0);
        write_reuse_list(&dir, &[]);
        write_nodes(&dir, &[[0; 32], [1; 32]]);

        // create storage and call set with existing and new nodes
        {
            let storage = NodeFileStorage::open(&dir).unwrap();
            storage.next_idx.store(5, Ordering::Relaxed);

            // overwrite existing node
            storage.set(0, &[3; 32]).unwrap();
            // add new node at end
            storage.set(2, &[4; 32]).unwrap();
            // add new node after end
            storage.set(4, &[5; 32]).unwrap();
        }

        let mut node_file = File::open(dir.join(NodeFileStorage::NODE_STORE_FILE)).unwrap();
        let mut buf = [0; size_of::<TestNode>() * 5];
        node_file.read_exact(&mut buf).unwrap();

        // overwritten node
        assert_eq!(&buf[..size_of::<TestNode>()], &[3; 32]);
        // second node remains unchanged
        assert_eq!(
            &buf[size_of::<TestNode>()..size_of::<TestNode>() * 2],
            &[1; 32]
        );
        // new node at index 2
        assert_eq!(
            &buf[size_of::<TestNode>() * 2..size_of::<TestNode>() * 3],
            &[4; 32]
        );
        // new node at index 4
        assert_eq!(&buf[size_of::<TestNode>() * 4..], &[5; 32]);
    }

    #[test]
    fn set_returns_error_if_index_out_of_bounds() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let storage = NodeFileStorage::open(&dir).unwrap();
        assert!(matches!(
            storage.set(123, &[0; 32]).unwrap_err(),
            Error::NotFound
        ));
    }

    #[test]
    fn delete_adds_index_to_reuse_list() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        write_metadata(&dir, 0, 2, 0);
        write_reuse_list(&dir, &[]);
        write_nodes(&dir, &[[0; 32], [1; 32]]);

        let storage = NodeFileStorage::open(&dir).unwrap();
        storage.delete(0).unwrap();
        storage.delete(1).unwrap();
        let mut reuse_list_file = storage.reuse_list_file.lock().unwrap();
        assert_eq!(reuse_list_file.pop(), Some(1));
        assert_eq!(reuse_list_file.pop(), Some(0));
        assert_eq!(reuse_list_file.pop(), None);
    }

    #[test]
    fn delete_returns_error_if_index_out_of_bounds() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let storage = NodeFileStorage::open(&dir).unwrap();
        assert!(matches!(storage.delete(0).unwrap_err(), Error::NotFound));
    }

    #[test]
    fn flush_writes_reuse_list_to_file_and_updates_frozen_count() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let storage = NodeFileStorage::open(&dir).unwrap();
        {
            let mut reuse_list_file = storage.reuse_list_file.lock().unwrap();
            reuse_list_file.push(0);
            reuse_list_file.push(1);
        }

        storage.flush().unwrap();

        // all current elements should be frozen
        assert!(storage.reuse_list_file.lock().unwrap().pop().is_none());

        let mut reuse_file = File::open(dir.join(NodeFileStorage::REUSE_LIST_FILE)).unwrap();
        assert_eq!(
            reuse_file.metadata().unwrap().len(),
            size_of::<u64>() as u64 * 2
        );
        let mut buf = [0u64; 2];
        reuse_file.seek(SeekFrom::Start(0)).unwrap();
        reuse_file.read_exact(buf.as_mut_bytes()).unwrap();
        assert_eq!(buf, [0, 1]);
    }

    impl<T, F> super::NodeFileStorage<T, F>
    where
        T: FromBytes + IntoBytes + Immutable + 'static,
        F: FileBackend + 'static,
    {
        /// Creates all files for a file-based node storage in the specified directory
        /// and populates them with the provided nodes and reusable indices.
        /// Both the nodes and the reusable indices are frozen, as [`NodeFileStorage::open`] only
        /// considers frozen data to exist.
        pub fn create_files_for_nodes(path: impl AsRef<Path>, nodes: &[T]) -> Result<(), Error> {
            let path = path.as_ref();

            fs::create_dir_all(path)?;

            Metadata {
                checkpoint: 0,
                frozen_nodes: nodes.len() as u64,
                frozen_reuse_indices: 0,
            }
            .write(path.join(Self::METADATA_FILE))?;
            fs::write(path.join(Self::REUSE_LIST_FILE), [])?;
            fs::write(path.join(Self::NODE_STORE_FILE), nodes.as_bytes())?;

            Ok(())
        }
    }

    fn write_metadata(
        dir: impl AsRef<Path>,
        checkpoint: u64,
        frozen_nodes: u64,
        frozen_reuse_indices: u64,
    ) {
        Metadata {
            checkpoint,
            frozen_nodes,
            frozen_reuse_indices,
        }
        .write(dir.as_ref().join(NodeFileStorage::METADATA_FILE))
        .unwrap();
    }

    fn write_reuse_list(dir: impl AsRef<Path>, indices: &[u64]) {
        fs::write(
            dir.as_ref().join(NodeFileStorage::REUSE_LIST_FILE),
            indices.as_bytes(),
        )
        .unwrap();
    }

    fn write_nodes(dir: impl AsRef<Path>, nodes: &[TestNode]) {
        fs::write(
            dir.as_ref().join(NodeFileStorage::NODE_STORE_FILE),
            nodes.as_bytes(),
        )
        .unwrap();
    }
}
