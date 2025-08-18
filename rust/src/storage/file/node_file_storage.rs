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
    fs::{File, OpenOptions},
    io::{Read, Write},
    marker::PhantomData,
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use zerocopy::{FromBytes, Immutable, IntoBytes, transmute_ref};

use crate::storage::{Error, Storage, file::FileBackend};

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
    node_file: F,
    reuse_list_file: Mutex<CachedReuseListFile>,
    next_idx: AtomicU64,
    _node_type: PhantomData<T>,
}

impl<T, F> NodeFileStorage<T, F>
where
    T: FromBytes + IntoBytes + Immutable + 'static,
    F: FileBackend + 'static,
{
    pub const NODE_STORE_FILE: &'static str = "node_store.bin";
    pub const REUSE_LIST_FILE: &'static str = "reuse_list.bin";
}

#[cfg_attr(test, mockall::automock)]
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
        std::fs::create_dir_all(dir)?;

        let mut file_opts = OpenOptions::new();
        file_opts
            .create(true)
            .truncate(false)
            .read(true)
            .write(true);

        let node_file = F::open(dir.join(Self::NODE_STORE_FILE).as_path(), &file_opts)?;
        let len = node_file.len()?;
        if len % size_of::<T>() as u64 != 0 {
            return Err(Error::DatabaseCorruption);
        }
        let next_idx = AtomicU64::new(len / size_of::<T>() as u64);

        let mut reuse_file = file_opts.open(dir.join(Self::REUSE_LIST_FILE))?;
        let len = reuse_file.metadata()?.len();
        if len % size_of::<u64>() as u64 != 0 {
            return Err(Error::DatabaseCorruption);
        }
        let mut reuse_idxs = Vec::with_capacity(len as usize / size_of::<u64>());
        reuse_file.read_to_end(&mut reuse_idxs)?;
        // we could also transmute here, but since new is only called once, performance is not
        // critical
        let reuse_idxs = reuse_idxs
            .chunks_exact(size_of::<u64>())
            .map(|chunk| {
                chunk.try_into().map(u64::from_be_bytes).unwrap() // slices are guaranteed to be of size 8
            })
            .collect();
        let reuse_file = Mutex::new(CachedReuseListFile {
            file: reuse_file,
            cache: reuse_idxs,
        });

        Ok(Self {
            node_file,
            reuse_list_file: reuse_file,
            next_idx,
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
            .cache
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
        self.reuse_list_file.lock().unwrap().cache.push(idx);
        Ok(())
    }

    fn flush(&self) -> Result<(), Error> {
        self.node_file.flush()?;

        let mut reuse_file = self.reuse_list_file.lock().unwrap();
        let reuse_file = &mut *reuse_file;
        let data: &[u8] = transmute_ref!(reuse_file.cache.as_slice());
        reuse_file.file.write_all(data)?;
        reuse_file.file.set_len(data.len() as u64)?;
        reuse_file.file.flush()?;

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

#[derive(Debug)]
struct CachedReuseListFile {
    file: File,
    cache: Vec<u64>,
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, Permissions},
        io::{Seek, SeekFrom},
        os::unix::fs::PermissionsExt,
    };

    use super::*;
    use crate::storage::{Error, file::SeekFile};

    type TestNode = [u8; 32];

    type NodeFileStorage = super::NodeFileStorage<TestNode, SeekFile>;

    #[test]
    fn open_creates_new_directory_and_files_for_non_existing_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("non_existing_dir");
        let path = path.as_path();

        assert!(NodeFileStorage::open(path).is_ok());

        assert!(fs::exists(path.join(NodeFileStorage::NODE_STORE_FILE)).unwrap());
        assert!(fs::exists(path.join(NodeFileStorage::REUSE_LIST_FILE)).unwrap());
    }

    #[test]
    fn open_creates_new_files_in_empty_directory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        assert!(NodeFileStorage::open(path).is_ok());

        assert!(fs::exists(path.join(NodeFileStorage::NODE_STORE_FILE)).unwrap());
        assert!(fs::exists(path.join(NodeFileStorage::REUSE_LIST_FILE)).unwrap());
    }

    #[test]
    fn open_performs_consistency_checks_on_existing_files() {
        // files have valid sizes
        {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path();
            fs::write(
                path.join(NodeFileStorage::NODE_STORE_FILE),
                [0; size_of::<TestNode>()],
            )
            .unwrap();
            fs::write(
                path.join(NodeFileStorage::REUSE_LIST_FILE),
                [0; size_of::<u64>()],
            )
            .unwrap();

            assert!(NodeFileStorage::open(path).is_ok());
        }
        // node store has invalid size
        {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path();
            fs::write(path.join(NodeFileStorage::NODE_STORE_FILE), [0; 1]).unwrap();

            assert!(matches!(
                NodeFileStorage::open(path),
                Err(Error::DatabaseCorruption)
            ));
        }
        // reuse list has invalid size
        {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path();
            fs::write(path.join(NodeFileStorage::REUSE_LIST_FILE), [0; 1]).unwrap();

            assert!(matches!(
                NodeFileStorage::open(path),
                Err(Error::DatabaseCorruption)
            ));
        }
    }

    #[test]
    fn open_forwards_io_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        fs::set_permissions(path, Permissions::from_mode(0o000)).unwrap();

        assert!(matches!(NodeFileStorage::open(path), Err(Error::Io(_))));

        fs::set_permissions(path, Permissions::from_mode(0o777)).unwrap();
    }

    #[test]
    fn get_reads_data_if_index_in_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        let mut node_file = File::create(path.join(NodeFileStorage::NODE_STORE_FILE)).unwrap();
        node_file.write_all(&[1; size_of::<TestNode>()]).unwrap();
        node_file.write_all(&[2; size_of::<TestNode>()]).unwrap();

        let storage = NodeFileStorage::open(path).unwrap();

        assert_eq!(storage.get(0).unwrap(), [1; 32]);
        assert_eq!(storage.get(1).unwrap(), [2; 32]);
    }

    #[test]
    fn get_returns_error_if_index_out_of_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        let mut node_file = File::create(path.join(NodeFileStorage::NODE_STORE_FILE)).unwrap();
        node_file.write_all(&[1; size_of::<TestNode>()]).unwrap();
        node_file.write_all(&[2; size_of::<TestNode>()]).unwrap();

        let storage = NodeFileStorage::open(path).unwrap();

        assert!(matches!(storage.get(2).unwrap_err(), Error::NotFound));
    }

    #[test]
    fn reserve_returns_last_index_from_reuse_list() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        // write index 1 and 3 in reuse list
        {
            let mut node_file = File::create(path.join(NodeFileStorage::REUSE_LIST_FILE)).unwrap();
            node_file.write_all(&1u64.to_be_bytes()).unwrap();
            node_file.write_all(&3u64.to_be_bytes()).unwrap();
        }

        let storage = NodeFileStorage::open(path).unwrap();

        assert_eq!(storage.reserve(&[0; 32]), 3);
    }

    #[test]
    fn reserve_returns_new_index_if_no_reuse_available() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        // create a single node -> index 0 is used
        fs::write(
            path.join(NodeFileStorage::NODE_STORE_FILE),
            [0; size_of::<TestNode>()],
        )
        .unwrap();

        let storage = NodeFileStorage::open(path).unwrap();

        assert_eq!(storage.reserve(&[0; 32]), 1);
    }

    #[test]
    fn set_writes_data_to_node_file_at_index() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        // prepare file: write some nodes into the file
        {
            let mut node_file = File::create(path.join(NodeFileStorage::NODE_STORE_FILE)).unwrap();
            node_file.write_all(&[1; size_of::<TestNode>()]).unwrap();
            node_file.write_all(&[2; size_of::<TestNode>()]).unwrap();
        }

        // create storage and call set with existing and new nodes
        {
            let storage = NodeFileStorage::open(path).unwrap();
            storage.next_idx.store(5, Ordering::Relaxed);

            // overwrite existing node
            storage.set(0, &[3; 32]).unwrap();
            // add new node at end
            storage.set(2, &[4; 32]).unwrap();
            // add new node after end
            storage.set(4, &[5; 32]).unwrap();
        }

        let mut node_file = File::open(path.join(NodeFileStorage::NODE_STORE_FILE)).unwrap();
        let mut buf = [0; size_of::<TestNode>() * 5];
        node_file.read_exact(&mut buf).unwrap();

        // overwritten node
        assert_eq!(&buf[..size_of::<TestNode>()], &[3; 32]);
        // second node remains unchanged
        assert_eq!(
            &buf[size_of::<TestNode>()..size_of::<TestNode>() * 2],
            &[2; 32]
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
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        let storage = NodeFileStorage::open(path).unwrap();
        assert!(matches!(
            storage.set(123, &[1; 32]).unwrap_err(),
            Error::NotFound
        ));
    }

    #[test]
    fn delete_adds_index_to_reuse_list() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        let mut node_file = File::create(path.join(NodeFileStorage::NODE_STORE_FILE)).unwrap();
        node_file.write_all(&[1; size_of::<TestNode>()]).unwrap();
        node_file.write_all(&[2; size_of::<TestNode>()]).unwrap();

        let storage = NodeFileStorage::open(path).unwrap();
        storage.delete(0).unwrap();
        storage.delete(1).unwrap();
        assert_eq!(storage.reuse_list_file.lock().unwrap().cache, [0, 1]);
    }

    #[test]
    fn delete_returns_error_if_index_out_of_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        let storage = NodeFileStorage::open(path).unwrap();
        assert!(matches!(storage.delete(0).unwrap_err(), Error::NotFound));
    }

    #[test]
    fn flush_writes_reuse_list_to_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        let storage = NodeFileStorage::open(path).unwrap();
        storage.reuse_list_file.lock().unwrap().cache = vec![0, 1];

        storage.flush().unwrap();

        let mut reuse_file = File::open(path.join(NodeFileStorage::REUSE_LIST_FILE)).unwrap();
        assert_eq!(
            reuse_file.metadata().unwrap().len(),
            size_of::<u64>() as u64 * 2
        );
        let mut buf = [0; size_of::<u64>() * 2];
        reuse_file.seek(SeekFrom::Start(0)).unwrap();
        reuse_file.read_exact(&mut buf).unwrap();
        assert_eq!(buf, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
    }
}
