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
};

use crate::{
    error::BTResult,
    storage::{
        CheckpointParticipant, Error, Storage,
        file::{
            FileBackend, FromToFile,
            node_file_storage::{
                node_file_storage_metadata::{
                    NodeFileStorageCheckpointMetadata, NodeFileStorageMetadata,
                },
                reuse_list_file::ReuseListFile,
            },
        },
    },
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    types::DiskRepresentable,
};
mod node_file_storage_metadata;
mod reuse_list_file;

/// A file-based storage backend for elements of type `T`.
///
/// Concurrent operations on non-overlapping index ranges are thread safe. Concurrent access to
/// overlapping index ranges is undefined behavior.
#[derive(Debug)]
pub struct NodeFileStorage<T, F> {
    checkpoint: AtomicU64,
    metadata_path: PathBuf,
    committed_metadata_path: PathBuf,
    prepared_metadata_path: PathBuf,

    node_file: F,
    next_idx: AtomicU64,
    frozen_nodes: AtomicU64,

    reuse_list_file: Mutex<ReuseListFile>,

    _node_type: PhantomData<T>,
}

impl<T, F> NodeFileStorage<T, F> {
    pub const NODE_STORE_FILE: &'static str = "node_store.bin";
    pub const REUSE_LIST_FILE: &'static str = "reuse_list.bin";
    pub const METADATA_FILE: &'static str = "metadata.bin";
    pub const COMMITTED_METADATA_FILE: &'static str = "committed_metadata.bin";
    pub const PREPARED_METADATA_FILE: &'static str = "prepared_metadata.bin";
}

impl<T, F> Storage for NodeFileStorage<T, F>
where
    T: DiskRepresentable + Send + Sync,
    F: FileBackend,
{
    type Id = u64;
    type Item = T;

    /// Creates all files for a file-based node storage in the specified directory.
    /// If the directory does not exist, it will be created.
    /// If the files do not exist, they will be created.
    /// If the files exist, they will be opened and their data verified.
    fn open(dir: &Path) -> BTResult<Self, Error> {
        fs::create_dir_all(dir)?;

        let metadata = NodeFileStorageMetadata::read_or_init(dir.join(Self::METADATA_FILE))?;
        if metadata.frozen_nodes > metadata.nodes
            || metadata.frozen_reuse_indices > metadata.reuse_indices
        {
            return Err(Error::DatabaseCorruption.into());
        }

        let reuse_list_file = ReuseListFile::open(
            dir.join(Self::REUSE_LIST_FILE),
            metadata.reuse_indices,
            metadata.frozen_reuse_indices,
        )?;
        if reuse_list_file
            .frozen_indices()
            .any(|&idx| idx >= metadata.frozen_nodes)
        {
            return Err(Error::DatabaseCorruption.into());
        }
        if reuse_list_file
            .reusable_indices()
            .any(|&idx| idx < metadata.frozen_nodes || idx >= metadata.nodes)
        {
            return Err(Error::DatabaseCorruption.into());
        }

        let mut file_opts = OpenOptions::new();
        file_opts
            .create(true)
            .truncate(false)
            .read(true)
            .write(true);

        let node_file = F::open(
            dir.join(Self::NODE_STORE_FILE).as_path(),
            file_opts,
            T::size(),
        )?;
        let len = node_file.len()?;
        if len < metadata.nodes * T::size() as u64 {
            return Err(Error::DatabaseCorruption.into());
        }

        Ok(Self {
            checkpoint: AtomicU64::new(metadata.last_checkpoint),
            metadata_path: dir.join(Self::METADATA_FILE),
            committed_metadata_path: dir.join(Self::COMMITTED_METADATA_FILE),
            prepared_metadata_path: dir.join(Self::PREPARED_METADATA_FILE),

            node_file,
            next_idx: AtomicU64::new(metadata.nodes),
            frozen_nodes: AtomicU64::new(metadata.frozen_nodes),

            reuse_list_file: Mutex::new(reuse_list_file),

            _node_type: PhantomData,
        })
    }

    fn get(&self, idx: Self::Id) -> BTResult<Self::Item, Error> {
        let offset = idx * T::size() as u64;
        if idx >= self.next_idx.load(Ordering::Relaxed)
            || self.node_file.len()? < offset + T::size() as u64
            || self.reuse_list_file.lock().unwrap().contains(idx)
        {
            return Err(Error::NotFound.into());
        }
        let node = T::from_disk_repr(|buf| self.node_file.read_exact_at(buf, offset))?;
        Ok(node)
    }

    fn reserve(&self, _node: &Self::Item) -> Self::Id {
        self.reuse_list_file
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| self.next_idx.fetch_add(1, Ordering::Relaxed))
    }

    fn set(&self, idx: Self::Id, node: &Self::Item) -> BTResult<(), Error> {
        if idx >= self.next_idx.load(Ordering::Relaxed)
            || self.reuse_list_file.lock().unwrap().contains(idx)
        {
            return Err(Error::NotFound.into());
        } else if idx < self.frozen_nodes.load(Ordering::Relaxed) {
            return Err(Error::Frozen.into());
        }
        let offset = idx * T::size() as u64;
        self.node_file
            .write_all_at(node.to_disk_repr().as_ref(), offset)?;
        Ok(())
    }

    fn delete(&self, idx: Self::Id) -> BTResult<(), Error> {
        if idx >= self.next_idx.load(Ordering::Relaxed)
            || self.reuse_list_file.lock().unwrap().contains(idx)
        {
            Err(Error::NotFound.into())
        } else if idx < self.frozen_nodes.load(Ordering::Relaxed) {
            Err(Error::Frozen.into())
        } else {
            self.reuse_list_file.lock().unwrap().push(idx);
            Ok(())
        }
    }

    fn close(self) -> BTResult<(), Error> {
        // Zero pad the file to ensure its size matches the number of nodes in case nodes were
        // reserved but not written out because of transformations. To make sure no existing
        // nodes are overwritten, we write at the current next_idx position.
        self.node_file.write_all_at(
            &vec![0; T::size()],
            self.next_idx.load(Ordering::Relaxed) * T::size() as u64,
        )?;
        self.node_file.flush()?;
        let mut reuse_list_file = self.reuse_list_file.lock().unwrap();
        reuse_list_file.write_to_disk()?;

        let metadata = NodeFileStorageMetadata {
            last_checkpoint: self.checkpoint.load(Ordering::Acquire),
            nodes: self.next_idx.load(Ordering::Acquire),
            frozen_nodes: self.frozen_nodes.load(Ordering::Acquire),
            reuse_indices: reuse_list_file.count() as u64,
            frozen_reuse_indices: reuse_list_file.frozen_count() as u64,
        };
        metadata.write(&self.metadata_path)?;
        Ok(())
    }
}

impl<T, F> CheckpointParticipant for NodeFileStorage<T, F>
where
    T: DiskRepresentable,
    F: FileBackend,
{
    fn ensure(&self, checkpoint: u64) -> BTResult<(), Error> {
        if checkpoint != self.checkpoint.load(Ordering::Relaxed) {
            return Err(Error::Checkpoint.into());
        }
        Ok(())
    }

    fn prepare(&self, checkpoint: u64) -> BTResult<(), Error> {
        if checkpoint != self.checkpoint.load(Ordering::Relaxed) + 1 {
            return Err(Error::Checkpoint.into());
        }

        // Zero pad the file to ensure its size matches the number of nodes in case nodes were
        // reserved but not written out because of transformations. To make sure no existing
        // nodes are overwritten, we write at the current next_idx position.
        self.node_file.write_all_at(
            &vec![0; T::size()],
            self.next_idx.load(Ordering::Relaxed) * T::size() as u64,
        )?;
        self.node_file.flush()?;
        let mut reuse_list_file = self.reuse_list_file.lock().unwrap();
        reuse_list_file.write_to_disk()?;
        reuse_list_file.freeze_temporarily();
        let frozen_reuse_indices = reuse_list_file.frozen_count() as u64;

        let frozen_nodes = self.next_idx.load(Ordering::Relaxed);
        let prepared_metadata = NodeFileStorageCheckpointMetadata {
            checkpoint,
            frozen_nodes,
            frozen_reuse_indices,
        };
        self.frozen_nodes.store(frozen_nodes, Ordering::Relaxed);
        prepared_metadata.write(&self.prepared_metadata_path)?;
        Ok(())
    }

    fn commit(&self, checkpoint: u64) -> BTResult<(), Error> {
        if checkpoint != self.checkpoint.load(Ordering::Relaxed) + 1 {
            return Err(Error::Checkpoint.into());
        }
        let prepared_metadata =
            NodeFileStorageCheckpointMetadata::read_or_init(&self.prepared_metadata_path)?;
        if checkpoint != prepared_metadata.checkpoint {
            return Err(Error::Checkpoint.into());
        }
        self.reuse_list_file.lock().unwrap().freeze_permanently();
        fs::rename(&self.prepared_metadata_path, &self.committed_metadata_path)?;
        self.checkpoint.store(checkpoint, Ordering::Relaxed);
        Ok(())
    }

    fn abort(&self, checkpoint: u64) -> BTResult<(), Error> {
        if checkpoint != self.checkpoint.load(Ordering::Relaxed) + 1 {
            return Err(Error::Checkpoint.into());
        }
        fs::remove_file(&self.prepared_metadata_path)?;
        let committed_metadata =
            NodeFileStorageCheckpointMetadata::read_or_init(&self.committed_metadata_path)?;
        let mut reuse_list_file = self.reuse_list_file.lock().unwrap();
        reuse_list_file.unfreeze_temp();
        if reuse_list_file.frozen_count() != committed_metadata.frozen_reuse_indices as usize {
            return Err(Error::Checkpoint.into());
        }
        self.frozen_nodes
            .store(committed_metadata.frozen_nodes, Ordering::Relaxed);
        Ok(())
    }

    fn restore(path: &Path, checkpoint: u64) -> BTResult<(), Error> {
        let committed_metadata = NodeFileStorageCheckpointMetadata::read_or_init(
            path.join(Self::COMMITTED_METADATA_FILE),
        )?;
        if checkpoint != committed_metadata.checkpoint {
            return Err(Error::Checkpoint.into());
        }
        let metadata = NodeFileStorageMetadata {
            last_checkpoint: committed_metadata.checkpoint,
            nodes: committed_metadata.frozen_nodes,
            frozen_nodes: committed_metadata.frozen_nodes,
            reuse_indices: committed_metadata.frozen_reuse_indices,
            frozen_reuse_indices: committed_metadata.frozen_reuse_indices,
        };
        metadata.write(path.join(Self::METADATA_FILE))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, File},
        io::Read,
    };

    use zerocopy::IntoBytes;

    use super::*;
    use crate::{
        error::BTError,
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
            write_metadata(&dir, 0, 1, 0, 1, 0);
            write_reuse_list(&dir, &[0]);
            write_nodes(&dir, &[[0; 32]]);

            assert!(NodeFileStorage::open(&dir).is_ok());
        }
        // frozen node count larger than total node count
        {
            let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
            write_metadata(&dir, 0, 0, 1, 1, 0);

            assert!(matches!(
                NodeFileStorage::open(&dir).map_err(BTError::into_inner),
                Err(Error::DatabaseCorruption)
            ));
        }
        // frozen reuse indices count larger than total reuse indices count
        {
            let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
            write_metadata(&dir, 0, 1, 0, 0, 1);

            assert!(matches!(
                NodeFileStorage::open(&dir).map_err(BTError::into_inner),
                Err(Error::DatabaseCorruption)
            ));
        }
        // node file size smaller than what metadata claims
        {
            let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
            write_metadata(&dir, 0, 2, 0, 0, 0);
            write_reuse_list(&dir, &[0]);
            write_nodes(&dir, &[[0; 32]]);

            assert!(matches!(
                NodeFileStorage::open(&dir).map_err(BTError::into_inner),
                Err(Error::DatabaseCorruption)
            ));
        }
        // reuse list file smaller than what metadata claims
        {
            let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
            write_metadata(&dir, 0, 0, 0, 2, 0);
            write_reuse_list(&dir, &[0]);
            write_nodes(&dir, &[[0; 32]]);

            assert!(matches!(
                NodeFileStorage::open(&dir).map_err(BTError::into_inner),
                Err(Error::DatabaseCorruption)
            ));
        }
        // frozen reuse indices larger than frozen node count
        {
            let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
            write_metadata(&dir, 0, 1, 0, 1, 1);
            write_reuse_list(&dir, &[0]);
            write_nodes(&dir, &[[0; 32]]);

            assert!(matches!(
                NodeFileStorage::open(&dir).map_err(BTError::into_inner),
                Err(Error::DatabaseCorruption)
            ));
        }
        // reusable reuse indices larger than node count
        {
            let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
            write_metadata(&dir, 0, 1, 0, 1, 0);
            write_reuse_list(&dir, &[1]);
            write_nodes(&dir, &[[0; 32]]);

            assert!(matches!(
                NodeFileStorage::open(&dir).map_err(BTError::into_inner),
                Err(Error::DatabaseCorruption)
            ));
        }
        // reusable reuse indices smaller or equal to frozen node count
        {
            let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
            write_metadata(&dir, 0, 1, 1, 1, 0);
            write_reuse_list(&dir, &[0]);
            write_nodes(&dir, &[[0; 32]]);

            assert!(matches!(
                NodeFileStorage::open(&dir).map_err(BTError::into_inner),
                Err(Error::DatabaseCorruption)
            ));
        }
    }

    #[test]
    fn open_forwards_io_errors() {
        let dir = TestDir::try_new(Permissions::ReadOnly).unwrap();

        assert!(matches!(
            NodeFileStorage::open(&dir).map_err(BTError::into_inner),
            Err(Error::Io(_))
        ));
    }

    #[test]
    fn get_reads_data_when_index_in_bounds_and_not_in_reuse_list() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        write_metadata(&dir, 0, 3, 0, 1, 0);
        write_reuse_list(&dir, &[1]);
        write_nodes(&dir, &[[0; 32], [1; 32], [2; 32]]);

        let storage = NodeFileStorage::open(&dir).unwrap();

        assert_eq!(storage.get(0).unwrap(), [0; 32]);
        assert_eq!(storage.get(2).unwrap(), [2; 32]);
    }

    #[test]
    fn get_returns_error_when_index_larger_than_highest_assigned_index() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let storage = NodeFileStorage::open(&dir).unwrap();
        assert_eq!(storage.next_idx.load(Ordering::Relaxed), 0);

        // index 0 is the next index to be assigned, and was therefore not yet assigned
        assert!(matches!(
            storage.get(0).map_err(BTError::into_inner),
            Err(Error::NotFound)
        ));
    }

    #[test]
    fn get_returns_error_when_index_in_reuse_list() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        write_metadata(&dir, 0, 1, 0, 1, 0);
        write_reuse_list(&dir, &[0]);
        write_nodes(&dir, &[[0; 32]]);

        let storage = NodeFileStorage::open(&dir).unwrap();

        // in reuse list
        assert!(matches!(
            storage.get(0).map_err(BTError::into_inner),
            Err(Error::NotFound)
        ));
    }

    #[test]
    fn reserve_returns_last_index_from_non_frozen_part_of_reuse_list() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        write_metadata(&dir, 0, 4, 2, 3, 1);
        write_reuse_list(&dir, &[1, 3, 2]);
        write_nodes(&dir, &[[0; 32], [1; 32], [2; 32], [3; 32]]);

        let storage = NodeFileStorage::open(&dir).unwrap();

        assert_eq!(storage.reserve(&[0; 32]), 2); // last index in reuse list
        assert_eq!(storage.reserve(&[0; 32]), 3); // next index in reuse list
        assert_eq!(storage.reserve(&[0; 32]), 4); // new index
    }

    #[test]
    fn reserve_returns_new_index_if_no_reuse_available() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        // create a single node -> index 0 is used
        write_metadata(&dir, 0, 1, 0, 0, 0);
        write_reuse_list(&dir, &[]);
        write_nodes(&dir, &[[0; 32]]);

        let storage = NodeFileStorage::open(&dir).unwrap();

        assert_eq!(storage.reserve(&[0; 32]), 1);
    }

    #[test]
    fn set_writes_data_to_node_file_at_index_and_updates_node_count() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        // prepare file: write some nodes into the file
        write_metadata(&dir, 0, 2, 0, 0, 0);
        write_reuse_list(&dir, &[]);
        write_nodes(&dir, &[[0; 32], [1; 32]]);

        // create storage and call set with existing and new nodes
        {
            let storage = NodeFileStorage::open(&dir).unwrap();
            storage.next_idx.store(5, Ordering::Relaxed);

            // add new node at end
            storage.set(2, &[4; 32]).unwrap();
            // add new node after end
            storage.set(4, &[5; 32]).unwrap();
        }

        let mut node_file = File::open(dir.join(NodeFileStorage::NODE_STORE_FILE)).unwrap();
        let mut buf = [0; size_of::<TestNode>() * 5];
        node_file.read_exact(&mut buf).unwrap();

        // first node remains unchanged
        assert_eq!(&buf[..size_of::<TestNode>()], &[0; 32]);
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
    fn set_returns_error_when_updating_frozen_node() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        write_metadata(&dir, 0, 1, 1, 0, 0);
        write_reuse_list(&dir, &[]);
        write_nodes(&dir, &[[0; 32]]);

        let storage = NodeFileStorage::open(&dir).unwrap();

        assert!(matches!(
            storage.set(0, &[0; 32]).map_err(BTError::into_inner),
            Err(Error::Frozen)
        ));
    }

    #[test]
    fn set_returns_error_when_updating_index_in_reuse_list() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        write_metadata(&dir, 0, 1, 0, 1, 0);
        write_reuse_list(&dir, &[0]);
        write_nodes(&dir, &[[0; 32]]);

        let storage = NodeFileStorage::open(&dir).unwrap();

        assert!(matches!(
            storage.set(0, &[0; 32]).map_err(BTError::into_inner),
            Err(Error::NotFound)
        ));
    }

    #[test]
    fn set_returns_error_if_index_larger_than_highest_assigned_index() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let storage = NodeFileStorage::open(&dir).unwrap();
        assert_eq!(storage.next_idx.load(Ordering::Relaxed), 0);

        assert!(matches!(
            storage.set(0, &[0; 32]).map_err(BTError::into_inner),
            Err(Error::NotFound)
        ));
    }

    #[test]
    fn delete_adds_index_to_reuse_list() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        write_metadata(&dir, 0, 2, 0, 0, 0);
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
    fn delete_returns_error_when_deleting_frozen_node() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        write_metadata(&dir, 0, 1, 1, 0, 0);
        write_reuse_list(&dir, &[]);
        write_nodes(&dir, &[[0; 32]]);

        let storage = NodeFileStorage::open(&dir).unwrap();
        assert!(matches!(
            storage.delete(0).map_err(BTError::into_inner),
            Err(Error::Frozen)
        ));
    }

    #[test]
    fn delete_returns_error_if_index_larger_than_highest_assigned_index() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let storage = NodeFileStorage::open(&dir).unwrap();
        assert_eq!(storage.next_idx.load(Ordering::Relaxed), 0);

        assert!(matches!(
            storage.delete(0).map_err(BTError::into_inner),
            Err(Error::NotFound)
        ));
    }

    #[test]
    fn delete_returns_error_if_index_in_reuse_list() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        write_metadata(&dir, 0, 1, 0, 1, 0);
        write_reuse_list(&dir, &[0]);
        write_nodes(&dir, &[[0; 32]]);

        let storage = NodeFileStorage::open(&dir).unwrap();
        assert!(matches!(
            storage.delete(0).map_err(BTError::into_inner),
            Err(Error::NotFound)
        ));
    }

    #[test]
    fn close_flushes_files_and_writes_metadata() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        write_metadata(&dir, 0, 2, 1, 2, 1);
        write_reuse_list(&dir, &[0, 1]);
        write_nodes(&dir, &[[0; 32], [1; 32]]);

        let storage = NodeFileStorage::open(&dir).unwrap();
        {
            // freeze second node, and a third node, reserve a fourth node
            storage.frozen_nodes.store(2, Ordering::Relaxed);
            storage.next_idx.store(3, Ordering::Relaxed);
            storage.set(2, &[2; 32]).unwrap();
            storage.next_idx.store(4, Ordering::Relaxed);
            // freeze second reuse index add a third reuse index
            let mut reuse_list_file = storage.reuse_list_file.lock().unwrap();
            reuse_list_file.write_to_disk().unwrap();
            reuse_list_file.freeze_temporarily();
            reuse_list_file.freeze_permanently();
            reuse_list_file.push(2);
        }

        storage.close().unwrap();

        let metadata =
            NodeFileStorageMetadata::read_or_init(dir.join(NodeFileStorage::METADATA_FILE))
                .unwrap();
        assert_eq!(
            metadata,
            NodeFileStorageMetadata {
                last_checkpoint: 0,
                nodes: 4,
                frozen_nodes: 2,
                reuse_indices: 3,
                frozen_reuse_indices: 2,
            }
        );
        let nodes = fs::read(dir.join(NodeFileStorage::NODE_STORE_FILE)).unwrap();
        assert_eq!(
            nodes,
            // the existing two nodes, the newly added node, a zero node for the reserved but not
            // written node and another zero node for padding
            [[0u8; 32], [1; 32], [2; 32], [0; 32], [0; 32]].as_bytes()
        );
        let reuse_list_indices = fs::read(dir.join(NodeFileStorage::REUSE_LIST_FILE)).unwrap();
        assert_eq!(reuse_list_indices, [0u64, 1, 2].as_bytes());
    }

    #[test]
    fn open_recovers_closed_state() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        write_metadata(&dir, 0, 2, 1, 2, 1);
        write_reuse_list(&dir, &[0, 1]);
        write_nodes(&dir, &[[0; 32], [1; 32]]);

        let storage = NodeFileStorage::open(&dir).unwrap();
        // freeze one node and add another one and freeze one reuse index and add another one
        {
            storage.frozen_nodes.store(2, Ordering::Relaxed);
            storage.next_idx.store(3, Ordering::Relaxed);
            storage.set(2, &[2; 32]).unwrap();
            let mut reuse_list_file = storage.reuse_list_file.lock().unwrap();
            reuse_list_file.write_to_disk().unwrap();
            reuse_list_file.freeze_temporarily();
            reuse_list_file.freeze_permanently();
            reuse_list_file.push(2);
        }

        storage.close().unwrap();

        let storage = NodeFileStorage::open(&dir).unwrap();
        assert_eq!(storage.checkpoint.load(Ordering::Relaxed), 0);
        assert_eq!(storage.next_idx.load(Ordering::Relaxed), 3);
        assert_eq!(storage.frozen_nodes.load(Ordering::Relaxed), 2);
        let reuse_list_file = storage.reuse_list_file.lock().unwrap();
        assert_eq!(reuse_list_file.count(), 3);
        assert_eq!(reuse_list_file.frozen_count(), 2);
    }

    #[test]
    fn ensure_returns_if_checkpoint_matches() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let checkpoint = 1;
        write_metadata(&dir, checkpoint, 0, 0, 0, 0);

        let storage = NodeFileStorage::open(dir.path()).unwrap();
        assert!(storage.ensure(checkpoint).is_ok());
        assert!(matches!(
            storage.ensure(checkpoint - 1).map_err(BTError::into_inner),
            Err(Error::Checkpoint)
        ));
        assert!(matches!(
            storage.ensure(checkpoint + 1).map_err(BTError::into_inner),
            Err(Error::Checkpoint)
        ));
    }

    #[test]
    fn prepare_fails_if_requested_checkpoint_is_not_one_larger_than_current_one() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let checkpoint = 1;
        write_metadata(&dir, checkpoint, 0, 0, 0, 0);

        let storage = NodeFileStorage::open(dir.path()).unwrap();
        assert!(matches!(
            storage.prepare(checkpoint).map_err(BTError::into_inner),
            Err(Error::Checkpoint)
        ));
        assert!(matches!(
            storage.prepare(checkpoint + 2).map_err(BTError::into_inner),
            Err(Error::Checkpoint)
        ));
    }

    #[test]
    fn prepare_flushes_nodes_and_reuse_indices_then_freezes_them_and_writes_prepared_metadata() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        write_metadata(&dir, 0, 2, 1, 2, 1);
        write_reuse_list(&dir, &[0, 1]);
        write_nodes(&dir, &[[0; 32], [1; 32]]);

        let expected_prepared_metadata = NodeFileStorageCheckpointMetadata {
            checkpoint: 1,
            frozen_nodes: 2,
            frozen_reuse_indices: 2,
        };

        let storage = NodeFileStorage::open(dir.path()).unwrap();

        storage.prepare(1).unwrap();

        // check that nodes have been flushed
        let nodes = fs::read(dir.join(NodeFileStorage::NODE_STORE_FILE)).unwrap();
        // the existing two nodes and a zero node for padding
        assert_eq!(nodes, [[0u8; 32], [1; 32], [0; 32]].as_bytes());

        // check that reuse list has been flushed and frozen
        let reuse_indices = fs::read(dir.join(NodeFileStorage::REUSE_LIST_FILE)).unwrap();
        assert_eq!(reuse_indices, [0u64, 1u64].as_bytes());

        // check that reuse list in memory has been frozen
        let cached_file = storage.reuse_list_file.lock().unwrap();
        assert_eq!(cached_file.frozen_count(), 2);

        // check that in-memory frozen node count has been updated
        assert_eq!(
            storage.frozen_nodes.load(Ordering::Relaxed),
            expected_prepared_metadata.frozen_nodes
        );

        // check that prepared metadata has been written
        let prepared_metadata = NodeFileStorageCheckpointMetadata::read_or_init(
            dir.join(NodeFileStorage::PREPARED_METADATA_FILE),
        )
        .unwrap();
        assert_eq!(prepared_metadata, expected_prepared_metadata);
    }

    #[test]
    fn commit_fails_if_requested_checkpoint_is_not_one_larger_than_current_one() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let checkpoint = 1;
        write_metadata(&dir, checkpoint, 0, 0, 0, 0);

        let storage = NodeFileStorage::open(dir.path()).unwrap();
        assert!(matches!(
            storage.commit(checkpoint).map_err(BTError::into_inner),
            Err(Error::Checkpoint)
        ));
        assert!(matches!(
            storage.commit(checkpoint + 2).map_err(BTError::into_inner),
            Err(Error::Checkpoint)
        ));
    }

    #[test]
    fn commit_fails_if_requested_checkpoint_does_not_match_prepared_metadata() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let checkpoint = 1;
        write_metadata(&dir, checkpoint, 0, 0, 0, 0);

        let storage = NodeFileStorage::open(dir.path()).unwrap();

        // Attempting to commit without a prepared metadata file fails.
        assert!(matches!(
            storage.commit(checkpoint + 1).map_err(BTError::into_inner),
            Err(Error::Checkpoint)
        ));

        fs::write(
            dir.path().join(NodeFileStorage::PREPARED_METADATA_FILE),
            NodeFileStorageCheckpointMetadata {
                checkpoint: checkpoint + 2,
                frozen_nodes: 0,
                frozen_reuse_indices: 0,
            }
            .as_bytes(),
        )
        .unwrap();

        assert!(matches!(
            storage.commit(checkpoint + 1).map_err(BTError::into_inner),
            Err(Error::Checkpoint)
        ));
    }

    #[test]
    fn commit_renames_prepared_to_committed_metadata_and_sets_checkpoint() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let old_metadata = NodeFileStorageCheckpointMetadata {
            checkpoint: 1,
            frozen_nodes: 1,
            frozen_reuse_indices: 1,
        };

        write_metadata(
            &dir,
            old_metadata.checkpoint,
            old_metadata.frozen_nodes,
            0,
            old_metadata.frozen_reuse_indices,
            0,
        );
        write_reuse_list(&dir, &[0]);
        write_nodes(&dir, &[[0; 32]]);

        let storage = NodeFileStorage::open(dir.path()).unwrap();
        assert_eq!(storage.checkpoint.load(Ordering::Relaxed), 1);

        let new_metadata = NodeFileStorageCheckpointMetadata {
            checkpoint: 2,
            frozen_nodes: 1,
            frozen_reuse_indices: 1,
        };
        fs::write(
            dir.join(NodeFileStorage::PREPARED_METADATA_FILE),
            new_metadata.as_bytes(),
        )
        .unwrap();

        storage.commit(2).unwrap();

        assert!(!fs::exists(dir.join(NodeFileStorage::PREPARED_METADATA_FILE)).unwrap());
        assert_eq!(
            fs::read(dir.join(NodeFileStorage::COMMITTED_METADATA_FILE)).unwrap(),
            new_metadata.as_bytes()
        );
        assert_eq!(
            storage.checkpoint.load(Ordering::Relaxed),
            new_metadata.checkpoint
        );
    }

    #[test]
    fn abort_fails_if_requested_checkpoint_is_not_one_larger_than_current_one() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let checkpoint = 1;
        write_metadata(&dir, checkpoint, 0, 0, 0, 0);

        let storage = NodeFileStorage::open(dir.path()).unwrap();
        assert!(matches!(
            storage.abort(checkpoint).map_err(BTError::into_inner),
            Err(Error::Checkpoint)
        ));
        assert!(matches!(
            storage.abort(checkpoint + 2).map_err(BTError::into_inner),
            Err(Error::Checkpoint)
        ));
    }

    #[test]
    fn abort_removes_prepared_metadata_and_restores_committed_metadata() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let committed_metadata = NodeFileStorageCheckpointMetadata {
            checkpoint: 1,
            frozen_nodes: 2,
            frozen_reuse_indices: 1,
        };
        let prepared_metadata = NodeFileStorageCheckpointMetadata {
            checkpoint: 2,
            frozen_nodes: 4,
            frozen_reuse_indices: 2,
        };
        fs::write(
            dir.join(NodeFileStorage::COMMITTED_METADATA_FILE),
            committed_metadata.as_bytes(),
        )
        .unwrap();

        write_metadata(
            &dir,
            committed_metadata.checkpoint,
            4,
            committed_metadata.frozen_nodes,
            2,
            committed_metadata.frozen_reuse_indices,
        );
        write_reuse_list(&dir, &[0, 2]); // one frozen + one new
        write_nodes(&dir, &[[0; 32], [1; 32], [2; 32], [3; 32]]); // two frozen + two new

        let storage = NodeFileStorage::open(&dir).unwrap();

        // simulate prepared state
        {
            storage.frozen_nodes.store(4, Ordering::Release);
            {
                let mut reuse_list_file = storage.reuse_list_file.lock().unwrap();
                reuse_list_file.write_to_disk().unwrap();
                reuse_list_file.freeze_temporarily();
            }
            fs::write(
                dir.join(NodeFileStorage::PREPARED_METADATA_FILE),
                prepared_metadata.as_bytes(),
            )
            .unwrap();
        }

        storage.abort(2).unwrap();

        assert!(!fs::exists(dir.join(NodeFileStorage::PREPARED_METADATA_FILE)).unwrap());
        assert_eq!(
            fs::read(dir.join(NodeFileStorage::COMMITTED_METADATA_FILE)).unwrap(),
            committed_metadata.as_bytes()
        );
        assert_eq!(
            storage.reuse_list_file.lock().unwrap().frozen_count(),
            committed_metadata.frozen_reuse_indices as usize
        );
        assert_eq!(
            storage.frozen_nodes.load(Ordering::Relaxed),
            committed_metadata.frozen_nodes
        );
        // Committed data remains available
        assert_eq!(storage.get(1).unwrap(), [1; 32]);
        // Uncommitted data becomes modifiable again
        assert!(storage.set(3, &[3; 32]).is_ok());
        // Uncommitted reuse indices are usable again
        assert_eq!(storage.reserve(&[0; 32]), 2);
    }

    #[test]
    fn restore_overwrites_metadata_with_committed_metadata() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let committed_metadata = NodeFileStorageCheckpointMetadata {
            checkpoint: 1,
            frozen_nodes: 2,
            frozen_reuse_indices: 3,
        };
        committed_metadata
            .write(dir.join(NodeFileStorage::COMMITTED_METADATA_FILE))
            .unwrap();

        let metadata = NodeFileStorageMetadata {
            last_checkpoint: 1,
            nodes: 3,
            frozen_nodes: 4,
            reuse_indices: 5,
            frozen_reuse_indices: 6,
        };
        metadata
            .write(dir.join(NodeFileStorage::METADATA_FILE))
            .unwrap();

        NodeFileStorage::restore(&dir, 1).unwrap();

        let metadata =
            NodeFileStorageMetadata::read_or_init(dir.join(NodeFileStorage::METADATA_FILE))
                .unwrap();
        assert_eq!(
            metadata,
            NodeFileStorageMetadata {
                last_checkpoint: committed_metadata.checkpoint,
                nodes: committed_metadata.frozen_nodes,
                frozen_nodes: committed_metadata.frozen_nodes,
                reuse_indices: committed_metadata.frozen_reuse_indices,
                frozen_reuse_indices: committed_metadata.frozen_reuse_indices,
            }
        );
    }

    #[test]
    fn restore_fails_if_checkpoint_does_not_match_committed_metadata() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let committed_metadata = NodeFileStorageCheckpointMetadata {
            checkpoint: 1,
            frozen_nodes: 0,
            frozen_reuse_indices: 0,
        };
        committed_metadata
            .write(dir.join(NodeFileStorage::COMMITTED_METADATA_FILE))
            .unwrap();

        assert!(matches!(
            NodeFileStorage::restore(&dir, 0).map_err(BTError::into_inner),
            Err(Error::Checkpoint)
        ));
        assert!(matches!(
            NodeFileStorage::restore(&dir, 2).map_err(BTError::into_inner),
            Err(Error::Checkpoint)
        ));
    }

    impl<T, F> super::NodeFileStorage<T, F>
    where
        T: DiskRepresentable,
    {
        /// Creates all files for a file-based node storage in the specified directory
        /// and populates them with the provided nodes and reusable indices.
        pub fn create_files(
            path: impl AsRef<Path>,
            nodes: &[T],
            frozen_nodes: u64,
            reuse_indices: &[u64],
            frozen_reuse_indices: u64,
        ) -> BTResult<(), Error> {
            let path = path.as_ref();

            fs::create_dir_all(path)?;

            NodeFileStorageMetadata {
                last_checkpoint: 0,
                nodes: nodes.len() as u64,
                frozen_nodes,
                reuse_indices: reuse_indices.len() as u64,
                frozen_reuse_indices,
            }
            .write(path.join(Self::METADATA_FILE))?;
            let nodes_bytes: Vec<u8> = nodes
                .iter()
                .flat_map(|n| n.to_disk_repr().to_vec())
                .collect();
            fs::write(path.join(NodeFileStorage::NODE_STORE_FILE), &nodes_bytes).unwrap();
            fs::write(path.join(Self::REUSE_LIST_FILE), reuse_indices.as_bytes())?;

            Ok(())
        }
    }

    fn write_metadata(
        dir: impl AsRef<Path>,
        checkpoint: u64,
        nodes: u64,
        frozen_nodes: u64,
        reuse_indices: u64,
        frozen_reuse_indices: u64,
    ) {
        NodeFileStorageMetadata {
            last_checkpoint: checkpoint,
            nodes,
            frozen_nodes,
            reuse_indices,
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
