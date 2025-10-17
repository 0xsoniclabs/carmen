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
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{
    storage::{
        CheckpointParticipant, Checkpointable, Error, RootIdProvider, Storage, StorageX,
        file::FromToFile,
    },
    types::{AllVariants, ToNodeType, TreeId},
};

mod checkpoint_data;
mod root_ids_file;

use checkpoint_data::CheckpointData;
use root_ids_file::RootIdsFile;
use zerocopy::{FromBytes, IntoBytes};

/// A storage manager for Verkle trie nodes for file based storage backends.
///
/// In order for concurrent operations to be safe (in that there are not data races) they have to
/// operate on different [`NodeId`]s.
//#[derive(Debug)]
pub struct FileStorageManager<S, ID>
where
    S: StorageX<Id = u64>,
    S::Item: ToNodeType,
{
    dir: PathBuf,
    checkpoint: AtomicU64,
    nodes: Vec<(<S::Item as ToNodeType>::NodeType, S)>,
    root_ids_file: RootIdsFile<ID>,
}

impl<S, ID> FileStorageManager<S, ID>
where
    S: StorageX<Id = u64>,
    S::Item: ToNodeType,
{
    pub const COMMITTED_CHECKPOINT_FILE: &str = "committed_checkpoint.bin";
    pub const PREPARED_CHECKPOINT_FILE: &str = "prepared_checkpoint.bin";
    pub const ROOT_IDS_FILE: &str = "root_ids.bin";
}

impl<S, ID> Storage for FileStorageManager<S, ID>
where
    S: StorageX<Id = u64> + CheckpointParticipant,
    S::Item: ToNodeType,
    <S::Item as ToNodeType>::NodeType: AllVariants + Eq + Clone + Send + Sync + 'static,
    ID: TreeId<NodeType = <S::Item as ToNodeType>::NodeType>
        + FromBytes
        + IntoBytes
        + Copy
        + Send
        + Sync
        + 'static,
{
    type Id = ID;
    type Item = S::Item;

    /// Opens or creates the file backends for the individual node types in the specified directory.
    fn open(dir: &Path) -> Result<Self, Error> {
        std::fs::create_dir_all(dir)?;

        let checkpoint_data =
            CheckpointData::read_or_init(dir.join(Self::COMMITTED_CHECKPOINT_FILE))?;

        let nodes = <S::Item as ToNodeType>::NodeType::all_variants()
            .iter()
            .map(|(variant, name)| {
                S::open(dir.join(name).as_path(), variant.clone()).map(|s| (variant.clone(), s))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let root_ids_file =
            RootIdsFile::open(dir.join(Self::ROOT_IDS_FILE), checkpoint_data.root_id_count)?;

        for (_, node) in &nodes {
            node.ensure(checkpoint_data.checkpoint_number)?;
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            checkpoint: AtomicU64::new(checkpoint_data.checkpoint_number),
            nodes,
            root_ids_file,
        })
    }

    fn get(&self, id: Self::Id) -> Result<Self::Item, Error> {
        let idx = id.to_index();
        let nt = id.to_node_type().ok_or(Error::InvalidId)?;

        self.nodes
            .iter()
            .find(|(node_type, _)| *node_type == nt)
            .ok_or(Error::InvalidId)?
            .1
            .get(idx)
    }

    fn reserve(&self, node: &Self::Item) -> Self::Id {
        let nt = node.to_node_type();
        let idx = self
            .nodes
            .iter()
            .find(|(node_type, _)| *node_type == nt)
            .unwrap()
            //.ok_or(Error::InvalidId)?
            .1
            .reserve(node);
        ID::from_idx_and_node_type(idx, nt)
    }

    fn set(&self, id: Self::Id, node: &Self::Item) -> Result<(), Error> {
        let idx = id.to_index();
        let nt = id.to_node_type().unwrap();
        if node.to_node_type() != nt {
            return Err(Error::IdNodeTypeMismatch);
        }
        self.nodes
            .iter()
            .find(|(node_type, _)| *node_type == nt)
            .unwrap()
            //.ok_or(Error::InvalidId)?
            .1
            .set(idx, node)
    }

    fn delete(&self, id: Self::Id) -> Result<(), Error> {
        let idx = id.to_index();
        let nt = id.to_node_type().unwrap();
        self.nodes
            .iter()
            .find(|(node_type, _)| *node_type == nt)
            .unwrap()
            //.ok_or(Error::InvalidId)?
            .1
            .delete(idx)
    }
}

impl<S, ID> Checkpointable for FileStorageManager<S, ID>
where
    S: StorageX<Id = u64> + CheckpointParticipant,
    S::Item: ToNodeType,
    <S::Item as ToNodeType>::NodeType: Send + Sync,
    ID: Copy + FromBytes + IntoBytes + Send + Sync,
{
    fn checkpoint(&self) -> Result<(), Error> {
        let current_checkpoint = self.checkpoint.load(Ordering::Acquire);
        let new_checkpoint = current_checkpoint + 1;
        for (i, (_, participant)) in self.nodes.iter().enumerate() {
            if let Err(err) = participant.prepare(new_checkpoint) {
                for (_, participant) in self.nodes[..i].iter().rev() {
                    participant.abort(current_checkpoint)?;
                }
                return Err(err);
            }
        }
        CheckpointData {
            checkpoint_number: new_checkpoint,
            root_id_count: self.root_ids_file.count(),
        }
        .write(self.dir.join(Self::PREPARED_CHECKPOINT_FILE))?;
        fs::rename(
            self.dir.join(Self::PREPARED_CHECKPOINT_FILE),
            self.dir.join(Self::COMMITTED_CHECKPOINT_FILE),
        )?;
        for (_, participant) in self.nodes.iter() {
            participant.commit(new_checkpoint)?;
        }
        self.checkpoint.store(new_checkpoint, Ordering::Release);
        Ok(())
    }
}

impl<S, ID> RootIdProvider for FileStorageManager<S, ID>
where
    S: StorageX<Id = u64>,
    S::Item: ToNodeType,
    ID: Copy + FromBytes + IntoBytes,
{
    type Id = ID;

    fn get_root_id(&self, block_number: u64) -> Result<ID, Error> {
        self.root_ids_file.get(block_number)
    }

    fn set_root_id(&self, block_number: u64, id: ID) -> Result<(), Error> {
        self.root_ids_file.set(block_number, id)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use mockall::{Sequence, predicate::eq};

    use super::*;
    use crate::{
        storage::{
            file::{NodeFileStorage, SeekFile},
            test_utils::{EmptyTestNode, NonEmptyTestNode, TestNode, TestNodeId, TestNodeType},
        },
        utils::test_dir::{Permissions, TestDir},
    };

    #[test]
    fn open_creates_directory_and_calls_open_on_all_storages() {
        type FileStorageManager =
            super::FileStorageManager<NodeFileStorage<TestNode, SeekFile>, TestNodeId>;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let storage = FileStorageManager::open(&dir);
        assert!(storage.is_ok());
        let sub_dirs = TestNodeType::all_variants().iter().map(|(_, name)| name);
        let files = [
            NodeFileStorage::<TestNode, SeekFile>::NODE_STORE_FILE,
            NodeFileStorage::<TestNode, SeekFile>::REUSE_LIST_FILE,
            NodeFileStorage::<TestNode, SeekFile>::COMMITTED_METADATA_FILE,
        ];
        for sub_dir in sub_dirs {
            assert!(fs::exists(dir.join(sub_dir)).unwrap());
            for file in &files {
                assert!(fs::exists(dir.join(sub_dir).join(file)).unwrap());
            }
        }
    }

    #[test]
    fn open_opens_existing_files() {
        type FileStorageManager =
            super::FileStorageManager<NodeFileStorage<TestNode, SeekFile>, TestNodeId>;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let sub_dirs = TestNodeType::all_variants().iter().map(|(_, name)| name);
        for sub_dir in sub_dirs {
            fs::create_dir_all(dir.join(sub_dir)).unwrap();
            // because we are not writing any nodes, the node type does not matter
            NodeFileStorage::<TestNode, SeekFile>::create_files_for_nodes(&dir, &[], &[]).unwrap();
        }

        let storage = FileStorageManager::open(&dir);
        assert!(storage.is_ok());
    }

    #[test]
    fn open_propagates_io_errors() {
        type FileStorageManager =
            super::FileStorageManager<NodeFileStorage<TestNode, SeekFile>, TestNodeId>;

        let dir = TestDir::try_new(Permissions::ReadOnly).unwrap();

        let path = dir.join("non_existent_dir");

        assert!(matches!(FileStorageManager::open(&path), Err(Error::Io(_))));
    }

    #[test]
    fn get_forwards_to_get_of_corresponding_node_file_storage_depending_on_node_type() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = FileStorageManager {
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(0),
            nodes: TestNodeType::all_variants()
                .iter()
                .map(|(node_type, _)| (*node_type, MockStorage::new()))
                .collect(),
            root_ids_file: RootIdsFile::open(dir.path().join("root_ids"), 0).unwrap(),
        };

        // Node::Empty
        {
            let node_id = TestNodeId::from_idx_and_node_type(1, TestNodeType::Empty);
            let node = TestNode::Empty(EmptyTestNode);
            storage.nodes[0]
                .1
                .expect_get()
                .with(eq(node_id.to_index()))
                .returning(move |_| Ok(node));
            assert_eq!(
                storage.get(node_id).unwrap(),
                TestNode::Empty(EmptyTestNode)
            );
        }

        // Node::NonEmpty
        {
            let node_id = TestNodeId::from_idx_and_node_type(0, TestNodeType::NonEmpty);
            let node = TestNode::NonEmpty(NonEmptyTestNode::default());
            storage.nodes[1]
                .1
                .expect_get()
                .with(eq(node_id.to_index()))
                .returning(move |_| Ok(node));
            assert_eq!(
                storage.get(node_id).unwrap(),
                TestNode::NonEmpty(NonEmptyTestNode([0; 32]))
            );
        }
    }

    #[test]
    fn reserve_forwards_to_reserve_of_corresponding_node_file_storage_depending_on_node_type() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = FileStorageManager {
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(0),
            nodes: TestNodeType::all_variants()
                .iter()
                .map(|(node_type, _)| (*node_type, MockStorage::new()))
                .collect(),
            root_ids_file: RootIdsFile::<TestNodeId>::open(dir.path().join("root_ids"), 0).unwrap(),
        };

        // Node::Empty
        {
            let node_idx = 0;
            let node = TestNode::Empty(EmptyTestNode);
            storage.nodes[0]
                .1
                .expect_reserve()
                .with(eq(node))
                .returning(move |_| node_idx);
            assert_eq!(
                storage.reserve(&TestNode::Empty(EmptyTestNode)),
                TestNodeId::from_idx_and_node_type(node_idx, TestNodeType::Empty)
            );
        }

        // Node::NonEmpty
        {
            let node_idx = 1;
            let node = TestNode::NonEmpty(NonEmptyTestNode::default());
            storage.nodes[1]
                .1
                .expect_reserve()
                .with(eq(node))
                .returning(move |_| node_idx);
            assert_eq!(
                storage.reserve(&node),
                TestNodeId::from_idx_and_node_type(node_idx, TestNodeType::NonEmpty)
            );
        }
    }

    #[test]
    fn set_forwards_to_set_of_corresponding_node_file_storage_depending_on_node_type() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = FileStorageManager {
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(0),
            nodes: TestNodeType::all_variants()
                .iter()
                .map(|(node_type, _)| (*node_type, MockStorage::new()))
                .collect(),
            root_ids_file: RootIdsFile::open(dir.path().join("root_ids"), 0).unwrap(),
        };

        // Node::Empty
        {
            let node_id = TestNodeId::from_idx_and_node_type(0, TestNodeType::Empty);
            let node = TestNode::Empty(EmptyTestNode);
            storage.nodes[0]
                .1
                .expect_set()
                .with(eq(node_id.to_index()), eq(node))
                .returning(move |_, _| Ok(()));
            assert!(storage.set(node_id, &node).is_ok());
        }

        // Node::NonEmpty
        {
            let node_id = TestNodeId::from_idx_and_node_type(1, TestNodeType::NonEmpty);
            let node = TestNode::NonEmpty(NonEmptyTestNode::default());
            storage.nodes[1]
                .1
                .expect_set()
                .with(eq(node_id.to_index()), eq(node))
                .returning(move |_, _| Ok(()));
            assert!(storage.set(node_id, &node).is_ok());
        }
    }

    #[test]
    fn set_returns_error_if_node_id_prefix_and_node_type_mismatch() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let storage = FileStorageManager {
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(0),
            nodes: TestNodeType::all_variants()
                .iter()
                .map(|(node_type, _)| (*node_type, MockStorage::new()))
                .collect(),
            root_ids_file: RootIdsFile::open(dir.path().join("root_ids"), 0).unwrap(),
        };

        let id = TestNodeId::from_idx_and_node_type(0, TestNodeType::NonEmpty);
        let node = TestNode::Empty(EmptyTestNode);

        assert!(matches!(
            storage.set(id, &node),
            Err(Error::IdNodeTypeMismatch)
        ));
    }

    #[test]
    fn delete_forwards_to_delete_of_corresponding_node_file_storage_depending_on_node_type() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = FileStorageManager {
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(0),
            nodes: TestNodeType::all_variants()
                .iter()
                .map(|(node_type, _)| (*node_type, MockStorage::<TestNode>::new()))
                .collect(),
            root_ids_file: RootIdsFile::open(dir.path().join("root_ids"), 0).unwrap(),
        };

        // Node::Empty
        {
            let node_id = TestNodeId::from_idx_and_node_type(0, TestNodeType::Empty);
            storage.nodes[0]
                .1
                .expect_delete()
                .with(eq(node_id.to_index()))
                .returning(move |_| Ok(()));
            assert!(storage.delete(node_id).is_ok());
        }

        // Node::NonEmpty
        {
            let node_id = TestNodeId::from_idx_and_node_type(1, TestNodeType::NonEmpty);
            storage.nodes[1]
                .1
                .expect_delete()
                .with(eq(node_id.to_index()))
                .returning(move |_| Ok(()));
            assert!(storage.delete(node_id).is_ok());
        }
    }

    #[test]
    fn checkpoint_follows_correct_sequence_for_two_phase_commit() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let old_checkpoint = 1;

        let mut storage = FileStorageManager {
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(old_checkpoint),
            nodes: TestNodeType::all_variants()
                .iter()
                .map(|(node_type, _)| (*node_type, MockStorage::<TestNode>::new()))
                .collect(),
            root_ids_file: RootIdsFile::<TestNodeId>::open(dir.path().join("root_ids"), 0).unwrap(),
        };

        let mut seq = Sequence::new();
        storage.nodes[0]
            .1
            .expect_prepare()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);
        storage.nodes[1]
            .1
            .expect_prepare()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);

        storage.nodes[0]
            .1
            .expect_commit()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);
        storage.nodes[1]
            .1
            .expect_commit()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);

        assert!(storage.checkpoint().is_ok());

        // The prepared checkpoint file should not exist after a successful checkpoint.
        assert!(
            !fs::exists(dir.path().join(
                FileStorageManager::<MockStorage<TestNode>, TestNodeId>::PREPARED_CHECKPOINT_FILE
            ))
            .unwrap()
        );
        // The committed checkpoint file should exist and contain the new checkpoint.
        assert_eq!(
            CheckpointData::read_or_init(dir.path().join(
                FileStorageManager::<MockStorage<TestNode>, TestNodeId>::COMMITTED_CHECKPOINT_FILE,
            ))
            .unwrap()
            .checkpoint_number,
            old_checkpoint + 1
        );
        // The checkpoint variable should be updated to the new checkpoint.
        assert_eq!(
            storage.checkpoint.load(Ordering::Acquire),
            old_checkpoint + 1
        );
    }

    #[test]
    fn checkpoint_calls_prepare_then_calls_abort_on_previous_participants_if_prepare_failed() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let old_checkpoint = 1;
        let old_checkpoint_data = CheckpointData {
            checkpoint_number: old_checkpoint,
            root_id_count: 0,
        };
        old_checkpoint_data
            .write(dir.path().join(
                FileStorageManager::<MockStorage<TestNode>, TestNodeId>::COMMITTED_CHECKPOINT_FILE,
            ))
            .unwrap();

        let mut storage = FileStorageManager {
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(old_checkpoint),
            nodes: TestNodeType::all_variants()
                .iter()
                .map(|(node_type, _)| (*node_type, MockStorage::<TestNode>::new()))
                .collect(),
            root_ids_file: RootIdsFile::<TestNodeId>::open(dir.path().join("root_ids"), 0).unwrap(),
        };

        let mut seq = Sequence::new();
        storage.nodes[0]
            .1
            .expect_prepare()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);
        storage.nodes[1]
            .1
            .expect_prepare()
            .returning(|_| Err(Error::Io(std::io::Error::from(std::io::ErrorKind::Other))))
            .times(1)
            .in_sequence(&mut seq);
        storage.nodes[0]
            .1
            .expect_abort()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);

        assert!(matches!(storage.checkpoint(), Err(Error::Io(_))));

        // The prepared checkpoint file should not exist after a failed checkpoint.
        assert!(
            !fs::exists(dir.path().join(
                FileStorageManager::<MockStorage<TestNode>, TestNodeId>::PREPARED_CHECKPOINT_FILE
            ))
            .unwrap()
        );
        // The committed checkpoint file should exist and contain the old checkpoint.
        assert_eq!(
            CheckpointData::read_or_init(dir.path().join(
                FileStorageManager::<MockStorage<TestNode>, TestNodeId>::COMMITTED_CHECKPOINT_FILE,
            ))
            .unwrap()
            .checkpoint_number,
            old_checkpoint
        );
        // The checkpoint variable should still be the old checkpoint.
        assert_eq!(storage.checkpoint.load(Ordering::Acquire), old_checkpoint);
    }

    mockall::mock! {
        pub Storage<T: ToNodeType + Send + Sync + 'static> {}

        impl<T: ToNodeType + Send + Sync + 'static> CheckpointParticipant for Storage<T> {
            fn ensure(&self, checkpoint: u64) -> Result<(), Error>;

            fn prepare(&self, checkpoint: u64) -> Result<(), Error>;

            fn commit(&self, checkpoint: u64) -> Result<(), Error>;

            fn abort(&self, checkpoint: u64) -> Result<(), Error>;
        }

        impl<T: ToNodeType + Send + Sync + 'static> StorageX for Storage<T> {
            type Id = u64;
            type Item = T;

            fn open(path: &Path, et: <T as ToNodeType>::NodeType) -> Result<Self, Error>
            where
                Self: Sized;

            fn get(&self, id: <Self as StorageX>::Id) -> Result<<Self as StorageX>::Item, Error>;

            fn reserve(&self, item: &<Self as StorageX>::Item) -> <Self as StorageX>::Id;

            fn set(&self, id: <Self as StorageX>::Id, item: &<Self as StorageX>::Item) -> Result<(), Error>;

            fn delete(&self, id: <Self as StorageX>::Id) -> Result<(), Error>;
        }
    }
}
