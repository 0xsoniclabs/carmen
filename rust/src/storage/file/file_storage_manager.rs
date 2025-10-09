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
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use zerocopy::IntoBytes;

use crate::{
    database::verkle::variants::managed::{
        EmptyNode, FullLeafNode, InnerNode, Node, NodeId, NodeType, SparseLeafNode,
    },
    storage::{CheckpointParticipant, Checkpointable, Error, Storage},
};

/// A storage manager for Verkle trie nodes for file based storage backends.
///
/// In order for concurrent operations to be safe (in that there are not data races) they have to
/// operate on different [`NodeId`]s.
#[derive(Debug)]
pub struct FileStorageManager<S1, S2, S3>
where
    S1: Storage<Id = u64, Item = InnerNode> + CheckpointParticipant,
    S2: Storage<Id = u64, Item = SparseLeafNode<2>> + CheckpointParticipant,
    S3: Storage<Id = u64, Item = FullLeafNode> + CheckpointParticipant,
{
    dir: PathBuf,
    checkpoint: AtomicU64,
    inner_nodes: S1,
    leaf_nodes_2: S2,
    leaf_nodes_256: S3,
}

impl<S1, S2, S3> FileStorageManager<S1, S2, S3>
where
    S1: Storage<Id = u64, Item = InnerNode> + CheckpointParticipant,
    S2: Storage<Id = u64, Item = SparseLeafNode<2>> + CheckpointParticipant,
    S3: Storage<Id = u64, Item = FullLeafNode> + CheckpointParticipant,
{
    pub const INNER_NODE_DIR: &str = "inner_node";
    pub const LEAF_NODE_2_DIR: &str = "leaf_node_2";
    pub const LEAF_NODE_256_DIR: &str = "leaf_node_256";
    pub const COMMITTED_CHECKPOINT_FILE: &str = "committed_checkpoint.bin";
    pub const PREPARED_CHECKPOINT_FILE: &str = "prepared_checkpoint.bin";
}

impl<S1, S2, S3> Storage for FileStorageManager<S1, S2, S3>
where
    S1: Storage<Id = u64, Item = InnerNode> + CheckpointParticipant,
    S2: Storage<Id = u64, Item = SparseLeafNode<2>> + CheckpointParticipant,
    S3: Storage<Id = u64, Item = FullLeafNode> + CheckpointParticipant,
{
    type Id = NodeId;
    type Item = Node;

    /// Opens or creates the file backends for the individual node types in the specified directory.
    fn open(dir: &Path) -> Result<Self, Error> {
        std::fs::create_dir_all(dir)?;

        let commited_path = dir.join(Self::COMMITTED_CHECKPOINT_FILE);
        if !fs::exists(&commited_path)? {
            fs::write(&commited_path, 0u64.as_bytes())?;
        }

        let mut checkpoint = 0u64;
        File::open(&commited_path)?.read_exact(checkpoint.as_mut_bytes())?;

        let inner_nodes = S1::open(dir.join(Self::INNER_NODE_DIR).as_path())?;
        let leaf_nodes_2 = S2::open(dir.join(Self::LEAF_NODE_2_DIR).as_path())?;
        let leaf_nodes_256 = S3::open(dir.join(Self::LEAF_NODE_256_DIR).as_path())?;

        inner_nodes.ensure(checkpoint)?;
        leaf_nodes_2.ensure(checkpoint)?;
        leaf_nodes_256.ensure(checkpoint)?;

        Ok(Self {
            dir: dir.to_path_buf(),
            checkpoint: AtomicU64::new(checkpoint),
            inner_nodes,
            leaf_nodes_2,
            leaf_nodes_256,
        })
    }

    fn get(&self, id: NodeId) -> Result<Node, Error> {
        let idx = id.to_index();
        match id.to_node_type().ok_or(Error::InvalidId)? {
            NodeType::Empty => Ok(Node::Empty(EmptyNode)),
            NodeType::Inner => {
                let node = self.inner_nodes.get(idx)?;
                Ok(Node::Inner(Box::new(node)))
            }
            NodeType::Leaf2 => {
                let node = self.leaf_nodes_2.get(idx)?;
                Ok(Node::Leaf2(Box::new(node)))
            }
            NodeType::Leaf256 => {
                let node = self.leaf_nodes_256.get(idx)?;
                Ok(Node::Leaf256(Box::new(node)))
            }
        }
    }

    fn reserve(&self, node: &Node) -> NodeId {
        match node {
            Node::Empty(_) => NodeId::from_idx_and_node_type(0, NodeType::Empty),
            Node::Inner(node) => {
                let idx = self.inner_nodes.reserve(node);
                NodeId::from_idx_and_node_type(idx, NodeType::Inner)
            }
            Node::Leaf2(node) => {
                let idx = self.leaf_nodes_2.reserve(node);
                NodeId::from_idx_and_node_type(idx, NodeType::Leaf2)
            }
            Node::Leaf256(node) => {
                let idx = self.leaf_nodes_256.reserve(node);
                NodeId::from_idx_and_node_type(idx, NodeType::Leaf256)
            }
        }
    }

    fn set(&self, id: NodeId, node: &Node) -> Result<(), Error> {
        let idx = id.to_index();
        match (node, id.to_node_type().ok_or(Error::InvalidId)?) {
            (Node::Empty(_), NodeType::Empty) => Ok(()),
            (Node::Inner(node), NodeType::Inner) => self.inner_nodes.set(idx, node),
            (Node::Leaf2(node), NodeType::Leaf2) => self.leaf_nodes_2.set(idx, node),
            (Node::Leaf256(node), NodeType::Leaf256) => self.leaf_nodes_256.set(idx, node),
            (Node::Empty(_) | Node::Inner(_) | Node::Leaf2(_) | Node::Leaf256(_), _) => {
                Err(Error::IdNodeTypeMismatch)
            }
        }
    }

    fn delete(&self, id: NodeId) -> Result<(), Error> {
        let idx = id.to_index();
        match id.to_node_type().ok_or(Error::InvalidId)? {
            NodeType::Empty => Ok(()),
            NodeType::Inner => self.inner_nodes.delete(idx),
            NodeType::Leaf2 => self.leaf_nodes_2.delete(idx),
            NodeType::Leaf256 => self.leaf_nodes_256.delete(idx),
        }
    }
}

impl<S1, S2, S3> Checkpointable for FileStorageManager<S1, S2, S3>
where
    S1: Storage<Id = u64, Item = InnerNode> + CheckpointParticipant,
    S2: Storage<Id = u64, Item = SparseLeafNode<2>> + CheckpointParticipant,
    S3: Storage<Id = u64, Item = FullLeafNode> + CheckpointParticipant,
{
    fn checkpoint(&self) -> Result<(), Error> {
        let current_checkpoint = self.checkpoint.load(Ordering::Acquire);
        let new_checkpoint = current_checkpoint + 1;
        let participants = [
            &self.inner_nodes as &dyn CheckpointParticipant,
            &self.leaf_nodes_2,
            &self.leaf_nodes_256,
        ];
        for (i, participant) in participants.iter().enumerate() {
            if let Err(err) = participant.prepare(new_checkpoint) {
                for participant in participants[..i].iter().rev() {
                    participant.abort(current_checkpoint)?;
                }
                return Err(err);
            }
        }
        fs::write(
            self.dir.join(Self::PREPARED_CHECKPOINT_FILE),
            new_checkpoint.as_bytes(),
        )?;
        fs::rename(
            self.dir.join(Self::PREPARED_CHECKPOINT_FILE),
            self.dir.join(Self::COMMITTED_CHECKPOINT_FILE),
        )?;
        for participant in participants.iter() {
            participant.commit(new_checkpoint)?;
        }
        self.checkpoint.store(new_checkpoint, Ordering::Release);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use mockall::{Sequence, predicate::eq};

    use super::*;
    use crate::{
        storage::file::{NodeFileStorage, SeekFile},
        utils::test_dir::{Permissions, TestDir},
    };

    #[test]
    fn open_creates_directory_and_calls_open_on_all_storages() {
        type FileStorageManager = super::FileStorageManager<
            NodeFileStorage<InnerNode, SeekFile>,
            NodeFileStorage<SparseLeafNode<2>, SeekFile>,
            NodeFileStorage<FullLeafNode, SeekFile>,
        >;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let storage = FileStorageManager::open(&dir);
        assert!(storage.is_ok());
        let sub_dirs = [
            FileStorageManager::INNER_NODE_DIR,
            FileStorageManager::LEAF_NODE_2_DIR,
            FileStorageManager::LEAF_NODE_256_DIR,
        ];
        let files = [
            NodeFileStorage::<InnerNode, SeekFile>::NODE_STORE_FILE,
            NodeFileStorage::<InnerNode, SeekFile>::REUSE_LIST_FILE,
            NodeFileStorage::<InnerNode, SeekFile>::COMMITTED_METADATA_FILE,
        ];
        for sub_dir in &sub_dirs {
            assert!(fs::exists(dir.join(sub_dir)).unwrap());
            for file in &files {
                assert!(fs::exists(dir.join(sub_dir).join(file)).unwrap());
            }
        }
    }

    #[test]
    fn open_opens_existing_files() {
        type FileStorageManager = super::FileStorageManager<
            NodeFileStorage<InnerNode, SeekFile>,
            NodeFileStorage<SparseLeafNode<2>, SeekFile>,
            NodeFileStorage<FullLeafNode, SeekFile>,
        >;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let sub_dirs = [
            FileStorageManager::INNER_NODE_DIR,
            FileStorageManager::LEAF_NODE_2_DIR,
            FileStorageManager::LEAF_NODE_256_DIR,
        ];
        for sub_dir in &sub_dirs {
            fs::create_dir_all(dir.join(sub_dir)).unwrap();
            // because we are not writing any nodes, the node type does not matter
            NodeFileStorage::<InnerNode, SeekFile>::create_files_for_nodes(&dir, &[], &[]).unwrap();
        }

        let storage = FileStorageManager::open(&dir);
        assert!(storage.is_ok());
    }

    #[test]
    fn open_propagates_io_errors() {
        type FileStorageManager = super::FileStorageManager<
            MockStorage<InnerNode>,
            MockStorage<SparseLeafNode<2>>,
            MockStorage<FullLeafNode>,
        >;

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
            inner_nodes: MockStorage::new(),
            leaf_nodes_2: MockStorage::new(),
            leaf_nodes_256: MockStorage::new(),
        };

        // Node::Empty
        {
            // Empty nodes are not stored. Calling get with them returns a (default) empty node.
            let empty_node_id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            assert_eq!(storage.get(empty_node_id).unwrap(), Node::Empty(EmptyNode));
        }

        // Node::Inner
        {
            let inner_node_id = NodeId::from_idx_and_node_type(1, NodeType::Inner);
            let inner_node = InnerNode::default();
            storage
                .inner_nodes
                .expect_get()
                .with(eq(inner_node_id.to_index()))
                .returning({
                    let inner_node = inner_node.clone();
                    move |_| Ok(inner_node.clone())
                });
            assert_eq!(
                storage.get(inner_node_id).unwrap(),
                Node::Inner(Box::new(inner_node))
            );
        }

        // Node::Leaf2
        {
            let leaf_node_2_id = NodeId::from_idx_and_node_type(2, NodeType::Leaf2);
            let leaf_node_2 = SparseLeafNode::default();
            storage
                .leaf_nodes_2
                .expect_get()
                .with(eq(leaf_node_2_id.to_index()))
                .returning({
                    let leaf_node_2 = leaf_node_2.clone();
                    move |_| Ok(leaf_node_2.clone())
                });
            assert_eq!(
                storage.get(leaf_node_2_id).unwrap(),
                Node::Leaf2(Box::new(leaf_node_2))
            );
        }

        // Node::Leaf256
        {
            let leaf_node_256_id = NodeId::from_idx_and_node_type(3, NodeType::Leaf256);
            let leaf_node_256 = FullLeafNode::default();
            storage
                .leaf_nodes_256
                .expect_get()
                .with(eq(leaf_node_256_id.to_index()))
                .returning({
                    let leaf_node_256 = leaf_node_256.clone();
                    move |_| Ok(leaf_node_256.clone())
                });
            assert_eq!(
                storage.get(leaf_node_256_id).unwrap(),
                Node::Leaf256(Box::new(leaf_node_256))
            );
        }
    }

    #[test]
    fn reserve_forwards_to_reserve_of_corresponding_node_file_storage_depending_on_node_type() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = FileStorageManager {
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(0),
            inner_nodes: MockStorage::new(),
            leaf_nodes_2: MockStorage::new(),
            leaf_nodes_256: MockStorage::new(),
        };

        // Node::Empty
        {
            // Empty nodes are not stored. Calling reserve with them always returns ID 0.
            let empty_node_idx = 0;
            assert_eq!(
                storage.reserve(&Node::Empty(EmptyNode)),
                NodeId::from_idx_and_node_type(empty_node_idx, NodeType::Empty)
            );
        }

        // Node::Inner
        {
            let inner_node_idx = 1;
            let inner_node = InnerNode::default();
            storage
                .inner_nodes
                .expect_reserve()
                .with(eq(inner_node.clone()))
                .returning(move |_| inner_node_idx);
            assert_eq!(
                storage.reserve(&Node::Inner(Box::new(inner_node))),
                NodeId::from_idx_and_node_type(inner_node_idx, NodeType::Inner)
            );
        }

        // Node::Leaf2
        {
            let leaf_node_2_idx = 2;
            let leaf_node_2 = SparseLeafNode::<2>::default();
            storage
                .leaf_nodes_2
                .expect_reserve()
                .with(eq(leaf_node_2.clone()))
                .returning(move |_| leaf_node_2_idx);
            assert_eq!(
                storage.reserve(&Node::Leaf2(Box::new(leaf_node_2))),
                NodeId::from_idx_and_node_type(leaf_node_2_idx, NodeType::Leaf2)
            );
        }

        // Node::Leaf256
        {
            let leaf_node_256_idx = 3;
            let leaf_node_256 = FullLeafNode::default();
            storage
                .leaf_nodes_256
                .expect_reserve()
                .with(eq(leaf_node_256.clone()))
                .returning(move |_| leaf_node_256_idx);
            assert_eq!(
                storage.reserve(&Node::Leaf256(Box::new(leaf_node_256))),
                NodeId::from_idx_and_node_type(leaf_node_256_idx, NodeType::Leaf256)
            );
        }
    }

    #[test]
    fn set_forwards_to_set_of_corresponding_node_file_storage_depending_on_node_type() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = FileStorageManager {
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(0),
            inner_nodes: MockStorage::new(),
            leaf_nodes_2: MockStorage::new(),
            leaf_nodes_256: MockStorage::new(),
        };

        // Node::Empty
        {
            // Empty nodes are not stored. Calling set with them is a no-op.
            let empty_node_id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            let empty_node = Node::Empty(EmptyNode);
            assert!(storage.set(empty_node_id, &empty_node).is_ok());
        }

        // Node::Inner
        {
            let inner_node_id = NodeId::from_idx_and_node_type(1, NodeType::Inner);
            let inner_node = InnerNode::default();
            storage
                .inner_nodes
                .expect_set()
                .with(eq(inner_node_id.to_index()), eq(inner_node.clone()))
                .returning(move |_, _| Ok(()));
            let inner_node = Node::Inner(Box::new(inner_node));
            assert!(storage.set(inner_node_id, &inner_node).is_ok());
        }

        // Node::Leaf2
        {
            let leaf_node_2_id = NodeId::from_idx_and_node_type(2, NodeType::Leaf2);
            let leaf_node_2 = SparseLeafNode::default();
            storage
                .leaf_nodes_2
                .expect_set()
                .with(eq(leaf_node_2_id.to_index()), eq(leaf_node_2.clone()))
                .returning(move |_, _| Ok(()));
            let leaf_node_2 = Node::Leaf2(Box::new(leaf_node_2));
            assert!(storage.set(leaf_node_2_id, &leaf_node_2).is_ok());
        }

        // Node::Leaf256
        {
            let leaf_node_256_id = NodeId::from_idx_and_node_type(3, NodeType::Leaf256);
            let leaf_node_256 = FullLeafNode::default();
            storage
                .leaf_nodes_256
                .expect_set()
                .with(eq(leaf_node_256_id.to_index()), eq(leaf_node_256.clone()))
                .returning(move |_, _| Ok(()));
            let leaf_node_256 = Node::Leaf256(Box::new(leaf_node_256));
            assert!(storage.set(leaf_node_256_id, &leaf_node_256).is_ok());
        }
    }

    #[test]
    fn set_returns_error_if_node_id_prefix_and_node_type_mismatch() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let storage = FileStorageManager {
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(0),
            inner_nodes: MockStorage::new(),
            leaf_nodes_2: MockStorage::new(),
            leaf_nodes_256: MockStorage::new(),
        };

        let id = NodeId::from_idx_and_node_type(0, NodeType::Leaf2);
        let node = Node::Inner(Box::default());

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
            inner_nodes: MockStorage::new(),
            leaf_nodes_2: MockStorage::new(),
            leaf_nodes_256: MockStorage::new(),
        };

        // Node::Empty
        {
            // Empty nodes are not stored. Calling delete with them is a no-op.
            let empty_node_id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            assert!(storage.delete(empty_node_id).is_ok());
        }

        // Node::Inner
        {
            let inner_node_id = NodeId::from_idx_and_node_type(1, NodeType::Inner);
            storage
                .inner_nodes
                .expect_delete()
                .with(eq(inner_node_id.to_index()))
                .returning(move |_| Ok(()));
            assert!(storage.delete(inner_node_id).is_ok());
        }

        // Node::Leaf2
        {
            let leaf_node_2_id = NodeId::from_idx_and_node_type(2, NodeType::Leaf2);
            storage
                .leaf_nodes_2
                .expect_delete()
                .with(eq(leaf_node_2_id.to_index()))
                .returning(move |_| Ok(()));
            assert!(storage.delete(leaf_node_2_id).is_ok());
        }

        // Node::Leaf256
        {
            let leaf_node_256_id = NodeId::from_idx_and_node_type(3, NodeType::Leaf256);
            storage
                .leaf_nodes_256
                .expect_delete()
                .with(eq(leaf_node_256_id.to_index()))
                .returning(move |_| Ok(()));
            assert!(storage.delete(leaf_node_256_id).is_ok());
        }
    }

    #[test]
    fn checkpoint_follows_correct_sequence_for_two_phase_commit() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let old_checkpoint = 1;

        let mut storage = FileStorageManager {
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(old_checkpoint),
            inner_nodes: MockStorage::new(),
            leaf_nodes_2: MockStorage::new(),
            leaf_nodes_256: MockStorage::new(),
        };

        let mut seq = Sequence::new();
        storage
            .inner_nodes
            .expect_prepare()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);
        storage
            .leaf_nodes_2
            .expect_prepare()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);
        storage
            .leaf_nodes_256
            .expect_prepare()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);

        storage
            .inner_nodes
            .expect_commit()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);
        storage
            .leaf_nodes_2
            .expect_commit()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);
        storage
            .leaf_nodes_256
            .expect_commit()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);

        assert!(storage.checkpoint().is_ok());

        // The prepared checkpoint file should not exist after a successful checkpoint.
        assert!(!fs::exists(dir.path().join(
            FileStorageManager::<MockStorage<_>, MockStorage<_>, MockStorage<_>>::PREPARED_CHECKPOINT_FILE
        )).unwrap());
        // The committed checkpoint file should exist and contain the new checkpoint.
        assert_eq!(
            fs::read(
                dir.path()
                    .join(FileStorageManager::<MockStorage<_>, MockStorage<_>, MockStorage<_>>::COMMITTED_CHECKPOINT_FILE)
            )
            .unwrap(),
            (old_checkpoint + 1).as_bytes()
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
        fs::write(
                dir.path()
                    .join(FileStorageManager::<MockStorage<_>, MockStorage<_>, MockStorage<_>>::COMMITTED_CHECKPOINT_FILE)
            ,old_checkpoint.as_bytes()
            )
            .unwrap();

        let mut storage = FileStorageManager {
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(old_checkpoint),
            inner_nodes: MockStorage::new(),
            leaf_nodes_2: MockStorage::new(),
            leaf_nodes_256: MockStorage::new(),
        };

        let mut seq = Sequence::new();
        storage
            .inner_nodes
            .expect_prepare()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);
        storage
            .leaf_nodes_2
            .expect_prepare()
            .returning(|_| Err(Error::Io(std::io::Error::from(std::io::ErrorKind::Other))))
            .times(1)
            .in_sequence(&mut seq);
        storage
            .inner_nodes
            .expect_abort()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);

        assert!(matches!(storage.checkpoint(), Err(Error::Io(_))));

        // The prepared checkpoint file should not exist after a failed checkpoint.
        assert!(!fs::exists(dir.path().join(
            FileStorageManager::<MockStorage<_>, MockStorage<_>, MockStorage<_>>::PREPARED_CHECKPOINT_FILE
        )).unwrap());
        // The committed checkpoint file should exist and contain the old checkpoint.
        assert_eq!(
            fs::read(
                dir.path()
                    .join(FileStorageManager::<MockStorage<_>, MockStorage<_>, MockStorage<_>>::COMMITTED_CHECKPOINT_FILE)
            )
            .unwrap(),
            old_checkpoint.as_bytes()
        );
        // The checkpoint variable should still be the old checkpoint.
        assert_eq!(storage.checkpoint.load(Ordering::Acquire), old_checkpoint);
    }

    mockall::mock! {
        pub Storage<T: Send + Sync + 'static> {}

        impl<T: Send + Sync + 'static> CheckpointParticipant for Storage<T> {
            fn ensure(&self, checkpoint: u64) -> Result<(), Error>;

            fn prepare(&self, checkpoint: u64) -> Result<(), Error>;

            fn commit(&self, checkpoint: u64) -> Result<(), Error>;

            fn abort(&self, checkpoint: u64) -> Result<(), Error>;
        }

        impl<T: Send + Sync + 'static> Storage for Storage<T> {
            type Id = u64;
            type Item = T;

            fn open(path: &Path) -> Result<Self, Error>
            where
                Self: Sized;

            fn get(&self, id: <Self as Storage>::Id) -> Result<<Self as Storage>::Item, Error>;

            fn reserve(&self, item: &<Self as Storage>::Item) -> <Self as Storage>::Id;

            fn set(&self, id: <Self as Storage>::Id, item: &<Self as Storage>::Item) -> Result<(), Error>;

            fn delete(&self, id: <Self as Storage>::Id) -> Result<(), Error>;
        }
    }
}
