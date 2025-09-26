// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::path::Path;

use crate::{
    storage::{Error, Storage},
    types::{FullLeafNode, InnerNode, Node, NodeId, NodeType, SparseLeafNode},
};

/// A storage manager for Verkle trie nodes for file based storage backends.
///
/// In order for concurrent operations to be safe (in that there are not data races) they have to
/// operate on different [`NodeId`]s.
#[derive(Debug)]
pub struct FileStorageManager<S1, S2, S3>
where
    S1: Storage<Id = u64, Item = InnerNode>,
    S2: Storage<Id = u64, Item = SparseLeafNode<2>>,
    S3: Storage<Id = u64, Item = FullLeafNode>,
{
    inner_nodes: S1,
    leaf_nodes_2: S2,
    leaf_nodes_256: S3,
}

impl<S1, S2, S3> FileStorageManager<S1, S2, S3>
where
    S1: Storage<Id = u64, Item = InnerNode>,
    S2: Storage<Id = u64, Item = SparseLeafNode<2>>,
    S3: Storage<Id = u64, Item = FullLeafNode>,
{
    pub const INNER_NODE_DIR: &str = "inner_node";
    pub const LEAF_NODE_2_DIR: &str = "leaf_node_2";
    pub const LEAF_NODE_256_DIR: &str = "leaf_node_256";
}

impl<S1, S2, S3> Storage for FileStorageManager<S1, S2, S3>
where
    S1: Storage<Id = u64, Item = InnerNode>,
    S2: Storage<Id = u64, Item = SparseLeafNode<2>>,
    S3: Storage<Id = u64, Item = FullLeafNode>,
{
    type Id = NodeId;
    type Item = Node;

    /// Opens or creates the file backends for the individual node types in the specified directory.
    fn open(dir: &Path) -> Result<Self, Error> {
        std::fs::create_dir_all(dir)?;

        let inner_nodes = S1::open(dir.join(Self::INNER_NODE_DIR).as_path())?;
        let leaf_nodes_2 = S2::open(dir.join(Self::LEAF_NODE_2_DIR).as_path())?;
        let leaf_nodes_256 = S3::open(dir.join(Self::LEAF_NODE_256_DIR).as_path())?;

        Ok(Self {
            inner_nodes,
            leaf_nodes_2,
            leaf_nodes_256,
        })
    }

    fn get(&self, id: NodeId) -> Result<Node, Error> {
        let idx = id.to_index();
        match id.to_node_type().ok_or(Error::InvalidId)? {
            NodeType::Empty => Ok(Node::Empty),
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
            Node::Empty => NodeId::from_idx_and_node_type(0, NodeType::Empty),
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
            (Node::Empty, NodeType::Empty) => Ok(()),
            (Node::Inner(node), NodeType::Inner) => self.inner_nodes.set(idx, node),
            (Node::Leaf2(node), NodeType::Leaf2) => self.leaf_nodes_2.set(idx, node),
            (Node::Leaf256(node), NodeType::Leaf256) => self.leaf_nodes_256.set(idx, node),
            (Node::Empty | Node::Inner(_) | Node::Leaf2(_) | Node::Leaf256(_), _) => {
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

    fn flush(&self) -> Result<(), Error> {
        self.inner_nodes.flush()?;
        self.leaf_nodes_2.flush()?;
        self.leaf_nodes_256.flush()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, Permissions},
        os::unix::fs::PermissionsExt,
    };

    use super::*;
    use crate::{
        storage::file::{NodeFileStorage, SeekFile},
        types::NodeId,
    };

    #[test]
    fn open_creates_directory_and_opens_committed_checkpoint_and_calls_open_on_all_storages() {
        type FileStorageManager = super::FileStorageManager<
            NodeFileStorage<InnerNode, SeekFile>,
            NodeFileStorage<SparseLeafNode<2>, SeekFile>,
            NodeFileStorage<FullLeafNode, SeekFile>,
        >;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        let storage = FileStorageManager::open(path);
        assert!(storage.is_ok());
        let sub_dirs = [
            FileStorageManager::INNER_NODE_DIR,
            FileStorageManager::LEAF_NODE_2_DIR,
            FileStorageManager::LEAF_NODE_256_DIR,
        ];
        let files = [
            NodeFileStorage::<InnerNode, SeekFile>::NODE_STORE_FILE,
            NodeFileStorage::<InnerNode, SeekFile>::REUSE_LIST_FILE,
            NodeFileStorage::<InnerNode, SeekFile>::METADATA_FILE,
        ];
        for sub_dir in &sub_dirs {
            assert!(fs::exists(path.join(sub_dir)).unwrap());
            for file in &files {
                assert!(fs::exists(path.join(sub_dir).join(file)).unwrap());
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

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        let sub_dirs = [
            FileStorageManager::INNER_NODE_DIR,
            FileStorageManager::LEAF_NODE_2_DIR,
            FileStorageManager::LEAF_NODE_256_DIR,
        ];
        for sub_dir in &sub_dirs {
            fs::create_dir_all(path.join(sub_dir)).unwrap();
            // because we are not writing any nodes, the node type does not matter
            NodeFileStorage::<InnerNode, SeekFile>::create_files_for_nodes(path, &[]).unwrap();
        }

        let storage = FileStorageManager::open(path);
        assert!(storage.is_ok());
    }

    #[test]
    fn open_propagates_io_errors() {
        type FileStorageManager = super::FileStorageManager<
            MockStorage<InnerNode>,
            MockStorage<SparseLeafNode<2>>,
            MockStorage<FullLeafNode>,
        >;

        let dir = tempfile::tempdir().unwrap();

        fs::set_permissions(&dir, Permissions::from_mode(0o000)).unwrap();

        let path = dir.path().join("non_existent_dir");

        assert!(matches!(
            FileStorageManager::open(path.as_path()),
            Err(Error::Io(_))
        ));

        fs::set_permissions(&dir, Permissions::from_mode(0o777)).unwrap();
    }

    #[test]
    fn get_forwards_to_get_of_corresponding_node_file_storage_depending_on_node_type() {
        let mut storage = FileStorageManager {
            inner_nodes: MockStorage::new(),
            leaf_nodes_2: MockStorage::new(),
            leaf_nodes_256: MockStorage::new(),
        };

        // Node::Empty
        {
            // Empty nodes are not stored. Calling get with them returns a (default) empty node.
            let empty_node_id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            assert_eq!(storage.get(empty_node_id).unwrap(), Node::Empty);
        }

        // Node::Inner
        {
            let inner_node_id = NodeId::from_idx_and_node_type(1, NodeType::Inner);
            let inner_node = InnerNode::default();
            storage
                .inner_nodes
                .expect_get()
                .withf(move |id| *id == inner_node_id.to_index())
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
                .withf(move |id| *id == leaf_node_2_id.to_index())
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
                .withf(move |id| *id == leaf_node_256_id.to_index())
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
        let mut storage = FileStorageManager {
            inner_nodes: MockStorage::new(),
            leaf_nodes_2: MockStorage::new(),
            leaf_nodes_256: MockStorage::new(),
        };

        // Node::Empty
        {
            // Empty nodes are not stored. Calling reserve with them always returns ID 0.
            let empty_node_idx = 0;
            assert_eq!(
                storage.reserve(&Node::Empty),
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
                .withf({
                    let inner_node = inner_node.clone();
                    move |node| *node == inner_node
                })
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
                .withf({
                    let leaf_node_2 = leaf_node_2.clone();
                    move |node| *node == leaf_node_2
                })
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
                .withf({
                    let leaf_node_256 = leaf_node_256.clone();
                    move |node| *node == leaf_node_256
                })
                .returning(move |_| leaf_node_256_idx);
            assert_eq!(
                storage.reserve(&Node::Leaf256(Box::new(leaf_node_256))),
                NodeId::from_idx_and_node_type(leaf_node_256_idx, NodeType::Leaf256)
            );
        }
    }

    #[test]
    fn set_forwards_to_set_of_corresponding_node_file_storage_depending_on_node_type() {
        let mut storage = FileStorageManager {
            inner_nodes: MockStorage::new(),
            leaf_nodes_2: MockStorage::new(),
            leaf_nodes_256: MockStorage::new(),
        };

        // Node::Empty
        {
            // Empty nodes are not stored. Calling set with them is a no-op.
            let empty_node_id = NodeId::from_idx_and_node_type(0, NodeType::Empty);
            let empty_node = Node::Empty;
            assert!(storage.set(empty_node_id, &empty_node).is_ok());
        }

        // Node::Inner
        {
            let inner_node_id = NodeId::from_idx_and_node_type(1, NodeType::Inner);
            let inner_node = InnerNode::default();
            storage
                .inner_nodes
                .expect_set()
                .withf({
                    let inner_node = inner_node.clone();
                    move |id, node| *id == inner_node_id.to_index() && *node == inner_node
                })
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
                .withf({
                    let leaf_node_2 = leaf_node_2.clone();
                    move |id, node| *id == leaf_node_2_id.to_index() && *node == leaf_node_2
                })
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
                .withf({
                    let leaf_node_256 = leaf_node_256.clone();
                    move |id, node| *id == leaf_node_256_id.to_index() && *node == leaf_node_256
                })
                .returning(move |_, _| Ok(()));
            let leaf_node_256 = Node::Leaf256(Box::new(leaf_node_256));
            assert!(storage.set(leaf_node_256_id, &leaf_node_256).is_ok());
        }
    }

    #[test]
    fn set_returns_error_if_node_id_prefix_and_node_type_mismatch() {
        let storage = FileStorageManager {
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
        let mut storage = FileStorageManager {
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
                .withf(move |id| *id == inner_node_id.to_index())
                .returning(move |_| Ok(()));
            assert!(storage.delete(inner_node_id).is_ok());
        }

        // Node::Leaf2
        {
            let leaf_node_2_id = NodeId::from_idx_and_node_type(2, NodeType::Leaf2);
            storage
                .leaf_nodes_2
                .expect_delete()
                .withf(move |id| *id == leaf_node_2_id.to_index())
                .returning(move |_| Ok(()));
            assert!(storage.delete(leaf_node_2_id).is_ok());
        }

        // Node::Leaf256
        {
            let leaf_node_256_id = NodeId::from_idx_and_node_type(3, NodeType::Leaf256);
            storage
                .leaf_nodes_256
                .expect_delete()
                .withf(move |id| *id == leaf_node_256_id.to_index())
                .returning(move |_| Ok(()));
            assert!(storage.delete(leaf_node_256_id).is_ok());
        }
    }

    #[test]
    fn flush_calls_flushes_on_all_storages() {
        let mut storage = FileStorageManager {
            inner_nodes: MockStorage::new(),
            leaf_nodes_2: MockStorage::new(),
            leaf_nodes_256: MockStorage::new(),
        };

        storage.inner_nodes.expect_flush().returning(|| Ok(()));
        storage.leaf_nodes_2.expect_flush().returning(|| Ok(()));
        storage.leaf_nodes_256.expect_flush().returning(|| Ok(()));

        assert!(storage.flush().is_ok());
    }

    mockall::mock! {
        pub Storage<T: 'static> {}


        impl<T: 'static> Storage for Storage<T> {
            type Id = u64;
            type Item = T;

            fn open(_path: &Path) -> Result<Self, Error>
            where
                Self: Sized;

            fn get(&self, _id: <Self as Storage>::Id) -> Result<<Self as Storage>::Item, Error>;

            fn reserve(&self, _item: &<Self as Storage>::Item) -> <Self as Storage>::Id;

            fn set(&self, _id: <Self as Storage>::Id, _item: &<Self as Storage>::Item) -> Result<(), Error>;

            fn delete(&self, _id: <Self as Storage>::Id) -> Result<(), Error>;

            fn flush(&self) -> Result<(), Error>;
        }
    }
}
