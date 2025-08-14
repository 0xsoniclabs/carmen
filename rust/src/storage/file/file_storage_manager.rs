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
    storage::{
        Error, Storage,
        file::{FileBackend, NodeFileStorage},
    },
    types::{FullLeafNode, InnerNode, Node, NodeId, NodeType, SparseLeafNode},
};

/// A storage manager for Verkle trie nodes for file based storage backends.
///
/// In order for concurrent operations to be safe (in that there are not data races) they have to
/// operate on different [`NodeId`]s.
#[derive(Debug)]
pub struct FileStorageManager<F: FileBackend + 'static> {
    inner_nodes: NodeFileStorage<InnerNode, F>,
    leaf_nodes_2: NodeFileStorage<SparseLeafNode<2>, F>,
    leaf_nodes_256: NodeFileStorage<FullLeafNode, F>,
}

impl<F: FileBackend> FileStorageManager<F> {
    const INNER_NODE_DIR: &str = "inner_node";
    const LEAF_NODE_2_DIR: &str = "leaf_node_2";
    const LEAF_NODE_256_DIR: &str = "leaf_node_256";
}

#[cfg_attr(test, mockall::automock)]
impl<F> Storage for FileStorageManager<F>
where
    F: FileBackend + 'static,
{
    type Id = NodeId;
    type Item = Node;

    /// Opens or creates the file backends for the individual node types in the specified directory.
    fn open(dir: &Path) -> Result<Self, Error> {
        Ok(Self {
            inner_nodes: NodeFileStorage::open(dir.join(Self::INNER_NODE_DIR).as_path())?,
            leaf_nodes_2: NodeFileStorage::open(dir.join(Self::LEAF_NODE_2_DIR).as_path())?,
            leaf_nodes_256: NodeFileStorage::open(dir.join(Self::LEAF_NODE_256_DIR).as_path())?,
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
        fs::{self, File, Permissions},
        io::Write,
        os::unix::fs::PermissionsExt,
    };

    use zerocopy::IntoBytes;

    use super::*;
    use crate::{storage::file::SeekFile, types::NodeId};

    type FileStorageManager = super::FileStorageManager<SeekFile>;
    type NodeFileStorage = super::NodeFileStorage<InnerNode, SeekFile>;

    #[test]
    fn open_creates_files_if_they_do_not_exist() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        let storage = FileStorageManager::open(path);
        assert!(storage.is_ok());
        assert!(fs::exists(path.join(FileStorageManager::INNER_NODE_DIR)).unwrap());
        assert!(fs::exists(path.join(FileStorageManager::LEAF_NODE_2_DIR)).unwrap());
        assert!(fs::exists(path.join(FileStorageManager::LEAF_NODE_256_DIR)).unwrap());
    }

    #[test]
    fn open_opens_existing_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        let sub_dirs = [
            FileStorageManager::INNER_NODE_DIR,
            FileStorageManager::LEAF_NODE_2_DIR,
            FileStorageManager::LEAF_NODE_256_DIR,
        ];
        let files = [
            NodeFileStorage::NODE_STORE_FILE,
            NodeFileStorage::REUSE_LIST_FILE,
        ];
        for sub_dir in &sub_dirs {
            fs::create_dir_all(path.join(sub_dir)).unwrap();
            for file in &files {
                File::create(path.join(sub_dir).join(file)).unwrap();
            }
        }

        let storage = FileStorageManager::open(path);
        assert!(storage.is_ok());
    }

    #[test]
    fn open_propagates_io_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        fs::set_permissions(path, Permissions::from_mode(0o000)).unwrap();

        assert!(matches!(FileStorageManager::open(path), Err(Error::Io(_))));

        fs::set_permissions(path, Permissions::from_mode(0o777)).unwrap();
    }

    #[test]
    fn get_reads_node_from_corresponding_node_file_storage_depending_on_node_type() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        let inner_node = Box::new(InnerNode::default());
        let leaf_node_2 = Box::new(SparseLeafNode::default());
        let leaf_node_256 = Box::new(FullLeafNode::default());

        fs::create_dir_all(path.join(FileStorageManager::INNER_NODE_DIR)).unwrap();
        fs::create_dir_all(path.join(FileStorageManager::LEAF_NODE_2_DIR)).unwrap();
        fs::create_dir_all(path.join(FileStorageManager::LEAF_NODE_256_DIR)).unwrap();

        File::create(
            path.join(FileStorageManager::INNER_NODE_DIR)
                .join(NodeFileStorage::NODE_STORE_FILE),
        )
        .unwrap()
        .write_all(inner_node.as_bytes())
        .unwrap();

        File::create(
            path.join(FileStorageManager::LEAF_NODE_2_DIR)
                .join(NodeFileStorage::NODE_STORE_FILE),
        )
        .unwrap()
        .write_all(leaf_node_2.as_bytes())
        .unwrap();

        File::create(
            path.join(FileStorageManager::LEAF_NODE_256_DIR)
                .join(NodeFileStorage::NODE_STORE_FILE),
        )
        .unwrap()
        .write_all(leaf_node_256.as_bytes())
        .unwrap();

        let storage = FileStorageManager::open(path).unwrap();

        // the empty node is not stored but is returned as a default value
        assert_eq!(
            storage
                .get(NodeId::from_idx_and_node_type(0, NodeType::Empty))
                .unwrap(),
            Node::Empty
        );
        assert_eq!(
            storage
                .get(NodeId::from_idx_and_node_type(0, NodeType::Inner))
                .unwrap(),
            Node::Inner(inner_node)
        );
        assert_eq!(
            storage
                .get(NodeId::from_idx_and_node_type(0, NodeType::Leaf2))
                .unwrap(),
            Node::Leaf2(leaf_node_2)
        );
        assert_eq!(
            storage
                .get(NodeId::from_idx_and_node_type(0, NodeType::Leaf256))
                .unwrap(),
            Node::Leaf256(leaf_node_256)
        );
    }

    #[test]
    fn get_non_existent_id_returns_not_found_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        let storage = FileStorageManager::open(path).unwrap();

        let node = Node::Inner(Box::default());
        let id = storage.reserve(&node);

        // id has not been set yet
        assert!(matches!(storage.get(id), Err(Error::NotFound)));

        storage.set(id, &node).unwrap();
        // id has been set, and get should succeed
        assert_eq!(storage.get(id).unwrap(), node);
    }

    #[test]
    fn reserve_retrieves_node_id_from_node_file_storage_depending_on_node_type() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        let inner_count = 1;
        let leaf_2_count = 2;
        let leaf_256_count = 3;

        // to tests that the id comes from the correct file storage, we create different number of
        // nodes for each type
        let inner_node = Box::new(InnerNode::default());
        let leaf_node_2 = Box::new(SparseLeafNode::default());
        let leaf_node_256 = Box::new(FullLeafNode::default());

        fs::create_dir_all(path.join(FileStorageManager::INNER_NODE_DIR)).unwrap();
        fs::create_dir_all(path.join(FileStorageManager::LEAF_NODE_2_DIR)).unwrap();
        fs::create_dir_all(path.join(FileStorageManager::LEAF_NODE_256_DIR)).unwrap();

        let mut file = File::create(
            path.join(FileStorageManager::INNER_NODE_DIR)
                .join(NodeFileStorage::NODE_STORE_FILE),
        )
        .unwrap();
        for _ in 0..inner_count {
            file.write_all(inner_node.as_bytes()).unwrap();
        }

        let mut file = File::create(
            path.join(FileStorageManager::LEAF_NODE_2_DIR)
                .join(NodeFileStorage::NODE_STORE_FILE),
        )
        .unwrap();
        for _ in 0..leaf_2_count {
            file.write_all(leaf_node_2.as_bytes()).unwrap();
        }

        let mut file = File::create(
            path.join(FileStorageManager::LEAF_NODE_256_DIR)
                .join(NodeFileStorage::NODE_STORE_FILE),
        )
        .unwrap();
        for _ in 0..leaf_256_count {
            file.write_all(leaf_node_256.as_bytes()).unwrap();
        }

        let storage = FileStorageManager::open(path).unwrap();

        // all empty nodes have id 0
        assert_eq!(
            storage.reserve(&Node::Empty),
            NodeId::from_idx_and_node_type(0, NodeType::Empty)
        );
        assert_eq!(
            storage.reserve(&Node::Inner(inner_node)),
            NodeId::from_idx_and_node_type(inner_count, NodeType::Inner)
        );
        assert_eq!(
            storage.reserve(&Node::Leaf2(leaf_node_2)),
            NodeId::from_idx_and_node_type(leaf_2_count, NodeType::Leaf2)
        );
        assert_eq!(
            storage.reserve(&Node::Leaf256(leaf_node_256)),
            NodeId::from_idx_and_node_type(leaf_256_count, NodeType::Leaf256)
        );
    }

    #[test]
    fn set_updates_node_in_correct_node_file_storage() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        let inner_node = Box::new(InnerNode::default());
        let leaf_node_2 = Box::new(SparseLeafNode::default());
        let leaf_node_256 = Box::new(FullLeafNode::default());

        {
            let storage = FileStorageManager::open(path).unwrap();

            // the empty node is not stored
            let empty_node = Node::Empty;
            storage
                .set(storage.reserve(&empty_node), &empty_node)
                .unwrap();

            let inner_node = Node::Inner(inner_node.clone());
            storage
                .set(storage.reserve(&inner_node), &inner_node)
                .unwrap();

            let leaf_node_2 = Node::Leaf2(leaf_node_2.clone());
            storage
                .set(storage.reserve(&leaf_node_2), &leaf_node_2)
                .unwrap();

            let leaf_node_256 = Node::Leaf256(leaf_node_256.clone());
            storage
                .set(storage.reserve(&leaf_node_256), &leaf_node_256)
                .unwrap();
        }

        let reuse_list_file = path
            .join(FileStorageManager::INNER_NODE_DIR)
            .join(NodeFileStorage::NODE_STORE_FILE);
        assert_eq!(fs::read(reuse_list_file).unwrap(), inner_node.as_bytes());

        let reuse_list_file = path
            .join(FileStorageManager::LEAF_NODE_2_DIR)
            .join(NodeFileStorage::NODE_STORE_FILE);
        assert_eq!(fs::read(reuse_list_file).unwrap(), leaf_node_2.as_bytes());

        let reuse_list_file = path
            .join(FileStorageManager::LEAF_NODE_256_DIR)
            .join(NodeFileStorage::NODE_STORE_FILE);
        assert_eq!(fs::read(reuse_list_file).unwrap(), leaf_node_256.as_bytes());
    }

    #[test]
    fn set_non_existent_id_returns_not_found_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        let storage = FileStorageManager::open(path).unwrap();

        let id = NodeId::from_idx_and_node_type(0, NodeType::Inner);
        let node = Node::Inner(Box::default());

        assert!(matches!(storage.set(id, &node), Err(Error::NotFound)));
    }

    #[test]
    fn set_returns_error_if_node_id_prefix_and_node_type_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        let storage = FileStorageManager::open(path).unwrap();

        let id = NodeId::from_idx_and_node_type(0, NodeType::Leaf2);
        let node = Node::Inner(Box::default());

        assert!(matches!(
            storage.set(id, &node),
            Err(Error::IdNodeTypeMismatch)
        ));
    }

    #[test]
    fn delete_adds_node_id_to_reuse_list_in_reuse_list_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        let inner_node = Box::new(InnerNode::default());
        let leaf_node_2 = Box::new(SparseLeafNode::<2>::default());
        let leaf_node_256 = Box::new(FullLeafNode::default());

        fs::create_dir_all(path.join(FileStorageManager::INNER_NODE_DIR)).unwrap();
        fs::create_dir_all(path.join(FileStorageManager::LEAF_NODE_2_DIR)).unwrap();
        fs::create_dir_all(path.join(FileStorageManager::LEAF_NODE_256_DIR)).unwrap();

        File::create(
            path.join(FileStorageManager::INNER_NODE_DIR)
                .join(NodeFileStorage::NODE_STORE_FILE),
        )
        .unwrap()
        .write_all(inner_node.as_bytes())
        .unwrap();
        File::create(
            path.join(FileStorageManager::LEAF_NODE_2_DIR)
                .join(NodeFileStorage::NODE_STORE_FILE),
        )
        .unwrap()
        .write_all(leaf_node_2.as_bytes())
        .unwrap();
        File::create(
            path.join(FileStorageManager::LEAF_NODE_256_DIR)
                .join(NodeFileStorage::NODE_STORE_FILE),
        )
        .unwrap()
        .write_all(leaf_node_256.as_bytes())
        .unwrap();

        let storage = FileStorageManager::open(path).unwrap();
        // the empty node is not stored so deleting it is a no-op
        storage
            .delete(NodeId::from_idx_and_node_type(0, NodeType::Empty))
            .unwrap();
        storage
            .delete(NodeId::from_idx_and_node_type(0, NodeType::Inner))
            .unwrap();
        storage
            .delete(NodeId::from_idx_and_node_type(0, NodeType::Leaf2))
            .unwrap();
        storage
            .delete(NodeId::from_idx_and_node_type(0, NodeType::Leaf256))
            .unwrap();

        drop(storage);

        let reuse_list_file = path
            .join(FileStorageManager::INNER_NODE_DIR)
            .join(NodeFileStorage::REUSE_LIST_FILE);
        assert_eq!(fs::read(reuse_list_file).unwrap(), [0; 8]);
        let reuse_list_file = path
            .join(FileStorageManager::LEAF_NODE_2_DIR)
            .join(NodeFileStorage::REUSE_LIST_FILE);
        assert_eq!(fs::read(reuse_list_file).unwrap(), [0; 8]);
        let reuse_list_file = path
            .join(FileStorageManager::LEAF_NODE_256_DIR)
            .join(NodeFileStorage::REUSE_LIST_FILE);
        assert_eq!(fs::read(reuse_list_file).unwrap(), [0; 8]);
    }

    #[test]
    fn flush_flushes_all_node_file_storages() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        let inner_node = Box::new(InnerNode::default());
        let leaf_node_2 = Box::new(SparseLeafNode::<_>::default());
        let leaf_node_256 = Box::new(FullLeafNode::default());

        let storage = FileStorageManager::open(path).unwrap();
        let id = storage.reserve(&Node::Empty);
        storage.set(id, &Node::Empty).unwrap();
        let id = storage.reserve(&Node::Inner(inner_node.clone()));
        storage.set(id, &Node::Inner(inner_node.clone())).unwrap();
        let id = storage.reserve(&Node::Leaf2(leaf_node_2.clone()));
        storage.set(id, &Node::Leaf2(leaf_node_2.clone())).unwrap();
        let id = storage.reserve(&Node::Leaf256(leaf_node_256.clone()));
        storage
            .set(id, &Node::Leaf256(leaf_node_256.clone()))
            .unwrap();

        storage.flush().unwrap();

        let reuse_list_file = path
            .join(FileStorageManager::INNER_NODE_DIR)
            .join(NodeFileStorage::NODE_STORE_FILE);
        assert_eq!(fs::read(reuse_list_file).unwrap(), inner_node.as_bytes());

        let reuse_list_file = path
            .join(FileStorageManager::LEAF_NODE_2_DIR)
            .join(NodeFileStorage::NODE_STORE_FILE);
        assert_eq!(fs::read(reuse_list_file).unwrap(), leaf_node_2.as_bytes());

        let reuse_list_file = path
            .join(FileStorageManager::LEAF_NODE_256_DIR)
            .join(NodeFileStorage::NODE_STORE_FILE);
        assert_eq!(fs::read(reuse_list_file).unwrap(), leaf_node_256.as_bytes());
    }
}
