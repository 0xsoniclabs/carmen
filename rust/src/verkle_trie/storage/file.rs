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
    io::{Seek, SeekFrom},
    marker::PhantomData,
    path::Path,
};

use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::verkle_trie::storage::{
    AllChildrenLeafNode, Error, Id, InnerNode, Node, Storage, TwoChildrenLeafNode,
};

struct FileStorageManager {
    inner_nodes: NodeFileStorage<InnerNode>,
    two_children_leaf_nodes: NodeFileStorage<TwoChildrenLeafNode>,
    all_children_leaf_nodes: NodeFileStorage<AllChildrenLeafNode>,
}

impl FileStorageManager {
    const INNER_NODE_PREFIX: u64 = 0x00_00_00_00_00_00_00_00;
    const TWO_CHILDREN_LEAF_NODE_PREFIX: u64 = 0x01_00_00_00_00_00_00_00;
    const ALL_CHILDREN_LEAF_NODE_PREFIX: u64 = 0x02_00_00_00_00_00_00_00;
    const PREFIX_MASK: u64 = 0xff_00_00_00_00_00_00_00;

    const INNER_NODE_PATH: &str = "inner_node_storage.bin";
    const TWO_CHILDREN_LEAF_NODE_PATH: &str = "two_children_leaf_node_storage.bin";
    const ALL_CHILDREN_LEAF_NODE_PATH: &str = "all_children_leaf_node_storage.bin";

    fn new_in_dir(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let dir = dir.as_ref();
        Self::new(
            dir.join(Self::INNER_NODE_PATH),
            dir.join(Self::TWO_CHILDREN_LEAF_NODE_PATH),
            dir.join(Self::ALL_CHILDREN_LEAF_NODE_PATH),
        )
    }

    fn new(
        inner_node_path: impl AsRef<Path>,
        two_children_leaf_node_path: impl AsRef<Path>,
        all_children_leaf_node_path: impl AsRef<Path>,
    ) -> Result<Self, Error> {
        Ok(Self {
            inner_nodes: NodeFileStorage::new(inner_node_path)?,
            two_children_leaf_nodes: NodeFileStorage::new(two_children_leaf_node_path)?,
            all_children_leaf_nodes: NodeFileStorage::new(all_children_leaf_node_path)?,
        })
    }
}

impl Storage for FileStorageManager {
    fn read(&self, id: Id) -> Result<Node, Error> {
        let idx = id & !Self::PREFIX_MASK;
        match id & Self::PREFIX_MASK {
            Self::INNER_NODE_PREFIX => {
                let node = self.inner_nodes.read(idx)?;
                Ok(Node::Inner(Box::new(node)))
            }
            Self::TWO_CHILDREN_LEAF_NODE_PREFIX => {
                let node = self.two_children_leaf_nodes.read(idx)?;
                Ok(Node::TwoChildren(Box::new(node)))
            }
            Self::ALL_CHILDREN_LEAF_NODE_PREFIX => {
                let node = self.all_children_leaf_nodes.read(idx)?;
                Ok(Node::AllChildren(Box::new(node)))
            }
            _ => Err(Error::NotFound),
        }
    }

    fn create(&self, node: &Node) -> Result<Id, Error> {
        match node {
            Node::Inner(inner_node) => {
                let idx = self.inner_nodes.create(inner_node)?;
                let id = idx | Self::INNER_NODE_PREFIX;
                Ok(id)
            }
            Node::TwoChildren(two_children_leaf_node) => {
                let idx = self
                    .two_children_leaf_nodes
                    .create(two_children_leaf_node)?;
                let id = idx | Self::TWO_CHILDREN_LEAF_NODE_PREFIX;
                Ok(id)
            }
            Node::AllChildren(all_children_leaf_node) => {
                let idx = self
                    .all_children_leaf_nodes
                    .create(all_children_leaf_node)?;
                let id = idx | Self::ALL_CHILDREN_LEAF_NODE_PREFIX;
                Ok(id)
            }
        }
    }

    fn flush(&self) -> Result<(), Error> {
        self.inner_nodes.flush()?;
        self.two_children_leaf_nodes.flush()?;
        self.all_children_leaf_nodes.flush()
    }
}

struct NodeFileStorage<T> {
    file: File,
    _node_type: PhantomData<T>,
}

impl<T: FromBytes + IntoBytes + Immutable> NodeFileStorage<T> {
    fn new(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        Ok(Self {
            file,
            _node_type: PhantomData,
        })
    }

    fn read(&self, idx: Id) -> Result<T, Error> {
        let offset = idx * size_of::<T>() as u64;
        if self.file.metadata()?.len() < offset + size_of::<T>() as u64 {
            return Err(Error::NotFound);
        }
        (&self.file).seek(SeekFrom::Start(offset))?;
        let node = T::read_from_io(&self.file)?;
        Ok(node)
    }

    fn create(&self, node: &T) -> Result<Id, Error> {
        let len = (&self.file).seek(SeekFrom::End(0))?;
        node.write_to_io(&self.file)?;
        let idx = len / size_of::<T>() as u64;
        Ok(idx)
    }

    fn flush(&self) -> Result<(), Error> {
        self.file.sync_data()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{io::Write, sync::Arc, time::Duration};

    use super::*;

    #[test]
    fn new_at_default_paths_creates_files_if_they_do_not_exist() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        let storage = FileStorageManager::new_in_dir(path);
        assert!(storage.is_ok());
        assert!(std::fs::exists(path.join(FileStorageManager::INNER_NODE_PATH)).unwrap());
        assert!(
            std::fs::exists(path.join(FileStorageManager::TWO_CHILDREN_LEAF_NODE_PATH)).unwrap()
        );
        assert!(
            std::fs::exists(path.join(FileStorageManager::ALL_CHILDREN_LEAF_NODE_PATH)).unwrap()
        );
    }

    #[test]
    fn read_write() {
        let dir = tempfile::tempdir().unwrap();
        let storage = FileStorageManager::new_in_dir(dir).unwrap();

        let node_1 = Node::Inner(Box::new(InnerNode {
            values: [[1; 32]; 256],
        }));
        let node_2 = Node::TwoChildren(Box::new(TwoChildrenLeafNode {
            values: [[2; 32]; 2],
        }));
        let node_3 = Node::AllChildren(Box::new(AllChildrenLeafNode {
            values: [[3; 32]; 256],
        }));
        let node_4 = Node::Inner(Box::new(InnerNode {
            values: [[4; 32]; 256],
        }));
        let node_5 = Node::TwoChildren(Box::new(TwoChildrenLeafNode {
            values: [[5; 32]; 2],
        }));
        let node_6 = Node::AllChildren(Box::new(AllChildrenLeafNode {
            values: [[6; 32]; 256],
        }));

        let id_1 = storage.create(&node_1).unwrap();
        let id_2 = storage.create(&node_2).unwrap();
        let id_3 = storage.create(&node_3).unwrap();
        let id_4 = storage.create(&node_4).unwrap();
        let id_5 = storage.create(&node_5).unwrap();
        let id_6 = storage.create(&node_6).unwrap();

        assert_eq!(storage.read(id_1).unwrap(), node_1);
        assert_eq!(storage.read(id_2).unwrap(), node_2);
        assert_eq!(storage.read(id_3).unwrap(), node_3);
        assert_eq!(storage.read(id_4).unwrap(), node_4);
        assert_eq!(storage.read(id_5).unwrap(), node_5);
        assert_eq!(storage.read(id_6).unwrap(), node_6);
    }

    #[test]
    fn read_non_existent_node_returns_not_found_error() {
        let dir = tempfile::tempdir().unwrap();
        let storage = FileStorageManager::new_in_dir(dir).unwrap();

        assert!(matches!(storage.read(0), Err(Error::NotFound)));

        let node = Node::Inner(Box::new(InnerNode {
            values: [[1; 32]; 256],
        }));
        let id = storage.create(&node).unwrap();
        assert_eq!(storage.read(id).unwrap(), node);
        assert!(matches!(storage.read(id + 1), Err(Error::NotFound)));
    }

    #[test]
    fn read_write_concurrently() {
        let dir = tempfile::tempdir().unwrap();

        let storage =
            NodeFileStorage::<InnerNode>::new(dir.path().join("test_storage.bin")).unwrap();
        let storage = Arc::new(storage);

        let node_1 = Box::new(InnerNode {
            values: [[1; 32]; 256],
        });
        let node_2 = Box::new(InnerNode {
            values: [[2; 32]; 256],
        });

        let t1 = std::thread::spawn({
            let storage = Arc::clone(&storage);
            let node_1 = node_1.clone();
            let node_2 = node_2.clone();
            move || {
                let idx_1 = storage.create(&node_1).unwrap();
                assert_eq!(&storage.read(idx_1).unwrap(), &*node_1);

                std::thread::sleep(Duration::from_millis(200));
                assert_eq!(&storage.read(idx_1 + 1).unwrap(), &*node_2);
            }
        });
        let t2 = std::thread::spawn({
            let storage = Arc::clone(&storage);
            move || {
                std::thread::sleep(Duration::from_millis(100));
                let idx_2 = storage.create(&node_2).unwrap();
                assert_eq!(&storage.read(idx_2).unwrap(), &*node_2);

                std::thread::sleep(Duration::from_millis(100));
                assert_eq!(&storage.read(idx_2 - 1).unwrap(), &*node_1);
            }
        });
        t1.join().unwrap();
        t2.join().unwrap();
    }

    #[test]
    fn cursor_is_shared() {
        let dir = tempfile::tempdir().unwrap();
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(dir.path().join("test_storage.bin"))
            .unwrap();
        let file = Arc::new(file);
        let t1 = std::thread::spawn({
            let file = Arc::clone(&file);
            move || {
                assert_eq!(file.metadata().unwrap().len(), 0);
                std::thread::sleep(Duration::from_millis(100));
                (&*file).write_all([1; 32].as_slice()).unwrap();
                assert_eq!(file.metadata().unwrap().len(), 32);
            }
        });
        let t2 = std::thread::spawn({
            let file = Arc::clone(&file);
            move || {
                assert_eq!(file.metadata().unwrap().len(), 0);
                std::thread::sleep(Duration::from_millis(200));
                assert_eq!(file.metadata().unwrap().len(), 32);
            }
        });
        t1.join().unwrap();
        t2.join().unwrap();
    }

    #[test]
    fn read_write_multiple_fds_interleaved() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_storage.bin");

        let node_1 = Box::new(InnerNode {
            values: [[1; 32]; 256],
        });
        let node_2 = Box::new(InnerNode {
            values: [[2; 32]; 256],
        });

        std::thread::scope(|s| {
            s.spawn({
                let path = path.clone();
                let node_1 = node_1.clone();
                let node_2 = node_2.clone();
                move || {
                    let storage = NodeFileStorage::<InnerNode>::new(path).unwrap();

                    let idx_1 = storage.create(&node_1).unwrap();
                    assert_eq!(&storage.read(idx_1).unwrap(), &*node_1);

                    std::thread::sleep(Duration::from_millis(200));
                    assert_eq!(&storage.read(idx_1 + 1).unwrap(), &*node_2);
                }
            });
            s.spawn({
                move || {
                    let storage = NodeFileStorage::<InnerNode>::new(path).unwrap();

                    std::thread::sleep(Duration::from_millis(100));
                    let idx_2 = storage.create(&node_2).unwrap();
                    assert_eq!(&storage.read(idx_2).unwrap(), &*node_2);

                    std::thread::sleep(Duration::from_millis(100));
                    assert_eq!(&storage.read(idx_2 - 1).unwrap(), &*node_1);
                }
            });
        });
    }
}
