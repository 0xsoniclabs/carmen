// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use derive_deftly::define_derive_deftly;

pub mod metadata;
pub mod root_ids_file;

define_derive_deftly! {
    /// This macro is supposed to be used on node enums and generates a FileStorageManager.
    ///
    /// The macro expects the following:
    ///   - The node enum must have the following shape assuming it is named `<NodeName>`:
    ///     - There must be a variant named `Empty` that contains a type called `Empty<NodeName>`
    ///     - All other variants must contain boxed types named `<VariantName><NodeName>`
    ///     ```ignore
    ///     enum Xx {
    ///       Empty(EmptyXx),
    ///       Aa(Box<AaXx>),
    ///       Bb(Box<BbXx>),
    ///     }
    ///     ```
    ///   - The node type enum must be called `<NodeName>Type` and its variants must have the same
    ///     names as the ones of the node enum itself.
    ///   - The ID type for this tree must be called `<NodeName>Id` and it must implement
    ///     `TreeId<NodeType = NodeNameType> + Copy + FromBytes + IntoBytes + Immutable + Send +
    ///     Sync`.
    ///   - The ID type, the node type enum and all variant type have to be in scope at the call/
    ///     expansion site.
    ///
    /// The generated FileStorageManager is named `<NodeName>FileStorageManager` and implements
    /// `Storage<Id = <NodeName>Id, Item = <NodeName>>`.
    FileStorageManager for enum:

${define MANAGER_TYPE ${paste $ttype FileStorageManager}}
${define ID ${paste $ttype Id}}
${define STORAGE_GENERICS
    $(
        ${if not(approx_equal($vname, Empty)) {
            ${paste ${upper_camel_case $vname} Storage},
        }}
    )
}
${define STORAGE_GENERICS_BOUNDS
    $(
        ${if not(approx_equal($vname, Empty)) {
            ${paste ${upper_camel_case $vname} Storage}: $crate::storage::Storage<Id = u64, Item = ${paste $vname $tname}> + $crate::storage::CheckpointParticipant,
        }}
    )
}

/// A storage manager for trie nodes for file based storage backends.
///
/// In order for concurrent operations to be safe (in that there are not data races) they have to
/// operate on different node ids.
#[derive(Debug)]
pub struct $MANAGER_TYPE<$STORAGE_GENERICS>
where
    $STORAGE_GENERICS_BOUNDS
{
    dir: ::std::path::PathBuf,
    checkpoint: $crate::sync::atomic::AtomicU64,
    $(
        ${if not(approx_equal($vname, Empty)) {
            ${snake_case $vname}: ${paste ${upper_camel_case $vname} Storage},
        }}
    )
    root_ids_file: $crate::storage::file::file_storage_manager::root_ids_file::RootIdsFile<$ID>,
}

impl<$STORAGE_GENERICS> $MANAGER_TYPE<$STORAGE_GENERICS>
where
    $STORAGE_GENERICS_BOUNDS
{
    $(
        ${if not(approx_equal($vname, Empty)) {
            pub const ${paste ${shouty_snake_case $vname} _DIR}: &str = std::stringify!(${snake_case $vname});
        }}
    )
    pub const DB_DIRTY_FILE: &str = "db_dirty.bin";
    pub const METADATA_FILE: &str = "metadata.bin";
    pub const COMMITTED_METADATA_FILE: &str = "committed_metadata.bin";
    pub const PREPARED_METADATA_FILE: &str = "prepared_metadata.bin";
    pub const ROOT_IDS_FILE: &str = "root_ids.bin";
}

impl<$STORAGE_GENERICS> $crate::storage::Storage for $MANAGER_TYPE<$STORAGE_GENERICS>
where
    $STORAGE_GENERICS_BOUNDS
{
    type Id = $ID;
    type Item = $tname;

    /// Opens or creates the file backends for the individual node types in the specified directory.
    fn open(dir: &::std::path::Path) -> $crate::error::BTResult<Self, $crate::storage::Error> {
        std::fs::create_dir_all(dir)?;

        if ::std::fs::exists(dir.join(Self::DB_DIRTY_FILE))? {
            return Err($crate::storage::Error::DirtyOpen.into());
        }

        ::std::fs::File::create(dir.join(Self::DB_DIRTY_FILE))?;

        let metadata =
            <$crate::storage::file::file_storage_manager::metadata::Metadata as $crate::storage::file::FromToFile>::read_or_init(dir.join(Self::METADATA_FILE))?;

        $(
            ${if not(approx_equal($vname, Empty)) {
                let ${snake_case $vname} = ${paste ${upper_camel_case $vname} Storage}::open(dir.join(Self::${paste ${shouty_snake_case $vname} _DIR}).as_path())?;
            }}
        )

        let root_ids_file =
            $crate::storage::file::file_storage_manager::root_ids_file::RootIdsFile::open(dir.join(Self::ROOT_IDS_FILE), metadata.root_id_count)?;

        $(
            ${if not(approx_equal($vname, Empty)) {
                ${snake_case $vname}.ensure(metadata.checkpoint_number)?;
            }}
        )

        Ok(Self {
            dir: dir.to_path_buf(),
            checkpoint: $crate::sync::atomic::AtomicU64::new(metadata.checkpoint_number),
            $(
                ${if not(approx_equal($vname, Empty)) {
                    ${snake_case $vname},
                }}
            )
            root_ids_file,
        })
    }

    fn get(&self, id: Self::Id) -> $crate::error::BTResult<Self::Item, $crate::storage::Error> {
        let idx = $crate::types::TreeId::to_index(id);
        match $crate::types::TreeId::to_node_type(id).ok_or($crate::storage::Error::InvalidId)? {
            ${paste $tname Type}::Empty => Ok(Self::Item::Empty(${paste Empty $tname})),
            $(
                ${if not(approx_equal($vname, Empty)) {
                    ${paste $tname Type}::$vname => {
                        let node = self.${snake_case $vname}.get(idx)?;
                        Ok(Self::Item::$vname(Box::new(node)))
                    }
                }}
            )
        }
    }

    fn reserve(&self, node: &Self::Item) -> Self::Id {
        match node {
            Self::Item::Empty(_) => <Self::Id as $crate::types::TreeId>::from_idx_and_node_type(0, ${paste $tname Type}::Empty),
            $(
                ${if not(approx_equal($vname, Empty)) {
                    Self::Item::$vname(node) => {
                        let idx = self.${snake_case $vname}.reserve(node);
                        <Self::Id as $crate::types::TreeId>::from_idx_and_node_type(idx, ${paste $tname Type}::$vname)
                    }
                }}
            )
        }
    }

    fn set(&self, id: Self::Id, node: &Self::Item) -> $crate::error::BTResult<(), $crate::storage::Error> {
        let idx = $crate::types::TreeId::to_index(id);
        match (node, $crate::types::TreeId::to_node_type(id).ok_or($crate::storage::Error::InvalidId)?) {
            (Self::Item::Empty(_), ${paste $tname Type}::Empty) => Ok(()),
            $(
                ${if not(approx_equal($vname, Empty)) {
                    ($tname::$vname(node), ${paste $tname Type}::$vname) => self.${snake_case $vname}.set(idx, node),
                }}
            )
            _ => Err($crate::storage::Error::IdNodeTypeMismatch.into())
        }
    }

    fn delete(&self, id: Self::Id) -> $crate::error::BTResult<(), $crate::storage::Error> {
        let idx = $crate::types::TreeId::to_index(id);
        match $crate::types::TreeId::to_node_type(id).ok_or($crate::storage::Error::InvalidId)? {
            ${paste $ttype Type}::Empty => Ok(()),
            $(
                ${if not(approx_equal($vname, Empty)) {
                    ${paste $tname Type}::$vname => self.${snake_case $vname}.delete(idx),
                }}
            )
        }
    }

    fn close(self) -> $crate::error::BTResult<(), $crate::storage::Error> {
        $(
            ${if not(approx_equal($vname, Empty)) {
                self.${snake_case $vname}.close()?;
            }}
        )
        <$crate::storage::file::file_storage_manager::metadata::Metadata as $crate::storage::file::FromToFile>::write(
            &$crate::storage::file::file_storage_manager::metadata::Metadata {
                checkpoint_number: self.checkpoint.load(::std::sync::atomic::Ordering::Acquire),
                root_id_count: self.root_ids_file.count(),
            },
            self.dir.join(Self::METADATA_FILE)
        )?;
        ::std::fs::remove_file(self.dir.join(Self::DB_DIRTY_FILE))?;
        Ok(())
    }
}

impl<$STORAGE_GENERICS> $crate::storage::Checkpointable for $MANAGER_TYPE<$STORAGE_GENERICS>
where
    $STORAGE_GENERICS_BOUNDS
{
    fn checkpoint(&self) -> $crate::error::BTResult<u64, $crate::storage::Error> {
        let current_checkpoint = self.checkpoint.load($crate::sync::atomic::Ordering::Acquire);
        let new_checkpoint = current_checkpoint + 1;
        let participants = [
            $(
                ${if not(approx_equal($vname, Empty)) {
                    &self.${snake_case $vname} as &dyn $crate::storage::CheckpointParticipant,
                }}
            )
        ];
        for (i, participant) in participants.iter().enumerate() {
            if let Err(err) = participant.prepare(new_checkpoint) {
                for participant in participants[..i].iter().rev() {
                    participant.abort(current_checkpoint)?;
                }
                return Err(err);
            }
        }
        $crate::storage::file::FromToFile::write(
            &$crate::storage::file::file_storage_manager::metadata::Metadata {
                checkpoint_number: new_checkpoint,
                root_id_count: self.root_ids_file.count(),
            },
            self.dir.join(Self::PREPARED_METADATA_FILE)
        )?;
        ::std::fs::rename(
            self.dir.join(Self::PREPARED_METADATA_FILE),
            self.dir.join(Self::COMMITTED_METADATA_FILE),
        )?;
        for participant in participants.iter() {
            participant.commit(new_checkpoint)?;
        }
        self.checkpoint.store(new_checkpoint, $crate::sync::atomic::Ordering::Release);

        Ok(new_checkpoint)
    }

    fn restore(dir: &::std::path::Path, checkpoint: u64) -> $crate::error::BTResult<(), $crate::storage::Error> {
        let committed_metadata = <$crate::storage::file::file_storage_manager::metadata::Metadata as $crate::storage::file::FromToFile>::read_or_init(
            dir.join(Self::COMMITTED_METADATA_FILE)
        )?;
        if checkpoint != committed_metadata.checkpoint_number {
            return Err($crate::storage::Error::Checkpoint.into());
        }
        $(
            ${if not(approx_equal($vname, Empty)) {
                ${paste ${upper_camel_case $vname} Storage}::restore(dir.join(Self::${paste ${shouty_snake_case $vname} _DIR}).as_path(), checkpoint)?;
            }}
        )
        <$crate::storage::file::file_storage_manager::metadata::Metadata as $crate::storage::file::FromToFile>::write(
            &committed_metadata,
            dir.join(Self::METADATA_FILE)
        )?;
        let dirty_file_path = dir.join(Self::DB_DIRTY_FILE);
        if ::std::fs::exists(&dirty_file_path)? {
            ::std::fs::remove_file(&dirty_file_path)?;
        }
        Ok(())
    }
}

impl<$STORAGE_GENERICS> $crate::storage::RootIdProvider for $MANAGER_TYPE<$STORAGE_GENERICS>
where
    $STORAGE_GENERICS_BOUNDS
{
    type Id = $ID;

    fn get_root_id(&self, block_number: u64) -> $crate::error::BTResult<Self::Id, $crate::storage::Error> {
        self.root_ids_file.get(block_number)
    }

    fn set_root_id(&self, block_number: u64, id: Self::Id) -> $crate::error::BTResult<(), $crate::storage::Error> {
        self.root_ids_file.set(block_number, id)
    }
}
}

pub(crate) use derive_deftly_template_FileStorageManager;
#[cfg(test)]
pub use tests::{TestNode, TestNodeFileStorageManager, TestNodeId, TestNodeType};

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, File},
        path::Path,
    };

    use derive_deftly::Deftly;
    use mockall::{Sequence, predicate::eq};
    use zerocopy::{FromBytes, Immutable, IntoBytes};

    use crate::{
        error::{BTError, BTResult},
        storage::{
            CheckpointParticipant, Checkpointable, Error, Storage,
            file::{
                FromToFile, NodeFileStorage, SeekFile,
                file_storage_manager::{metadata::Metadata, root_ids_file::RootIdsFile},
            },
        },
        sync::atomic::{AtomicBool, AtomicU64, Ordering},
        types::{NodeSize, TreeId},
        utils::test_dir::{Permissions, TestDir},
    };

    #[test]
    fn open_creates_directory_and_calls_open_on_all_storages_and_creates_db_dirty_file() {
        type FileStorageManager = TestNodeFileStorageManager<
            NodeFileStorage<NonEmpty1TestNode, SeekFile>,
            NodeFileStorage<NonEmpty2TestNode, SeekFile>,
        >;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let storage = FileStorageManager::open(&dir);
        assert!(storage.is_ok());
        let sub_dirs = [
            FileStorageManager::NON_EMPTY1_DIR,
            FileStorageManager::NON_EMPTY2_DIR,
        ];
        let files = [
            NodeFileStorage::<NonEmpty1TestNode, SeekFile>::NODE_STORE_FILE,
            NodeFileStorage::<NonEmpty1TestNode, SeekFile>::REUSE_LIST_FILE,
            NodeFileStorage::<NonEmpty1TestNode, SeekFile>::METADATA_FILE,
        ];
        for sub_dir in &sub_dirs {
            assert!(fs::exists(dir.join(sub_dir)).unwrap());
            for file in &files {
                assert!(fs::exists(dir.join(sub_dir).join(file)).unwrap());
            }
        }
        assert!(fs::exists(dir.join(FileStorageManager::DB_DIRTY_FILE)).unwrap());
    }

    #[test]
    fn open_opens_existing_files() {
        type FileStorageManager = TestNodeFileStorageManager<
            NodeFileStorage<NonEmpty1TestNode, SeekFile>,
            NodeFileStorage<NonEmpty2TestNode, SeekFile>,
        >;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let sub_dirs = [
            FileStorageManager::NON_EMPTY1_DIR,
            FileStorageManager::NON_EMPTY2_DIR,
        ];
        for sub_dir in &sub_dirs {
            fs::create_dir_all(dir.join(sub_dir)).unwrap();
            // because we are not writing any nodes, the node type does not matter
            NodeFileStorage::<NonEmpty1TestNode, SeekFile>::create_files(
                dir.join(sub_dir),
                &[],
                0,
                &[],
                0,
            )
            .unwrap();
        }

        let storage = FileStorageManager::open(&dir);
        assert!(storage.is_ok());
    }

    #[test]
    fn open_fails_if_db_dirty_flag_file_exists() {
        type FileStorageManager = TestNodeFileStorageManager<
            MockStorage<NonEmpty1TestNode>,
            MockStorage<NonEmpty2TestNode>,
        >;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        File::create(dir.join(FileStorageManager::DB_DIRTY_FILE)).unwrap();

        assert!(matches!(
            FileStorageManager::open(&dir).map_err(BTError::into_inner),
            Err(Error::DirtyOpen)
        ));
    }

    #[test]
    fn open_propagates_io_errors() {
        type FileStorageManager = TestNodeFileStorageManager<
            MockStorage<NonEmpty1TestNode>,
            MockStorage<NonEmpty2TestNode>,
        >;

        let dir = TestDir::try_new(Permissions::ReadOnly).unwrap();

        let path = dir.join("non_existent_dir");

        assert!(matches!(
            FileStorageManager::open(&path).map_err(BTError::into_inner),
            Err(Error::Io(_))
        ));
    }

    #[test]
    fn get_forwards_to_get_of_corresponding_node_file_storage_depending_on_node_type() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = TestNodeFileStorageManager {
            dir: dir.to_path_buf(),
            checkpoint: AtomicU64::new(0),
            non_empty1: MockStorage::new(),
            non_empty2: MockStorage::new(),
            root_ids_file: RootIdsFile::open(dir.join("root_ids"), 0).unwrap(),
        };

        // TestNode::Empty
        {
            let node_id = TestNodeId::from_idx_and_node_type(0, TestNodeType::Empty);
            let node = EmptyTestNode;
            assert_eq!(storage.get(node_id).unwrap(), TestNode::Empty(node));
        }

        // TestNode::NonEmpty1
        {
            let node_id = TestNodeId::from_idx_and_node_type(1, TestNodeType::NonEmpty1);
            let node = NonEmpty1TestNode::default();
            storage
                .non_empty1
                .expect_get()
                .with(eq(node_id.to_index()))
                .returning({
                    let node = node.clone();
                    move |_| Ok(node.clone())
                });
            assert_eq!(
                storage.get(node_id).unwrap(),
                TestNode::NonEmpty1(Box::new(node))
            );
        }

        // TestNode::NonEmpty2
        {
            let node_id = TestNodeId::from_idx_and_node_type(1, TestNodeType::NonEmpty2);
            let node = NonEmpty2TestNode::default();
            storage
                .non_empty2
                .expect_get()
                .with(eq(node_id.to_index()))
                .returning({
                    let node = node.clone();
                    move |_| Ok(node.clone())
                });
            assert_eq!(
                storage.get(node_id).unwrap(),
                TestNode::NonEmpty2(Box::new(node))
            );
        }
    }

    #[test]
    fn reserve_forwards_to_reserve_of_corresponding_node_file_storage_depending_on_node_type() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = TestNodeFileStorageManager {
            dir: dir.to_path_buf(),
            checkpoint: AtomicU64::new(0),
            non_empty1: MockStorage::new(),
            non_empty2: MockStorage::new(),
            root_ids_file: RootIdsFile::<TestNodeId>::open(dir.join("root_ids"), 0).unwrap(),
        };

        // TestNode::Empty
        {
            let node_idx = 0;
            let node = EmptyTestNode;
            assert_eq!(
                storage.reserve(&TestNode::Empty(node)),
                TestNodeId::from_idx_and_node_type(node_idx, TestNodeType::Empty)
            );
        }

        // TestNode::NonEmpty1
        {
            let node_idx = 1;
            let node = NonEmpty1TestNode::default();
            storage
                .non_empty1
                .expect_reserve()
                .with(eq(node.clone()))
                .returning(move |_| node_idx);
            assert_eq!(
                storage.reserve(&TestNode::NonEmpty1(Box::new(node))),
                TestNodeId::from_idx_and_node_type(node_idx, TestNodeType::NonEmpty1)
            );
        }

        // TestNode::NonEmpty
        {
            let node_idx = 1;
            let node = NonEmpty2TestNode::default();
            storage
                .non_empty2
                .expect_reserve()
                .with(eq(node.clone()))
                .returning(move |_| node_idx);
            assert_eq!(
                storage.reserve(&TestNode::NonEmpty2(Box::new(node))),
                TestNodeId::from_idx_and_node_type(node_idx, TestNodeType::NonEmpty2)
            );
        }
    }

    #[test]
    fn set_forwards_to_set_of_corresponding_node_file_storage_depending_on_node_type() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = TestNodeFileStorageManager {
            dir: dir.to_path_buf(),
            checkpoint: AtomicU64::new(0),
            non_empty1: MockStorage::new(),
            non_empty2: MockStorage::new(),
            root_ids_file: RootIdsFile::open(dir.join("root_ids"), 0).unwrap(),
        };

        // TestNode::Empty
        {
            let node_id = TestNodeId::from_idx_and_node_type(0, TestNodeType::Empty);
            let node = EmptyTestNode;
            let node = TestNode::Empty(node);
            assert!(storage.set(node_id, &node).is_ok());
        }

        // TestNode::NonEmpty1
        {
            let node_id = TestNodeId::from_idx_and_node_type(1, TestNodeType::NonEmpty1);
            let node = NonEmpty1TestNode::default();
            storage
                .non_empty1
                .expect_set()
                .with(eq(node_id.to_index()), eq(node.clone()))
                .returning(move |_, _| Ok(()));
            let node = TestNode::NonEmpty1(Box::new(node));
            assert!(storage.set(node_id, &node).is_ok());
        }

        // TestNode::NonEmpty2
        {
            let node_id = TestNodeId::from_idx_and_node_type(1, TestNodeType::NonEmpty2);
            let node = NonEmpty2TestNode::default();
            storage
                .non_empty2
                .expect_set()
                .with(eq(node_id.to_index()), eq(node.clone()))
                .returning(move |_, _| Ok(()));
            let node = TestNode::NonEmpty2(Box::new(node));
            assert!(storage.set(node_id, &node).is_ok());
        }
    }

    #[test]
    fn set_returns_error_if_node_id_prefix_and_node_type_mismatch() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let storage = TestNodeFileStorageManager {
            dir: dir.to_path_buf(),
            checkpoint: AtomicU64::new(0),
            non_empty1: MockStorage::new(),
            non_empty2: MockStorage::new(),
            root_ids_file: RootIdsFile::open(dir.join("root_ids"), 0).unwrap(),
        };

        let id = TestNodeId::from_idx_and_node_type(0, TestNodeType::NonEmpty2);
        let node = TestNode::NonEmpty1(Box::default());

        assert!(matches!(
            storage.set(id, &node).map_err(BTError::into_inner),
            Err(Error::IdNodeTypeMismatch)
        ));
    }

    #[test]
    fn delete_forwards_to_delete_of_corresponding_node_file_storage_depending_on_node_type() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = TestNodeFileStorageManager {
            dir: dir.to_path_buf(),
            checkpoint: AtomicU64::new(0),
            non_empty1: MockStorage::new(),
            non_empty2: MockStorage::new(),
            root_ids_file: RootIdsFile::open(dir.join("root_ids"), 0).unwrap(),
        };

        // TestNode::Empty
        {
            let node_id = TestNodeId::from_idx_and_node_type(0, TestNodeType::Empty);
            assert!(storage.delete(node_id).is_ok());
        }

        // TestNode::NonEmpty1
        {
            let node_id = TestNodeId::from_idx_and_node_type(1, TestNodeType::NonEmpty1);
            storage
                .non_empty1
                .expect_delete()
                .with(eq(node_id.to_index()))
                .returning(move |_| Ok(()));
            assert!(storage.delete(node_id).is_ok());
        }

        // TestNode::NonEmpty2
        {
            let node_id = TestNodeId::from_idx_and_node_type(1, TestNodeType::NonEmpty2);
            storage
                .non_empty2
                .expect_delete()
                .with(eq(node_id.to_index()))
                .returning(move |_| Ok(()));
            assert!(storage.delete(node_id).is_ok());
        }
    }

    #[test]
    fn close_calls_close_on_all_storages_and_writes_metadata_and_removes_db_dirty_file() {
        type TestNodeFileStorageManager = super::TestNodeFileStorageManager<
            MockStorage<NonEmpty1TestNode>,
            MockStorage<NonEmpty2TestNode>,
        >;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = TestNodeFileStorageManager {
            dir: dir.to_path_buf(),
            checkpoint: AtomicU64::new(0),
            non_empty1: MockStorage::new(),
            non_empty2: MockStorage::new(),
            root_ids_file: RootIdsFile::open(dir.join("root_ids"), 0).unwrap(),
        };

        File::create(dir.join(TestNodeFileStorageManager::DB_DIRTY_FILE)).unwrap();

        storage
            .non_empty1
            .expect_close()
            .returning(|| Ok(()))
            .times(1);
        storage
            .non_empty2
            .expect_close()
            .returning(|| Ok(()))
            .times(1);
        storage
            .root_ids_file
            .set(
                0,
                TestNodeId::from_idx_and_node_type(0, TestNodeType::Empty),
            )
            .unwrap();

        storage.close().unwrap();

        assert_eq!(
            fs::read(dir.join(TestNodeFileStorageManager::METADATA_FILE)).unwrap(),
            Metadata {
                checkpoint_number: 0,
                root_id_count: 1,
            }
            .as_bytes()
        );
        assert!(!fs::exists(dir.join(TestNodeFileStorageManager::DB_DIRTY_FILE)).unwrap());
    }

    #[test]
    fn checkpoint_follows_correct_sequence_for_two_phase_commit() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let old_checkpoint = 1;

        let mut storage = TestNodeFileStorageManager {
            dir: dir.to_path_buf(),
            checkpoint: AtomicU64::new(old_checkpoint),
            non_empty1: MockStorage::new(),
            non_empty2: MockStorage::new(),
            root_ids_file: RootIdsFile::<TestNodeId>::open(dir.join("root_ids"), 0).unwrap(),
        };

        let mut seq = Sequence::new();
        storage
            .non_empty1
            .expect_prepare()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);
        storage
            .non_empty2
            .expect_prepare()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);

        storage
            .non_empty1
            .expect_commit()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);
        storage
            .non_empty2
            .expect_commit()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);

        assert!(storage.checkpoint().is_ok());

        // The prepared metadata file should not exist after a successful checkpoint.
        assert!(!fs::exists(dir.join(
            TestNodeFileStorageManager::<MockStorage<_>, MockStorage<_>>::PREPARED_METADATA_FILE
        )).unwrap());
        // The committed metadata file should exist and contain the new checkpoint.
        assert_eq!(
            Metadata::read_or_init(dir.join(
                TestNodeFileStorageManager::<MockStorage<_>, MockStorage<_>>::COMMITTED_METADATA_FILE,
            )).unwrap().checkpoint_number,
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
        let old_metadata = Metadata {
            checkpoint_number: old_checkpoint,
            root_id_count: 0,
        };
        old_metadata.write(dir.join(
            TestNodeFileStorageManager::<MockStorage<_>, MockStorage<_>>::COMMITTED_METADATA_FILE,
        )).unwrap();

        let mut storage = TestNodeFileStorageManager {
            dir: dir.to_path_buf(),
            checkpoint: AtomicU64::new(old_checkpoint),
            non_empty1: MockStorage::new(),
            non_empty2: MockStorage::new(),
            root_ids_file: RootIdsFile::<TestNodeId>::open(dir.join("root_ids"), 0).unwrap(),
        };

        let mut seq = Sequence::new();
        storage
            .non_empty1
            .expect_prepare()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);
        storage
            .non_empty2
            .expect_prepare()
            .returning(|_| Err(Error::Io(std::io::Error::from(std::io::ErrorKind::Other)).into()))
            .times(1)
            .in_sequence(&mut seq);
        storage
            .non_empty1
            .expect_abort()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);

        assert!(matches!(
            storage.checkpoint().map_err(BTError::into_inner),
            Err(Error::Io(_))
        ));

        // The prepared metadata file should not exist after a failed checkpoint.
        assert!(!fs::exists(dir.join(
            TestNodeFileStorageManager::<MockStorage<_>, MockStorage<_>>::PREPARED_METADATA_FILE
        )).unwrap());
        // The committed metadata file should exist and contain the old checkpoint.
        assert_eq!(
            Metadata::read_or_init(dir.join(
                TestNodeFileStorageManager::<MockStorage<_>, MockStorage<_>>::COMMITTED_METADATA_FILE,
            )).unwrap().checkpoint_number,
            old_checkpoint
        );
        // The checkpoint variable should still be the old checkpoint.
        assert_eq!(storage.checkpoint.load(Ordering::Acquire), old_checkpoint);
    }

    #[test]
    fn restore_calls_restore_on_all_storages_and_overwrites_metadata_with_committed_metadata_and_removes_db_dirty_file()
     {
        static RESTORE_CALLED1: AtomicBool = AtomicBool::new(false);
        static RESTORE_CALLED2: AtomicBool = AtomicBool::new(false);

        struct CheckRestoreStorage1;

        impl Storage for CheckRestoreStorage1 {
            type Id = u64;
            type Item = NonEmpty1TestNode;

            fn open(_path: &Path) -> BTResult<Self, Error> {
                unimplemented!()
            }

            fn get(&self, _id: Self::Id) -> BTResult<Self::Item, Error> {
                unimplemented!()
            }

            fn reserve(&self, _item: &Self::Item) -> Self::Id {
                unimplemented!()
            }

            fn set(&self, _id: Self::Id, _item: &Self::Item) -> BTResult<(), Error> {
                unimplemented!()
            }

            fn delete(&self, _id: Self::Id) -> BTResult<(), Error> {
                unimplemented!()
            }

            fn close(self) -> BTResult<(), Error> {
                unimplemented!()
            }
        }

        impl CheckpointParticipant for CheckRestoreStorage1 {
            fn ensure(&self, _checkpoint: u64) -> BTResult<(), Error> {
                todo!()
            }

            fn prepare(&self, _checkpoint: u64) -> BTResult<(), Error> {
                todo!()
            }

            fn commit(&self, _checkpoint: u64) -> BTResult<(), Error> {
                todo!()
            }

            fn abort(&self, _checkpoint: u64) -> BTResult<(), Error> {
                todo!()
            }

            fn restore(_path: &Path, _checkpoint: u64) -> BTResult<(), Error> {
                RESTORE_CALLED1.store(true, Ordering::Relaxed);
                Ok(())
            }
        }

        struct CheckRestoreStorage2;

        impl Storage for CheckRestoreStorage2 {
            type Id = u64;
            type Item = NonEmpty2TestNode;

            fn open(_path: &Path) -> BTResult<Self, Error> {
                unimplemented!()
            }

            fn get(&self, _id: Self::Id) -> BTResult<Self::Item, Error> {
                unimplemented!()
            }

            fn reserve(&self, _item: &Self::Item) -> Self::Id {
                unimplemented!()
            }

            fn set(&self, _id: Self::Id, _item: &Self::Item) -> BTResult<(), Error> {
                unimplemented!()
            }

            fn delete(&self, _id: Self::Id) -> BTResult<(), Error> {
                unimplemented!()
            }

            fn close(self) -> BTResult<(), Error> {
                unimplemented!()
            }
        }

        impl CheckpointParticipant for CheckRestoreStorage2 {
            fn ensure(&self, _checkpoint: u64) -> BTResult<(), Error> {
                todo!()
            }

            fn prepare(&self, _checkpoint: u64) -> BTResult<(), Error> {
                todo!()
            }

            fn commit(&self, _checkpoint: u64) -> BTResult<(), Error> {
                todo!()
            }

            fn abort(&self, _checkpoint: u64) -> BTResult<(), Error> {
                todo!()
            }

            fn restore(_path: &Path, _checkpoint: u64) -> BTResult<(), Error> {
                RESTORE_CALLED2.store(true, Ordering::Relaxed);
                Ok(())
            }
        }

        type FileStorageManager =
            TestNodeFileStorageManager<CheckRestoreStorage1, CheckRestoreStorage2>;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        File::create(dir.join(FileStorageManager::DB_DIRTY_FILE)).unwrap();

        let checkpoint_number = 1;
        let metadata = Metadata {
            checkpoint_number,
            root_id_count: 0,
        };
        metadata
            .write(dir.join(FileStorageManager::COMMITTED_METADATA_FILE))
            .unwrap();

        FileStorageManager::restore(&dir, checkpoint_number).unwrap();

        assert!(RESTORE_CALLED1.load(Ordering::Relaxed));
        assert!(RESTORE_CALLED2.load(Ordering::Relaxed));

        assert_eq!(
            fs::read(dir.join(FileStorageManager::METADATA_FILE)).unwrap(),
            metadata.as_bytes()
        );

        assert!(!fs::exists(dir.join(FileStorageManager::DB_DIRTY_FILE)).unwrap());
    }

    #[test]
    fn restore_fails_if_checkpoint_does_not_match_committed_checkpoint() {
        type FileStorageManager = TestNodeFileStorageManager<
            MockStorage<NonEmpty1TestNode>,
            MockStorage<NonEmpty2TestNode>,
        >;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let committed_metadata = Metadata {
            checkpoint_number: 1,
            root_id_count: 0,
        };
        committed_metadata
            .write(dir.join(FileStorageManager::COMMITTED_METADATA_FILE))
            .unwrap();

        assert!(matches!(
            FileStorageManager::restore(&dir, 0).map_err(BTError::into_inner),
            Err(Error::Checkpoint)
        ));
        assert!(matches!(
            FileStorageManager::restore(&dir, 2).map_err(BTError::into_inner),
            Err(Error::Checkpoint)
        ));
    }

    #[test]
    fn open_modify_checkpoint_modify_close_open_close_restore__works_as_expected() {
        // This is an integration-style test that checks that the interactions between open,
        // checkpoint, close and restore work as expected.

        type FileStorageManager = TestNodeFileStorageManager<
            NodeFileStorage<NonEmpty1TestNode, SeekFile>,
            NodeFileStorage<NonEmpty2TestNode, SeekFile>,
        >;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        // Open
        let storage = FileStorageManager::open(&dir).unwrap();

        // Modify: add non_empty1_1 and non_empty2_1
        let non_empty1_1 = TestNode::NonEmpty1(Box::new(NonEmpty1TestNode([1u8; 16])));
        let non_empty_1_1_id = storage.reserve(&non_empty1_1);
        storage.set(non_empty_1_1_id, &non_empty1_1).unwrap();
        let non_empty2_1 = TestNode::NonEmpty2(Box::new(NonEmpty2TestNode([1u8; 32])));
        let non_empty_2_1_id = storage.reserve(&non_empty2_1);
        storage.set(non_empty_2_1_id, &non_empty2_1).unwrap();

        // Checkpoint
        let checkpoint = storage.checkpoint().unwrap();

        // Modify: add non_empty1_2 and non_empty2_2
        let non_empty1_2 = TestNode::NonEmpty1(Box::new(NonEmpty1TestNode([2u8; 16])));
        let non_empty_1_2_id = storage.reserve(&non_empty1_2);
        storage.set(non_empty_1_2_id, &non_empty1_2).unwrap();
        let non_empty2_2 = TestNode::NonEmpty2(Box::new(NonEmpty2TestNode([2u8; 32])));
        let non_empty_2_2_id = storage.reserve(&non_empty2_2);
        storage.set(non_empty_2_2_id, &non_empty2_2).unwrap();

        // Close
        storage.close().unwrap();

        // Open again
        let storage = FileStorageManager::open(&dir).unwrap();
        // Check that all nodes are present, but the ones added before the checkpoint are frozen
        assert_eq!(storage.get(non_empty_1_1_id), Ok(non_empty1_1.clone()));
        assert_eq!(storage.get(non_empty_2_1_id), Ok(non_empty2_1.clone()));
        assert_eq!(storage.get(non_empty_1_2_id), Ok(non_empty1_2.clone()));
        assert_eq!(storage.get(non_empty_2_2_id), Ok(non_empty2_2.clone()));
        assert_eq!(
            storage
                .set(non_empty_1_1_id, &non_empty1_1)
                .map_err(BTError::into_inner),
            Err(Error::Frozen)
        );
        assert_eq!(
            storage
                .set(non_empty_2_1_id, &non_empty2_1)
                .map_err(BTError::into_inner),
            Err(Error::Frozen)
        );
        assert_eq!(storage.set(non_empty_1_2_id, &non_empty1_2), Ok(()));
        assert_eq!(storage.set(non_empty_2_2_id, &non_empty2_2), Ok(()));

        // Close
        storage.close().unwrap();

        // Restore to checkpoint 1
        FileStorageManager::restore(&dir, checkpoint).unwrap();

        // Open again
        let storage = FileStorageManager::open(&dir).unwrap();
        // Check that only the nodes added before the checkpoint are present and that they are
        // frozen
        assert_eq!(storage.get(non_empty_1_1_id), Ok(non_empty1_1.clone()));
        assert_eq!(storage.get(non_empty_2_1_id), Ok(non_empty2_1.clone()));
        assert_eq!(
            storage.get(non_empty_1_2_id).map_err(BTError::into_inner),
            Err(Error::NotFound)
        );
        assert_eq!(
            storage.get(non_empty_2_2_id).map_err(BTError::into_inner),
            Err(Error::NotFound)
        );
        assert_eq!(
            storage
                .set(non_empty_1_1_id, &non_empty1_1)
                .map_err(BTError::into_inner),
            Err(Error::Frozen)
        );
        assert_eq!(
            storage
                .set(non_empty_2_1_id, &non_empty2_1)
                .map_err(BTError::into_inner),
            Err(Error::Frozen)
        );
    }

    mockall::mock! {
        pub Storage<T: Send + Sync + 'static> {}

        impl<T: Send + Sync + 'static> CheckpointParticipant for Storage<T> {
            fn ensure(&self, checkpoint: u64) -> BTResult<(), Error>;

            fn prepare(&self, checkpoint: u64) -> BTResult<(), Error>;

            fn commit(&self, checkpoint: u64) -> BTResult<(), Error>;

            fn abort(&self, checkpoint: u64) -> BTResult<(), Error>;

            fn restore(path: &Path, checkpoint: u64) -> BTResult<(), Error>;
        }

        impl<T: Send + Sync + 'static> Storage for Storage<T> {
            type Id = u64;
            type Item = T;

            fn open(path: &std::path::Path) -> BTResult<Self, Error>
            where
                Self: Sized;

            fn get(&self, id: <Self as Storage>::Id) -> BTResult<<Self as Storage>::Item, Error>;

            fn reserve(&self, item: &<Self as Storage>::Item) -> <Self as Storage>::Id;

            fn set(&self, id: <Self as Storage>::Id, item: &<Self as Storage>::Item) -> BTResult<(), Error>;

            fn delete(&self, id: <Self as Storage>::Id) -> BTResult<(), Error>;

            fn close(self) -> BTResult<(), Error>;
        }
    }

    pub type TestNodeId = [u8; 9];

    impl TreeId for TestNodeId {
        type NodeType = TestNodeType;

        fn from_idx_and_node_type(idx: u64, node_type: Self::NodeType) -> Self {
            let upper = match node_type {
                TestNodeType::Empty => 0x00,
                TestNodeType::NonEmpty1 => 0x01,
                TestNodeType::NonEmpty2 => 0x02,
            };
            let mut id = [0; 9];
            id[0] = upper;
            id[1..].copy_from_slice(&idx.to_be_bytes());
            id
        }

        fn to_index(self) -> u64 {
            let mut idx = [0; 8];
            idx.copy_from_slice(&self[1..]);
            u64::from_be_bytes(idx)
        }

        fn to_node_type(self) -> Option<Self::NodeType> {
            match self[0] {
                0x00 => Some(TestNodeType::Empty),
                0x01 => Some(TestNodeType::NonEmpty1),
                0x02 => Some(TestNodeType::NonEmpty2),
                _ => None,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Deftly)]
    #[derive_deftly(FileStorageManager)]
    pub enum TestNode {
        Empty(EmptyTestNode),
        NonEmpty1(Box<NonEmpty1TestNode>),
        NonEmpty2(Box<NonEmpty2TestNode>),
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TestNodeType {
        Empty,
        NonEmpty1,
        NonEmpty2,
    }

    impl NodeSize for TestNodeType {
        fn node_byte_size(&self) -> usize {
            match self {
                TestNodeType::Empty => size_of::<EmptyTestNode>(),
                TestNodeType::NonEmpty1 => size_of::<NonEmpty1TestNode>(),
                TestNodeType::NonEmpty2 => size_of::<NonEmpty2TestNode>(),
            }
        }

        fn min_non_empty_node_size() -> usize {
            size_of::<NonEmpty1TestNode>()
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
    pub struct EmptyTestNode;

    #[derive(Debug, Clone, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
    pub struct NonEmpty1TestNode(pub [u8; 16]);

    #[derive(Debug, Clone, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
    pub struct NonEmpty2TestNode(pub [u8; 32]);
}
