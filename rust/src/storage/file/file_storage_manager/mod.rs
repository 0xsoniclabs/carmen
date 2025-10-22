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

pub mod checkpoint_data;
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
    ///   - The ID type for this tree must be called `<NodeName>Id` and it must implement
    ///     `TreeId<NodeType = NodeNameType> + Copy + FromBytes + IntoBytes + Immutable + Send +
    ///     Sync`.
    ///   - The node type enum must be called `<NodeName>Type` and its variants must have the same
    ///     names as the ones of the node enum itself.
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
    checkpoint: ::std::sync::atomic::AtomicU64,
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
    pub const COMMITTED_CHECKPOINT_FILE: &str = "committed_checkpoint.bin";
    pub const PREPARED_CHECKPOINT_FILE: &str = "prepared_checkpoint.bin";
    pub const ROOT_IDS_FILE: &str = "root_ids.bin";
}

impl<$STORAGE_GENERICS> $crate::storage::Storage for $MANAGER_TYPE<$STORAGE_GENERICS>
where
    $STORAGE_GENERICS_BOUNDS
{
    type Id = $ID;
    type Item = $tname;

    /// Opens or creates the file backends for the individual node types in the specified directory.
    fn open(dir: &::std::path::Path) -> Result<Self, $crate::storage::Error> {
        std::fs::create_dir_all(dir)?;

        let checkpoint_data =
            <$crate::storage::file::file_storage_manager::checkpoint_data::CheckpointData as $crate::storage::file::FromToFile>::read_or_init(dir.join(Self::COMMITTED_CHECKPOINT_FILE))?;

        $(
            ${if not(approx_equal($vname, Empty)) {
                let ${snake_case $vname} = ${paste ${upper_camel_case $vname} Storage}::open(dir.join(Self::${paste ${shouty_snake_case $vname} _DIR}).as_path())?;
            }}
        )

        let root_ids_file =
            $crate::storage::file::file_storage_manager::root_ids_file::RootIdsFile::open(dir.join(Self::ROOT_IDS_FILE), checkpoint_data.root_id_count)?;

        $(
            ${if not(approx_equal($vname, Empty)) {
                ${snake_case $vname}.ensure(checkpoint_data.checkpoint_number)?;
            }}
        )

        Ok(Self {
            dir: dir.to_path_buf(),
            checkpoint: ::std::sync::atomic::AtomicU64::new(checkpoint_data.checkpoint_number),
            $(
                ${if not(approx_equal($vname, Empty)) {
                    ${snake_case $vname},
                }}
            )
            root_ids_file,
        })
    }

    fn get(&self, id: Self::Id) -> Result<Self::Item, $crate::storage::Error> {
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

    fn set(&self, id: Self::Id, node: &Self::Item) -> Result<(), $crate::storage::Error> {
        let idx = $crate::types::TreeId::to_index(id);
        match (node, $crate::types::TreeId::to_node_type(id).ok_or($crate::storage::Error::InvalidId)?) {
            (Self::Item::Empty(_), ${paste $tname Type}::Empty) => Ok(()),
            $(
                ${if not(approx_equal($vname, Empty)) {
                    ($tname::$vname(node), ${paste $tname Type}::$vname) => self.${snake_case $vname}.set(idx, node),
                }}
            )
            _ => Err($crate::storage::Error::IdNodeTypeMismatch)
        }
    }

    fn delete(&self, id: Self::Id) -> Result<(), $crate::storage::Error> {
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
}

impl<$STORAGE_GENERICS> $crate::storage::Checkpointable for $MANAGER_TYPE<$STORAGE_GENERICS>
where
    $STORAGE_GENERICS_BOUNDS
{
    fn checkpoint(&self) -> Result<(), $crate::storage::Error> {
        let current_checkpoint = self.checkpoint.load(::std::sync::atomic::Ordering::Acquire);
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
            &$crate::storage::file::file_storage_manager::checkpoint_data::CheckpointData {
                checkpoint_number: new_checkpoint,
                root_id_count: self.root_ids_file.count(),
            },
            self.dir.join(Self::PREPARED_CHECKPOINT_FILE)
        )?;
        ::std::fs::rename(
            self.dir.join(Self::PREPARED_CHECKPOINT_FILE),
            self.dir.join(Self::COMMITTED_CHECKPOINT_FILE),
        )?;
        for participant in participants.iter() {
            participant.commit(new_checkpoint)?;
        }
        self.checkpoint.store(new_checkpoint, ::std::sync::atomic::Ordering::Release);
        Ok(())
    }
}

impl<$STORAGE_GENERICS> $crate::storage::RootIdProvider for $MANAGER_TYPE<$STORAGE_GENERICS>
where
    $STORAGE_GENERICS_BOUNDS
{
    type Id = $ID;

    fn get_root_id(&self, block_number: u64) -> Result<Self::Id, $crate::storage::Error> {
        self.root_ids_file.get(block_number)
    }

    fn set_root_id(&self, block_number: u64, id: Self::Id) -> Result<(), $crate::storage::Error> {
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
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };

    use derive_deftly::Deftly;
    use mockall::{Sequence, predicate::eq};
    use zerocopy::{FromBytes, Immutable, IntoBytes};

    use crate::{
        storage::{
            CheckpointParticipant, Checkpointable, Error, Storage,
            file::{
                FromToFile, NodeFileStorage, SeekFile,
                file_storage_manager::{
                    checkpoint_data::CheckpointData, root_ids_file::RootIdsFile,
                },
            },
        },
        types::{NodeSize, TreeId},
        utils::test_dir::{Permissions, TestDir},
    };

    #[test]
    fn open_creates_directory_and_calls_open_on_all_storages() {
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
            NodeFileStorage::<NonEmpty1TestNode, SeekFile>::COMMITTED_METADATA_FILE,
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
            NodeFileStorage::<NonEmpty1TestNode, SeekFile>::create_files_for_nodes(&dir, &[], &[])
                .unwrap();
        }

        let storage = FileStorageManager::open(&dir);
        assert!(storage.is_ok());
    }

    #[test]
    fn open_propagates_io_errors() {
        type FileStorageManager = TestNodeFileStorageManager<
            MockStorage<NonEmpty1TestNode>,
            MockStorage<NonEmpty2TestNode>,
        >;

        let dir = TestDir::try_new(Permissions::ReadOnly).unwrap();

        let path = dir.join("non_existent_dir");

        assert!(matches!(FileStorageManager::open(&path), Err(Error::Io(_))));
    }

    #[test]
    fn get_forwards_to_get_of_corresponding_node_file_storage_depending_on_node_type() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = TestNodeFileStorageManager {
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(0),
            non_empty1: MockStorage::new(),
            non_empty2: MockStorage::new(),
            root_ids_file: RootIdsFile::open(dir.path().join("root_ids"), 0).unwrap(),
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
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(0),
            non_empty1: MockStorage::new(),
            non_empty2: MockStorage::new(),
            root_ids_file: RootIdsFile::<TestNodeId>::open(dir.path().join("root_ids"), 0).unwrap(),
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
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(0),
            non_empty1: MockStorage::new(),
            non_empty2: MockStorage::new(),
            root_ids_file: RootIdsFile::open(dir.path().join("root_ids"), 0).unwrap(),
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
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(0),
            non_empty1: MockStorage::new(),
            non_empty2: MockStorage::new(),
            root_ids_file: RootIdsFile::open(dir.path().join("root_ids"), 0).unwrap(),
        };

        let id = TestNodeId::from_idx_and_node_type(0, TestNodeType::NonEmpty2);
        let node = TestNode::NonEmpty1(Box::default());

        assert!(matches!(
            storage.set(id, &node),
            Err(Error::IdNodeTypeMismatch)
        ));
    }

    #[test]
    fn delete_forwards_to_delete_of_corresponding_node_file_storage_depending_on_node_type() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = TestNodeFileStorageManager {
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(0),
            non_empty1: MockStorage::new(),
            non_empty2: MockStorage::new(),
            root_ids_file: RootIdsFile::open(dir.path().join("root_ids"), 0).unwrap(),
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
    fn checkpoint_follows_correct_sequence_for_two_phase_commit() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let old_checkpoint = 1;

        let mut storage = TestNodeFileStorageManager {
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(old_checkpoint),
            non_empty1: MockStorage::new(),
            non_empty2: MockStorage::new(),
            root_ids_file: RootIdsFile::<TestNodeId>::open(dir.path().join("root_ids"), 0).unwrap(),
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

        // The prepared checkpoint file should not exist after a successful checkpoint.
        assert!(!fs::exists(dir.path().join(
            TestNodeFileStorageManager::<MockStorage<_>, MockStorage<_>>::PREPARED_CHECKPOINT_FILE
        )).unwrap());
        // The committed checkpoint file should exist and contain the new checkpoint.
        assert_eq!(
            CheckpointData::read_or_init(dir.path().join(
                TestNodeFileStorageManager::<MockStorage<_>, MockStorage<_>>::COMMITTED_CHECKPOINT_FILE,
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
        let old_checkpoint_data = CheckpointData {
            checkpoint_number: old_checkpoint,
            root_id_count: 0,
        };
        old_checkpoint_data.write(dir.path().join(
            TestNodeFileStorageManager::<MockStorage<_>, MockStorage<_>>::COMMITTED_CHECKPOINT_FILE,
        )).unwrap();

        let mut storage = TestNodeFileStorageManager {
            dir: dir.path().to_path_buf(),
            checkpoint: AtomicU64::new(old_checkpoint),
            non_empty1: MockStorage::new(),
            non_empty2: MockStorage::new(),
            root_ids_file: RootIdsFile::<TestNodeId>::open(dir.path().join("root_ids"), 0).unwrap(),
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
            .returning(|_| Err(Error::Io(std::io::Error::from(std::io::ErrorKind::Other))))
            .times(1)
            .in_sequence(&mut seq);
        storage
            .non_empty1
            .expect_abort()
            .returning(|_| Ok(()))
            .times(1)
            .in_sequence(&mut seq);

        assert!(matches!(storage.checkpoint(), Err(Error::Io(_))));

        // The prepared checkpoint file should not exist after a failed checkpoint.
        assert!(!fs::exists(dir.path().join(
            TestNodeFileStorageManager::<MockStorage<_>, MockStorage<_>>::PREPARED_CHECKPOINT_FILE
        )).unwrap());
        // The committed checkpoint file should exist and contain the old checkpoint.
        assert_eq!(
            CheckpointData::read_or_init(dir.path().join(
                TestNodeFileStorageManager::<MockStorage<_>, MockStorage<_>>::COMMITTED_CHECKPOINT_FILE,
            )).unwrap().checkpoint_number,
            old_checkpoint
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

            fn open(path: &std::path::Path) -> Result<Self, Error>
            where
                Self: Sized;

            fn get(&self, id: <Self as Storage>::Id) -> Result<<Self as Storage>::Item, Error>;

            fn reserve(&self, item: &<Self as Storage>::Item) -> <Self as Storage>::Id;

            fn set(&self, id: <Self as Storage>::Id, item: &<Self as Storage>::Item) -> Result<(), Error>;

            fn delete(&self, id: <Self as Storage>::Id) -> Result<(), Error>;
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
