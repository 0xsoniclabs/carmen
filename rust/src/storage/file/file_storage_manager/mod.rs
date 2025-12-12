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
    ///   - The node kind enum must be called `<NodeName>Kind` and its variants must have the same
    ///     names as the ones of the node enum itself.
    ///   - The ID type for this tree must be called `<NodeName>Id` and it must implement
    ///     `TreeId<NodeKind = NodeNameKind> + Copy + FromBytes + IntoBytes + Immutable + Send +
    ///     Sync`.
    ///   - The ID type, the node kind enum and all variant types have to be in scope at the call/
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
    db_mode: $crate::storage::DbMode,
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
    fn open(dir: &::std::path::Path, db_mode: $crate::storage::DbMode) -> $crate::error::BTResult<Self, $crate::storage::Error> {
        if db_mode.has_write_access() {
            std::fs::create_dir_all(dir)?;
        }

        if ::std::fs::exists(dir.join(Self::DB_DIRTY_FILE))? {
            return Err($crate::storage::Error::DirtyOpen.into());
        }

        if db_mode.has_write_access() {
            ::std::fs::File::create(dir.join(Self::DB_DIRTY_FILE))?;
        }

        let metadata =
            <$crate::storage::file::file_storage_manager::metadata::Metadata as $crate::storage::file::FromToFile>::read_or_init(dir.join(Self::METADATA_FILE), db_mode)?;

        $(
            ${if not(approx_equal($vname, Empty)) {
                let ${snake_case $vname} = ${paste ${upper_camel_case $vname} Storage}::open(dir.join(Self::${paste ${shouty_snake_case $vname} _DIR}).as_path(), db_mode)?;
            }}
        )

        let root_ids_file =
            $crate::storage::file::file_storage_manager::root_ids_file::RootIdsFile::open(dir.join(Self::ROOT_IDS_FILE), metadata.root_id_count, db_mode)?;

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
            db_mode,
        })
    }

    fn get(&self, id: Self::Id) -> $crate::error::BTResult<Self::Item, $crate::storage::Error> {
        let idx = $crate::types::TreeId::to_index(id);
        match $crate::types::ToNodeKind::to_node_kind(&id).ok_or($crate::storage::Error::InvalidId)? {
            ${paste $tname Kind}::Empty => Ok(Self::Item::Empty(${paste Empty $tname}{})),
            $(
                ${if not(approx_equal($vname, Empty)) {
                    ${paste $tname Kind}::$vname => {
                        let node = self.${snake_case $vname}.get(idx)?;
                        Ok(Self::Item::$vname(Box::new(node)))
                    }
                }}
            )
        }
    }

    fn reserve(&self, node: &Self::Item) -> Self::Id {
        match node {
            Self::Item::Empty(_) => <Self::Id as $crate::types::TreeId>::from_idx_and_node_kind(0, ${paste $tname Kind}::Empty),
            $(
                ${if not(approx_equal($vname, Empty)) {
                    Self::Item::$vname(node) => {
                        let idx = self.${snake_case $vname}.reserve(node);
                        <Self::Id as $crate::types::TreeId>::from_idx_and_node_kind(idx, ${paste $tname Kind}::$vname)
                    }
                }}
            )
        }
    }

    fn set(&self, id: Self::Id, node: &Self::Item) -> $crate::error::BTResult<(), $crate::storage::Error> {
        if self.db_mode.read_only() {
            return Err($crate::storage::Error::ReadOnly.into());
        }
        let idx = $crate::types::TreeId::to_index(id);
        match (node, $crate::types::ToNodeKind::to_node_kind(&id).ok_or($crate::storage::Error::InvalidId)?) {
            (Self::Item::Empty(_), ${paste $tname Kind}::Empty) => Ok(()),
            $(
                ${if not(approx_equal($vname, Empty)) {
                    ($tname::$vname(node), ${paste $tname Kind}::$vname) => self.${snake_case $vname}.set(idx, node),
                }}
            )
            _ => Err($crate::storage::Error::IdNodeVariantMismatch.into())
        }
    }

    fn delete(&self, id: Self::Id) -> $crate::error::BTResult<(), $crate::storage::Error> {
        if self.db_mode.read_only() {
            return Err($crate::storage::Error::ReadOnly.into());
        }
        let idx = $crate::types::TreeId::to_index(id);
        match $crate::types::ToNodeKind::to_node_kind(&id).ok_or($crate::storage::Error::InvalidId)? {
            ${paste $ttype Kind}::Empty => Ok(()),
            $(
                ${if not(approx_equal($vname, Empty)) {
                    ${paste $tname Kind}::$vname => self.${snake_case $vname}.delete(idx),
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
        if self.db_mode.read_only() {
            return Ok(());
        }
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
        if self.db_mode.read_only() {
            return Err($crate::storage::Error::ReadOnly.into());
        }
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
            dir.join(Self::COMMITTED_METADATA_FILE),
            $crate::storage::DbMode::Read,
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
        if self.db_mode.read_only() {
            return Err($crate::storage::Error::ReadOnly.into());
        }
        self.root_ids_file.set(block_number, id)
    }

    fn highest_block_number(&self) -> $crate::error::BTResult<Option<u64>, $crate::storage::Error> {
        let count = self.root_ids_file.count();
        if count == 0 {
            Ok(None)
        } else {
            Ok(Some(count - 1))
        }
    }
}
}

pub(crate) use derive_deftly_template_FileStorageManager;
#[cfg(test)]
pub use tests::{
    EmptyTestNode, NonEmpty1TestNode, NonEmpty2TestNode, TestNode, TestNodeFileStorageManager,
    TestNodeId, TestNodeKind,
};

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
            CheckpointParticipant, Checkpointable, DbMode, Error, RootIdProvider, Storage,
            all_db_modes,
            file::{
                FromToFile, NodeFileStorage, SeekFile,
                file_storage_manager::{metadata::Metadata, root_ids_file::RootIdsFile},
            },
        },
        sync::atomic::{AtomicU64, Ordering},
        types::{NodeSize, ToNodeKind, TreeId},
        utils::test_dir::{Permissions, TestDir},
    };

    #[test]
    fn open_creates_directory_and_calls_open_on_all_storages_and_creates_db_dirty_file_if_db_mode_has_write_access()
     {
        type FileStorageManager = TestNodeFileStorageManager<
            NodeFileStorage<NonEmpty1TestNode, SeekFile>,
            NodeFileStorage<NonEmpty2TestNode, SeekFile>,
        >;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let storage = FileStorageManager::open(&dir, DbMode::ReadWrite);
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

    #[rstest_reuse::apply(all_db_modes)]
    fn open_opens_existing_files(#[case] db_mode: DbMode) {
        type FileStorageManager = TestNodeFileStorageManager<
            NodeFileStorage<NonEmpty1TestNode, SeekFile>,
            NodeFileStorage<NonEmpty2TestNode, SeekFile>,
        >;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        fs::create_dir_all(dir.join(FileStorageManager::NON_EMPTY1_DIR)).unwrap();
        NodeFileStorage::<NonEmpty1TestNode, SeekFile>::create_files(
            dir.join(FileStorageManager::NON_EMPTY1_DIR),
            &[NonEmpty1TestNode([1u8; 16])],
            0,
            &[],
            0,
        )
        .unwrap();
        fs::create_dir_all(dir.join(FileStorageManager::NON_EMPTY2_DIR)).unwrap();
        NodeFileStorage::<NonEmpty2TestNode, SeekFile>::create_files(
            dir.join(FileStorageManager::NON_EMPTY2_DIR),
            &[NonEmpty2TestNode([2u8; 32])],
            0,
            &[],
            0,
        )
        .unwrap();
        // Write metadata
        Metadata {
            checkpoint_number: 0,
            root_id_count: 0,
        }
        .write(dir.join(FileStorageManager::METADATA_FILE))
        .unwrap();
        // Write root ids file
        RootIdsFile::<TestNodeId>::open(
            dir.join(FileStorageManager::ROOT_IDS_FILE),
            0,
            DbMode::ReadWrite,
        )
        .unwrap();

        let storage = FileStorageManager::open(&dir, db_mode).unwrap();
        assert_eq!(
            storage
                .get(TestNodeId::from_idx_and_node_kind(
                    0,
                    TestNodeKind::NonEmpty1
                ))
                .unwrap(),
            TestNode::NonEmpty1(Box::new(NonEmpty1TestNode([1u8; 16])))
        );
        assert_eq!(
            storage
                .get(TestNodeId::from_idx_and_node_kind(
                    0,
                    TestNodeKind::NonEmpty2
                ))
                .unwrap(),
            TestNode::NonEmpty2(Box::new(NonEmpty2TestNode([2u8; 32])))
        );
    }

    #[test]
    fn open_does_not_create_db_dirty_file_if_db_mode_is_read_only() {
        type FileStorageManager = TestNodeFileStorageManager<
            NodeFileStorage<NonEmpty1TestNode, SeekFile>,
            NodeFileStorage<NonEmpty2TestNode, SeekFile>,
        >;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let storage = FileStorageManager::open(&dir, DbMode::ReadWrite).unwrap();
        assert!(fs::exists(dir.join(FileStorageManager::DB_DIRTY_FILE)).unwrap());
        storage.close().unwrap();
        assert!(!fs::exists(dir.join(FileStorageManager::DB_DIRTY_FILE)).unwrap());
        let storage = FileStorageManager::open(&dir, DbMode::Read).unwrap();
        assert!(!fs::exists(dir.join(FileStorageManager::DB_DIRTY_FILE)).unwrap());
        storage.close().unwrap();
    }

    #[test]
    fn open_fails_for_invalid_directory_if_db_mode_is_read_only() {
        type FileStorageManager = TestNodeFileStorageManager<
            NodeFileStorage<NonEmpty1TestNode, SeekFile>,
            NodeFileStorage<NonEmpty2TestNode, SeekFile>,
        >;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        assert!(matches!(
            FileStorageManager::open(&dir, DbMode::Read).map_err(BTError::into_inner),
            Err(Error::Io(_))
        ));
    }

    #[rstest_reuse::apply(all_db_modes)]
    fn open_fails_if_db_dirty_flag_file_exists(#[case] db_mode: DbMode) {
        type FileStorageManager = TestNodeFileStorageManager<
            MockStorage<NonEmpty1TestNode>,
            MockStorage<NonEmpty2TestNode>,
        >;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        File::create(dir.join(FileStorageManager::DB_DIRTY_FILE)).unwrap();

        assert!(matches!(
            FileStorageManager::open(&dir, db_mode).map_err(BTError::into_inner),
            Err(Error::DirtyOpen)
        ));
    }

    #[rstest_reuse::apply(all_db_modes)]
    fn get_forwards_to_get_of_corresponding_node_file_storage_depending_on_node_type(
        #[case] db_mode: DbMode,
    ) {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = make_test_node_file_storage(dir.path(), db_mode, 0);

        // TestNode::Empty
        {
            let node_id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::Empty);
            let node = EmptyTestNode;
            assert_eq!(storage.get(node_id).unwrap(), TestNode::Empty(node));
        }

        // TestNode::NonEmpty1
        {
            let node_id = TestNodeId::from_idx_and_node_kind(1, TestNodeKind::NonEmpty1);
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
            let node_id = TestNodeId::from_idx_and_node_kind(1, TestNodeKind::NonEmpty2);
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

    #[rstest_reuse::apply(all_db_modes)]
    fn reserve_forwards_to_reserve_of_corresponding_node_file_storage_depending_on_node_type(
        #[case] db_mode: DbMode,
    ) {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = make_test_node_file_storage(dir.path(), db_mode, 0);

        // TestNode::Empty
        {
            let node_idx = 0;
            let node = EmptyTestNode;
            assert_eq!(
                storage.reserve(&TestNode::Empty(node)),
                TestNodeId::from_idx_and_node_kind(node_idx, TestNodeKind::Empty)
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
                TestNodeId::from_idx_and_node_kind(node_idx, TestNodeKind::NonEmpty1)
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
                TestNodeId::from_idx_and_node_kind(node_idx, TestNodeKind::NonEmpty2)
            );
        }
    }

    #[test]
    fn set_forwards_to_set_of_corresponding_node_file_storage_depending_on_node_type() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = make_test_node_file_storage(dir.path(), DbMode::ReadWrite, 0);

        // TestNode::Empty
        {
            let node_id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::Empty);
            let node = EmptyTestNode;
            let node = TestNode::Empty(node);
            assert!(storage.set(node_id, &node).is_ok());
        }

        // TestNode::NonEmpty1
        {
            let node_id = TestNodeId::from_idx_and_node_kind(1, TestNodeKind::NonEmpty1);
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
            let node_id = TestNodeId::from_idx_and_node_kind(1, TestNodeKind::NonEmpty2);
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

        let storage = make_test_node_file_storage(dir.path(), DbMode::ReadWrite, 0);

        let id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::NonEmpty2);
        let node = TestNode::NonEmpty1(Box::default());

        assert!(matches!(
            storage.set(id, &node).map_err(BTError::into_inner),
            Err(Error::IdNodeVariantMismatch)
        ));
    }

    #[test]
    fn set_returns_error_if_db_is_read_only() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let storage = make_test_node_file_storage(dir.path(), DbMode::Read, 0);

        let id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::NonEmpty2);
        let node = TestNode::NonEmpty2(Box::default());

        assert!(matches!(
            storage.set(id, &node).map_err(BTError::into_inner),
            Err(Error::ReadOnly)
        ));
    }

    #[test]
    fn delete_forwards_to_delete_of_corresponding_node_file_storage_depending_on_node_type() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = make_test_node_file_storage(dir.path(), DbMode::ReadWrite, 0);

        // TestNode::Empty
        {
            let node_id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::Empty);
            assert!(storage.delete(node_id).is_ok());
        }

        // TestNode::NonEmpty1
        {
            let node_id = TestNodeId::from_idx_and_node_kind(1, TestNodeKind::NonEmpty1);
            storage
                .non_empty1
                .expect_delete()
                .with(eq(node_id.to_index()))
                .returning(move |_| Ok(()));
            assert!(storage.delete(node_id).is_ok());
        }

        // TestNode::NonEmpty2
        {
            let node_id = TestNodeId::from_idx_and_node_kind(1, TestNodeKind::NonEmpty2);
            storage
                .non_empty2
                .expect_delete()
                .with(eq(node_id.to_index()))
                .returning(move |_| Ok(()));
            assert!(storage.delete(node_id).is_ok());
        }
    }

    #[test]
    fn delete_returns_error_if_db_is_read_only() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let storage = make_test_node_file_storage(dir.path(), DbMode::Read, 0);

        let id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::NonEmpty2);

        assert!(matches!(
            storage.delete(id).map_err(BTError::into_inner),
            Err(Error::ReadOnly)
        ));
    }

    #[test]
    fn close_calls_close_on_all_storages_and_writes_metadata_and_removes_db_dirty_file_if_db_has_write_access()
     {
        type TestNodeFileStorageManager = super::TestNodeFileStorageManager<
            MockStorage<NonEmpty1TestNode>,
            MockStorage<NonEmpty2TestNode>,
        >;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = make_test_node_file_storage(dir.path(), DbMode::ReadWrite, 0);

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
                TestNodeId::from_idx_and_node_kind(0, TestNodeKind::Empty),
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
    fn close_only_calls_close_on_all_storages_if_db_mode_is_read_only() {
        type TestNodeFileStorageManager = super::TestNodeFileStorageManager<
            MockStorage<NonEmpty1TestNode>,
            MockStorage<NonEmpty2TestNode>,
        >;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let mut storage = make_test_node_file_storage(dir.path(), DbMode::Read, 0);

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
        storage.close().unwrap(); // Would panic if it tried to write metadata.
    }

    #[test]
    fn checkpoint_follows_correct_sequence_for_two_phase_commit() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let old_checkpoint = 1;

        let mut storage =
            make_test_node_file_storage(dir.path(), DbMode::ReadWrite, old_checkpoint);

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
               ), DbMode::Read).unwrap().checkpoint_number,
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

        let mut storage =
            make_test_node_file_storage(dir.path(), DbMode::ReadWrite, old_checkpoint);

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
               ), DbMode::Read).unwrap().checkpoint_number,
               old_checkpoint
           );
        // The checkpoint variable should still be the old checkpoint.
        assert_eq!(storage.checkpoint.load(Ordering::Acquire), old_checkpoint);
    }

    #[test]
    fn checkpoint_fails_in_read_only_mode() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let storage = make_test_node_file_storage(dir.path(), DbMode::Read, 0);

        assert!(matches!(
            storage.checkpoint().map_err(BTError::into_inner),
            Err(Error::ReadOnly)
        ));
    }

    #[test]
    fn restore_calls_restore_on_all_storages_and_overwrites_metadata_with_committed_metadata_and_removes_db_dirty_file()
     {
        type FileStorageManager = TestNodeFileStorageManager<
            MockStorage<NonEmpty1TestNode>,
            MockStorage<NonEmpty2TestNode>,
        >;

        let dir = TestDir::try_new(Permissions::ReadWrite)
            .unwrap()
            .join("restore_test");
        fs::create_dir_all(&dir).unwrap();

        let ctx = MockStorage::<NonEmpty1TestNode>::restore_context();
        ctx.expect()
            .withf(|path: &Path, checkpoint| {
                path.to_str().unwrap().contains("restore_test") && *checkpoint == 1
            })
            .returning(|_, _| Ok(()))
            .times(1);
        let ctx = MockStorage::<NonEmpty2TestNode>::restore_context();
        ctx.expect()
            .withf(|path: &Path, checkpoint| {
                path.to_str().unwrap().contains("restore_test") && *checkpoint == 1
            })
            .returning(|_, _| Ok(()))
            .times(1);

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
        let storage = FileStorageManager::open(&dir, DbMode::ReadWrite).unwrap();

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
        let storage = FileStorageManager::open(&dir, DbMode::ReadWrite).unwrap();
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
        let storage = FileStorageManager::open(&dir, DbMode::ReadWrite).unwrap();
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

    #[rstest_reuse::apply(all_db_modes)]
    fn get_root_id_gets_root_id_from_root_id_file(#[case] db_mode: DbMode) {
        let root_id0 = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::Empty);
        let root_id1 = TestNodeId::from_idx_and_node_kind(1, TestNodeKind::Empty);

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let storage = make_test_node_file_storage(dir.path(), db_mode, 0);

        assert_eq!(
            storage.root_ids_file.get(0).map_err(BTError::into_inner),
            Err(Error::NotFound)
        );
        assert_eq!(
            storage.get_root_id(0).map_err(BTError::into_inner),
            Err(Error::NotFound)
        );

        storage.root_ids_file.set(0, root_id0).unwrap();
        assert_eq!(
            storage.get_root_id(0).map_err(BTError::into_inner),
            Ok(root_id0)
        );
        assert_eq!(
            storage.get_root_id(1).map_err(BTError::into_inner),
            Err(Error::NotFound)
        );

        storage.root_ids_file.set(1, root_id1).unwrap();
        assert_eq!(
            storage.get_root_id(0).map_err(BTError::into_inner),
            Ok(root_id0)
        );
        assert_eq!(
            storage.get_root_id(1).map_err(BTError::into_inner),
            Ok(root_id1)
        );
    }

    #[test]
    fn set_root_id_sets_root_id_in_root_id_file() {
        let root_id0 = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::Empty);
        let root_id1 = TestNodeId::from_idx_and_node_kind(1, TestNodeKind::Empty);

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let storage = make_test_node_file_storage(dir.path(), DbMode::ReadWrite, 0);

        assert_eq!(storage.root_ids_file.count(), 0);
        assert_eq!(
            storage.root_ids_file.get(0).map_err(BTError::into_inner),
            Err(Error::NotFound)
        );
        assert_eq!(
            storage.root_ids_file.get(1).map_err(BTError::into_inner),
            Err(Error::NotFound)
        );

        storage.set_root_id(0, root_id0).unwrap();
        assert_eq!(storage.root_ids_file.count(), 1);
        assert_eq!(storage.root_ids_file.get(0).unwrap(), root_id0);
        assert_eq!(
            storage.root_ids_file.get(1).map_err(BTError::into_inner),
            Err(Error::NotFound)
        );

        storage.set_root_id(1, root_id1).unwrap();
        assert_eq!(storage.root_ids_file.count(), 2);
        assert_eq!(storage.root_ids_file.get(0).unwrap(), root_id0);
        assert_eq!(storage.root_ids_file.get(1).unwrap(), root_id1);
    }

    #[test]
    fn set_root_id_fails_if_db_mode() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let root_id0 = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::Empty);
        let storage = make_test_node_file_storage(dir.path(), DbMode::Read, 0);

        assert!(matches!(
            storage
                .set_root_id(0, root_id0)
                .map_err(BTError::into_inner),
            Err(Error::ReadOnly)
        ));
    }

    #[rstest_reuse::apply(all_db_modes)]
    fn get_highest_block_number_gets_highest_root_id_from_root_id_file_count_and_subtracts_one(
        #[case] db_mode: DbMode,
    ) {
        let root_id = TestNodeId::from_idx_and_node_kind(0, TestNodeKind::Empty);

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();

        let storage = make_test_node_file_storage(dir.path(), db_mode, 0);

        assert_eq!(storage.root_ids_file.count(), 0);
        assert_eq!(storage.highest_block_number().unwrap(), None);

        storage.root_ids_file.set(0, root_id).unwrap();
        assert_eq!(storage.root_ids_file.count(), 1);
        assert_eq!(storage.highest_block_number().unwrap(), Some(0));

        storage.root_ids_file.set(1, root_id).unwrap();
        assert_eq!(storage.root_ids_file.count(), 2);
        assert_eq!(storage.highest_block_number().unwrap(), Some(1));
    }

    #[allow(clippy::disallowed_types)]
    mod mock {
        use super::*;

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

                fn open(path: &std::path::Path, db_mode: DbMode) -> BTResult<Self, Error>
                where
                    Self: Sized;

                fn get(&self, id: <Self as Storage>::Id) -> BTResult<<Self as Storage>::Item, Error>;

                fn reserve(&self, item: &<Self as Storage>::Item) -> <Self as Storage>::Id;

                fn set(&self, id: <Self as Storage>::Id, item: &<Self as Storage>::Item) -> BTResult<(), Error>;

                fn delete(&self, id: <Self as Storage>::Id) -> BTResult<(), Error>;

                fn close(self) -> BTResult<(), Error>;
            }
        }
    }
    use mock::MockStorage;

    pub type TestNodeId = [u8; 9];

    impl ToNodeKind for TestNodeId {
        type Target = TestNodeKind;

        fn to_node_kind(&self) -> Option<Self::Target> {
            match self[0] {
                0x00 => Some(TestNodeKind::Empty),
                0x01 => Some(TestNodeKind::NonEmpty1),
                0x02 => Some(TestNodeKind::NonEmpty2),
                _ => None,
            }
        }
    }

    impl TreeId for TestNodeId {
        fn from_idx_and_node_kind(idx: u64, node_type: Self::Target) -> Self {
            let upper = match node_type {
                TestNodeKind::Empty => 0x00,
                TestNodeKind::NonEmpty1 => 0x01,
                TestNodeKind::NonEmpty2 => 0x02,
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
    }

    #[derive(Debug, Clone, PartialEq, Eq, Deftly)]
    #[derive_deftly(FileStorageManager)]
    pub enum TestNode {
        Empty(EmptyTestNode),
        NonEmpty1(Box<NonEmpty1TestNode>),
        NonEmpty2(Box<NonEmpty2TestNode>),
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TestNodeKind {
        Empty,
        NonEmpty1,
        NonEmpty2,
    }

    impl NodeSize for TestNodeKind {
        fn node_byte_size(&self) -> usize {
            match self {
                TestNodeKind::Empty => size_of::<EmptyTestNode>(),
                TestNodeKind::NonEmpty1 => size_of::<NonEmpty1TestNode>(),
                TestNodeKind::NonEmpty2 => size_of::<NonEmpty2TestNode>(),
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

    fn make_test_node_file_storage(
        dir: &Path,
        db_mode: DbMode,
        checkpoint: u64,
    ) -> TestNodeFileStorageManager<MockStorage<NonEmpty1TestNode>, MockStorage<NonEmpty2TestNode>>
    {
        TestNodeFileStorageManager {
            dir: dir.to_path_buf(),
            checkpoint: AtomicU64::new(checkpoint),
            non_empty1: MockStorage::new(),
            non_empty2: MockStorage::new(),
            root_ids_file: RootIdsFile::<TestNodeId>::open(
                dir.join("root_ids"),
                0,
                DbMode::ReadWrite,
            )
            .unwrap(),
            db_mode,
        }
    }
}
