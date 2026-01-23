// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{path::Path, sync::Arc};

use carmen_rust::{
    database::{
        self,
        verkle::variants::managed::{
            FullInnerNode, FullLeafNode, InnerDeltaNode, LeafDeltaNode, SparseInnerNode,
            SparseLeafNode, VerkleNode, VerkleNodeFileStorageManager,
        },
        visitor::AcceptVisitor,
    },
    node_manager::cached_node_manager::CachedNodeManager,
    statistics::node_count::{NodeCountVisitor, NodeCountsByLevelAndKind},
    storage::{
        DbMode, Storage,
        file::{NoSeekFile, NodeFileStorage},
    },
};

type VerkleStorageManager = VerkleNodeFileStorageManager<
    NodeFileStorage<SparseInnerNode<9>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<15>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<21>, NoSeekFile>,
    NodeFileStorage<FullInnerNode, NoSeekFile>,
    NodeFileStorage<InnerDeltaNode, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<1>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<2>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<5>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<18>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<146>, NoSeekFile>,
    NodeFileStorage<FullLeafNode, NoSeekFile>,
    NodeFileStorage<LeafDeltaNode, NoSeekFile>,
>;

/// Perform tree traversal based statistics collection on the Carmen DB located at `db_path`.
pub fn tree_traversal_stats(db_path: &Path) -> NodeCountsByLevelAndKind {
    if !db_path.ends_with("live") {
        eprintln!("The tree traversal stats only work with live DBs");
        std::process::exit(1);
    }
    let storage = VerkleStorageManager::open(db_path, DbMode::ReadOnly)
        .map_err(|e| {
            eprintln!("error: could not open database at the specified path: {e}");
            std::process::exit(1);
        })
        .unwrap();
    let is_pinned = |_n: &VerkleNode| false; // We don't care about the pinned status for stats
    let manager = Arc::new(CachedNodeManager::new(100_000, storage, is_pinned));

    let mut count_visitor = NodeCountVisitor::default();
    // NOTE: the `ManagedVerkleTrie` must be dropped before closing the DB, hence the
    // inner scope.
    {
        let managed_trie =
            database::ManagedVerkleTrie::<_>::try_from_block_height(manager.clone(), 0).unwrap();
        managed_trie.accept(&mut count_visitor).unwrap();
    }

    // Close the DB
    Arc::into_inner(manager).unwrap().close().unwrap();

    count_visitor.node_count
}
