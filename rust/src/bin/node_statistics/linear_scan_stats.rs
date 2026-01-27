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
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    thread,
};

use carmen_rust::{
    database::verkle::variants::managed::{
        FullInnerNode, FullLeafNode, SparseInnerNode, SparseLeafNode,
    },
    error::BTError,
    statistics::node_count::{NodeCountBySize, NodeCountsByKindStatistic},
    storage::{
        DbOpenMode, Error, Storage,
        file::{FromToFile, NoSeekFile, NodeFileStorage, NodeFileStorageMetadata},
    },
    types::{DiskRepresentable, HasEmptyId},
};

const PRINT_INTERVAL: u64 = 1000000;

/// Perform linear scan based statistics collection on the Carmen DB located at `db_path`.
pub fn linear_scan_stats(db_path: &Path) -> NodeCountsByKindStatistic {
    thread::scope(|s| {
        let mut node_variants: Vec<_> = db_path
            .read_dir()
            .unwrap()
            .filter(|e| e.as_ref().unwrap().file_type().unwrap().is_dir())
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        let mut expected_node_variants = [
            "inner9", "inner15", "inner21", "inner256", "leaf1", "leaf2", "leaf5", "leaf18",
            "leaf146", "leaf256",
        ];
        node_variants.sort();
        expected_node_variants.sort();
        if node_variants != expected_node_variants {
            eprintln!(
                "Unexpected node variants in DB path:\n\
                found    {node_variants:?}\n\
                expected {expected_node_variants:?}"
            );
            std::process::exit(1);
        }

        let inner = [
            s.spawn(|| analyze_sparse_inner::<9>(&db_path.join("inner9"))),
            s.spawn(|| analyze_sparse_inner::<15>(&db_path.join("inner15"))),
            s.spawn(|| analyze_sparse_inner::<21>(&db_path.join("inner21"))),
            s.spawn(|| analyze_full_inner(&db_path.join("inner256"))),
        ];

        let leaf = [
            s.spawn(|| analyze_sparse_leaf::<1>(&db_path.join("leaf1"))),
            s.spawn(|| analyze_sparse_leaf::<2>(&db_path.join("leaf2"))),
            s.spawn(|| analyze_sparse_leaf::<5>(&db_path.join("leaf5"))),
            s.spawn(|| analyze_sparse_leaf::<18>(&db_path.join("leaf18"))),
            s.spawn(|| analyze_sparse_leaf::<146>(&db_path.join("leaf146"))),
            s.spawn(|| analyze_full_leaf(&db_path.join("leaf256"))),
        ];

        let mut inner_sizes = [0; 257];
        for handle in inner {
            let res = handle.join().unwrap();
            for i in 0..=256 {
                inner_sizes[i] += res[i];
            }
        }

        let mut leaf_sizes = [0; 257];
        for handle in leaf {
            let res = handle.join().unwrap();
            for i in 0..=256 {
                leaf_sizes[i] += res[i];
            }
        }

        NodeCountsByKindStatistic {
            aggregated_node_statistics: [
                (
                    "Inner",
                    NodeCountBySize {
                        size_count: inner_sizes
                            .into_iter()
                            .enumerate()
                            .filter(|(_, c)| *c > 0)
                            .map(|(i, c)| (i as u64, c as u64))
                            .collect(),
                    },
                ),
                (
                    "Leaf",
                    NodeCountBySize {
                        size_count: leaf_sizes
                            .into_iter()
                            .enumerate()
                            .filter(|(_, c)| *c > 0)
                            .map(|(i, c)| (i as u64, c as u64))
                            .collect(),
                    },
                ),
            ]
            .into_iter()
            .collect(),
            total_nodes: inner_sizes.iter().sum::<usize>() as u64
                + leaf_sizes.iter().sum::<usize>() as u64,
        }
    })
}

/// Analyze full inner nodes at the given path, returning an array where the index is the number
/// of filled child slots and the value is the count of nodes with that many filled slots.
fn analyze_full_inner(path: &Path) -> [usize; 257] {
    analyze_node::<FullInnerNode>(path, |node| {
        node.children.iter().filter(|c| !c.is_empty_id()).count()
    })
}

/// Analyze sparse inner nodes at the given path, returning an array where the index is the number
/// of filled child slots and the value is the count of nodes with that many filled slots.
fn analyze_sparse_inner<const N: usize>(path: &Path) -> [usize; 257] {
    analyze_node::<SparseInnerNode<N>>(path, |node| {
        node.children
            .iter()
            .filter(|c| !c.item.is_empty_id())
            .count()
    })
}

/// Analyze full leaf nodes at the given path, returning an array where the index is the number
/// of filled value slots and the value is the count of nodes with that many filled slots.
fn analyze_full_leaf(path: &Path) -> [usize; 257] {
    analyze_node::<FullLeafNode>(path, |node| {
        node.values.iter().filter(|c| **c != [0; 32]).count()
    })
}

/// Analyze sparse leaf nodes at the given path, returning an array where the index is the number
/// of filled value slots and the value is the count of nodes with that many filled slots.
fn analyze_sparse_leaf<const N: usize>(path: &Path) -> [usize; 257] {
    analyze_node::<SparseLeafNode<N>>(path, |node| {
        node.values.iter().filter(|c| c.item != [0; 32]).count()
    })
}

/// Analyzes nodes of type `N` at the given path, using the provided `count_fn` to determine the
/// number of filled slots in each node. Returns an array where the index is the number of filled
/// slots and the value is the count of nodes with that many filled slots.
fn analyze_node<N>(path: &Path, count_fn: impl Fn(&N) -> usize) -> [usize; 257]
where
    N: DiskRepresentable + Send + Sync,
{
    // Total number of nodes analyzed, across all `N`s.
    static COUNT: AtomicU64 = AtomicU64::new(0);

    let metadata = NodeFileStorageMetadata::read_or_init(
        path.join(NodeFileStorage::<N, NoSeekFile>::METADATA_FILE),
        DbOpenMode::ReadOnly,
    )
    .unwrap();
    let nodes = metadata.nodes;

    let mut sizes = [0; 257];
    let s = NodeFileStorage::<N, NoSeekFile>::open(path, DbOpenMode::ReadOnly).unwrap();
    for idx in 0..nodes {
        let node = match s.get(idx).map_err(BTError::into_inner) {
            Ok(n) => n,
            Err(Error::NotFound) => continue, // this index is reusable
            Err(e) => panic!("{e}"),
        };
        let filled_slots = count_fn(&node);
        sizes[filled_slots] += 1;
        let count = COUNT.fetch_add(1, Ordering::Relaxed);
        if count.is_multiple_of(PRINT_INTERVAL) {
            eprintln!("analyzed {count} nodes");
        }
    }
    sizes
}
