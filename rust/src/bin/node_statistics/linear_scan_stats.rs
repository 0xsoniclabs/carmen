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
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    thread,
};

use carmen_rust::{
    database::verkle::variants::managed::{
        FullInnerNode, FullLeafNode, InnerDeltaNode, LeafDeltaNode, SparseInnerNode,
        SparseLeafNode, VerkleNode, VerkleNodeFileStorageManager,
    },
    error::BTError,
    node_manager::{NodeManager, cached_node_manager::CachedNodeManager},
    statistics::node_count::{NodeCountBySize, NodeCountsByKindStatistic},
    storage::{
        DbMode, Error, Storage,
        file::{FromToFile, NoSeekFile, NodeFileStorage, NodeFileStorageMetadata},
    },
    types::{DiskRepresentable, HasDeltaVariant, HasEmptyId},
};

// Number of analyzed nodes, across all `N`s.
static PROCESSED: AtomicU64 = AtomicU64::new(0);
// Total number of nodes, across all `N`s.
static TOTAL: AtomicU64 = AtomicU64::new(0);

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

/// Perform linear scan based statistics collection on the Carmen DB located at `db_path`.
pub fn linear_scan_stats(db_path: &Path) -> NodeCountsByKindStatistic {
    if !matches!(
        db_path.file_name().and_then(|n| n.to_str()),
        Some("live" | "archive")
    ) {
        eprintln!(
            "Expected DB path to end with 'live' or 'archive', got {:?}",
            db_path.file_name()
        );
        std::process::exit(1);
    }
    let mut node_variants: Vec<_> = db_path
        .read_dir()
        .unwrap()
        .filter(|e| e.as_ref().unwrap().file_type().unwrap().is_dir())
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    let mut expected_node_variants = [
        "inner9",
        "inner15",
        "inner21",
        "inner256",
        "inner_delta",
        "leaf1",
        "leaf2",
        "leaf5",
        "leaf18",
        "leaf146",
        "leaf256",
        "leaf_delta",
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

    let stop = AtomicBool::new(false);
    thread::scope(|s| {
        let inner = [
            s.spawn(|| analyze_sparse_inner::<9>(&db_path.join("inner9"))),
            s.spawn(|| analyze_sparse_inner::<15>(&db_path.join("inner15"))),
            s.spawn(|| analyze_sparse_inner::<21>(&db_path.join("inner21"))),
            s.spawn(|| analyze_full_inner(&db_path.join("inner256"))),
            s.spawn(|| analyze_inner_delta(&db_path.join("inner_delta"))),
        ];

        let leaf = [
            s.spawn(|| analyze_sparse_leaf::<1>(&db_path.join("leaf1"))),
            s.spawn(|| analyze_sparse_leaf::<2>(&db_path.join("leaf2"))),
            s.spawn(|| analyze_sparse_leaf::<5>(&db_path.join("leaf5"))),
            s.spawn(|| analyze_sparse_leaf::<18>(&db_path.join("leaf18"))),
            s.spawn(|| analyze_sparse_leaf::<146>(&db_path.join("leaf146"))),
            s.spawn(|| analyze_full_leaf(&db_path.join("leaf256"))),
            s.spawn(|| analyze_leaf_delta(&db_path.join("leaf_delta"))),
        ];

        let progress = s.spawn(|| print_progress(&stop));

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

        stop.store(true, Ordering::Relaxed);
        progress.join().unwrap();

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

/// Analyze inner delta nodes at the given path, returning an array where the index is the number
/// of filled child slots and the value is the count of nodes with that many filled slots.
fn analyze_inner_delta(path: &Path) -> [usize; 257] {
    analyze_delta_node::<InnerDeltaNode>(
        path,
        |node| {
            node.children.iter().filter(|c| !c.is_empty_id()).count()
                + node
                    .children_delta
                    .iter()
                    .filter(|c| {
                        !c.item.is_empty_id() && node.children[c.index as usize].is_empty_id()
                    })
                    .count()
        },
        |n| VerkleNode::InnerDelta(Box::new(n)),
        |v| match v {
            VerkleNode::InnerDelta(n) => *n,
            _ => panic!("expected InnerDelta"),
        },
    )
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

/// Analyze leaf delta nodes at the given path, returning an array where the index is the number
/// of filled value slots and the value is the count of nodes with that many filled slots.
fn analyze_leaf_delta(path: &Path) -> [usize; 257] {
    analyze_delta_node::<LeafDeltaNode>(
        path,
        |node| {
            node.values.iter().filter(|c| **c != [0; 32]).count()
                + node
                    .values_delta
                    .iter()
                    .filter(|c| c.item.is_some() && node.values[c.index as usize] == [0; 32])
                    .count()
        },
        |n| VerkleNode::LeafDelta(Box::new(n)),
        |v| match v {
            VerkleNode::LeafDelta(n) => *n,
            _ => panic!("expected LeafDelta"),
        },
    )
}

/// Analyzes nodes of type `N` at the given path, using the provided `count_fn` to determine the
/// number of filled slots in each node. Returns an array where the index is the number of filled
/// slots and the value is the count of nodes with that many filled slots.
fn analyze_node<N>(path: &Path, count_fn: impl Fn(&N) -> usize) -> [usize; 257]
where
    N: DiskRepresentable + Send + Sync,
{
    let metadata = NodeFileStorageMetadata::read_or_init(
        path.join(NodeFileStorage::<N, NoSeekFile>::METADATA_FILE),
        DbMode::ReadOnly,
    )
    .unwrap();
    let nodes = metadata.nodes;

    TOTAL.fetch_add(nodes, Ordering::Relaxed);

    let mut sizes = [0; 257];
    let s = NodeFileStorage::<N, NoSeekFile>::open(path, DbMode::ReadOnly).unwrap();
    for idx in 0..nodes {
        let node = match s.get(idx).map_err(BTError::into_inner) {
            Ok(n) => n,
            Err(Error::NotFound) => continue, // this index is reusable
            Err(e) => panic!("{e}"),
        };
        let filled_slots = count_fn(&node);
        sizes[filled_slots] += 1;
        PROCESSED.fetch_add(1, Ordering::Relaxed);
    }
    sizes
}

/// Analyzes nodes of type `N` at the given path, using the provided `count_fn` to determine the
/// number of filled slots in each node. Returns an array where the index is the number of filled
/// slots and the value is the count of nodes with that many filled slots.
fn analyze_delta_node<N>(
    path: &Path,
    count_fn: impl Fn(&N) -> usize,
    wrap_fn: impl Fn(N) -> VerkleNode,
    unwrap_fn: impl Fn(VerkleNode) -> N,
) -> [usize; 257]
where
    N: DiskRepresentable + Send + Sync,
{
    let metadata = NodeFileStorageMetadata::read_or_init(
        path.join(NodeFileStorage::<N, NoSeekFile>::METADATA_FILE),
        DbMode::ReadOnly,
    )
    .unwrap();
    let nodes = metadata.nodes;

    TOTAL.fetch_add(nodes, Ordering::Relaxed);

    let mut sizes = [0; 257];
    let s = NodeFileStorage::<N, NoSeekFile>::open(path, DbMode::ReadOnly).unwrap();
    let m = CachedNodeManager::new(
        100_000,
        VerkleStorageManager::open(path.parent().unwrap(), DbMode::ReadOnly).unwrap(),
        |_| false,
    );
    for idx in 0..nodes {
        let node = match s.get(idx).map_err(BTError::into_inner) {
            Ok(n) => n,
            Err(Error::NotFound) => continue, // this index is reusable
            Err(e) => panic!("{e}"),
        };
        let mut wrapped = wrap_fn(node);
        if let Some(full_id) = wrapped.needs_full() {
            let base_node = m.get_read_access(full_id).unwrap();
            wrapped.copy_from_base(&base_node).unwrap();
        }
        let filled_slots = count_fn(&unwrap_fn(wrapped));
        sizes[filled_slots] += 1;
        PROCESSED.fetch_add(1, Ordering::Relaxed);
    }
    sizes
}

fn print_progress(stop: &AtomicBool) {
    while !stop.load(Ordering::Relaxed) {
        thread::sleep(std::time::Duration::from_secs(10));
        let processed = PROCESSED.load(Ordering::Relaxed);
        let total = TOTAL.load(Ordering::Relaxed);
        eprintln!(
            "analyzed {processed} / {total} nodes ({:.2}%)",
            (processed as f64 / total as f64) * 100.0
        );
    }
}
