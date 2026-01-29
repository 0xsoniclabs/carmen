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
    time::Duration,
};

use carmen_rust::{
    VerkleStorageManager,
    database::verkle::variants::managed::{
        FullInnerNode, FullLeafNode, InnerDeltaNode, LeafDeltaNode, SparseInnerNode,
        SparseLeafNode, VerkleNode, VerkleNodeId, VerkleNodeKind,
    },
    error::BTError,
    statistics::node_count::{NodeCountBySize, NodeCountsByKindStatistic},
    storage::{
        DbOpenMode, Error, Storage,
        file::{FromToFile, NoSeekFile, NodeFileStorage, NodeFileStorageMetadata},
    },
    types::{DiskRepresentable, HasDeltaVariant, HasEmptyId, TreeId},
};

// Number of analyzed nodes, across all node kinds.
static PROCESSED: AtomicU64 = AtomicU64::new(0);
// Total number of nodes, across all node kinds.
static TOTAL: AtomicU64 = AtomicU64::new(0);

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

    let mut inner_sizes = [0; 257];
    let mut leaf_sizes = [0; 257];

    let stop = AtomicBool::new(false);
    let m = VerkleStorageManager::open(db_path, DbOpenMode::ReadOnly).unwrap();
    thread::scope(|s| {
        let inner = [
            s.spawn(|| {
                analyze_node::<SparseInnerNode<9>>(
                    &db_path.join("inner9"),
                    &m,
                    VerkleNodeKind::Inner9,
                )
            }),
            s.spawn(|| {
                analyze_node::<SparseInnerNode<15>>(
                    &db_path.join("inner15"),
                    &m,
                    VerkleNodeKind::Inner15,
                )
            }),
            s.spawn(|| {
                analyze_node::<SparseInnerNode<21>>(
                    &db_path.join("inner21"),
                    &m,
                    VerkleNodeKind::Inner21,
                )
            }),
            s.spawn(|| {
                analyze_node::<FullInnerNode>(
                    &db_path.join("inner256"),
                    &m,
                    VerkleNodeKind::Inner256,
                )
            }),
            s.spawn(|| {
                analyze_node::<InnerDeltaNode>(
                    &db_path.join("inner_delta"),
                    &m,
                    VerkleNodeKind::InnerDelta,
                )
            }),
        ];

        let leaf = [
            s.spawn(|| {
                analyze_node::<SparseLeafNode<1>>(&db_path.join("leaf1"), &m, VerkleNodeKind::Leaf1)
            }),
            s.spawn(|| {
                analyze_node::<SparseLeafNode<2>>(&db_path.join("leaf2"), &m, VerkleNodeKind::Leaf2)
            }),
            s.spawn(|| {
                analyze_node::<SparseLeafNode<5>>(&db_path.join("leaf5"), &m, VerkleNodeKind::Leaf5)
            }),
            s.spawn(|| {
                analyze_node::<SparseLeafNode<18>>(
                    &db_path.join("leaf18"),
                    &m,
                    VerkleNodeKind::Leaf18,
                )
            }),
            s.spawn(|| {
                analyze_node::<SparseLeafNode<146>>(
                    &db_path.join("leaf146"),
                    &m,
                    VerkleNodeKind::Leaf146,
                )
            }),
            s.spawn(|| {
                analyze_node::<FullLeafNode>(&db_path.join("leaf256"), &m, VerkleNodeKind::Leaf256)
            }),
            s.spawn(|| {
                analyze_node::<LeafDeltaNode>(
                    &db_path.join("leaf_delta"),
                    &m,
                    VerkleNodeKind::LeafDelta,
                )
            }),
        ];

        let progress = s.spawn(|| print_progress(&stop));

        for handle in inner {
            let res = handle.join().unwrap();
            for i in 0..=256 {
                inner_sizes[i] += res[i];
            }
        }

        for handle in leaf {
            let res = handle.join().unwrap();
            for i in 0..=256 {
                leaf_sizes[i] += res[i];
            }
        }

        stop.store(true, Ordering::Relaxed);
        progress.join().unwrap();
    });

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
}

/// Analyzes nodes of type `N` at the given path. Returns an array where the index is the number of
/// filled slots and the value is the count of nodes with that many filled slots.
fn analyze_node<N>(path: &Path, m: &VerkleStorageManager, node_kind: VerkleNodeKind) -> [usize; 257]
where
    N: DiskRepresentable + Send + Sync,
{
    let metadata = NodeFileStorageMetadata::read_or_init(
        path.join(NodeFileStorage::<N, NoSeekFile>::METADATA_FILE),
        DbOpenMode::ReadOnly,
    )
    .unwrap();
    let nodes = metadata.nodes as usize;

    TOTAL.fetch_add(nodes as u64, Ordering::Relaxed);

    // This uses all available parallelism for each node type. Since most node types are processed
    // fairly quickly, this ensures that the few remaining node types are processed in parallel as
    // well.
    let num_workers = thread::available_parallelism().unwrap().get();
    let chunk_size = nodes.div_ceil(num_workers);
    let mut sizes = [0usize; 257];

    thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_workers);
        for worker in 0..num_workers {
            let start = worker * chunk_size;
            let end = ((worker + 1) * chunk_size).min(nodes);
            handles.push(s.spawn(move || {
                let mut local_sizes = [0usize; 257];
                for idx in start..end {
                    let mut node = match m
                        .get(VerkleNodeId::from_idx_and_node_kind(idx as u64, node_kind))
                        .map_err(BTError::into_inner)
                    {
                        Ok(n) => n,
                        Err(Error::NotFound) => continue, // this index is reusable
                        Err(e) => panic!("{e}"),
                    };
                    if let Some(full_id) = node.needs_delta_base() {
                        let base_node = m.get(full_id).unwrap();
                        node.copy_from_delta_base(&base_node).unwrap();
                    }
                    let filled_slots = count_used_slots(&node);
                    local_sizes[filled_slots] += 1;
                    PROCESSED.fetch_add(1, Ordering::Relaxed);
                }
                local_sizes
            }));
        }
        for handle in handles {
            let res = handle.join().unwrap();
            for i in 0..=256 {
                sizes[i] += res[i];
            }
        }
    });

    sizes
}

/// Counts the number of filled slots in the given [`VerkleNode`].
fn count_used_slots(node: &VerkleNode) -> usize {
    match node {
        VerkleNode::Empty(_) => 0,
        VerkleNode::Inner9(n) => n.children.iter().filter(|c| !c.item.is_empty_id()).count(),
        VerkleNode::Inner15(n) => n.children.iter().filter(|c| !c.item.is_empty_id()).count(),
        VerkleNode::Inner21(n) => n.children.iter().filter(|c| !c.item.is_empty_id()).count(),
        VerkleNode::Inner256(n) => n.children.iter().filter(|c| !c.is_empty_id()).count(),
        VerkleNode::InnerDelta(n) => {
            n.children.iter().filter(|c| !c.is_empty_id()).count()
                + n.children_delta
                    .iter()
                    .filter(|c| !c.item.is_empty_id() && n.children[c.index as usize].is_empty_id())
                    .count()
        }
        VerkleNode::Leaf1(n) => n
            .commitment
            .committed_used_indices
            .iter()
            .map(|b| b.count_ones() as usize)
            .sum(),
        VerkleNode::Leaf2(n) => n
            .commitment
            .committed_used_indices
            .iter()
            .map(|b| b.count_ones() as usize)
            .sum(),
        VerkleNode::Leaf5(n) => n
            .commitment
            .committed_used_indices
            .iter()
            .map(|b| b.count_ones() as usize)
            .sum(),
        VerkleNode::Leaf18(n) => n
            .commitment
            .committed_used_indices
            .iter()
            .map(|b| b.count_ones() as usize)
            .sum(),
        VerkleNode::Leaf146(n) => n
            .commitment
            .committed_used_indices
            .iter()
            .map(|b| b.count_ones() as usize)
            .sum(),
        VerkleNode::Leaf256(n) => n
            .commitment
            .committed_used_indices
            .iter()
            .map(|b| b.count_ones() as usize)
            .sum(),
        VerkleNode::LeafDelta(n) => {
            n.commitment
                .committed_used_indices
                .iter()
                .map(|b| b.count_ones() as usize)
                .sum()
            // (n.values.iter().filter(|c| **c != [0; 32]).count() as isize
            //     + n.values_delta .iter() .map(|c| { c.item.is_some_and(|v| v != [0; 32]) as isize
            //                 - (n.values[c.index as usize] != [0; 32]) as isize
            //         })
            //         .sum::<isize>()) as usize
        }
    }
}

/// Periodically prints progress of the analysis until `stop` is set to true.
fn print_progress(stop: &AtomicBool) {
    while !stop.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_secs(10));
        let processed = PROCESSED.load(Ordering::Relaxed);
        let total = TOTAL.load(Ordering::Relaxed);
        eprintln!(
            "analyzed {processed} / {total} nodes ({:.2}%)",
            (processed as f64 / total as f64) * 100.0
        );
    }
}
