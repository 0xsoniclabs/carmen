use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    thread,
};

use carmen_rust::{
    database::verkle::variants::managed::{
        FullInnerNode, FullLeafNode, SparseInnerNode, SparseLeafNode,
    },
    error::BTError,
    storage::{
        Error, Storage,
        file::{NoSeekFile, NodeFileStorage},
    },
    types::{DiskRepresentable, HasEmptyId},
};

const PRINT_INTERVAL: u64 = 1000000;

fn main() {
    let dir = std::env::args().nth(1).expect("expected path argument");
    let dir = PathBuf::from(dir);

    thread::scope(|s| {
        let inner = [
            s.spawn(|| analyze_sparse_inner::<3>(&dir.join("inner3"))),
            s.spawn(|| analyze_sparse_inner::<47>(&dir.join("inner47"))),
            s.spawn(|| analyze_full_inner(&dir.join("inner256"))),
        ];

        let leaf = [
            s.spawn(|| analyze_sparse_leaf::<1>(&dir.join("leaf1"))),
            s.spawn(|| analyze_sparse_leaf::<2>(&dir.join("leaf2"))),
            s.spawn(|| analyze_sparse_leaf::<21>(&dir.join("leaf21"))),
            s.spawn(|| analyze_sparse_leaf::<64>(&dir.join("leaf64"))),
            s.spawn(|| analyze_sparse_leaf::<141>(&dir.join("leaf141"))),
            s.spawn(|| analyze_full_leaf(&dir.join("leaf256"))),
        ];

        let mut inner_sizes = [0; 257];
        for handle in inner {
            let res = handle.join().unwrap();
            for i in 0..257 {
                inner_sizes[i] += res[i];
            }
        }

        let mut leaf_sizes = [0; 257];
        for handle in leaf {
            let res = handle.join().unwrap();
            for i in 0..257 {
                leaf_sizes[i] += res[i];
            }
        }

        println!("inner nodes:");
        for (i, size) in inner_sizes.iter().enumerate() {
            println!("{i}: {size}");
        }
        println!("leaf nodes:");
        for (i, size) in leaf_sizes.iter().enumerate() {
            println!("{i}: {size}");
        }
    });
}

fn analyze_full_inner(path: &Path) -> [usize; 257] {
    analyze_node::<FullInnerNode>(path, |node| {
        node.children.iter().filter(|c| !c.is_empty_id()).count()
    })
}

fn analyze_sparse_inner<const N: usize>(path: &Path) -> [usize; 257] {
    analyze_node::<SparseInnerNode<N>>(path, |node| {
        node.children
            .iter()
            .filter(|c| !c.item.is_empty_id())
            .count()
    })
}

fn analyze_full_leaf(path: &Path) -> [usize; 257] {
    analyze_node::<FullLeafNode>(path, |node| {
        node.values.iter().filter(|c| **c != [0; 32]).count()
    })
}

fn analyze_sparse_leaf<const N: usize>(path: &Path) -> [usize; 257] {
    analyze_node::<SparseLeafNode<N>>(path, |node| {
        node.values.iter().filter(|c| c.item != [0; 32]).count()
    })
}

fn analyze_node<N>(path: &Path, count_fn: impl Fn(&N) -> usize) -> [usize; 257]
where
    N: DiskRepresentable + Send + Sync,
{
    static COUNT: AtomicU64 = AtomicU64::new(0);

    let mut sizes = [0; 257];
    let s = NodeFileStorage::<N, NoSeekFile>::open(path).unwrap();
    for idx in 0..s.next_idx.load(Ordering::Relaxed) {
        let node = match s.get(idx).map_err(BTError::into_inner) {
            Ok(n) => n,
            Err(Error::NotFound) => continue,
            Err(e) => panic!("{}", e),
        };
        let filled_slots = count_fn(&node);
        sizes[filled_slots] += 1;
        let count = COUNT.fetch_add(1, Ordering::Relaxed);
        if count.is_multiple_of(PRINT_INTERVAL) {
            println!("analyzed {count} nodes");
        }
    }
    sizes
}
