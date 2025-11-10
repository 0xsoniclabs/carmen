use std::{
    ops::DerefMut,
    path::Path,
    sync::{
        Arc, Mutex, Once,
        atomic::{AtomicU8, AtomicU64, Ordering},
    },
};

use carmen_rust::{
    database::verkle::variants::managed::nodes::{
        Node, NodeFileStorageManager, NodeType, empty::EmptyNode, id::NodeId, inner::InnerNode,
        leaf::FullLeafNode, sparse_leaf::SparseLeafNode,
    },
    error::{BTResult, Error},
    node_manager::{NodeManager, cached_node_manager::CachedNodeManager},
    storage::{
        Storage,
        file::{MultiPageCachedFile, NoSeekFile, NodeFileStorage},
        storage_with_flush_buffer::StorageWithFlushBuffer,
    },
    types::TreeId,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dashmap::DashMap;

use crate::utils::{check_proportions, execute_with_threads, pow_2_threads, with_prob};

pub mod utils;

/// A simple in-memory storage for benchmarking purposes.
/// It does not check if an ID has been previously reserved before using it.
#[derive(Default, Debug)]
struct SimpleInMemoryStorage {
    nodes: DashMap<NodeId, Node>,
    next_id: AtomicU64,
}

impl Storage for SimpleInMemoryStorage {
    type Id = NodeId;
    type Item = Node;

    fn open(_path: &std::path::Path) -> BTResult<Self, carmen_rust::storage::Error>
    where
        Self: Sized,
    {
        Ok(SimpleInMemoryStorage::default())
    }

    fn get(&self, id: Self::Id) -> BTResult<Self::Item, carmen_rust::storage::Error> {
        match self.nodes.get(&id) {
            Some(node) => Ok(node.value().clone()),
            None => Err(carmen_rust::storage::Error::NotFound.into()),
        }
    }

    fn reserve(&self, item: &Self::Item) -> Self::Id {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        NodeId::from_idx_and_node_type(id, item.to_node_type())
    }

    fn set(&self, id: Self::Id, item: &Self::Item) -> BTResult<(), carmen_rust::storage::Error> {
        self.nodes.insert(id, item.clone());
        Ok(())
    }

    fn delete(&self, id: Self::Id) -> BTResult<(), carmen_rust::storage::Error> {
        self.nodes.remove(&id);
        Ok(())
    }

    fn close(self) -> BTResult<(), carmen_rust::storage::Error> {
        Ok(())
    }
}

/// An enum to represent the different storage types to benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum StorageType {
    InMemory,
    FileBased,
    FileBasedWithFlushBuffer,
}

/// Construct a storage of the given type.
fn make_storage(storage_type: StorageType) -> Box<dyn Storage<Id = NodeId, Item = Node>> {
    match storage_type {
        StorageType::InMemory => Box::new(SimpleInMemoryStorage::open(Path::new("")).unwrap())
            as Box<dyn Storage<Id = NodeId, Item = Node>>,
        StorageType::FileBased | StorageType::FileBasedWithFlushBuffer => {
            type FileStorageImpl = MultiPageCachedFile<16, NoSeekFile, false>;
            type FileStorage = NodeFileStorageManager<
                NodeFileStorage<InnerNode, FileStorageImpl>,
                NodeFileStorage<SparseLeafNode<2>, FileStorageImpl>,
                NodeFileStorage<FullLeafNode, FileStorageImpl>,
            >;
            // Clear up bench_storage path
            let bench_storage_dir = "bench_storage";
            let _ = std::fs::remove_dir_all(bench_storage_dir); // This may fail for non-existing dir, ignore errors
            if storage_type == StorageType::FileBasedWithFlushBuffer {
                Box::new(
                    StorageWithFlushBuffer::<FileStorage>::open(Path::new(bench_storage_dir))
                        .unwrap(),
                ) as Box<dyn Storage<Id = NodeId, Item = Node>>
            } else {
                Box::new(FileStorage::open(Path::new(bench_storage_dir)).unwrap())
                    as Box<dyn Storage<Id = NodeId, Item = Node>>
            }
        }
    }
}

/// Initializes the storage with the given number of nodes and proportions.
fn init_storage(
    storage: &mut dyn Storage<Id = NodeId, Item = Node>,
    num_nodes: u64,
    node_proportions: &[(NodeType, f32)],
) -> BTResult<Vec<NodeId>, Error> {
    let mut ids = Vec::new();
    for (node_type, proportion) in node_proportions {
        let count = (num_nodes as f32 * proportion) as u64;
        for _ in 0..count {
            let node = match node_type {
                NodeType::Inner => Node::Inner(Box::default()),
                NodeType::Leaf2 => Node::Leaf2(Box::default()),
                NodeType::Leaf256 => Node::Leaf256(Box::default()),
                NodeType::Empty => Node::Empty(EmptyNode),
            };
            let id = storage.reserve(&node);
            storage.set(id, &node)?;
            ids.push(id);
        }
    }

    Ok(ids)
}

/// An operation to perform on the [`NodeManager`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
enum Op {
    Add,
    GetR,
    GetW,
    Delete,
}

impl From<Op> for u8 {
    fn from(op: Op) -> Self {
        match op {
            Op::Add => 0,
            Op::GetR => 1,
            Op::GetW => 2,
            Op::Delete => 3,
        }
    }
}

impl TryFrom<u8> for Op {
    type Error = std::num::IntErrorKind;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Op::Add),
            1 => Ok(Op::GetR),
            2 => Ok(Op::GetW),
            3 => Ok(Op::Delete),
            _ => Err(std::num::IntErrorKind::PosOverflow),
        }
    }
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Add => write!(f, "A"),
            Op::GetR => write!(f, "GR"),
            Op::GetW => write!(f, "GW"),
            Op::Delete => write!(f, "D"),
        }
    }
}

impl Op {
    /// Execute the operation on the  with the provided node IDs.
    /// The delete operation uses the provided delete_id mutex to get IDs to delete.
    /// The other operations use random IDs from the provided list.
    fn execute<S>(
        self,
        manager: &Arc<impl NodeManager<Id = S::Id, NodeType = S::Item>>,
        ids: &[NodeId],
        delete_id: &Arc<Mutex<Vec<NodeId>>>, // Maybe use a crossbeam queue
    ) where
        S: Storage<Id = NodeId, Item = Node> + 'static,
    {
        let get_random_id = || {
            let idx = fastrand::usize(0..ids.len());
            ids[idx]
        };

        match self {
            Op::Add => {
                let _ = manager.add(Node::Leaf2(Box::default())).unwrap();
            }
            Op::GetR => {
                let _unused = manager.get_read_access(get_random_id()).unwrap();
            }
            Op::GetW => {
                let mut guard = manager.get_write_access(get_random_id()).unwrap();
                let _unused = guard.deref_mut().deref_mut();
            }
            Op::Delete => {
                if let Some(id) = delete_id.lock().unwrap().pop() {
                    manager.delete(id).unwrap();
                }
            }
        }
    }
}

/// Execute the benchmark with the given parameters
/// In order:
/// - Initialize storage and node manager
/// - Fill the cache with some nodes
/// - Mark nodes in the cache as dirty according to the given proportion
/// - Reserve some nodes for delete operations according to the given proportion
/// - Create threads that perform random operations according to the given proportions
#[allow(clippy::too_many_arguments)]
fn execute(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    storage_type: StorageType,
    manager_size: u64,
    storage_size: u64,
    node_proportions: &[(NodeType, f32)],
    operations_proportions: &[(Op, f32)],
    pinning_prob: u8,
    dirty_nodes_proportions: u8,
    num_threads: usize,
) {
    static PINNING_PROB: AtomicU8 = AtomicU8::new(0);
    PINNING_PROB.store(pinning_prob, Ordering::Relaxed);

    let mut storage = make_storage(storage_type);
    let mut node_ids = init_storage(&mut *storage, storage_size, node_proportions).unwrap();
    let manager = Arc::new(CachedNodeManager::new(
        manager_size as usize,
        storage,
        move |_| with_prob(PINNING_PROB.load(Ordering::Relaxed)),
    ));

    // Take a portion of nodes to insert in the cache
    fastrand::shuffle(&mut node_ids);
    let node_in_manager = node_ids
        .iter()
        .take(manager_size as usize)
        .inspect(|&id| {
            // Insert in the cache
            let mut guard = manager.get_write_access(*id).unwrap();
            if with_prob(dirty_nodes_proportions) {
                // Mark as dirty
                let _ = guard.deref_mut().deref_mut();
            }
        })
        .copied()
        .collect::<Vec<_>>();
    // Reserve some nodes for delete operations with probability
    // operation_proportions[OP::delete]
    let reserved_node_delete = node_in_manager
        .iter()
        .filter(|_| {
            fastrand::f32()
                < operations_proportions
                    .iter()
                    .find(|(op, _)| *op == Op::Delete)
                    .map(|(_, p)| *p)
                    .unwrap_or(0.0)
        })
        .copied()
        .collect::<Vec<_>>();
    // Filter nodes reserved for deletion, as other concurrent operations on
    // them are invalid
    let available_node_ids = node_ids
        .iter()
        .filter(|id| !reserved_node_delete.contains(id))
        .copied()
        .collect::<Vec<_>>();
    let reserved_node_delete = Arc::new(Mutex::new(reserved_node_delete));

    // Each thread executes an operation based on the given proportions
    let thread_fn = {
        let reserved_node_delete = reserved_node_delete.clone();
        move |_, _: &mut ()| {
            let mut cumulative_prob = 0.0;
            let rand = fastrand::f32();
            for (operation, prob) in operations_proportions {
                cumulative_prob += *prob;
                if rand <= cumulative_prob {
                    operation.execute::<SimpleInMemoryStorage>(
                        &manager,
                        &available_node_ids,
                        &reserved_node_delete,
                    );
                    return;
                }
            }
            unreachable!("Probabilities should sum to 1.0");
        }
    };
    let mut completed_iterations = 0u64;
    group.bench_with_input(
        BenchmarkId::from_parameter(format!("{num_threads}threads/{storage_type:?}")),
        &(),
        |b, _| {
            b.iter_custom(|iters| {
                execute_with_threads(
                    num_threads as u64,
                    iters,
                    &mut completed_iterations,
                    || (),
                    &thread_fn,
                )
            });
        },
    );
}

/// Benchmark performing random operations on the CachedNodeManager with different storage
/// backends.
/// It varies:
/// - Storage type
/// - Node manager size
/// - Node type proportions (Which nodes are stored)
/// - Operations proportions (Which operations the threads are going to perform)
/// - Pinning probability (influences eviction time)
/// - Percentage of dirty nodes in the cache (triggers write-backs to storage)
/// - Number of threads
fn random_op_benchmark(c: &mut criterion::Criterion) {
    fastrand::seed(123);
    let storage_types = [
        // StorageType::InMemory,
        StorageType::FileBased,
        StorageType::FileBasedWithFlushBuffer,
    ];
    let node_proportions_vec = [vec![
        (NodeType::Leaf2, 0.7f32),
        (NodeType::Leaf256, 0.2),
        (NodeType::Inner, 0.1),
    ]];
    let operations_proportions_vec = [vec![
        (Op::Add, 0.1f32),
        (Op::GetR, 0.5),
        (Op::GetW, 0.3),
        (Op::Delete, 0.1),
    ]];

    check_proportions(&node_proportions_vec);
    check_proportions(&operations_proportions_vec);

    for manager_size in [100_000] {
        let storage_size = manager_size * 5; // Storage is bigger than the manager capacity
        for node_proportions in &node_proportions_vec {
            for operations_proportions in &operations_proportions_vec {
                for pinning_prob in [0, 10, 25] {
                    for dirty_nodes_proportion in [0, 10, 25, 50, 100] {
                        let mut group = c
                .benchmark_group(format!("random_ops/{manager_size}/{node_proportions:?}/{operations_proportions:?}/{pinning_prob}pinning_prob/{dirty_nodes_proportion}dirty_nodes_prop"));
                        for num_threads in pow_2_threads() {
                            for storage_type in storage_types {
                                execute(
                                    &mut group,
                                    storage_type,
                                    manager_size,
                                    storage_size,
                                    node_proportions,
                                    operations_proportions,
                                    pinning_prob,
                                    dirty_nodes_proportion,
                                    num_threads,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

criterion_group!(name = node_manager; config = Criterion::default(); targets = random_op_benchmark);
criterion_main!(node_manager);
