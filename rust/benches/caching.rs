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
    self,
    hash::RandomState,
    sync::{
        Arc, LazyLock,
        atomic::{AtomicU8, AtomicU64, Ordering},
    },
};

use carmen_rust::{
    error::{BTResult, Error},
    node_manager::{
        NodeManager,
        cached_node_manager::CachedNodeManager,
        lock_cache::{EvictionHooks, LockCache},
    },
    storage::{self, DbMode, Storage},
    types::{HasDeltaVariant, HasEmptyId, HasEmptyNode},
};
use criterion::{BenchmarkId, criterion_group, criterion_main};
use quick_cache::{Lifecycle, UnitWeighter};

use crate::utils::{execute_with_threads, pow_2_threads, with_prob};
pub mod utils;

// Quick-cache will start evicting items before it is completely full.
// To avoid this, we have to overprovision the cache size by a certain amount.
const CACHE_SIZE_OVERPROVISION_FACTOR: f64 = 0.2;

/// A simple identifier for benchmarking purposes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
struct BenchId(u64);

impl HasEmptyId for BenchId {
    fn empty_id() -> Self {
        BenchId(u64::MAX)
    }

    fn is_empty_id(&self) -> bool {
        self.0 == u64::MAX
    }
}

/// A simple value for benchmarking purposes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
struct BenchValue(i64);

impl HasEmptyNode for BenchValue {
    fn empty_node() -> Self {
        BenchValue(i64::MAX)
    }

    fn is_empty_node(&self) -> bool {
        self.0 == i64::MAX
    }
}

impl HasDeltaVariant for BenchValue {
    type Id = BenchId;

    fn needs_full(&self) -> Option<Self::Id> {
        None
    }

    fn copy_from_base(&mut self, _full: &Self) -> BTResult<(), Error> {
        Ok(())
    }
}

/// A component that randomly pins items based on a given probability.
#[derive(Clone, Default)]
struct RandomPinner {
    prob: u8,
}

impl<K, V> Lifecycle<K, V> for RandomPinner {
    type RequestState = ();

    fn begin_request(&self) -> Self::RequestState {}

    fn on_evict(&self, _state: &mut Self::RequestState, _key: K, _val: V) {}

    fn is_pinned(&self, _key: &K, _val: &V) -> bool {
        with_prob(self.prob)
    }
}

impl EvictionHooks for RandomPinner {
    type Key = BenchId;
    type Value = BenchValue;

    fn is_pinned(&self, _key: &Self::Key, _value: &Self::Value) -> bool {
        with_prob(self.prob)
    }
}

/// Storage implementation that produces constant data for any requested id, with atomically
/// incrementing ids.
struct ProducerStorage {
    id_counter: AtomicU64,
}

impl ProducerStorage {
    fn new() -> Self {
        Self {
            id_counter: AtomicU64::new(0),
        }
    }
}

impl Storage for ProducerStorage {
    type Id = BenchId;
    type Item = BenchValue;

    fn open(_path: &std::path::Path, _mode: DbMode) -> BTResult<Self, storage::Error> {
        Ok(Self::new())
    }

    fn get(&self, _id: Self::Id) -> BTResult<Self::Item, storage::Error> {
        Ok(BenchValue(42))
    }

    fn reserve(&self, _item: &Self::Item) -> Self::Id {
        BenchId(self.id_counter.fetch_add(1, Ordering::Relaxed))
    }

    fn set(&self, _id: Self::Id, _item: &Self::Item) -> BTResult<(), carmen_rust::storage::Error> {
        Ok(())
    }

    fn delete(&self, _id: Self::Id) -> BTResult<(), carmen_rust::storage::Error> {
        Ok(())
    }

    fn close(self) -> BTResult<(), storage::Error> {
        Ok(())
    }
}

/// Enum wrapping the different cache implementations used in the benchmarks
#[allow(clippy::type_complexity)]
#[allow(clippy::enum_variant_names)]
#[allow(clippy::large_enum_variant)]
enum Cache {
    QuickCache(
        quick_cache::sync::Cache<BenchId, BenchValue, UnitWeighter, RandomState, RandomPinner>,
    ),
    CachedNodeManager(CachedNodeManager<ProducerStorage>),
    LockCache(LockCache<BenchId, BenchValue>),
}

/// Enum representing the different cache implementations used in the benchmarks
#[derive(Debug, Clone, Copy)]
pub enum CacheKind {
    QuickCache,
    LockCache,
    CachedNodeManager,
}

impl CacheKind {
    /// Initializes a cache of the given type with the given size and pinning probability.
    fn make_cache(self, size: u64, pinning_prob: u8) -> Cache {
        static PINNING_PROB: AtomicU8 = AtomicU8::new(0);
        let size = (size as f64 * (1.0 + CACHE_SIZE_OVERPROVISION_FACTOR)) as u64;
        match self {
            CacheKind::QuickCache => Cache::QuickCache(quick_cache::sync::Cache::with(
                size as usize,
                size,
                UnitWeighter,
                RandomState::default(),
                RandomPinner { prob: pinning_prob },
            )),
            CacheKind::CachedNodeManager => {
                PINNING_PROB.store(pinning_prob, Ordering::Relaxed);
                let storage = ProducerStorage::new();
                Cache::CachedNodeManager(CachedNodeManager::new(
                    size as usize,
                    storage,
                    move |_| with_prob(PINNING_PROB.load(Ordering::Relaxed)),
                ))
            }
            CacheKind::LockCache => Cache::LockCache(LockCache::new(
                size as usize,
                Arc::new(RandomPinner { prob: pinning_prob }),
            )),
        }
    }

    pub fn variants() -> impl Iterator<Item = CacheKind> {
        [
            CacheKind::QuickCache,
            CacheKind::LockCache,
            CacheKind::CachedNodeManager,
        ]
        .into_iter()
    }
}

impl Cache {
    /// Fills the cache with `num_entries` values, using consecutive ids starting from 0.
    fn fill(&self, num_entries: u64) {
        for i in 0..num_entries {
            match self {
                Cache::QuickCache(cache) => {
                    cache.insert(BenchId(i), BenchValue(42));
                }
                Cache::LockCache(lock_cache) => {
                    let _unused = lock_cache
                        .get_read_access_or_insert(BenchId(i), || Ok(BenchValue(42)))
                        .unwrap();
                }
                Cache::CachedNodeManager(node_manager) => {
                    let _unused = node_manager.get_read_access(BenchId(i)).unwrap();
                }
            }
        }
    }

    /// Executes a read operation on the cache for the given id.
    fn read(&self, id: BenchId) {
        match self {
            Cache::QuickCache(cache) => match cache.get_value_or_guard(&id, None) {
                quick_cache::sync::GuardResult::Value(_) => {}
                quick_cache::sync::GuardResult::Guard(_)
                | quick_cache::sync::GuardResult::Timeout => {
                    panic!("Cache miss on QuickCache for id {id:?}");
                }
            },
            Cache::CachedNodeManager(node_manager) => {
                let _node = node_manager.get_read_access(id).unwrap();
            }
            Cache::LockCache(lock_cache) => {
                let _node = lock_cache
                    .get_read_access_or_insert(id, || {
                        panic!("Cache miss on LockCache for id {id:?}")
                    })
                    .unwrap();
            }
        }
    }

    fn add(&self, id: BenchId) {
        match self {
            Cache::QuickCache(cache) => match cache.get_value_or_guard(&id, None) {
                quick_cache::sync::GuardResult::Guard(guard) => {
                    guard.insert(BenchValue(42)).unwrap();
                }
                quick_cache::sync::GuardResult::Value(_) => {
                    panic!("Cache hit on QuickCache for id {id:?}");
                }
                quick_cache::sync::GuardResult::Timeout => {
                    unreachable!();
                }
            },
            Cache::CachedNodeManager(node_manager) => {
                let _node = node_manager.add(BenchValue(42)).unwrap();
            }
            Cache::LockCache(lock_cache) => {
                // No real way to enforce an insert
                let _unused = lock_cache
                    .get_read_access_or_insert(id, || Ok(BenchValue(42)))
                    .unwrap();
            }
        }
    }
}

/// Benchmark caches read performance.
/// It varies:
/// - Cache size (influence contention)
/// - Whether the accessed ids are in cache or not (cache hit/miss)
/// - Number of threads (influences contention)
fn read_benchmark(c: &mut criterion::Criterion) {
    fastrand::seed(123);

    let cache_sizes = if cfg!(debug_assertions) {
        [100, 1_000]
    } else {
        [100_000, 1_000_000]
    };
    for cache_size in cache_sizes {
        let mut bench_group = c.benchmark_group(format!("caching/read/{cache_size}capacity"));
        for num_threads in pow_2_threads(None) {
            for cache_type in CacheKind::variants() {
                let cache = LazyLock::new(|| {
                    let cache = cache_type.make_cache(cache_size, 0);
                    cache.fill(cache_size);
                    cache
                });
                let mut completed_iterations = 0u64;
                bench_group.bench_with_input(
                    BenchmarkId::from_parameter(format!("{num_threads:02}threads/{cache_type:?}")),
                    &(),
                    |b, _| {
                        b.iter_custom(|iters| {
                            execute_with_threads(
                                num_threads as u64,
                                iters,
                                &mut completed_iterations,
                                || (),
                                |iter, _| {
                                    // Make sure the accessed id is always in cache
                                    cache.read(BenchId(iter % cache_size));
                                },
                            )
                        });
                    },
                );
            }
        }
    }
}

/// Benchmark caches add performance.
/// It varies:
/// - Pinning probability (forces linear search for evictable items)
/// - Cache size (influence contention)
/// - Number of threads (influences contention)
fn add_benchmark(c: &mut criterion::Criterion) {
    fastrand::seed(123);

    let cache_sizes = if cfg!(debug_assertions) {
        [10]
    } else {
        [1_000_000]
    };
    for cache_size in cache_sizes {
        for pinning_prob in [0, 10, 25, 50] {
            let mut bench_group = c.benchmark_group(format!(
                "caching/add/{cache_size}capacity/{pinning_prob}pinning_prob"
            ));
            for num_threads in pow_2_threads(None) {
                for cache_type in CacheKind::variants() {
                    let cache = LazyLock::new(|| cache_type.make_cache(cache_size, pinning_prob));
                    let mut completed_iterations = 0u64;
                    bench_group.bench_with_input(
                        BenchmarkId::from_parameter(format!(
                            "{num_threads:02}threads/{cache_type:?}"
                        )),
                        &(),
                        |b, _| {
                            cache.fill(cache_size);
                            b.iter_custom(|iters| {
                                execute_with_threads(
                                    num_threads as u64,
                                    iters,
                                    &mut completed_iterations,
                                    || (),
                                    |iter, _| {
                                        // Force eviction on every read by only requesting ids that
                                        // are not in the cache
                                        cache.add(BenchId(iter + cache_size));
                                    },
                                )
                            });
                        },
                    );
                }
            }
        }
    }
}

criterion_group!(name = caching; config = criterion::Criterion::default(); targets = read_benchmark, add_benchmark);
criterion_main!(caching);
