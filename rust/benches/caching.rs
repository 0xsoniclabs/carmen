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
        Arc,
        atomic::{AtomicU8, AtomicU64, Ordering},
    },
};

use carmen_rust::{
    error::BTResult,
    node_manager::{
        NodeManager,
        cached_node_manager::CachedNodeManager,
        lock_cache::{EvictionHooks, LockCache},
    },
    storage::{self, Storage},
};
use criterion::{BenchmarkId, criterion_group, criterion_main};
use quick_cache::{Lifecycle, UnitWeighter};

use crate::utils::{execute_with_threads, pow_2_threads, with_prob};
pub mod utils;

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
    type Key = u64;
    type Value = i64;

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
    type Id = u64;
    type Item = i64;

    fn open(_path: &std::path::Path) -> BTResult<Self, storage::Error> {
        Ok(Self::new())
    }

    fn get(&self, _id: Self::Id) -> BTResult<Self::Item, storage::Error> {
        Ok(42i64)
    }

    fn reserve(&self, _item: &Self::Item) -> Self::Id {
        self.id_counter.fetch_add(1, Ordering::Relaxed)
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
enum Cache {
    QuickCache(quick_cache::sync::Cache<u64, i64, UnitWeighter, RandomState, RandomPinner>),
    CachedNodeManager(CachedNodeManager<ProducerStorage>),
    LockCache(LockCache<u64, i64>),
}

/// Enum representing the different cache implementations used in the benchmarks
#[derive(Debug, Clone, Copy)]
pub enum CacheType {
    QuickCache,
    LockCache,
    CachedNodeManager,
}

impl std::fmt::Display for CacheType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheType::QuickCache => write!(f, "QuickCache"),
            CacheType::LockCache => write!(f, "LockCache"),
            CacheType::CachedNodeManager => write!(f, "CachedNodeManager"),
        }
    }
}

impl CacheType {
    /// Initializes a cache of the given type with the given size and pinning probability.
    fn make_cache(self, size: u64, pinning_prob: u8) -> Cache {
        static PINNING_PROB: AtomicU8 = AtomicU8::new(0);
        match self {
            CacheType::QuickCache => Cache::QuickCache(quick_cache::sync::Cache::with(
                size as usize,
                size,
                UnitWeighter,
                RandomState::default(),
                RandomPinner { prob: pinning_prob },
            )),
            CacheType::CachedNodeManager => {
                PINNING_PROB.store(pinning_prob, Ordering::Relaxed);
                let storage = ProducerStorage::new();
                Cache::CachedNodeManager(CachedNodeManager::new(
                    size as usize,
                    storage,
                    move |_| with_prob(PINNING_PROB.load(Ordering::Relaxed)),
                ))
            }
            CacheType::LockCache => Cache::LockCache(LockCache::new(
                size as usize,
                Arc::new(RandomPinner { prob: pinning_prob }),
            )),
        }
    }
}

impl Cache {
    /// Fills the cache to its capacity, using ids in range `0..capacity`
    fn fill(&mut self) {
        for i in 0..self.capacity() {
            match self {
                Cache::QuickCache(cache) => {
                    cache.insert(i, 42i64);
                }
                Cache::LockCache(lock_cache) => {
                    let _unused = lock_cache
                        .get_read_access_or_insert(i, || Ok(42i64))
                        .unwrap();
                }
                Cache::CachedNodeManager(node_manager) => {
                    let _unused = node_manager.get_read_access(i).unwrap();
                }
            }
        }
    }

    /// Returns the capacity of the cache.
    fn capacity(&self) -> u64 {
        match self {
            Cache::QuickCache(cache) => cache.capacity(),
            Cache::CachedNodeManager(node_manager) => node_manager.capacity(),
            Cache::LockCache(lock_cache) => lock_cache.capacity(),
        }
    }

    /// Executes a read operation on the cache for the given id.
    fn execute_read_op(&self, iter: u64) {
        match self {
            Cache::QuickCache(cache) => match cache.get_value_or_guard(&iter, None) {
                quick_cache::sync::GuardResult::Guard(guard) => {
                    let _ = guard.insert(42i64);
                }
                quick_cache::sync::GuardResult::Value(_)
                | quick_cache::sync::GuardResult::Timeout => {}
            },
            Cache::CachedNodeManager(node_manager) => {
                let _node = node_manager.get_read_access(iter).unwrap();
            }
            Cache::LockCache(lock_cache) => {
                let _node = lock_cache
                    .get_read_access_or_insert(iter, || Ok(42i64))
                    .unwrap();
            }
        }
    }
}

/// Benchmark caches read performance.
/// It varies:
/// - Cache size (influence contention)
/// - Whether the accessed ids are in cache or not (cache hit/miss)
/// - Number of threads (influence contention)
fn read_benchmark(c: &mut criterion::Criterion) {
    fastrand::seed(123);

    let cache_sizes = [100_000, 1_000_000];
    for cache_size in cache_sizes {
        for in_cache in [true, false] {
            let get_id = |i: u64| {
                if in_cache {
                    i % cache_size
                } else {
                    i + cache_size
                }
            };
            let mut bench_group = c.benchmark_group(format!(
                "caching/read/{cache_size}capacity/{in_cache}_cached"
            ));
            for num_threads in pow_2_threads() {
                for cache_type in [
                    CacheType::QuickCache,
                    CacheType::LockCache,
                    CacheType::CachedNodeManager,
                ] {
                    let mut cache = cache_type.make_cache(cache_size, 0);
                    cache.fill();
                    let mut completed_iterations = 0u64;
                    bench_group.bench_with_input(
                        BenchmarkId::from_parameter(format!(
                            "{num_threads:02}threads/{cache_type}"
                        )),
                        &(),
                        |b, _| {
                            b.iter_custom(|iters| {
                                execute_with_threads(
                                    num_threads as u64,
                                    iters,
                                    &mut completed_iterations,
                                    || (),
                                    |iter, _| {
                                        let id = get_id(iter);
                                        cache.execute_read_op(id);
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

/// Benchmark the effect of different pinning probabilities on cache performance
/// It varies:
/// - Pinning probability (forces linear scans on evictable items)
/// - Cache size (influence contention)
/// - Number of threads (influence contention)
fn pinning_benchmark(c: &mut criterion::Criterion) {
    fastrand::seed(123);

    for pinning_prob in [0, 10, 25, 50] {
        for cache_size in [100_000, 1_000_000] {
            let mut bench_group = c.benchmark_group(format!(
                "caching/pinning/{cache_size}capacity/{pinning_prob}pinning_prob"
            ));
            for num_threads in pow_2_threads() {
                for cache_type in [
                    CacheType::QuickCache,
                    CacheType::LockCache,
                    CacheType::CachedNodeManager,
                ] {
                    let mut cache = cache_type.make_cache(cache_size, pinning_prob);
                    cache.fill();
                    let mut completed_iterations = 0u64;
                    bench_group.bench_with_input(
                        BenchmarkId::from_parameter(format!(
                            "{num_threads:02}threads/{cache_type}"
                        )),
                        &(),
                        |b, _| {
                            b.iter_custom(|iters| {
                                execute_with_threads(
                                    num_threads as u64,
                                    iters,
                                    &mut completed_iterations,
                                    || (),
                                    |iter, _| {
                                        // Force eviction on every read by only requesting ids that
                                        // are not in the cache
                                        cache.execute_read_op(iter + cache_size);
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

criterion_group!(name = caching; config = criterion::Criterion::default(); targets = pinning_benchmark, read_benchmark);
criterion_main!(caching);
