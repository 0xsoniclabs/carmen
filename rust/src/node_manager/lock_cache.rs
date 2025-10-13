// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use dashmap::DashSet;
use quick_cache::{
    DefaultHashBuilder, Lifecycle, UnitWeighter,
    sync::{Cache, DefaultLifecycle},
};

use crate::error::Error;

/// A trait for handling eviction events in the cache.
pub trait OnEvict<K, V>: Send + Sync {
    fn on_evict(&self, key: K, value: V);
}

/// A cache that holds items (`K`/`V` pairs) on which read/write locks can be acquired.
///
/// The cache has a fixed capacity and evicts items when full.
/// An eviction callback can be provided to handle evicted items.
/// If an item is currently locked for reading or writing, it will not be evicted.
pub struct LockCache<K, V> {
    locks: Arc<[RwLock<V>]>,
    free_slots: Arc<DashSet<usize>>,
    cache: Cache<K, usize, UnitWeighter, DefaultHashBuilder, ItemLifecycle<K, V>>,
}

impl<K, V> LockCache<K, V>
where
    K: Copy + Eq + std::hash::Hash,
    V: Default,
{
    /// Creates a new cache with the given capacity and eviction callback.
    ///
    /// The actual capacity might differ slightly due to rounding performed by quick-cache.
    pub fn new(capacity: usize, on_evict: Arc<dyn OnEvict<K, V>>) -> Self {
        let options = quick_cache::OptionsBuilder::new()
            .estimated_items_capacity(capacity)
            .weight_capacity(capacity as u64) // unit weight per value
            .build()
            .unwrap();

        let true_capacity = {
            // Create temporary quick-cache instance to determine true capacity.
            let tmp_cache = Cache::<K, usize>::with_options(
                options.clone(),
                UnitWeighter,
                DefaultHashBuilder::default(),
                DefaultLifecycle::default(),
            );
            tmp_cache.num_shards() * tmp_cache.shard_capacity() as usize
        };

        let num_slots = true_capacity + 1;
        let locks: Arc<[_]> = (0..num_slots).map(|_| RwLock::default()).collect();
        let free_slots = Arc::new(DashSet::from_iter(0..num_slots));

        let cache = Cache::with_options(
            options,
            UnitWeighter,
            DefaultHashBuilder::default(),
            ItemLifecycle {
                locks: locks.clone(),
                free_slots: free_slots.clone(),
                callback: on_evict,
            },
        );

        LockCache {
            locks,
            free_slots,
            cache,
        }
    }

    /// Accesses the value for the given key for reading.
    /// Multiple concurrent read accesses to the same item are allowed.
    /// While a read lock is held, the item will not be evicted.
    ///
    /// If the key is not present, it is inserted using `insert_fn`.
    /// Any error returned by `insert_fn` is propagated to the caller.
    pub fn get_read_access_or_insert(
        &self,
        key: K,
        insert_fn: impl Fn() -> Result<V, Error>,
    ) -> Result<RwLockReadGuard<'_, V>, Error> {
        self.get_access_or_insert(key, insert_fn, |lock| lock.read().unwrap())
    }

    /// Accesses the value for the given key for writing.
    /// While a write lock is held, no other read or write access to the same item is allowed,
    /// and the item will not be evicted.
    ///
    /// If the key is not present, it is inserted using `insert_fn`.
    /// Any error returned by `insert_fn` is propagated to the caller.
    pub fn get_write_access_or_insert(
        &self,
        key: K,
        insert_fn: impl Fn() -> Result<V, Error>,
    ) -> Result<RwLockWriteGuard<'_, V>, Error> {
        self.get_access_or_insert(key, insert_fn, |lock| lock.write().unwrap())
    }

    /// Removes the item with the given key from the cache, if it exists.
    ///
    /// If the item is currently locked for reading or writing, this method will block
    /// until the lock is released.
    pub fn remove(&self, key: K) {
        if let Some(slot) = self.cache.get(&key) {
            // Get exclusive write access before removing the key,
            // ensuring that no other thread is holding a reference to it.
            let mut guard = self.locks[slot].write().unwrap();
            self.cache.remove(&key);
            *guard = V::default();
            self.free_slots.insert(slot);
        }
    }

    /// Iterates over all items in the cache, returning a write lock guard for each.
    ///
    /// The iterator will yield all items that are present in the cache at the time of
    /// creation, unless they are removed or evicted concurrently. The iterator may
    /// also yield items that have been added after the iterator was created.
    pub fn iter_write(&self) -> impl Iterator<Item = (K, RwLockWriteGuard<'_, V>)> {
        self.cache
            .iter()
            .map(|(key, slot)| (key, slot, self.locks[slot].write().unwrap()))
            .filter(|(key, slot, _)| {
                // Ensure the slot is still valid (has not been evicted/removed concurrently).
                let current_slot = self.cache.peek(key);
                current_slot.is_some() && current_slot.unwrap() == *slot
            })
            .map(|(key, _, guard)| (key, guard))
    }

    /// Shared implementation for [`get_read_access_or_insert`] and [`get_write_access_or_insert`].
    /// `access_fn` should either return a read or write lock guard.
    fn get_access_or_insert<'a, T>(
        &'a self,
        key: K,
        insert_fn: impl Fn() -> Result<V, Error>,
        access_fn: impl Fn(&'a RwLock<V>) -> T + 'a,
    ) -> Result<T, Error> {
        loop {
            match self.cache.get_value_or_guard(&key, None) {
                quick_cache::sync::GuardResult::Value(slot) => {
                    let slot_guard = access_fn(&self.locks[slot]);
                    if let Some(current_slot) = self.cache.peek(&key)
                        && current_slot == slot
                    {
                        return Ok(slot_guard);
                    }
                    continue;
                }
                quick_cache::sync::GuardResult::Guard(cache_guard) => {
                    // Get value first to avoid unnecessarily allocating a slot in case it fails.
                    let value = insert_fn()?;
                    let slot = loop {
                        // While there should always be a free slot, another thread may
                        // simultaneously be inserting a key and temporarily hold the
                        // last free slot. Since this can only happen if the cache is full,
                        // that thread will eventually evict an item and free up a slot.
                        let slot = self.free_slots.iter().next().map(|s| *s);
                        if let Some(slot) = slot
                            && let Some(slot) = self.free_slots.remove(&slot)
                        {
                            break slot;
                        }
                        std::hint::spin_loop();
                    };
                    let mut slot_guard = self.locks[slot].write().unwrap();
                    *slot_guard = value;
                    // Re-acquire the type of lock the caller requested (read or write).
                    // We do not risk racing on the slot here since we haven't inserted it into
                    // the cache yet.
                    drop(slot_guard);
                    let slot_guard = access_fn(&self.locks[slot]);
                    // We hold the lock on the slot while inserting the key into the cache,
                    // thereby avoiding the key from immediately being evicted again.
                    // This is important since we always have to return a valid lock.
                    cache_guard
                        .insert(slot)
                        .expect("cache entry should not be modified concurrently");
                    assert!(self.cache.len() < self.locks.len());
                    return Ok(slot_guard);
                }
                quick_cache::sync::GuardResult::Timeout => unreachable!(),
            }
        }
    }
}

struct ItemLifecycle<K, V> {
    locks: Arc<[RwLock<V>]>,
    free_slots: Arc<DashSet<usize>>,
    callback: Arc<dyn OnEvict<K, V>>,
}

impl<K, V> Clone for ItemLifecycle<K, V> {
    fn clone(&self) -> Self {
        ItemLifecycle {
            locks: self.locks.clone(),
            free_slots: self.free_slots.clone(),
            callback: self.callback.clone(),
        }
    }
}

impl<K, V> Lifecycle<K, usize> for ItemLifecycle<K, V>
where
    K: Copy,
    V: Default,
{
    type RequestState = ();

    fn begin_request(&self) -> Self::RequestState {}

    fn is_pinned(&self, _key: &K, slot: &usize) -> bool {
        // If the lock is currently held for writing, we consider the item pinned.
        self.locks[*slot].try_write().is_err()
    }

    fn on_evict(&self, _state: &mut Self::RequestState, key: K, slot: usize) {
        let value = {
            let mut lock = self.locks[slot].write().unwrap();
            std::mem::take(&mut *lock)
        };
        self.callback.on_evict(key, value);
        self.free_slots.insert(slot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage;

    #[derive(Default)]
    struct EvictionLogger {
        evicted: DashSet<(u32, i32)>,
    }

    impl OnEvict<u32, i32> for EvictionLogger {
        fn on_evict(&self, key: u32, value: i32) {
            self.evicted.insert((key, value));
        }
    }

    fn not_found() -> Result<i32, Error> {
        Err(Error::Storage(storage::Error::NotFound))
    }

    /// Helper function for performing a get/insert where we don't care about the returned guard.
    fn ignore_guard<T>(result: Result<T, Error>) {
        let _guard = result.unwrap();
    }

    #[test]
    fn new_creates_cache_with_correct_capacity() {
        let logger = Arc::new(EvictionLogger::default());
        let capacity = 10;
        let cache = LockCache::<u32, i32>::new(capacity, logger);

        assert_eq!(cache.locks.len(), capacity + 1);
        assert_eq!(cache.cache.capacity(), capacity as u64); // Unit weight per value
        // Check slots are correctly initialized
        for i in 0..(capacity + 1) {
            assert!(cache.free_slots.contains(&i));
            assert_eq!(*cache.locks[i].read().unwrap(), i32::default());
        }
    }

    #[rstest_reuse::apply(get_method)]
    fn items_can_be_inserted_and_removed(
        #[case] get_fn: GetOrInsertMethod<fn() -> Result<i32, Error>>,
    ) {
        let logger = Arc::new(EvictionLogger::default());
        let cache = LockCache::<u32, i32>::new(10, logger.clone());

        ignore_guard(get_fn(&cache, 1u32, || Ok(123)));
        ignore_guard(get_fn(&cache, 2u32, || Ok(456)));
        ignore_guard(get_fn(&cache, 3u32, || Ok(789)));

        {
            assert_eq!(get_fn(&cache, 1u32, not_found).unwrap(), 123);
            assert_eq!(get_fn(&cache, 2u32, not_found).unwrap(), 456);
            assert_eq!(get_fn(&cache, 3u32, not_found).unwrap(), 789);
        }

        cache.remove(2u32);
        let res = get_fn(&cache, 2u32, not_found);
        assert!(matches!(res, Err(Error::Storage(storage::Error::NotFound))));

        cache.remove(9999u32); // Removing non-existing key is a no-op
    }

    #[test]
    fn iter_returns_all_items() {
        let logger = Arc::new(EvictionLogger::default());
        let cache = LockCache::<u32, i32>::new(3, logger.clone());

        ignore_guard(cache.get_read_access_or_insert(1u32, || Ok(123)));
        ignore_guard(cache.get_read_access_or_insert(2u32, || Ok(456)));
        ignore_guard(cache.get_read_access_or_insert(3u32, || Ok(789)));

        let mut found = vec![];
        for (key, guard) in cache.iter_write() {
            found.push((key, *guard));
        }
        found.sort_unstable();
        assert_eq!(found, vec![(1, 123), (2, 456), (3, 789)]);
    }

    #[test]
    fn exceeding_capacity_causes_eviction() {
        let logger = Arc::new(EvictionLogger::default());
        let cache = LockCache::<u32, i32>::new(2, logger.clone());

        ignore_guard(cache.get_read_access_or_insert(1u32, || Ok(123)));
        ignore_guard(cache.get_read_access_or_insert(2u32, || Ok(456)));
        assert!(logger.evicted.is_empty());

        // By default quick-cache would immediately evict key 3.
        // Since we keep a lock on it during get_read_access_or_insert (thereby pinning it), key 1
        // is evicted instead.
        ignore_guard(cache.get_read_access_or_insert(3u32, || Ok(789)));
        assert_eq!(logger.evicted.len(), 1);
        assert!(logger.evicted.contains(&(1, 123)));

        // Key 3 is now in the cache
        {
            let guard = cache.get_read_access_or_insert(3u32, not_found).unwrap();
            assert_eq!(*guard, 789);
        }

        // Key 1 is not
        let res = cache.get_read_access_or_insert(1u32, not_found);
        assert!(matches!(res, Err(Error::Storage(storage::Error::NotFound))));

        assert_eq!(cache.free_slots.len(), 1);
        for slot in cache.free_slots.iter() {
            // The evicted slot is reset to the default value.
            assert_eq!(*cache.locks[*slot].read().unwrap(), i32::default());
        }
    }

    #[test]
    fn holding_lock_prevents_eviction() {
        let logger = Arc::new(EvictionLogger::default());
        let cache = LockCache::<u32, i32>::new(2, logger.clone());

        let _outside_guard = cache.get_read_access_or_insert(1u32, || Ok(123)).unwrap();

        {
            let _guard = cache.get_read_access_or_insert(2u32, || Ok(456)).unwrap();
            assert!(logger.evicted.is_empty());
        }

        {
            // Since we now hold a lock on key 1, key 2 is evicted instead.
            let _guard = cache.get_read_access_or_insert(3u32, || Ok(789)).unwrap();
            assert!(logger.evicted.contains(&(2, 456)));
        }
    }

    #[test]
    fn removing_keys_frees_up_slots() {
        let logger = Arc::new(EvictionLogger::default());
        let cache = LockCache::<u32, i32>::new(2, logger.clone());

        assert_eq!(cache.free_slots.len(), 3); // 2 + 1

        ignore_guard(cache.get_read_access_or_insert(1u32, || Ok(123)));
        ignore_guard(cache.get_read_access_or_insert(2u32, || Ok(456)));
        assert_eq!(cache.free_slots.len(), 1);

        cache.remove(1u32);
        assert_eq!(cache.free_slots.len(), 2);

        for slot in cache.free_slots.iter() {
            // The removed slot is reset to the default value.
            assert_eq!(*cache.locks[*slot].read().unwrap(), i32::default());
        }
    }

    #[test]
    fn removed_items_are_not_considered_evicted() {
        let logger = Arc::new(EvictionLogger::default());
        let cache = LockCache::<u32, i32>::new(2, logger.clone());

        ignore_guard(cache.get_read_access_or_insert(1u32, || Ok(123)));
        assert!(logger.evicted.is_empty());
        cache.remove(1u32);
        assert!(logger.evicted.is_empty());
    }

    #[test]
    fn item_lifecycle_is_pinned_checks_lock() {
        let nodes: Arc<[_]> = Arc::from(vec![RwLock::default()].into_boxed_slice());
        let lifecycle = ItemLifecycle {
            locks: nodes,
            free_slots: Arc::new(DashSet::new()),
            callback: Arc::new(EvictionLogger::default()),
        };

        // Element is not pinned as it's not locked
        assert!(!lifecycle.is_pinned(&0, &0));

        // Element is pinned as another thread holds a lock
        let _guard = lifecycle.locks[0].read().unwrap(); // Lock item at pos 0
        assert!(lifecycle.is_pinned(&0, &0));
    }

    #[test]
    fn item_lifecycle_on_evict_invokes_callback_and_resets_slot() {
        let nodes: Arc<[_]> = Arc::from(vec![RwLock::new(123)].into_boxed_slice());
        let free_slots = Arc::new(DashSet::new());
        let logger = Arc::new(EvictionLogger::default());
        let lifecycle = ItemLifecycle {
            locks: nodes,
            free_slots: free_slots.clone(),
            callback: logger.clone(),
        };
        lifecycle.on_evict(&mut (), 42, 0);
        assert!(logger.evicted.contains(&(42, 123)));
        assert!(free_slots.contains(&0));
        assert_eq!(*lifecycle.locks[0].read().unwrap(), i32::default());
    }

    /// Type alias for a closure that calls either `get_read_access` or `get_write_access`
    type GetOrInsertMethod<F> = fn(&LockCache<u32, i32>, u32, F) -> Result<i32, Error>;

    /// Reusable rstest template to test both `get_read_access` and `get_write_access`
    #[rstest_reuse::template]
    #[rstest::rstest]
    #[case::get_read_access((|cache, id, insert_fn| {
        let guard = cache.get_read_access_or_insert(id, insert_fn)?;
        Ok(*guard)
    }) as GetOrInsertMethod<_>)]
    #[case::get_write_access((|cache, id, insert_fn| {
        let guard = cache.get_write_access_or_insert(id, insert_fn)?;
        Ok(*guard)
    }) as GetOrInsertMethod<_>)]
    fn get_method(#[case] f: GetOrInsertMethod) {}
}
