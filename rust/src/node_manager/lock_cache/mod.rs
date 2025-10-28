// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{num::NonZero, sync::TryLockError};

use dashmap::DashSet;
use quick_cache::{
    DefaultHashBuilder, Lifecycle, UnitWeighter,
    sync::{Cache, DefaultLifecycle, GuardResult},
};

use crate::{
    error::Error,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard, hint},
};

mod test_utils;

/// A trait for handling eviction events in the cache.
pub trait EvictionHooks: Send + Sync {
    type Key;
    type Value;

    /// Determines whether the given item is currently pinned and should not be evicted.
    /// This hook is only called if the item is otherwise eligible for eviction.
    fn is_pinned(&self, _key: &Self::Key, _value: &Self::Value) -> bool {
        false
    }

    /// Called when an item is evicted from the cache.
    /// This function should be fast, otherwise cache performance might be negatively affected.
    fn on_evict(&self, _key: Self::Key, _value: Self::Value) -> Result<(), Error> {
        Ok(())
    }
}

/// A cache that holds items (`K`/`V` pairs) on which read/write locks can be acquired.
///
/// The cache allows for concurrent access to its items, with one caveat: During a call to
/// [`LockCache::remove`], no other operation on the same key is allowed. Attempting to do so
/// will return an [`Error::IllegalConcurrentOperation`], after which the cache is in an
/// indeterminate state.
///
/// The cache has a fixed capacity and evicts items when full.
/// An eviction callback can be provided to handle evicted items.
/// If an item is currently locked for reading or writing, it will not be evicted.
pub struct LockCache<K, V> {
    locks: Arc<[RwLock<V>]>,
    free_slots: Arc<DashSet<usize>>,
    /// The quick-cache instance holds `Arc`s of slot indices into `locks`.
    /// By using an `Arc`, we can track how many threads are currently accessing a slot,
    /// and thereby avoid evicting items that are currently in use.
    /// Importantly, since each interaction with the cache for a specific key locks its
    /// respective shard, the lookup of a slot and the increment of its [`Arc::strong_count`]
    /// is an atomic operation from the perspective of the [`LockCache`].
    cache: Cache<K, Arc<usize>, UnitWeighter, DefaultHashBuilder, ItemLifecycle<K, V>>,
}

impl<K, V> LockCache<K, V>
where
    K: Copy + Eq + std::hash::Hash,
    V: Default,
{
    /// Creates a new cache with the given capacity and eviction callback.
    ///
    /// The actual capacity might differ slightly due to rounding performed by quick-cache.
    pub fn new(capacity: usize, hooks: Arc<dyn EvictionHooks<Key = K, Value = V>>) -> Self {
        // We allocate a couple of additional slots, roughly one for each concurrent thread.
        // This way, when the cache is full, we always have a free slot we can use to insert a new
        // item into the cache and force the eviction of an old one.
        let extra_slots = std::thread::available_parallelism().unwrap_or(NonZero::new(1).unwrap());
        Self::new_internal(capacity, extra_slots, hooks)
    }

    /// Internal constructor that allows to specify the number of `extra_slots` directly.
    /// This is required for testing.
    fn new_internal(
        capacity: usize,
        extra_slots: NonZero<usize>,
        hooks: Arc<dyn EvictionHooks<Key = K, Value = V>>,
    ) -> Self {
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
            tmp_cache.capacity() as usize
        };

        let num_slots = true_capacity + extra_slots.get();
        let locks: Arc<[_]> = (0..num_slots).map(|_| RwLock::default()).collect();
        let free_slots = Arc::new(DashSet::from_iter(0..num_slots));

        let cache = Cache::with_options(
            options,
            UnitWeighter,
            DefaultHashBuilder::default(),
            ItemLifecycle {
                locks: locks.clone(),
                free_slots: free_slots.clone(),
                hooks,
            },
        );

        LockCache {
            locks,
            free_slots,
            cache,
        }
    }

    /// Accesses the value for the given key for reading.
    /// Multiple concurrent read accesses to the same item are allowed,
    /// but any attempt to acquire a write lock will block until all read locks are released.
    /// While a read lock is held, the item will not be evicted.
    ///
    /// If the key is not present, it is inserted using `insert_fn`.
    /// Any error returned by `insert_fn` is propagated to the caller.
    pub fn get_read_access_or_insert(
        &self,
        key: K,
        insert_fn: impl FnOnce() -> Result<V, Error>,
    ) -> Result<RwLockReadGuard<'_, V>, Error> {
        self.get_access_or_insert(key, insert_fn, |lock| lock.read().unwrap())
    }

    /// Accesses the value for the given key for writing.
    /// No concurrent read or write access to the same item is allowed,
    /// and any attempt to acquire a lock will block until the write lock is released.
    /// While a write lock is held, the item will not be evicted.
    ///
    /// If the key is not present, it is inserted using `insert_fn`.
    /// Any error returned by `insert_fn` is propagated to the caller.
    pub fn get_write_access_or_insert(
        &self,
        key: K,
        insert_fn: impl FnOnce() -> Result<V, Error>,
    ) -> Result<RwLockWriteGuard<'_, V>, Error> {
        self.get_access_or_insert(key, insert_fn, |lock| lock.write().unwrap())
    }

    /// Removes the item with the given key from the cache, if it exists.
    ///
    /// This function must not be called concurrently with any other operation on the same key.
    /// If the key is currently being accessed by another thread, an error is returned, after
    /// which the cache is in an indeterminate state.
    pub fn remove(&self, key: K) -> Result<(), Error> {
        if let Some(slot) = self.cache.get(&key) {
            // Try getting exclusive write access before removing the key,
            // ensuring that no other thread is holding a reference to it.
            match self.locks[*slot].try_write() {
                Ok(mut guard) => {
                    self.cache.remove(&key);
                    if Arc::strong_count(&slot) > 1 {
                        return Err(Error::IllegalConcurrentOperation(
                            "another thread is attempting to access a key while it is being removed"
                                .to_owned(),
                        ));
                    }
                    *guard = V::default();
                    self.free_slots.insert(*slot);
                }
                Err(TryLockError::WouldBlock) => {
                    return Err(Error::IllegalConcurrentOperation(
                        "another thread is holding a lock on a key that is being removed"
                            .to_owned(),
                    ));
                }
                Err(TryLockError::Poisoned(e)) => panic!("poisoned lock: {e:?}"),
            }
        }
        Ok(())
    }

    /// Iterates over all items in the cache, returning a write lock guard for each.
    ///
    /// The iterator will yield all items that are present in the cache at the time of
    /// creation, unless they are evicted concurrently. The iterator may also yield
    /// items that have been added after the iterator was created.
    pub fn iter_write(&self) -> impl Iterator<Item = (K, RwLockWriteGuard<'_, V>)> {
        self.cache
            .iter()
            .map(|(key, slot)| (key, self.locks[*slot].write().unwrap()))
    }

    /// Shared implementation for [`get_read_access_or_insert`] and [`get_write_access_or_insert`].
    /// `access_fn` should either return a read or write lock guard.
    fn get_access_or_insert<'a, T>(
        &'a self,
        key: K,
        insert_fn: impl FnOnce() -> Result<V, Error>,
        access_fn: impl FnOnce(&'a RwLock<V>) -> T + 'a,
    ) -> Result<T, Error> {
        match self.cache.get_value_or_guard(&key, None) {
            GuardResult::Value(slot) => {
                // NOTE: After we get the slot and before we acquire the lock (this line),
                // the `Arc::strong_count` of the slot is at least 2, so the item cannot be
                // evicted concurrently.
                Ok(access_fn(&self.locks[*slot]))
            }
            GuardResult::Guard(cache_guard) => {
                // Get value first to avoid unnecessarily allocating a slot in case it fails.
                let value = insert_fn()?;
                let slot = loop {
                    // While there should always be a free slot, concurrent threads may
                    // simultaneously be inserting keys and temporarily hold all remaining
                    // free slots. Since this can only happen if the cache is full,
                    // those threads will eventually each evict an item and free up a slot.
                    let slot = self.free_slots.iter().next().map(|s| *s);
                    if let Some(slot) = slot
                        && let Some(slot) = self.free_slots.remove(&slot)
                    {
                        break slot;
                    }
                    hint::spin_loop();
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
                cache_guard.insert(Arc::new(slot)).map_err(|_| {
                    Error::IllegalConcurrentOperation(
                        "another thread removed the key while it was being inserted".to_owned(),
                    )
                })?;
                // In theory quick-cache can exceed its capacity if all items are pinned.
                // This should however never happen in practice for our usage patterns.
                if self.cache.len() >= self.locks.len() {
                    return Err(Error::CorruptedState(
            "LockCache's cache size is equal or bigger than the number of slots. This may have happened because an insert operation was executed with all cache entries marked as pinned".to_owned(),
            ));
                }
                Ok(slot_guard)
            }
            GuardResult::Timeout => unreachable!(),
        }
    }
}

/// Helper type responsible for pinning items and invoking the eviction callback.
///
/// Items are considered pinned if they have an [`Arc::strong_count`] greater than 1,
/// or if their corresponding slot is currently locked.
struct ItemLifecycle<K, V> {
    locks: Arc<[RwLock<V>]>,
    free_slots: Arc<DashSet<usize>>,
    hooks: Arc<dyn EvictionHooks<Key = K, Value = V>>,
}

impl<K, V> Clone for ItemLifecycle<K, V> {
    fn clone(&self) -> Self {
        ItemLifecycle {
            locks: self.locks.clone(),
            free_slots: self.free_slots.clone(),
            hooks: self.hooks.clone(),
        }
    }
}

impl<K, V> Lifecycle<K, Arc<usize>> for ItemLifecycle<K, V>
where
    K: Copy,
    V: Default,
{
    type RequestState = ();

    fn begin_request(&self) -> Self::RequestState {}

    fn is_pinned(&self, key: &K, slot: &Arc<usize>) -> bool {
        // If another thread is currently holding a reference to the slot,
        // we cannot evict the contained item.
        if Arc::strong_count(slot) > 1 {
            return true;
        }
        // If the lock is currently locked, we consider the item pinned.
        // If not, we let the hook decide.
        match self.locks[**slot].try_write() {
            Ok(guard) => self.hooks.is_pinned(key, &guard),
            Err(_) => true,
        }
    }

    /// Invokes the eviction callback, resets the slot to its default value and
    /// marks the slot as free.
    /// NOTE: this will panic if the eviction callback fails.
    fn on_evict(&self, _state: &mut Self::RequestState, key: K, slot: Arc<usize>) {
        let value = {
            let mut lock = self.locks[*slot].write().unwrap();
            std::mem::take(&mut *lock)
        };
        self.free_slots.insert(*slot);
        self.hooks
            .on_evict(key, value)
            .expect("eviction callback failed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        node_manager::lock_cache::test_utils::{
            EvictionLogger, GetOrInsertMethod, get_method, ignore_guard,
        },
        storage,
    };

    fn not_found() -> Result<i32, Error> {
        Err(Error::Storage(storage::Error::NotFound))
    }

    #[test]
    fn new_creates_cache_with_correct_capacity() {
        let logger = Arc::new(EvictionLogger::default());
        let capacity = 10;
        let cache = LockCache::<u32, i32>::new(capacity, logger);

        let extra_slots = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);
        assert_eq!(cache.locks.len(), capacity + extra_slots);
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
        type InsertFn = fn() -> Result<i32, Error>;
        let logger = Arc::new(EvictionLogger::default());
        let cache = LockCache::<u32, i32>::new(10, logger.clone());

        ignore_guard(get_fn(&cache, 1u32, &((|| Ok(123)) as InsertFn)));
        ignore_guard(get_fn(&cache, 2u32, &((|| Ok(456)) as InsertFn)));
        ignore_guard(get_fn(&cache, 3u32, &((|| Ok(789)) as InsertFn)));

        {
            assert_eq!(get_fn(&cache, 1u32, &(not_found as InsertFn)).unwrap(), 123);
            assert_eq!(get_fn(&cache, 2u32, &(not_found as InsertFn)).unwrap(), 456);
            assert_eq!(get_fn(&cache, 3u32, &(not_found as InsertFn)).unwrap(), 789);
        }

        cache.remove(2u32).unwrap();
        let res = get_fn(&cache, 2u32, &(not_found as InsertFn));
        assert!(matches!(res, Err(Error::Storage(storage::Error::NotFound))));

        cache.remove(9999u32).unwrap(); // Removing non-existing key is a no-op
    }

    #[test]
    fn iter_write_returns_all_items() {
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

        let items_to_insert = [(1u32, 123), (2u32, 456)];

        for (key, value) in items_to_insert {
            ignore_guard(cache.get_read_access_or_insert(key, || Ok(value)));
        }
        assert!(logger.evicted.is_empty());
        let free_slots = cache.free_slots.len();

        // By default quick-cache would immediately evict key 3.
        // Since we keep a lock on it during get_read_access_or_insert (thereby pinning it), one
        // of the other two keys is evicted instead.
        ignore_guard(cache.get_read_access_or_insert(3u32, || Ok(789)));
        assert_eq!(logger.evicted.len(), 1);
        let evicted_item = logger.evicted.iter().next().unwrap();
        assert!(items_to_insert.contains(&evicted_item));
        assert_eq!(cache.free_slots.len(), free_slots);

        // Key 3 is now in the cache
        {
            let guard = cache.get_read_access_or_insert(3u32, not_found).unwrap();
            assert_eq!(*guard, 789);
        }

        // Evicted key is gone
        let res = cache.get_read_access_or_insert(evicted_item.0, not_found);
        assert!(matches!(res, Err(Error::Storage(storage::Error::NotFound))));

        assert!(!cache.free_slots.is_empty());
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
        let extra_slots = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);

        assert_eq!(cache.free_slots.len(), 2 + extra_slots);

        ignore_guard(cache.get_read_access_or_insert(1u32, || Ok(123)));
        ignore_guard(cache.get_read_access_or_insert(2u32, || Ok(456)));
        assert_eq!(cache.free_slots.len(), extra_slots);

        cache.remove(1u32).unwrap();
        assert_eq!(cache.free_slots.len(), 1 + extra_slots);

        for slot in cache.free_slots.iter() {
            // The removed slot is reset to the default value.
            assert_eq!(*cache.locks[*slot].read().unwrap(), i32::default());
        }
    }

    #[test]
    fn removing_locked_key_returns_error() {
        let logger = Arc::new(EvictionLogger::default());
        let cache = LockCache::<u32, i32>::new(2, logger.clone());

        let _guard = cache.get_read_access_or_insert(1u32, || Ok(123)).unwrap();
        let res = cache.remove(1u32);
        assert!(matches!(res, Err(Error::IllegalConcurrentOperation(_))));
    }

    #[test]
    fn removing_key_during_concurrent_access_returns_error() {
        let logger = Arc::new(EvictionLogger::default());
        let cache = Arc::new(LockCache::<u32, i32>::new(2, logger.clone()));
        ignore_guard(cache.get_read_access_or_insert(1u32, || Ok(123)));

        // We simulate a concurrent access by increasing the Arc's strong count.
        let _slot = cache.cache.get(&1u32).unwrap();
        let res = cache.remove(1u32);
        assert!(matches!(res, Err(Error::IllegalConcurrentOperation(_))));
    }

    #[test]
    fn removed_items_are_not_considered_evicted() {
        let logger = Arc::new(EvictionLogger::default());
        let cache = LockCache::<u32, i32>::new(2, logger.clone());

        ignore_guard(cache.get_read_access_or_insert(1u32, || Ok(123)));
        assert!(logger.evicted.is_empty());
        cache.remove(1u32).unwrap();
        assert!(logger.evicted.is_empty());
    }

    #[test]
    fn exceeding_maximum_size_returns_corrupted_state_error() {
        let logger = Arc::new(EvictionLogger::default());
        let cache =
            LockCache::<u32, i32>::new_internal(1, NonZero::new(1).unwrap(), logger.clone());

        let _guard = cache.get_read_access_or_insert(1u32, || Ok(123)).unwrap();
        let res = cache.get_read_access_or_insert(2u32, || Ok(456));
        assert!(matches!(res, Err(Error::CorruptedState(_))));
    }

    #[test]
    fn item_lifecycle_is_pinned_checks_strong_count_and_lock_and_hook() {
        struct PinnedHook {}
        impl EvictionHooks for PinnedHook {
            type Key = u32;
            type Value = usize;

            fn is_pinned(&self, _key: &u32, value: &usize) -> bool {
                *value == 42
            }
        }

        let locks: Arc<[_]> = Arc::from(vec![RwLock::new(123), RwLock::new(42)].into_boxed_slice());
        let lifecycle = ItemLifecycle {
            locks,
            free_slots: Arc::new(DashSet::new()),
            hooks: Arc::new(PinnedHook {}),
        };

        // The key does not matter for this test
        let some_key = 33;

        // Item is not pinned as it's not locked and the Arc's strong count is 1
        assert!(!lifecycle.is_pinned(&some_key, &Arc::new(0usize)));

        // Item is pinned as its Arc's strong count is > 1
        let arc = Arc::new(0usize);
        let arc2 = arc.clone();
        assert!(lifecycle.is_pinned(&some_key, &arc2));

        // Item is pinned as another thread holds a lock
        let _guard = lifecycle.locks[0].read().unwrap(); // Lock item at pos 0
        assert!(lifecycle.is_pinned(&some_key, &Arc::new(0usize)));

        // Item is pinned as the hook says so
        assert!(lifecycle.is_pinned(&some_key, &Arc::new(1usize))); // value at index 1 is 42
    }

    #[test]
    fn item_lifecycle_on_evict_invokes_callback_and_resets_slot() {
        let nodes: Arc<[_]> = Arc::from(vec![RwLock::new(123)].into_boxed_slice());
        let free_slots = Arc::new(DashSet::new());
        let logger = Arc::new(EvictionLogger::default());
        let lifecycle = ItemLifecycle {
            locks: nodes,
            free_slots: free_slots.clone(),
            hooks: logger.clone(),
        };
        lifecycle.on_evict(&mut (), 42, Arc::new(0usize));
        assert!(logger.evicted.contains(&(42, 123)));
        assert!(free_slots.contains(&0));
        assert_eq!(*lifecycle.locks[0].read().unwrap(), i32::default());
    }

    #[test]
    #[should_panic(expected = "eviction callback failed")]
    fn item_lifecycle_on_evict_fails_if_callback_fails() {
        struct FailingEvictionCallback;

        impl EvictionHooks for FailingEvictionCallback {
            type Key = u32;
            type Value = i32;

            fn on_evict(&self, _key: u32, _value: i32) -> Result<(), Error> {
                Err(Error::Storage(storage::Error::NotFound))
            }
        }

        let nodes: Arc<[_]> = Arc::from(vec![RwLock::new(123)].into_boxed_slice());
        let free_slots = Arc::new(DashSet::new());
        let logger = Arc::new(FailingEvictionCallback);
        let lifecycle = ItemLifecycle {
            locks: nodes,
            free_slots: free_slots.clone(),
            hooks: logger.clone(),
        };
        lifecycle.on_evict(&mut (), 42, Arc::new(0usize));
    }
}
