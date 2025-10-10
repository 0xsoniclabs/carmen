use std::sync::{Arc, RwLock};

use dashmap::DashSet;
use quick_cache::{
    DefaultHashBuilder, Lifecycle, UnitWeighter,
    sync::{Cache, DefaultLifecycle},
};

use crate::error::Error;

/// TODO: Docblock
pub trait OnEvict<K, V>: Send + Sync {
    fn on_evict(&self, key: K, value: V);
}

/// TODO: Docblock
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
    /// TODO: Docblock
    pub fn new(capacity: usize, on_evict: Arc<dyn OnEvict<K, V>>) -> Self {
        // TODO: Get rid of options builder? There is another ctor
        let options = quick_cache::OptionsBuilder::new()
            .estimated_items_capacity(capacity)
            .weight_capacity(capacity as u64) // unit weight per value
            .build()
            .unwrap();

        // FIXME: Can get rid of faux cache somehow?
        let faux_cache = Cache::<K, usize>::with_options(
            options.clone(),
            UnitWeighter,
            DefaultHashBuilder::default(),
            DefaultLifecycle::default(),
        );

        let true_capacity = faux_cache.num_shards() * faux_cache.shard_capacity() as usize;

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
                evict_callback: on_evict,
            },
        );

        LockCache {
            locks,
            free_slots,
            cache,
        }
    }

    /// TODO: Docblock
    pub fn get_or_insert(
        &self,
        key: K,
        insert_fn: impl FnOnce() -> Result<V, Error>,
    ) -> Result<&RwLock<V>, Error> {
        match self.cache.get_value_or_guard(&key, None) {
            quick_cache::sync::GuardResult::Value(v) => {
                // TODO: Need to check again here after locking if position is still valid
                Ok(&self.locks[v])
            }
            quick_cache::sync::GuardResult::Guard(guard) => {
                // Get value first to avoid unnecessarily allocating a slot in case it fails.
                let value = insert_fn()?;
                let slot = loop {
                    let slot = self.free_slots.iter().next().map(|s| *s);
                    if let Some(slot) = slot
                        && let Some(slot) = self.free_slots.remove(&slot)
                    {
                        break slot;
                    }
                    std::hint::spin_loop();
                };
                let mut lock = self.locks[slot].write().unwrap();
                *lock = value;
                // NOTE: We keep the lock on the slot while inserting the key into the cache,
                //       thereby avoiding the key from immediately being evicted again.
                //       This is important since we always have to return a valid lock.
                guard
                    .insert(slot)
                    .expect("cache entry should not be modified concurrently");
                assert!(self.cache.len() < self.locks.len());
                Ok(&self.locks[slot])
            }
            quick_cache::sync::GuardResult::Timeout => unreachable!(),
        }
    }

    /// TODO: Docblock
    pub fn remove(&self, key: K) {
        if let Some((_, slot)) = self.cache.remove(&key) {
            *self.locks[slot].write().unwrap() = V::default();
            self.free_slots.insert(slot);
        }
    }

    /// TODO: Docblock, test
    pub fn iter(&self) -> impl Iterator<Item = (K, &RwLock<V>)> + '_ {
        self.cache
            .iter()
            .map(|(key, slot)| (key, &self.locks[slot]))
    }
}

struct ItemLifecycle<K, V> {
    locks: Arc<[RwLock<V>]>,
    free_slots: Arc<DashSet<usize>>,
    // FIXME: Naming
    evict_callback: Arc<dyn OnEvict<K, V>>,
}

impl<K, V> Clone for ItemLifecycle<K, V> {
    fn clone(&self) -> Self {
        ItemLifecycle {
            locks: self.locks.clone(),
            free_slots: self.free_slots.clone(),
            evict_callback: self.evict_callback.clone(),
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
        self.evict_callback.on_evict(key, value);
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

    #[test]
    fn items_can_be_inserted_and_deleted() {
        let logger = Arc::new(EvictionLogger::default());
        let cache = LockCache::<u32, i32>::new(10, logger.clone());

        cache.get_or_insert(1u32, || Ok(123)).unwrap();
        cache.get_or_insert(2u32, || Ok(456)).unwrap();
        cache.get_or_insert(3u32, || Ok(789)).unwrap();

        let lock = cache.get_or_insert(1u32, not_found).unwrap();
        assert_eq!(*lock.read().unwrap(), 123);
        let lock = cache.get_or_insert(2u32, not_found).unwrap();
        assert_eq!(*lock.read().unwrap(), 456);
        let lock = cache.get_or_insert(3u32, not_found).unwrap();
        assert_eq!(*lock.read().unwrap(), 789);

        cache.remove(2u32);
        let res = cache.get_or_insert(2u32, not_found);
        assert!(matches!(res, Err(Error::Storage(storage::Error::NotFound))));
    }

    #[test]
    fn exceeding_capacity_causes_eviction() {
        let logger = Arc::new(EvictionLogger::default());
        let cache = LockCache::<u32, i32>::new(2, logger.clone());

        cache.get_or_insert(1u32, || Ok(123)).unwrap();
        assert!(logger.evicted.is_empty());
        cache.get_or_insert(2u32, || Ok(456)).unwrap();
        assert!(logger.evicted.is_empty());

        // By default quick-cache would immediately evict key 3.
        // Since we keep a lock on it during get_or_insert (thereby pinning it), key 1 is
        // evicted instead.
        cache.get_or_insert(3u32, || Ok(789)).unwrap();
        assert_eq!(logger.evicted.len(), 1);
        assert!(logger.evicted.contains(&(1, 123)));

        // Key 3 is now in the cache
        let lock = cache.get_or_insert(3u32, not_found).unwrap();
        assert_eq!(*lock.read().unwrap(), 789);

        // Key 1 is not
        let res = cache.get_or_insert(1u32, not_found);
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

        let _guard1 = cache
            .get_or_insert(1u32, || Ok(123))
            .unwrap()
            .read()
            .unwrap();
        cache.get_or_insert(2u32, || Ok(456)).unwrap();
        assert!(logger.evicted.is_empty());

        // Since we now hold a lock on key 1, key 2 is evicted instead.
        cache.get_or_insert(3u32, || Ok(789)).unwrap();
        assert!(logger.evicted.contains(&(2, 456)));
    }

    #[test]
    fn removing_keys_frees_up_slots() {
        let logger = Arc::new(EvictionLogger::default());
        let cache = LockCache::<u32, i32>::new(2, logger.clone());

        assert_eq!(cache.free_slots.len(), 3); // 2 + 1

        cache.get_or_insert(1u32, || Ok(123)).unwrap();
        cache.get_or_insert(2u32, || Ok(456)).unwrap();
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

        cache.get_or_insert(1u32, || Ok(123)).unwrap();
        assert!(logger.evicted.is_empty());
        cache.remove(1u32);
        assert!(logger.evicted.is_empty());
    }
}
