use dashmap::DashSet;
use quick_cache::{
    DefaultHashBuilder, Lifecycle, UnitWeighter,
    sync::{Cache, DefaultLifecycle},
};

use crate::{error::Error, sync::*};

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
    K: Copy + Eq + std::hash::Hash + std::fmt::Debug,
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
                    // NOTE: We keep the lock on the slot while inserting the key into the cache,
                    //       thereby avoiding the key from immediately being evicted again.
                    //       This is important since we always have to return a valid lock.
                    cache_guard
                        .insert(slot)
                        .expect("cache entry should not be modified concurrently");
                    assert!(self.cache.len() < self.locks.len());
                    drop(slot_guard);

                    // TODO: Confirm that while holding a guard inserting another key can trigger
                    // this guard's key to be evicted. If yes, we need the
                    // following re-check loop:

                    let slot_guard = access_fn(&self.locks[slot]);
                    if let Some(current_slot) = self.cache.peek(&key)
                        && current_slot == slot
                    {
                        return Ok(slot_guard);
                    }
                    continue;
                }
                quick_cache::sync::GuardResult::Timeout => unreachable!(),
            }
        }
    }

    /// TODO: Docblock
    pub fn get_read_access_or_insert(
        &self,
        key: K,
        insert_fn: impl Fn() -> Result<V, Error>,
    ) -> Result<RwLockReadGuard<'_, V>, Error> {
        self.get_access_or_insert(key, insert_fn, |lock| lock.read().unwrap())
    }

    /// TODO: Docblock
    pub fn get_write_access_or_insert(
        &self,
        key: K,
        insert_fn: impl Fn() -> Result<V, Error>,
    ) -> Result<RwLockWriteGuard<'_, V>, Error> {
        self.get_access_or_insert(key, insert_fn, |lock| lock.write().unwrap())
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

    /// Helper function for performing a get/insert where we don't care about the returned guard.
    fn get_read_access_or_insert_no_guard(
        cache: &LockCache<u32, i32>,
        key: u32,
        insert_fn: impl Fn() -> Result<i32, Error>,
    ) {
        let _guard = cache.get_read_access_or_insert(key, insert_fn);
    }

    #[track_caller]
    fn run_shuttle_check(test: impl Fn() + Send + Sync + 'static, _num_iter: usize) {
        #[cfg(feature = "shuttle")]
        {
            use std::{env, os};

            match env::var("SHUTTLE_REPLAY") {
                Ok(schedule) => {
                    let path = if !schedule.is_empty() {
                        schedule
                    } else {
                        // find all the files in the current directory that start with "schedule"
                        let mut schedules = vec![];
                        for entry in std::fs::read_dir(".").unwrap() {
                            let entry = entry.unwrap();
                            let path = entry.path();
                            if path.is_file() {
                                if let Some(name) = path.file_name() {
                                    if name.to_string_lossy().starts_with("schedule") {
                                        schedules.push(path.to_string_lossy().to_string());
                                    }
                                }
                            }
                        }
                        schedules.sort();
                        schedules.last().cloned().expect("No schedule files found")
                    };
                    shuttle::replay_from_file(test, &path);
                }
                Err(_) => {
                    let mut shuttle_config = shuttle::Config::new();
                    shuttle_config.failure_persistence = shuttle::FailurePersistence::File(None);
                    let runner = shuttle::Runner::new(
                        shuttle::scheduler::RandomScheduler::new(_num_iter),
                        shuttle_config,
                    );
                    runner.run(test);
                }
            }
        }
        #[cfg(not(feature = "shuttle"))]
        test();
    }

    #[test]
    fn shuttle__cached_node_manager_never_returns_a_reference_to_a_non_existing_node_on_evict() {
        const VALUE: i32 = i32::MAX;
        run_shuttle_check(
            move || {
                let num_elements: u64 = 3;
                // Evict the element previous to the last one
                let logger = Arc::new(EvictionLogger::default());
                let insert_fn = move || Ok(VALUE);

                let lock_cache = Arc::new(LockCache::<u32, i32>::new(
                    num_elements as usize,
                    logger.clone(),
                ));
                // TODO: Is it requried? can we get rid of it?
                // assert_eq!(lock_cache.cache.num_shards(), 1);

                // now we spawn two threads: one that adds a new node to a full cache, which
                // will cause eviction of an existing node, and one that
                // tries to get a reference to the node being evicted. The
                // condition we want to test is that the get thread never
                // returns a reference to an evicted (aka. empty) node.
                for i in 0..num_elements {
                    get_read_access_or_insert_no_guard(&lock_cache, i as u32, insert_fn);
                }
                assert_eq!(lock_cache.cache.len(), num_elements as usize);

                let add_thread = thread::spawn({
                    let lock_cache = lock_cache.clone();
                    move || {
                        // Replace the entire cache with new elements
                        for i in 0..num_elements {
                            get_read_access_or_insert_no_guard(
                                &lock_cache,
                                (i + num_elements) as u32,
                                insert_fn,
                            );
                            // Make the hotter element
                            get_read_access_or_insert_no_guard(
                                &lock_cache,
                                (i + num_elements) as u32,
                                insert_fn,
                            );
                        }
                    }
                });

                let get_thread = thread::spawn({
                    let lock_cache = lock_cache.clone();
                    move || {
                        for id_to_get in 0..num_elements {
                            let res =
                                lock_cache.get_read_access_or_insert(id_to_get as u32, insert_fn);
                            if let Ok(guard) = res {
                                assert!(*guard != i32::default());
                            }
                        }
                    }
                });

                add_thread.join().unwrap();
                get_thread.join().unwrap();
            },
            10000,
        );
    }
    #[test]
    fn items_can_be_inserted_and_removed() {
        let logger = Arc::new(EvictionLogger::default());
        let cache = LockCache::<u32, i32>::new(10, logger.clone());

        get_read_access_or_insert_no_guard(&cache, 1u32, || Ok(123));
        get_read_access_or_insert_no_guard(&cache, 2u32, || Ok(456));
        get_read_access_or_insert_no_guard(&cache, 3u32, || Ok(789));

        {
            let guard = cache.get_read_access_or_insert(1u32, not_found).unwrap();
            assert_eq!(*guard, 123);
            let guard = cache.get_read_access_or_insert(2u32, not_found).unwrap();
            assert_eq!(*guard, 456);
            let guard = cache.get_read_access_or_insert(3u32, not_found).unwrap();
            assert_eq!(*guard, 789);
        }

        cache.remove(2u32);
        let res = cache.get_read_access_or_insert(2u32, not_found);
        assert!(matches!(res, Err(Error::Storage(storage::Error::NotFound))));
    }

    #[test]
    fn exceeding_capacity_causes_eviction() {
        let logger = Arc::new(EvictionLogger::default());
        let cache = LockCache::<u32, i32>::new(2, logger.clone());

        get_read_access_or_insert_no_guard(&cache, 1u32, || Ok(123));
        get_read_access_or_insert_no_guard(&cache, 2u32, || Ok(456));
        assert!(logger.evicted.is_empty());

        // By default quick-cache would immediately evict key 3.
        // Since we keep a lock on it during get_read_access_or_insert (thereby pinning it), key 1
        // is evicted instead.
        get_read_access_or_insert_no_guard(&cache, 3u32, || Ok(789));
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

        get_read_access_or_insert_no_guard(&cache, 1u32, || Ok(123));
        get_read_access_or_insert_no_guard(&cache, 2u32, || Ok(456));
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

        get_read_access_or_insert_no_guard(&cache, 1u32, || Ok(123));
        assert!(logger.evicted.is_empty());
        cache.remove(1u32);
        assert!(logger.evicted.is_empty());
    }
}
