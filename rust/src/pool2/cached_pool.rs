
use std::{
    hash::RandomState,
    vec::Vec,
    collections::VecDeque,
    sync::{
        RwLock,
        Arc,
        Mutex,
        atomic::AtomicUsize,
    },
};
use std::cmp::Eq;
use std::hash::Hash;

use quick_cache::{Lifecycle, UnitWeighter};

use crate::{
    pool2::Pool,
    storage::{
        Storage,
        Error,
    }
};

pub struct CachedPool<K,V,S:Storage<Id=K,Item=V>> {
    elements: Vec<RwLock<V>>,                 // the owner of all values
    cache: quick_cache::sync::Cache<          // cache, managing the key to element position mapping as well as the element eviction
        K,                                    // key type to identify cached elements
        usize,                                // value type to identify element positions in the elements vector
        UnitWeighter,                         // all elements are considered to cost the same
        RandomState,                          // default hasher
        ElementLifecycle,                     // tracks and reports evicted elements
    >,
    free_list: Mutex<VecDeque<usize>>,        // free list of available element positions
    next_empty: AtomicUsize,                  // next empty position in elements vector
    storage: Arc<S>,                          // storage for managing IDs, fetching missing elements, and saving evicted elements to
}

impl<K: Eq+Hash+Copy,V: Default,S> CachedPool<K,V,S> where S:Storage<Id=K,Item=V> {
    pub fn new(capacity: usize, storage: Arc<S>) -> Self {
        let mut elements= Vec::with_capacity(capacity);
        for _ in 0..capacity {
            // Pre-allocate with default values. This requires V: Default.
            elements.push(RwLock::new(V::default()));
        }

        let options = quick_cache::OptionsBuilder::new()
            .estimated_items_capacity(capacity)
            .weight_capacity(capacity as u64)
            .build().unwrap(); // TODO: cleanup

        CachedPool {
            elements: elements,
            storage: storage.clone(),
            //cache: quick_cache::sync::Cache::new(capacity),
            cache: quick_cache::sync::Cache::with_options(
                options,
                UnitWeighter,
                RandomState::default(),
                ElementLifecycle {},
            ),
            free_list: Mutex::new(VecDeque::new()),
            next_empty: AtomicUsize::new(0),
        }
    }

    fn evict(&self, entries : Vec<(K,usize)>) {
        // TODO: handle this in extra thread and deal with issues
        for (key, pos) in entries {
            self.storage.set(key, &self.elements[pos].write().unwrap()).unwrap();
            self.free_list.lock().unwrap().push_back(pos);
        }
    }

    fn insert_into_free_slot(&self, key: K, item: V) -> Result<usize, Error> {
        let mut free_list = self.free_list.lock().map_err(|_| Error::NotFound)?;
        let mut pos = free_list.pop_front().unwrap_or_else(|| {
            self.next_empty.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        });
        if pos >= self.elements.len() {
            self.next_empty.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

            // The cache is full, we need to evict an element to make space.
            let dummy_pos = 0; // position is irrelevant here
            let evicted = self.cache.insert_with_lifecycle(key, dummy_pos);
            self.evict(evicted);

            // Now, there should be an element in the free list. If not, the
            // pool eviction failed (e.g. since all elements are pinned) and
            // the insertion cannot proceed.
            pos = free_list.pop_front().ok_or(Error::NotFound)?;
        }
        let mut guard = self.elements[pos].write().map_err(|_| Error::NotFound)?;
                *guard = item;

        // Include new element in cache, evict old elements if necessary.
        let evicted = self.cache.insert_with_lifecycle(key, pos);
        self.evict(evicted);
        Ok(pos)
    }

}

impl<K: Eq+Hash+Copy,V:Default,S> Pool for CachedPool<K,V,S> where S:Storage<Id=K,Item=V> {
    type Id = K;
    type Item = V;

    fn add(&self, item: Self::Item) -> Result<Self::Id, Error> {
        let id = self.storage.reserve(&item);
        self.insert_into_free_slot(id, item)?;
        Ok(id)
    }

    fn get_read_access(
        &self,
        id: Self::Id,
    ) -> Result<std::sync::RwLockReadGuard<'_, Self::Item>, Error> {
        match self.cache.get(&id) {
            Some(pos) => return self.elements[pos].read().map_err(|_| Error::NotFound),
            None => {
                let item =self.storage.get(id)?; // check if element exists in storage
                let pos = self.insert_into_free_slot(id, item)?;
                return self.elements[pos].read().map_err(|_| Error::NotFound)
            },
        }
    }

    fn get_write_access(
        &self,
        id: Self::Id,
    ) -> Result<std::sync::RwLockWriteGuard<'_, Self::Item>, Error> {
        match self.cache.get(&id) {
            Some(pos) => return self.elements[pos].write().map_err(|_| Error::NotFound),
            None => {
                let item =self.storage.get(id)?; // check if element exists in storage
                let pos = self.insert_into_free_slot(id, item)?;
                return self.elements[pos].write().map_err(|_| Error::NotFound);
            },
        }
    }

    fn delete(&self, id: Self::Id) -> Result<(), Error> {
        if let Some(pos) = self.cache.get(&id) {
            // get exclusive write access before dropping the element
            let _guard = self.elements[pos].write().map_err(|_| Error::NotFound)?;
            self.cache.remove(&id);
            let mut free_list = self.free_list.lock().map_err(|_| Error::NotFound)?;
            free_list.push_back(pos);
            return Ok(())
        }
        Err(Error::NotFound)
    }
}


pub struct ElementLifecycle {}

impl<K: Eq+Hash+Copy> Lifecycle<K,usize> for ElementLifecycle {

    type RequestState = Vec<(K,usize)>;

    fn is_pinned(&self, _key: &K, _value: &usize) -> bool {
        false // TODO: could check whether a write-lock can be obtained
    }

    fn begin_request(&self) -> Self::RequestState {
        Vec::new()
    }

    fn on_evict(&self, state: &mut Self::RequestState, _key: K,_value: usize) {
        state.push((_key, _value));
    }
}

impl Clone for ElementLifecycle {
    fn clone(&self) -> Self {
        ElementLifecycle {}
    }
}


mod tests {
    use super::*;

    use std::sync::RwLock;
    use crate::storage::Error;

    // I did not manage to get the mock system to work, so I had to use a fake.
    struct FakeStorage{
        data: RwLock<std::collections::HashMap<i32, i32>>,
    }


    impl FakeStorage {
        pub fn new() -> Self {
            FakeStorage {
                data: RwLock::new(std::collections::HashMap::new()),
            }
        }
    }

    impl Storage for FakeStorage {
        type Id = i32;
        type Item = i32;

        fn open(_path: &std::path::Path) -> Result<Self, crate::storage::Error> where Self: Sized {
            Err(crate::storage::Error::NotFound)
        }

        fn get(&self, id: Self::Id) -> Result<Self::Item, crate::storage::Error> {
            match self.data.read() {
                Ok(data) => data.get(&id).ok_or(Error::NotFound).copied(),
                Err(_) => Err(Error::NotFound),
            }
        }

        fn reserve(&self, item: &Self::Item) -> Self::Id {
            match self.data.write() {
                Ok(mut data) => {
                    let id = data.len() as i32;
                    data.insert(id, *item);
                    id
                },
                Err(_) => 0,
            }
        }

        fn set(&self, _id: Self::Id, _item: &Self::Item) -> Result<(), crate::storage::Error> {
            match self.data.write() {
                Ok(mut data) => {
                    data.insert(_id, *_item);
                    Ok(())
                },
                Err(_) => Err(crate::storage::Error::NotFound),
            }
        }

        fn delete(&self, _id: Self::Id) -> Result<(), crate::storage::Error> {
            match self.data.write() {
                Ok(mut data) => {
                    data.remove(&_id);
                    Ok(())
                },
                Err(_) => Err(crate::storage::Error::NotFound),
            }
        }

        fn flush(&self) -> Result<(), crate::storage::Error> {
            Ok(())
        }
    }


    #[test]
    fn test_cached_pool_add_and_retrieve() {
        let storage = FakeStorage::new();
        let pool  = CachedPool::new(200, Arc::new(storage));
        let id1 = pool.add(42).unwrap();
        assert_eq!(id1, 0);
        let id2 = pool.add(24).unwrap();
        assert_eq!(id2, 1);
        assert_eq!(*pool.get_read_access(id1).unwrap(), 42);
        assert_eq!(*pool.get_read_access(id2).unwrap(), 24);
    }


    #[test]
    fn test_cached_pool_elements_can_be_modified() {
        let storage = FakeStorage::new();
        let pool  = CachedPool::new(200, Arc::new(storage));
        let id = pool.add(42).unwrap();
        assert_eq!(*pool.get_read_access(id).unwrap(), 42);

        *pool.get_write_access(id).unwrap() = 24;
        assert_eq!(*pool.get_read_access(id).unwrap(), 24);
    }

    #[test]
    fn test_cached_pool_elements_can_hold_write_access_to_multiple_elements() {
        let storage = FakeStorage::new();
        let pool  = CachedPool::new(200, Arc::new(storage));
        let id1 = pool.add(1).unwrap();
        let id2 = pool.add(2).unwrap();
        assert_ne!(id1, id2);

        // Demonstrating that write access can be obtained for multiple elements
        let _guard1 = pool.get_write_access(id1).unwrap();
        let _guard2 = pool.get_write_access(id2).unwrap();
    }


    #[test]
    fn test_cached_pool_elements_can_be_deleted() {
        let storage = FakeStorage::new();
        let pool  = CachedPool::new(200, Arc::new(storage));
        let id = pool.add(42).unwrap();
        assert_eq!(*pool.get_read_access(id).unwrap(), 42);

        assert!(pool.delete(id).is_ok());
        //assert!(pool.get_read_access(id).is_err());
    }


}

