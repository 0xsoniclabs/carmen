use std::{
    ops::DerefMut,
    sync::{Arc, LockResult, RwLock},
};

use crate::error::Error;
pub mod node_pool_with_storage;

/// A collection of thread-safe *items* that dereference to [`Pool::Type`].
///
/// Items in the pool are uniquely identified by a [`Pool::Id`].
/// Calling [`Pool::get`] with the same ID twice is guaranteed to yield the same item.
/// IDs are managed by the pool itself, which hands out new IDs upon insertion of an item.
/// IDs are not globally unique and may be reused after deletion.
///
/// The concrete type returned by [`Pool::get`] may not be [`Pool::Type`] but instead a wrapper type
/// which dereferences to [`Pool::Type`]. This abstraction allows for the pool to associate metadata
/// with each item, for example to implement smart cache eviction.
pub trait Pool {
    /// The id type used to identify items in the pool.
    type Id;
    /// The type of items indexed by the pool.
    type Item;

    /// Adds the item in the pool and returns an ID for it.
    fn add(&self, item: Self::Item) -> Result<Self::Id, Error>;

    /// Retrieves an item from the pool, if it exists. Returns [`Error::NotFound`] otherwise.
    fn get(
        &self,
        id: Self::Id,
    ) -> Result<PoolItem<impl DerefMut<Target = Self::Item> + Send + Sync + 'static>, Error>;

    /// Deletes an item with the given ID from the pool
    /// The ID may be reused in the future, when creating a new item by calling [`Pool::set`].
    fn delete(&self, id: Self::Id) -> Result<(), Error>;

    /// Flushes all pending operations to the underlying storage layer (if one exists).
    #[allow(dead_code)]
    fn flush(&self) -> Result<(), Error>;
}

/// An item retrieved from the pool which can be locked for reading or writing to enable safe
/// concurrent access.
#[derive(Debug)]
pub struct PoolItem<T>(Arc<RwLock<T>>);

impl<T> PoolItem<T> {
    /// Creates a new [`PoolItem`] by wrapping the given [`Arc<RwLock<T>>`].
    pub fn new(item: Arc<RwLock<T>>) -> Self {
        Self(item)
    }

    /// Acquires a read lock on the item.
    pub fn read(&self) -> LockResult<std::sync::RwLockReadGuard<'_, T>> {
        self.0.read()
    }

    /// Acquires a write lock on the item.
    pub fn write(&self) -> LockResult<std::sync::RwLockWriteGuard<'_, T>> {
        self.0.write()
    }
}

#[cfg(test)]
mod tests {

    use std::{
        collections::HashMap,
        ops::{Deref, DerefMut},
        sync::{Arc, Mutex, RwLock},
        thread,
    };

    use crate::{
        error::Error,
        pool::{Pool, PoolItem},
        storage,
    };

    const TREE_DEPTH: u32 = 6;
    const CHILDREN_PER_NODE: u32 = 3;

    /// A simple in-memory pool of nodes for testing purposes.
    struct FakeNodePool {
        nodes: Mutex<HashMap<u32, Arc<RwLock<TestNode>>>>,
    }

    impl Pool for FakeNodePool {
        type Id = u32;
        type Item = TestNode;

        fn get(
            &self,
            id: Self::Id,
        ) -> Result<
            PoolItem<impl DerefMut<Target = Self::Item> + Send + Sync + 'static>,
            crate::error::Error,
        > {
            self.nodes
                .lock()
                .unwrap()
                .get(&id)
                .cloned()
                .map(super::PoolItem::new)
                .ok_or(Error::Storage(storage::Error::NotFound))
        }

        fn add(&self, value: TestNode) -> Result<Self::Id, crate::error::Error> {
            let node = Arc::new(RwLock::new(value));
            let mut nodes = self.nodes.lock().unwrap();
            let id = nodes.len() as u32 + 1;
            nodes.insert(id, node);
            Ok(id)
        }

        fn delete(&self, id: Self::Id) -> Result<(), crate::error::Error> {
            self.nodes
                .lock()
                .unwrap()
                .remove(&id)
                .map(|_| ())
                .ok_or(Error::Storage(storage::Error::NotFound))
        }

        fn flush(&self) -> Result<(), crate::error::Error> {
            Ok(())
        }
    }

    /// A simple tree node for testing purposes.
    /// It stores a u32 payload a list of children node IDs.
    struct TestNode {
        value: u32,
        children: Vec<u32>,
    }

    impl Deref for TestNode {
        type Target = Self;

        fn deref(&self) -> &Self::Target {
            self
        }
    }

    impl DerefMut for TestNode {
        fn deref_mut(&mut self) -> &mut Self::Target {
            self
        }
    }

    /// A utility wrapper for holding a path with its expected value at the end of the path.
    struct PathWithValue {
        path: Vec<u32>,
        expected: u32,
    }

    /// Recursively generates all possible paths of a given depth in a tree with the expected value
    /// at the end of each path, as it is initialized in [`populate_tree`].
    fn generate_cases_recursive(
        cases: &mut Vec<PathWithValue>,
        current_path: &mut Vec<u32>,
        remaining_depth: u32,
    ) {
        if remaining_depth == 0 {
            let expected = current_path
                .iter()
                .rev()
                .enumerate()
                .map(|(i, &val)| (val + 1) * 10u32.pow(i.try_into().unwrap()))
                .sum();

            // Push the new Case.
            cases.push(PathWithValue {
                path: current_path.clone(),
                expected,
            });
            return;
        }

        for i in 0..CHILDREN_PER_NODE {
            current_path.push(i);
            generate_cases_recursive(cases, current_path, remaining_depth - 1);
            current_path.pop();
        }
    }

    /// Recursively sets up a tree with a [`Pool`] as backing store.
    /// A thread is spawned for each child node to populate its subtree. Each node's value is set to
    /// a unique number based on its position in the tree.
    fn populate_tree(
        cur_node: &mut TestNode,
        depth: u32,
        max_depth: u32,
        root_id: u32,
        pool: &Arc<impl Pool<Id = u32, Item = TestNode> + Send + Sync + 'static>,
    ) {
        if depth == max_depth {
            return;
        }
        let mut handles = vec![];
        for i in 1..=CHILDREN_PER_NODE {
            let child = TestNode {
                value: root_id * 10 + i,
                children: vec![],
            };
            let child_id = pool.add(child).unwrap();
            cur_node.children.push(child_id);
            let child = pool.get(child_id).unwrap();
            handles.push(thread::spawn({
                let pool = pool.clone();
                move || {
                    let mut child = child.write().unwrap();
                    let new_root_id = child.value;
                    populate_tree(&mut child, depth + 1, TREE_DEPTH, new_root_id, &pool);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    fn set_value_at_path(
        cur_node: &TestNode,
        path: &[u32],
        value: u32,
        pool: &Arc<impl Pool<Id = u32, Item = TestNode> + Send + Sync + 'static>,
    ) {
        let child_id = path
            .first()
            .copied()
            .unwrap_or_else(|| panic!("Empty path. You may have recursed too deep."));
        let path = &path[1..];
        let child = pool.get(cur_node.children[child_id as usize]).unwrap();
        if path.is_empty() {
            child.write().unwrap().value = value;
        } else {
            set_value_at_path(&child.read().unwrap(), path, value, pool);
        }
    }

    /// Recursively reads a value from the tree following the given path.
    fn read_from_tree_path(
        cur_node: &TestNode,
        path: &[u32],
        pool: &Arc<impl Pool<Id = u32, Item = TestNode> + Send + Sync + 'static>,
    ) -> u32 {
        let child_id = path
            .first()
            .copied()
            .unwrap_or_else(|| panic!("Empty path. You may have recursed too deep."));
        let path = &path[1..];
        let child = pool.get(cur_node.children[child_id as usize]).unwrap();
        if path.is_empty() {
            return child.read().unwrap().value;
        } else {
            read_from_tree_path(&child.read().unwrap(), path, pool)
        }
    }

    /// Recursively deletes all nodes at the given level in the tree.
    fn delete_tree_level(
        node: &TestNode,
        level: u32,
        id: u32,
        depth: u32,
        pool: &Arc<impl Pool<Id = u32, Item = TestNode> + Send + Sync + 'static>,
    ) {
        if depth == level {
            pool.delete(id).unwrap();
            return;
        }

        let mut handles = vec![];
        for &child_id in &node.children {
            let child = pool.get(child_id).unwrap();
            handles.push(thread::spawn({
                let pool = pool.clone();
                move || {
                    delete_tree_level(&child.read().unwrap(), level, child_id, depth + 1, &pool);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    pub fn pool_allows_concurrent_tree_construction() {
        let pool = Arc::new(FakeNodePool {
            nodes: Mutex::new(HashMap::new()),
        });

        let root = TestNode {
            value: 0,
            children: vec![],
        };
        let root_id = pool.add(root).unwrap();

        let root = pool.get(root_id).unwrap();
        populate_tree(&mut root.write().unwrap(), 3, TREE_DEPTH, 0, &pool);
    }

    #[test]
    pub fn pool_allows_concurrent_tree_lookup() {
        let pool = Arc::new(FakeNodePool {
            nodes: Mutex::new(HashMap::new()),
        });

        // Setting up the tree
        let root = TestNode {
            value: 0,
            children: vec![],
        };
        let root_id = pool.add(root).unwrap();
        let root = pool.get(root_id).unwrap();
        populate_tree(&mut root.write().unwrap(), 0, TREE_DEPTH, 0, &pool.clone());

        let mut cases = vec![];
        generate_cases_recursive(&mut cases, &mut vec![], TREE_DEPTH);

        let mut handles = vec![];
        for case in cases {
            let root = pool.get(root_id).unwrap();
            let pool = pool.clone();
            handles.push(thread::spawn(move || {
                assert_eq!(
                    case.expected,
                    read_from_tree_path(&root.read().unwrap(), &case.path, &pool)
                );
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn pool_allows_concurrent_delete_on_different_branches() {
        let pool = Arc::new(FakeNodePool {
            nodes: Mutex::new(HashMap::new()),
        });

        // Setting up the tree
        let root = TestNode {
            value: 0,
            children: vec![],
        };
        let root_id = pool.add(root).unwrap();
        let root = pool.get(root_id).unwrap();
        populate_tree(&mut root.write().unwrap(), 0, TREE_DEPTH, 0, &pool.clone());

        let num_nodes = pool.nodes.lock().unwrap().len();
        delete_tree_level(&root.read().unwrap(), TREE_DEPTH, root_id, 0, &pool);

        assert_eq!(
            pool.nodes.lock().unwrap().len(),
            num_nodes - (CHILDREN_PER_NODE.pow(TREE_DEPTH) as usize)
        );
    }

    #[test]
    fn pool_allows_concurrent_set_and_get() {
        // The point of this test is to prove that one can model concurrent set and get operations
        // on a tree using the pool without deadlocking or panicking.
        let pool = Arc::new(FakeNodePool {
            nodes: Mutex::new(HashMap::new()),
        });
        // Setting up the tree
        let root = TestNode {
            value: 0,
            children: vec![],
        };
        let root_id = pool.add(root).unwrap();
        let root = pool.get(root_id).unwrap();
        populate_tree(&mut root.write().unwrap(), 0, TREE_DEPTH, 0, &pool.clone());

        let mut cases = vec![];
        generate_cases_recursive(&mut cases, &mut vec![], TREE_DEPTH);
        // IMO this is clearer than filtering the items in the for loop below.
        #[allow(clippy::needless_collect)]
        let paths: Vec<Vec<u32>> = cases.into_iter().map(|c| c.path).collect();

        let mut handles = vec![];
        for (i, path) in paths.into_iter().enumerate() {
            // Spawn set
            handles.push(thread::spawn({
                let pool = pool.clone();
                let root = pool.get(root_id).unwrap();
                let path = path.clone();
                move || {
                    set_value_at_path(&root.read().unwrap(), &path, i as u32, &pool.clone());
                }
            }));
            // Spawn get
            handles.push(thread::spawn({
                let pool = pool.clone();
                let root = pool.get(root_id).unwrap();
                move || {
                    let _ = read_from_tree_path(&root.read().unwrap(), &path, &pool);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }
}
