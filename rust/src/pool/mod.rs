use std::{
    ops::{Deref, DerefMut},
    sync::{Arc, LockResult, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use crate::error::Error;
pub mod cached_node_manager;
// pub mod node_pool_with_storage;

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
// pub trait Pool {
//     /// The id type used to identify items in the pool.
//     type Id;
//     /// The type of items indexed by the pool.
//     type Item;

//     /// Adds the item in the pool and returns an ID for it.
//     fn add(&self, item: Self::Item) -> Result<Self::Id, Error>;

//     /// Retrieves an item from the pool, if it exists. Returns [`Error::NotFound`] otherwise.
//     fn get(
//         &self,
//         id: Self::Id,
//     ) -> Result<PoolItem<impl DerefMut<Target = Self::Item> + Send + Sync + 'static>, Error>;

//     /// Deletes an item with the given ID from the pool
//     /// The ID may be reused in the future, when creating a new item by calling [`Pool::set`].
//     fn delete(&self, id: Self::Id) -> Result<(), Error>;

//     /// Flushes all pending operations to the underlying storage layer (if one exists).
//     fn flush(&self) -> Result<(), Error>;
// }

/// A collection of thread-safe *nodes* that dereference to [`NodeManager::NodeType`].
///
/// Nodes are uniquely identified by a [`NodeManager::Id`].
/// Nodes ownership is held by the [`NodeManager`] implementation and can be accessed through read
/// or write locks with the [`NodeManager::get_read_access`] and [`NodeManager::get_write_access`]
/// methods.
/// Calling a `get_*` method with the same ID twice is guaranteed to yield the same item.
/// IDs are managed by the pool itself, which hands out new IDs upon insertion of an item.
/// IDs are not globally unique and may be reused after deletion.
///
/// The concrete type returned by the [`NodeManager`] may not be [`NodeManager::NodeType`] but
/// instead a wrapper type which dereferences to [`NodeManager::NodeType`]. This abstraction allows
/// for the pool to associate metadata with each item, for example to implement smart cache
/// eviction.
pub trait NodeManager {
    /// The id type used to identify items in the pool.
    type Id;
    /// The node type indexed by the pool, which is specialized depending on the trie
    /// implementation.
    type NodeType;

    /// Adds the item in the node manager and returns an ID for it.
    fn add(&self, item: Self::NodeType) -> Result<Self::Id, Error>;

    /// Retrieves and lock an item from the node manager with read access, if it exists. Returns
    /// [`crate::storage::Error::NotFound`] otherwise.
    fn get_read_access(
        &self,
        id: Self::Id,
    ) -> Result<RwLockReadGuard<'_, impl Deref<Target = Self::NodeType>>, Error>;

    /// Retrieves and lock an item from the node manager with write access, if it exists. Returns
    /// [`crate::storage::Error::NotFound`] otherwise.
    fn get_write_access(
        &self,
        id: Self::Id,
    ) -> Result<RwLockWriteGuard<'_, impl DerefMut<Target = Self::NodeType>>, Error>;

    /// Deletes an item with the given ID from the pool
    /// The ID may be reused in the future, when creating a new item by calling
    /// [`NodeManager::add`].
    fn delete(&self, id: Self::Id) -> Result<(), Error>;

    /// Flushes all pending operations to the underlying storage layer (if one exists).
    fn flush(&self) -> Result<(), Error>;
}

// /// An item retrieved from the pool which can be locked for reading or writing to enable safe
// /// concurrent access.
// #[derive(Debug)]
// pub struct PoolItem<T>(Arc<RwLock<T>>);

// impl<T> PoolItem<T> {
//     /// Creates a new [`PoolItem`] by wrapping the given [`Arc<RwLock<T>>`].
//     pub fn new(item: Arc<RwLock<T>>) -> Self {
//         Self(item)
//     }

//     /// Acquires a read lock on the item.
//     pub fn read(&self) -> LockResult<std::sync::RwLockReadGuard<'_, T>> {
//         self.0.read()
//     }

//     /// Acquires a write lock on the item.
//     pub fn write(&self) -> LockResult<std::sync::RwLockWriteGuard<'_, T>> {
//         self.0.write()
//     }
// }

#[cfg(test)]
mod tests {

    use std::{
        ops::{Deref, DerefMut},
        sync::{
            Arc, Mutex, RwLock,
            atomic::{AtomicUsize, Ordering},
        },
        thread, vec,
    };

    use crate::{error::Error, pool::NodeManager, storage};

    const FAKE_POOL_SIZE: usize = 1_000_000;
    const TREE_DEPTH: u32 = 6;
    const CHILDREN_PER_NODE: u32 = 3;

    /// A simple in-memory pool of nodes for testing purposes.
    struct FakeNodePool {
        nodes: Vec<RwLock<TestNode>>,   // Fixed-size pool of nodes
        next_pos: AtomicUsize,          // Next position to insert a node
        deleted_nodes: Mutex<Vec<u32>>, // To keep track of deleted node IDs
    }

    impl FakeNodePool {
        pub fn new() -> Self {
            let mut nodes = vec![];
            for _ in 0..FAKE_POOL_SIZE {
                nodes.push(RwLock::new(TestNode {
                    value: 0,
                    children: vec![],
                }));
            }
            Self {
                nodes,
                next_pos: AtomicUsize::new(0),
                deleted_nodes: Mutex::new(vec![]),
            }
        }
    }

    impl NodeManager for FakeNodePool {
        type Id = u32;

        type NodeType = TestNode;

        fn add(&self, item: Self::NodeType) -> Result<Self::Id, Error> {
            let id = self.next_pos.fetch_add(1, Ordering::Relaxed);
            if id >= FAKE_POOL_SIZE {
                return Err(Error::Storage(storage::Error::NotFound));
            }
            *self.nodes[id].write().unwrap() = item;
            Ok(id as u32)
        }

        fn get_read_access(
            &self,
            id: Self::Id,
        ) -> Result<std::sync::RwLockReadGuard<'_, impl Deref<Target = Self::NodeType>>, Error>
        {
            if (id as usize) >= FAKE_POOL_SIZE {
                return Err(Error::Storage(storage::Error::NotFound));
            }
            Ok(self.nodes[id as usize].read().unwrap())
        }

        fn get_write_access(
            &self,
            id: Self::Id,
        ) -> Result<std::sync::RwLockWriteGuard<'_, impl DerefMut<Target = Self::NodeType>>, Error>
        {
            if (id as usize) >= FAKE_POOL_SIZE {
                return Err(Error::Storage(storage::Error::NotFound));
            }
            Ok(self.nodes[id as usize].write().unwrap())
        }

        /// Deletes an item with the given ID from the pool
        /// The ID will NOT be reused in the future.
        fn delete(&self, id: Self::Id) -> Result<(), Error> {
            if (id as usize) >= FAKE_POOL_SIZE {
                return Err(Error::Storage(storage::Error::NotFound));
            }
            let mut deleted = self.deleted_nodes.lock().unwrap();
            if deleted.contains(&id) {
                return Err(Error::Storage(storage::Error::NotFound));
            }
            deleted.push(id);
            Ok(())
        }

        /// No-op for the fake pool.
        fn flush(&self) -> Result<(), Error> {
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
        pool: &Arc<impl NodeManager<Id = u32, NodeType = TestNode> + Send + Sync + 'static>,
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
            handles.push(thread::spawn({
                let pool = pool.clone();
                move || {
                    let mut child = pool.get_write_access(child_id).unwrap();
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
        cur_node: &mut TestNode,
        path: &[u32],
        value: u32,
        pool: &Arc<impl NodeManager<Id = u32, NodeType = TestNode> + Send + Sync + 'static>,
    ) {
        let child_id = path
            .first()
            .copied()
            .unwrap_or_else(|| panic!("Empty path. You may have recursed too deep."));
        let path = &path[1..];
        let mut child = pool
            .get_write_access(cur_node.children[child_id as usize])
            .unwrap();
        if path.is_empty() {
            child.value = value;
        } else {
            set_value_at_path(&mut child, path, value, pool);
        }
    }

    /// Recursively reads a value from the tree following the given path.
    fn read_from_tree_path(
        cur_node: &TestNode,
        path: &[u32],
        pool: &Arc<impl NodeManager<Id = u32, NodeType = TestNode> + Send + Sync + 'static>,
    ) -> u32 {
        let child_id = path
            .first()
            .copied()
            .unwrap_or_else(|| panic!("Empty path. You may have recursed too deep."));
        let path = &path[1..];
        let child = pool
            .get_read_access(cur_node.children[child_id as usize])
            .unwrap();
        if path.is_empty() {
            return child.value;
        } else {
            read_from_tree_path(&child, path, pool)
        }
    }

    /// Recursively deletes all nodes at the given level in the tree.
    fn delete_tree_level(
        node: &TestNode,
        level: u32,
        id: u32,
        depth: u32,
        pool: &Arc<impl NodeManager<Id = u32, NodeType = TestNode> + Send + Sync + 'static>,
    ) {
        if depth == level {
            pool.delete(id).unwrap();
            return;
        }

        let mut handles = vec![];
        for &child_id in &node.children {
            handles.push(thread::spawn({
                let pool = pool.clone();
                move || {
                    let child = pool.get_read_access(child_id).unwrap();
                    delete_tree_level(&child, level, child_id, depth + 1, &pool);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    pub fn pool_allows_concurrent_tree_construction() {
        let pool = Arc::new(FakeNodePool::new());

        let root = TestNode {
            value: 0,
            children: vec![],
        };
        let root_id = pool.add(root).unwrap();

        let mut root = pool.get_write_access(root_id).unwrap();
        populate_tree(&mut root, 3, TREE_DEPTH, 0, &pool);
    }

    #[test]
    pub fn pool_allows_concurrent_tree_lookup() {
        let pool = Arc::new(FakeNodePool::new());

        // Setting up the tree
        let root = TestNode {
            value: 0,
            children: vec![],
        };
        let root_id = pool.add(root).unwrap();
        {
            let mut root = pool.get_write_access(root_id).unwrap();
            populate_tree(&mut root, 0, TREE_DEPTH, 0, &pool.clone());
        }

        let mut cases = vec![];
        generate_cases_recursive(&mut cases, &mut vec![], TREE_DEPTH);
        let mut handles = vec![];
        for case in cases {
            let pool = pool.clone();
            handles.push(thread::spawn(move || {
                let root = pool.get_read_access(root_id).unwrap();
                assert_eq!(case.expected, read_from_tree_path(&root, &case.path, &pool));
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn pool_allows_concurrent_delete_on_different_branches() {
        let pool = Arc::new(FakeNodePool::new());

        // Setting up the tree
        let root = TestNode {
            value: 0,
            children: vec![],
        };
        let root_id = pool.add(root).unwrap();
        {
            let mut root = pool.get_write_access(root_id).unwrap();
            populate_tree(&mut root, 0, TREE_DEPTH, 0, &pool.clone());
        }

        delete_tree_level(
            &pool.get_read_access(root_id).unwrap(),
            TREE_DEPTH,
            root_id,
            0,
            &pool,
        );
        assert_eq!(
            pool.deleted_nodes.lock().unwrap().len() as u32,
            CHILDREN_PER_NODE.pow(TREE_DEPTH)
        );
    }

    #[test]
    fn pool_allows_concurrent_set_and_get() {
        // The point of this test is to prove that one can model concurrent set and get operations
        // on a tree using the pool without deadlocking or panicking.
        let pool = Arc::new(FakeNodePool::new());

        // Setting up the tree
        let root = TestNode {
            value: 0,
            children: vec![],
        };
        let root_id = pool.add(root).unwrap();
        {
            let mut root = pool.get_write_access(root_id).unwrap();
            populate_tree(&mut root, 0, TREE_DEPTH, 0, &pool.clone());
        }

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
                let path = path.clone();
                move || {
                    let mut root = pool.get_write_access(root_id).unwrap();
                    set_value_at_path(&mut root, &path, i as u32, &pool.clone());
                }
            }));
            // Spawn get
            handles.push(thread::spawn({
                let pool = pool.clone();
                move || {
                    let root = pool.get_read_access(root_id).unwrap();
                    let _ = read_from_tree_path(&root, &path, &pool);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }
}
