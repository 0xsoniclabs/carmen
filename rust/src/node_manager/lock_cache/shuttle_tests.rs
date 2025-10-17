use std::panic::catch_unwind;

use itertools::Itertools;

use crate::{
    error::Error,
    node_manager::lock_cache::{
        LockCache,
        test_utils::{EvictionLogger, GetOrInsertMethod, get_method, ignore_guard},
    },
    storage,
    sync::*,
    utils::shuttle::{run_shuttle_check, set_name_for_shuttle_task},
};

#[rstest_reuse::apply(get_method)]
fn shuttle_cached_node_manager_multiple_get_on_same_id_insert_in_cache_only_once(
    #[case] get_fn: GetOrInsertMethod,
) {
    run_shuttle_check(
        move || {
            let insert_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let insert_fn = {
                let insert_count = insert_count.clone();
                Arc::new(move || {
                    insert_count.fetch_add(1, Ordering::SeqCst);
                    Ok(42)
                })
            };
            let id = 0;
            let cache = Arc::new(LockCache::new(10, Arc::new(EvictionLogger::default())));

            let mut handles = vec![];
            for _ in 0..2 {
                let cache = cache.clone();
                let insert_fn = insert_fn.clone();
                handles.push(thread::spawn(move || {
                    ignore_guard(get_fn(&cache, id, insert_fn));
                }));
            }

            for handle in handles {
                handle.join().unwrap();
            }

            assert_eq!(insert_count.load(Ordering::SeqCst), 1);
        },
        10000,
    );
}

/// An operation to perform on the lock cache in shuttle tests.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum Op {
    Add,
    Get,
    Delete,
    Iter,
}

impl Op {
    /// Execute the operation on the given lock cache and node ID, returning a handle to the spawned
    /// thread.
    /// It panics in two cases:
    /// - A reference to a non-existing node is returned
    /// - An unexpected error occurs
    fn execute(self, cache: Arc<LockCache<u32, i32>>, id: u32) -> thread::JoinHandle<()> {
        match self {
            Op::Add => {
                thread::spawn(
                    move || match cache.get_read_access_or_insert(id, || Ok(42)) {
                        Ok(_) => {}
                        // Add on full cache with all elements pinned
                        Err(Error::CorruptedState(s))
                            if s == "lock cache's cache size is equal or bigger than the number of slots. This may have happened because an insert operation was executed with all cache entries marked as pinned" =>
                            {}
                        Err(Error::IllegalConcurrentOperation(s))
                            if s == "another thread removed the key while it was being inserted" =>
                        {
                            panic!("Expected error on add: {s}");
                        }
                        Err(e) => panic!("Unexpected error on add: {e:?}"),
                    },
                )
            }
            Op::Get => thread::spawn(move || {
                let guard = cache.get_read_access_or_insert(id, || {
                    Err(Error::Storage(storage::Error::NotFound))
                });
                if let Ok(guard) = guard {
                    assert!(*guard != i32::default());
                }
            }),
            Op::Delete => thread::spawn(move || match cache.remove(id) {
                Ok(_) => {}
                Err(e) => panic!("Expected error: {e:?}"),
            }),
            Op::Iter => thread::spawn(move || {
                for (_, guard) in cache.iter_write() {
                    assert!(*guard != i32::default());
                }
            }),
        }
    }
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Add => write!(f, "Add"),
            Op::Get => write!(f, "Get"),
            Op::Delete => write!(f, "Delete"),
            Op::Iter => write!(f, "Iter"),
        }
    }
}

/// Tests all permutations of operations and node IDs on the lock cache using shuttle.
/// The test is repeated for different cache sizes.
/// The idea is to find cases where the lock cache guarantees are violated, such as:
/// - An operation returns a reference to a non-existing node.
#[test]
fn shuttle_operation_permutations() {
    let CONCURRENT_OPS: usize = 6; // Must be >= 2
    let cache_sizes = (2..=CONCURRENT_OPS + 1).collect::<Vec<usize>>();
    let node_ids = (0..CONCURRENT_OPS as u32).collect::<Vec<u32>>();
    let operations = vec![Op::Add, Op::Get, Op::Delete, Op::Iter];
    for cache_size in cache_sizes {
        for operation_case in operations
            .clone()
            .into_iter()
            .cartesian_product(node_ids.clone().into_iter())
            .combinations_with_replacement(CONCURRENT_OPS)
        {
            println!("Testing case: {operation_case:?} with cache size {cache_size}");
            if let Err(e) = catch_unwind(|| {
                run_shuttle_check(
                    {
                        let operation_case = operation_case.clone();
                        move || {
                            let lock_cache = Arc::new(LockCache::new_with_extra_slots(
                                cache_size,
                                0,
                                Arc::new(EvictionLogger::default()),
                            ));
                            assert_eq!(
                                lock_cache.cache.capacity() as usize,
                                cache_size,
                                "Cache capacity should match the requested size plus extra slots"
                            );

                            let mut handles = vec![];
                            for (op, id) in &operation_case {
                                set_name_for_shuttle_task(format!("{op}({id})"));
                                handles.push(op.execute(lock_cache.clone(), *id));
                            }

                            for handle in handles {
                                handle.join().unwrap();
                            }
                        }
                    },
                    1000,
                );
            }) {
                eprintln!("\n#############################################################");
                eprintln!("Test case failed: {operation_case:?} with cache size {cache_size}");
                eprintln!("--------------------------------------------------------------");
                if let Some(message) = e.downcast_ref::<String>()
                    && message.contains("Expected error")
                {
                    eprintln!("Ignoring expected error: {message}");
                    eprintln!("###########################################################\n");
                } else {
                    panic!("Panic in shuttle test: {e:?}");
                }
            }
        }
    }
}
