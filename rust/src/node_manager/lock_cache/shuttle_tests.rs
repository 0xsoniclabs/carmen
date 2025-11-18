// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use core::panic;
use std::{env, iter, num::NonZero, panic::catch_unwind};

use itertools::Itertools;

use crate::{
    error::{BTResult, Error},
    node_manager::lock_cache::{
        LockCache,
        shuttle_test_utils::{Op, OpPanicStatus, OpWithId, PermutationTestCase},
        test_utils::{EvictionLogger, GetOrInsertMethod, get_method, ignore_guard},
    },
    sync::{atomic::Ordering, *},
    utils::shuttle::{run_shuttle_check, set_name_for_shuttle_task},
};

#[rstest_reuse::apply(get_method)]
fn shuttletest_cached_node_manager_multiple_get_on_same_id_insert_in_cache_only_once(
    #[case] get_fn: GetOrInsertMethod<dyn Fn() -> BTResult<i32, Error>>,
) {
    run_shuttle_check(
        move || {
            const ID: u32 = 0;
            let insert_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let insert_fn = {
                let insert_count = insert_count.clone();
                move || {
                    insert_count.fetch_add(1, Ordering::SeqCst);
                    Ok(42)
                }
            };
            let cache = LockCache::new(10, Arc::new(EvictionLogger::default()));

            thread::scope(|s| {
                for _ in 0..2 {
                    s.spawn(|| {
                        ignore_guard(get_fn(&cache, ID, &insert_fn));
                    });
                }
            });

            assert_eq!(insert_count.load(Ordering::SeqCst), 1);
        },
        100,
    );
}

/// Tests all permutations of operations and a small set of node IDs on the lock cache using
/// shuttle. The test is repeated for different cache sizes to stress cache contention scenarios.
/// The idea is to find cases where the lock cache guarantees are violated, i.e. an operation
/// returns a reference to a non-existing node or an unexpected error occurs.
///
/// Not all permutations result in valid sequences of operations. These are expected to produce
/// errors but do not cause the test to fail.
/// Due to a current limitation of shuttle, they are still reported as errors however: https://github.com/awslabs/shuttle/issues/221
///
/// To facilitate debugging, each test case is serialized to a file before being executed, and
/// can be replayed by setting the `CARMEN_SHUTTLE_SERIALIZED_CASE` environment variable.
#[test]
fn shuttletest_operation_permutations() {
    const SHUTTLE_PERMUTATION_ITERATIONS: usize = 100;
    if cfg!(not(feature = "shuttle")) {
        return;
    }
    let current_dir = env::current_dir().unwrap();
    let CONCURRENT_OPS: usize = 4; // Must be >= 2
    let cache_sizes = 2..=CONCURRENT_OPS + 1;
    let node_ids = (0..CONCURRENT_OPS as u32).collect::<Vec<u32>>();
    let operations = [Op::Add, Op::Get, Op::Delete, Op::Iter];

    let case_yielder: Box<dyn Iterator<Item = PermutationTestCase>> =
        match std::env::var("CARMEN_SHUTTLE_SERIALIZED_CASE") {
            Ok(_) => Box::new(iter::once(PermutationTestCase::deserialize(&current_dir))),
            Err(_) => Box::new(cache_sizes.into_iter().flat_map(move |cache_size| {
                operations
                    .into_iter()
                    .cartesian_product(node_ids.clone())
                    .map(OpWithId::from)
                    .combinations_with_replacement(CONCURRENT_OPS)
                    .map(move |operations| PermutationTestCase {
                        cache_size,
                        operations,
                    })
            })),
        };

    for operation_case in case_yielder {
        operation_case.serialize(&current_dir);
        let PermutationTestCase {
            cache_size,
            operations,
        } = operation_case;

        let shuttle_res = catch_unwind(|| {
            run_shuttle_check(
                {
                    let operations = operations.clone();
                    move || {
                        let lock_cache = Arc::new(LockCache::new_internal(
                            cache_size,
                            NonZero::new(1).unwrap(),
                            Arc::new(EvictionLogger::default()),
                        ));

                        let mut handles = vec![];
                        for OpWithId { op, id } in &operations {
                            set_name_for_shuttle_task(format!("{op}({id})"));
                            handles.push(op.execute(lock_cache.clone(), *id));
                        }

                        for handle in handles {
                            handle.join().unwrap();
                        }
                    }
                },
                SHUTTLE_PERMUTATION_ITERATIONS,
            );
        });
        if let Err(e) = shuttle_res {
            if let Some(error) = e.downcast_ref::<OpPanicStatus>() {
                eprintln!("\n#############################################################");
                eprintln!("Test case failed: {operations:?} with cache size {cache_size}");
                eprintln!("--------------------------------------------------------------");
                if error.expected {
                    eprintln!("Ignoring expected error on {}: {}", error.op, error.error);
                    eprintln!("###########################################################\n");
                } else {
                    panic!(
                        "Unexpected error in shuttle test: {error:?}. Set CARMEN_SHUTTLE_SERIALIZED_CASE env var to reproduce."
                    );
                }
            } else {
                panic!("Unexpected error format in shuttle test: {e:?}");
            }
        }
    }
}
