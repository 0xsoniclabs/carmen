// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use crate::{
    error::{BTResult, Error},
    node_manager::lock_cache::{
        LockCache,
        test_utils::{EvictionLogger, GetOrInsertMethod, get_method, ignore_guard},
    },
    sync::{atomic::Ordering, *},
    utils::shuttle::run_shuttle_check,
};

#[rstest_reuse::apply(get_method)]
fn shuttle__cached_node_manager_multiple_get_on_same_id_insert_in_cache_only_once(
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
