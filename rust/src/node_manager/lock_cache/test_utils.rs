// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use dashmap::DashSet;

use crate::{
    error::{BTResult, Error},
    node_manager::lock_cache::{EvictionHooks, LockCache},
};

/// Logger that records evicted entries
#[derive(Default)]
pub struct EvictionLogger {
    pub evicted: DashSet<(u32, i32)>,
}

impl EvictionHooks for EvictionLogger {
    type Key = u32;
    type Value = i32;

    fn on_evict(&self, key: u32, value: i32) -> BTResult<(), Error> {
        self.evicted.insert((key, value));
        Ok(())
    }
}

/// Helper function for performing a get/insert where we don't care about the returned guard.
pub fn ignore_guard<T>(result: BTResult<T, Error>) {
    let _guard = result.unwrap();
}

/// Type alias for a closure that calls either `get_read_access_or_insert` or
/// `get_write_access_or_insert`
pub type GetOrInsertMethod<F> = fn(&LockCache<u32, i32>, u32, &F) -> BTResult<i32, Error>;

/// Reusable rstest template to test both `get_read_access_or_insert` and
/// `get_write_access_or_insert`
#[rstest_reuse::template]
#[rstest::rstest]
#[case::get_read_access((|cache, id, insert_fn| {
        let guard = cache.get_read_access_or_insert(id, insert_fn)?;
        Ok(*guard)
    }) as crate::node_manager::lock_cache::test_utils::GetOrInsertMethod<_>)]
#[case::get_write_access((|cache, id, insert_fn| {
        let guard = cache.get_write_access_or_insert(id, insert_fn)?;
        Ok(*guard)
    }) as crate::node_manager::lock_cache::test_utils::GetOrInsertMethod<_>)]
fn get_method(#[case] f: GetOrInsertMethod) {}
