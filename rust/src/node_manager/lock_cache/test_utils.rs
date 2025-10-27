use dashmap::DashSet;

use crate::{
    error::Error,
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

    fn on_evict(&self, key: u32, value: i32) -> Result<(), Error> {
        self.evicted.insert((key, value));
        Ok(())
    }
}

/// Helper function for performing a get/insert where we don't care about the returned guard.
pub fn ignore_guard<T>(result: Result<T, Error>) {
    let _guard = result.unwrap();
}

/// Type alias for a closure that calls either `get_read_access_or_insert` or
/// `get_write_access_or_insert`
pub type GetOrInsertMethod = fn(
    &LockCache<u32, i32>,
    u32,
    std::sync::Arc<dyn Fn() -> Result<i32, Error>>,
) -> Result<i32, Error>;

/// Reusable rstest template to test both `get_read_access_or_insert` and
/// `get_write_access_or_insert`
#[rstest_reuse::template]
#[rstest::rstest]
#[case::get_read_access((|cache, id, insert_fn| {
        let guard = cache.get_read_access_or_insert(id, || insert_fn())?;
        Ok(*guard)
    }) as crate::node_manager::lock_cache::test_utils::GetOrInsertMethod)]
#[case::get_write_access((|cache, id, insert_fn| {
        let guard = cache.get_write_access_or_insert(id, || insert_fn())?;
        Ok(*guard)
    }) as crate::node_manager::lock_cache::test_utils::GetOrInsertMethod)]
fn get_method(#[case] f: GetOrInsertMethod) {}
