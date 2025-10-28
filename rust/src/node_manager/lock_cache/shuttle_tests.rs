use core::panic;
use std::{
    collections::HashSet,
    env,
    io::Write,
    panic::{catch_unwind, panic_any},
    path::Path,
};

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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
enum Op {
    Add,
    Get,
    Delete,
    Iter,
}

impl From<Op> for u8 {
    fn from(op: Op) -> Self {
        match op {
            Op::Add => 0,
            Op::Get => 1,
            Op::Delete => 2,
            Op::Iter => 3,
        }
    }
}

impl TryFrom<u8> for Op {
    type Error = std::num::IntErrorKind;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Op::Add),
            1 => Ok(Op::Get),
            2 => Ok(Op::Delete),
            3 => Ok(Op::Iter),
            _ => Err(std::num::IntErrorKind::PosOverflow),
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

impl Op {
    /// Execute the operation on the given lock cache and node ID, returning a handle to the spawned
    /// thread.
    /// It panics in two cases:
    /// - A reference to a non-existing node is returned
    /// - An unexpected error occurs
    fn execute(self, cache: Arc<LockCache<u32, i32>>, id: u32) -> thread::JoinHandle<()> {
        match self {
            Op::Add => thread::spawn(move || {
                if let Err(e) = cache.get_read_access_or_insert(id, || Ok(42)) {
                    self.handle_invalid_state(&e);
                }
            }),
            Op::Get => thread::spawn(move || {
                let guard = cache.get_read_access_or_insert(id, || {
                    Err(Error::Storage(storage::Error::NotFound))
                });
                match guard {
                    Ok(guard) => {
                        assert!(*guard != i32::default());
                    }
                    Err(e) => self.handle_invalid_state(&e),
                }
            }),
            Op::Delete => thread::spawn(move || {
                if let Err(e) = cache.remove(id) {
                    self.handle_invalid_state(&e);
                }
            }),
            Op::Iter => thread::spawn(move || {
                for (_, guard) in cache.iter_write() {
                    assert!(*guard != i32::default());
                }
            }),
        }
    }

    /// Handle an invalid state error according to the operation type.
    /// Errors are classified into expected and unexpected ones, and panics are raised accordingly.
    /// If the error does not indicate an invalid state, the function returns silently.
    #[track_caller]
    fn handle_invalid_state(self, error: &Error) {
        let expected = match self {
            Op::Add => matches!(
                &error,
                Error::CorruptedState(s)
                    if s == "lock cache's cache size is equal or bigger than the number of slots. This may have happened because an insert operation was executed with all cache entries marked as pinned"
                    || s == "another thread removed the key while it was being inserted"
            ),
            Op::Get => matches!(error, Error::Storage(storage::Error::NotFound)),
            Op::Delete => true,
            Op::Iter => false,
        };
        if matches!(self, Op::Get) && expected {
            // For Get operations, expected errors doesn't indicate an invalid state
            return;
        }
        panic_any((
            self,
            format!(
                "{} error on {}: {:?}",
                if expected { "Expected" } else { "Unexpected" },
                self,
                error
            )
            .to_owned(),
        ));
    }
}

/// An operation with an associated node ID with serialization/deserialization support.
#[derive(Clone, Ord, PartialOrd, PartialEq, Eq, Hash)]
struct OpWithId {
    op: Op,
    id: u32,
}

impl OpWithId {
    /// Serialize the operation with ID into the given byte vector.
    fn serialize(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&u8::from(self.op).to_le_bytes());
        bytes.extend_from_slice(&self.id.to_le_bytes());
    }

    /// Deserialize the operation with ID from the given byte slice.
    fn deserialize(bytes: &[u8]) -> Self {
        let op: Op = Op::try_from(bytes[0]).unwrap();
        let id: [u8; 4] = bytes[1..5].try_into().unwrap();
        let id = u32::from_le_bytes(id);
        Self { op, id }
    }

    /// Get the size of the serialized operation with ID in bytes.
    fn size() -> usize {
        std::mem::size_of::<u8>() + std::mem::size_of::<u32>()
    }
}

impl From<(Op, u32)> for OpWithId {
    fn from(value: (Op, u32)) -> Self {
        Self {
            op: value.0,
            id: value.1,
        }
    }
}

impl std::fmt::Debug for OpWithId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", self.op, self.id)
    }
}

/// A test case for shuttle operation permutations with serialization/deserialization support.
#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
struct PermutationTestCase {
    cache_size: usize,
    operations: Vec<OpWithId>,
}

impl PermutationTestCase {
    const PATH: &str = "shuttle_permutation_case.bin";

    /// Serialize the permutation case to [`Self::PATH`]
    fn serialize(&self, path: &Path) {
        let mut file = std::fs::File::create(path.join(Self::PATH)).unwrap();
        let mut bytes = vec![];
        bytes.extend_from_slice(&self.cache_size.to_le_bytes());
        for operation in &self.operations {
            operation.serialize(&mut bytes);
        }
        file.write_all(&bytes).unwrap();
    }

    /// Deserialize the permutation case from [`Self::PATH`]
    fn deserialize(path: &Path) -> Self {
        let bytes = std::fs::read(path.join(Self::PATH)).unwrap();
        if bytes.len() < std::mem::size_of::<usize>() {
            panic!("Serialized case file is too small");
        }

        let mut operations = vec![];
        let sizeof_usize = std::mem::size_of::<usize>();
        let cache_size = usize::from_le_bytes(bytes[0..sizeof_usize].try_into().unwrap());
        let mut i = sizeof_usize;
        while i + OpWithId::size() <= bytes.len() {
            let op = OpWithId::deserialize(&bytes[i..]);
            operations.push(op);
            i += OpWithId::size();
        }
        if i != bytes.len() {
            panic!("Serialized case file has extra bytes");
        }
        Self {
            cache_size,
            operations,
        }
    }
}

/// Tests all permutations of operations and a small set of node IDs on the lock cache using
/// shuttle. The test is repeated for different cache sizes to stress cache contention scenarios.
/// The idea is to find cases where the lock cache guarantees are violated, i.e. an operation
/// returns a reference to a non-existing node or an unexpected error occurs.
/// The permutation generates known invalid operation sequences that are expected to fail, and
/// these are reported but do not cause the test to fail.
/// To facilitate debugging, the test case that caused a failure is serialized to a file, and
/// can be reloaded by setting the `SHUTTLE_SERIALIZED_CASE` environment variable.
#[test]
fn shuttle_operation_permutations() {
    #[cfg(not(feature = "shuttle"))]
    return;
    #[allow(unreachable_code)]
    let current_dir = env::current_dir().unwrap();
    let CONCURRENT_OPS: usize = 6; // Must be >= 2
    let cache_sizes = (2..=CONCURRENT_OPS + 1).collect::<Vec<usize>>();
    let node_ids = (0..CONCURRENT_OPS as u32).collect::<Vec<u32>>();
    let operations = vec![Op::Add, Op::Get, Op::Delete, Op::Iter];

    let case_yielder: Box<dyn Iterator<Item = PermutationTestCase>> =
        match std::env::var("SHUTTLE_SERIALIZED_CASE") {
            Ok(_) => Box::new(vec![PermutationTestCase::deserialize(&current_dir)].into_iter()),
            Err(_) => Box::new(cache_sizes.clone().into_iter().flat_map(move |cache_size| {
                operations
                    .clone()
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
        let case_error_pool: Arc<std::sync::Mutex<HashSet<(Op, String)>>> = Arc::default();
        operation_case.serialize(&current_dir);
        let PermutationTestCase {
            cache_size,
            operations,
        } = operation_case;
        case_error_pool.lock().unwrap().clear();

        println!("Testing case: {operations:?} with cache size {cache_size}");
        if let Err(e) = catch_unwind(|| {
            run_shuttle_check(
                {
                    let operations = operations.clone();
                    move || {
                        let lock_cache = Arc::new(LockCache::new_with_extra_slots(
                            cache_size,
                            1,
                            Arc::new(EvictionLogger::default()),
                        ));
                        assert_eq!(
                            lock_cache.cache.capacity() as usize,
                            cache_size,
                            "Cache capacity should match the requested size plus extra slots"
                        );
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
                1000,
            );
        }) {
            if let Some(error) = e.downcast_ref::<(Op, String)>() {
                let mut case_error_pool = case_error_pool.lock().unwrap();
                // Skip already known error cases
                if case_error_pool.contains(error) {
                    break;
                }
                case_error_pool.insert(error.clone());
                eprintln!("\n#############################################################");
                eprintln!("Test case failed: {operations:?} with cache size {cache_size}");
                eprintln!("--------------------------------------------------------------");
                if error.1.contains("Expected error") {
                    eprintln!("Ignoring expected error: {}", error.1);
                    eprintln!("###########################################################\n");
                } else {
                    panic!("Unexpected error in shuttle test: {e:?}");
                }
            } else {
                panic!("Unexpected error format in shuttle test: {e:?}");
            }
        }
    }
}

mod tests {
    use std::panic::AssertUnwindSafe;

    use super::*;
    use crate::utils::test_dir::{Permissions, TestDir};

    #[test]
    fn op_conversion_from_and_to_u8_works() {
        assert_eq!(Op::try_from(u8::from(Op::Add)).unwrap(), Op::Add);
        assert_eq!(Op::try_from(u8::from(Op::Get)).unwrap(), Op::Get);
        assert_eq!(Op::try_from(u8::from(Op::Delete)).unwrap(), Op::Delete);
        assert_eq!(Op::try_from(u8::from(Op::Iter)).unwrap(), Op::Iter);
        assert!(Op::try_from(4).is_err());
    }

    #[rstest::rstest]
    fn handle_invalid_state_panics_on_unexpected_errors(
        #[values(Op::Add, Op::Get, Op::Iter)] op: Op,
    ) {
        let res = catch_unwind(|| {
            op.handle_invalid_state(&Error::CorruptedState("unexpected string".into()));
        })
        .unwrap_err();
        let message = res.downcast_ref::<(Op, String)>().unwrap();
        assert_eq!(message.0, op);
        assert_eq!(
            message.1,
            format!("Unexpected error on {op}: CorruptedState(\"unexpected string\")")
        );
    }

    #[rstest::rstest]
    #[case(Op::Add, Error::CorruptedState("lock cache's cache size is equal or bigger than the number of slots. This may have happened because an insert operation was executed with all cache entries marked as pinned".into()))]
    #[case(Op::Add, Error::CorruptedState("another thread removed the key while it was being inserted".into()))]
    #[case(Op::Delete, Error::CorruptedState("some delete error".into()))]
    fn handle_invalid_state_panics_on_expected_error(#[case] op: Op, #[case] error: Error) {
        let res = catch_unwind(AssertUnwindSafe(|| {
            op.handle_invalid_state(&error);
        }))
        .unwrap_err();
        let message = res.downcast_ref::<(Op, String)>().unwrap();
        assert_eq!(message.0, op);
        assert_eq!(message.1, format!("Expected error on {op}: {error:?}"));
    }

    #[test]
    fn handle_invalid_state_ignores_expected_get_not_found_error() {
        let op = Op::Get;
        // Should not panic
        op.handle_invalid_state(&Error::Storage(storage::Error::NotFound));
    }

    #[rstest::rstest]
    #[case(Op::Add)]
    #[case(Op::Get)]
    #[case(Op::Delete)]
    #[case(Op::Iter)]
    fn op_execute_joins_without_panicking_on_successful_operations(#[case] op: Op) {
        let cache = Arc::new(LockCache::new(10, Arc::new(EvictionLogger::default())));
        let handle = op.execute(cache, 0);
        handle.join().unwrap();
    }

    #[test]
    fn op_with_id_serialization_and_deserialization_work() {
        let op_with_id = OpWithId {
            op: Op::Delete,
            id: 12345,
        };
        let mut bytes = vec![];
        op_with_id.serialize(&mut bytes);
        let deserialized_op_with_id = OpWithId::deserialize(&bytes);
        assert_eq!(op_with_id, deserialized_op_with_id);
    }

    #[test]
    fn op_with_id_deserialize_panics_on_invalid_buffer() {
        // Invalid op
        let bytes = [255];
        let res = catch_unwind(|| OpWithId::deserialize(&bytes));
        assert!(res.is_err());

        // Buffer too small
        let bytes = [2, 0]; // Valid Op but incomplete ID
        let res = catch_unwind(|| OpWithId::deserialize(&bytes));
        assert!(res.is_err());
    }

    #[test]
    fn permutation_test_case_serialize_and_deserialize_work() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let original_case = PermutationTestCase {
            cache_size: 5,
            operations: vec![
                OpWithId { op: Op::Add, id: 1 },
                OpWithId { op: Op::Get, id: 2 },
                OpWithId {
                    op: Op::Delete,
                    id: 3,
                },
            ],
        };
        original_case.serialize(&dir);
        let deserialized_case = PermutationTestCase::deserialize(&dir);
        assert_eq!(original_case, deserialized_case);
    }

    #[test]
    #[should_panic(expected = "Serialized case file is too small")]
    fn permutation_test_case_deserialize_panics_on_small_file() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        std::fs::write(dir.join(PermutationTestCase::PATH), vec![0u8; 2]).unwrap();
        let _ = PermutationTestCase::deserialize(&dir);
    }

    #[test]
    #[should_panic]
    fn permutation_test_case_deserialize_panics_on_invalid_op() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let mut bytes = vec![];
        let cache_size: usize = 10;
        bytes.extend_from_slice(&cache_size.to_le_bytes());
        bytes.push(255); // Invalid Op
        std::fs::write(dir.join(PermutationTestCase::PATH), &bytes).unwrap();
        let _ = PermutationTestCase::deserialize(&dir);
    }

    #[test]
    #[should_panic(expected = "Serialized case file has extra bytes")]
    fn permutation_test_case_deserialize_panics_on_extra_bytes() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let mut bytes = vec![];
        let cache_size: usize = 10;
        bytes.extend_from_slice(&cache_size.to_le_bytes());
        let op_with_id = OpWithId { op: Op::Add, id: 1 };
        op_with_id.serialize(&mut bytes);
        bytes.push(0); // Extra byte
        std::fs::write(dir.join(PermutationTestCase::PATH), &bytes).unwrap();
        let _ = PermutationTestCase::deserialize(&dir);
    }
}
