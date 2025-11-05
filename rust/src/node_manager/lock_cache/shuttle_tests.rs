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
use std::{
    io::Write,
    panic::{catch_unwind, panic_any},
    path::Path,
};

use crate::{
    error::{BTResult, Error},
    node_manager::lock_cache::{
        LockCache,
        test_utils::{EvictionLogger, GetOrInsertMethod, get_method, ignore_guard},
    },
    storage,
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

/// The status of an operation error, indicating whether it was expected or unexpected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum OpErrorStatus {
    Expected,
    Unexpected,
}

/// A utility struct to hold information about an operation that panicked.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct OpPanicStatus {
    op: Op,
    status: OpErrorStatus,
    error: String,
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
                    self.handle_invalid_state(&e.into_inner());
                }
            }),
            Op::Get => thread::spawn(move || {
                let guard = cache.get_read_access_or_insert(id, || {
                    Err(Error::Storage(storage::Error::NotFound).into())
                });
                match guard {
                    Ok(guard) => {
                        assert!(*guard != i32::default());
                    }
                    Err(e) => self.handle_invalid_state(&e.into_inner()),
                }
            }),
            Op::Delete => thread::spawn(move || {
                if let Err(e) = cache.remove(id) {
                    self.handle_invalid_state(&e.into_inner());
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
                error,
                Error::CorruptedState(s)
                    if s == "LockCache's cache size is equal or bigger than the number of slots. This may have happened because an insert operation was executed with all cache entries marked as pinned"
                    || s == "another thread removed the key while it was being inserted"
            ),
            Op::Get => matches!(error, Error::Storage(storage::Error::NotFound)),
            // Every error during Delete is expected
            Op::Delete => true,
            Op::Iter => false,
        };
        if matches!(self, Op::Get) && expected {
            // For Get operations, expected errors doesn't indicate an invalid state
            return;
        }
        panic_any(OpPanicStatus {
            op: self,
            status: if expected {
                OpErrorStatus::Expected
            } else {
                OpErrorStatus::Unexpected
            },
            error: error.to_string(),
        });
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
        let op: Op = Op::try_from(bytes[0]).expect("Invalid Op in serialized OpWithId");
        let id: [u8; 4] = bytes[1..5]
            .try_into()
            .expect("Invalid ID in serialized OpWithId");
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
        let message = res.downcast_ref::<OpPanicStatus>().unwrap();
        assert_eq!(message.op, op);
        assert_eq!(message.status, OpErrorStatus::Unexpected);
        assert_eq!(
            message.error,
            format!("{}", Error::CorruptedState("unexpected string".into()))
        );
    }

    #[rstest::rstest]
    #[case(Op::Add, Error::CorruptedState("LockCache's cache size is equal or bigger than the number of slots. This may have happened because an insert operation was executed with all cache entries marked as pinned".into()))]
    #[case(Op::Add, Error::CorruptedState("another thread removed the key while it was being inserted".into()))]
    #[case(Op::Delete, Error::CorruptedState("some delete error".into()))]
    fn handle_invalid_state_panics_on_expected_error(#[case] op: Op, #[case] error: Error) {
        let res = catch_unwind(AssertUnwindSafe(|| {
            op.handle_invalid_state(&error);
        }))
        .unwrap_err();
        let message = res.downcast_ref::<OpPanicStatus>().unwrap();
        assert_eq!(message.op, op);
        assert_eq!(message.status, OpErrorStatus::Expected);
        assert_eq!(message.error, format!("{error}"));
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
