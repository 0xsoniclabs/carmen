// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.
#![allow(unused_variables)]

use std::ffi::{c_char, c_int, c_void};

use crate::ffi::bindings;

/// Opens a new state object based on the provided implementation maintaining
/// its data in the given directory. If the directory does not exist, it is
/// created. If it is empty, a new, empty state is initialized. If it contains
/// state information, the information is loaded.
///
/// The function returns an opaque pointer to a state object that can be used
/// with the remaining functions in this file. Ownership is transferred to the
/// caller, which is required for releasing it eventually using Carmen_Rust_ReleaseState.
/// If for some reason the creation of the state instance failed, a nullptr is
/// returned.
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_OpenState(
    schema: u8,
    state: bindings::StateImpl,
    archive: bindings::ArchiveImpl,
    directory: *const c_char,
    length: c_int,
) -> *mut c_void {
    unimplemented!()
}

/// Flushes all committed state information to disk to guarantee permanent
/// storage. All internally cached modifications are synced to disk.
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_Flush(state: *mut c_void) {
    unimplemented!()
}

/// Closes this state, releasing all IO handles and locks on external resources.
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_Close(state: *mut c_void) {
    unimplemented!()
}

/// Releases a state object, thereby causing its destruction. After releasing it,
/// no more operations may be applied on it.
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_ReleaseState(state: *mut c_void) {
    unimplemented!()
}

/// Creates a state snapshot reflecting the state at the given block height. The
/// resulting state must be released and must not outlive the life time of the
/// provided state.
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetArchiveState(state: *mut c_void, block: u64) -> *mut c_void {
    unimplemented!()
}

/// Gets the current state of the given account.
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetAccountState(
    state: *mut c_void,
    addr: *mut c_void,
    out_state: *mut c_void,
) {
    unimplemented!()
}

/// Retrieves the balance of the given account.
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetBalance(
    state: *mut c_void,
    addr: *mut c_void,
    out_balance: *mut c_void,
) {
    unimplemented!()
}

/// Retrieves the nonce of the given account.
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetNonce(
    state: *mut c_void,
    addr: *mut c_void,
    out_nonce: *mut c_void,
) {
    unimplemented!()
}

/// Retrieves the value of storage location (addr,key) in the given state.
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetStorageValue(
    state: *mut c_void,
    addr: *mut c_void,
    key: *mut c_void,
    out_value: *mut c_void,
) {
    unimplemented!()
}

/// Retrieves the code stored under the given address.
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetCode(
    state: *mut c_void,
    addr: *mut c_void,
    out_code: *mut c_void,
    out_length: *mut u32,
) {
    unimplemented!()
}

/// Retrieves the hash of the code stored under the given address.
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetCodeHash(
    state: *mut c_void,
    addr: *mut c_void,
    out_hash: *mut c_void,
) {
    unimplemented!()
}

/// Retrieves the code length stored under the given address.
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetCodeSize(
    state: *mut c_void,
    addr: *mut c_void,
    out_length: *mut u32,
) {
    unimplemented!()
}

/// Applies the provided block update to the maintained state.
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_Apply(
    state: *mut c_void,
    block: u64,
    update: *mut c_void,
    length: u64,
) {
    unimplemented!()
}

/// Retrieves a global state hash of the given state.
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetHash(state: *mut c_void, out_hash: *mut c_void) {
    unimplemented!()
}

/// Retrieves a summary of the used memory. After the call the out variable will
/// point to a buffer with a serialized summary that needs to be freed by the
/// caller.
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetMemoryFootprint(
    state: *mut c_void,
    out: *mut *mut c_char,
    out_length: *mut u64,
) {
    unimplemented!()
}

/// Releases the buffer returned by GetMemoryFootprint.
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_ReleaseMemoryFootprintBuffer(buf: *mut c_char, buf_length: u64) {
    unimplemented!()
}

/// Compile-time check that the signatures of the exported functions match the ones generated from
/// the C header file.
#[allow(unused)]
const COMPILE_TIME_CHECK_THAT_SIGNATURES_MATCH_SIGNATURES_GENERATED_FROM_C_HEADER: () = {
    assert_same_signature(
        Carmen_Rust_OpenState as unsafe extern "C" fn(_, _, _, _, _) -> _,
        bindings::Carmen_Rust_OpenState,
    );
    assert_same_signature(
        Carmen_Rust_GetArchiveState as unsafe extern "C" fn(_, _) -> _,
        bindings::Carmen_Rust_GetArchiveState,
    );
    assert_same_signature(
        Carmen_Rust_Flush as unsafe extern "C" fn(_),
        bindings::Carmen_Rust_Flush,
    );
    assert_same_signature(
        Carmen_Rust_Close as unsafe extern "C" fn(_),
        bindings::Carmen_Rust_Close,
    );
    assert_same_signature(
        Carmen_Rust_ReleaseState as unsafe extern "C" fn(_),
        bindings::Carmen_Rust_ReleaseState,
    );
    assert_same_signature(
        Carmen_Rust_GetAccountState as unsafe extern "C" fn(_, _, _),
        bindings::Carmen_Rust_GetAccountState,
    );
    assert_same_signature(
        Carmen_Rust_GetBalance as unsafe extern "C" fn(_, _, _),
        bindings::Carmen_Rust_GetBalance,
    );
    assert_same_signature(
        Carmen_Rust_GetNonce as unsafe extern "C" fn(_, _, _),
        bindings::Carmen_Rust_GetNonce,
    );
    assert_same_signature(
        Carmen_Rust_GetStorageValue as unsafe extern "C" fn(_, _, _, _),
        bindings::Carmen_Rust_GetStorageValue,
    );
    assert_same_signature(
        Carmen_Rust_GetCode as unsafe extern "C" fn(_, _, _, _),
        bindings::Carmen_Rust_GetCode,
    );
    assert_same_signature(
        Carmen_Rust_GetCodeHash as unsafe extern "C" fn(_, _, _),
        bindings::Carmen_Rust_GetCodeHash,
    );
    assert_same_signature(
        Carmen_Rust_GetCodeSize as unsafe extern "C" fn(_, _, _),
        bindings::Carmen_Rust_GetCodeSize,
    );
    assert_same_signature(
        Carmen_Rust_Apply as unsafe extern "C" fn(_, _, _, _),
        bindings::Carmen_Rust_Apply,
    );
    assert_same_signature(
        Carmen_Rust_GetHash as unsafe extern "C" fn(_, _),
        bindings::Carmen_Rust_GetHash,
    );
    assert_same_signature(
        Carmen_Rust_GetMemoryFootprint as unsafe extern "C" fn(_, _, _),
        bindings::Carmen_Rust_GetMemoryFootprint,
    );
    assert_same_signature(
        Carmen_Rust_ReleaseMemoryFootprintBuffer as unsafe extern "C" fn(_, _),
        bindings::Carmen_Rust_ReleaseMemoryFootprintBuffer,
    );
};

const fn assert_same_signature<T>(a: T, b: T) -> (T, T) {
    (a, b)
}
