// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{
    ffi::{c_char, c_int, c_void},
    mem::MaybeUninit,
};

use crate::{
    CarmenDb, CarmenState,
    ffi::bindings,
    open_carmen_db,
    types::{Address, ArchiveImpl, Hash, Key, LiveImpl, U256, Update, Value},
};

/// The size of the buffer that gets passed to `Carmen_Rust_GetCode`.
/// This is also the maximum size.
/// When running in MIRI, the maximum code size is reduced to 25 bytes because MIRI is very slow
/// otherwise.
const MAX_CODE_SIZE: usize = if cfg!(miri) { 25 } else { 25000 };

/// Opens a new database object based on the provided implementation maintaining its data in the
/// given directory. If the directory does not exist, it is created. If it is empty, a new, empty
/// database is initialized. If it contains information, the information is loaded.
///
/// The function writes an opaque pointer to a database object into the output parameter
/// `out_database`, that can be used with the remaining functions in this file. Ownership is
/// transferred to the caller, which is required for releasing it eventually using
/// Carmen_Rust_ReleaseDatabase(). If for some reason the creation of the state instance failed, an
/// error is returned and the output pointer is not written to.
///
/// # Safety
/// - `directory` must be a valid pointer to a byte array of length `length`
/// - `directory` must be valid for reads for the duration of the call
/// - `directory` must not be mutated for the duration of the call
/// - `out_database` must be a valid pointer to a `*mut c_void`
/// - `out_database` must be valid for writes for the duration of the call
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_OpenDatabase(
    schema: u8,
    live_impl: bindings::LiveImpl,
    archive_impl: bindings::ArchiveImpl,
    directory: *const c_char,
    length: c_int,
    database_out: *mut *mut c_void,
) -> bindings::Result {
    let token = LifetimeToken;
    if directory.is_null() || length <= 0 || database_out.is_null() {
        return bindings::Result_kResult_InvalidArguments;
    }
    let live_impl = match live_impl {
        bindings::LiveImpl_kLive_Memory => LiveImpl::Memory,
        bindings::LiveImpl_kLive_File => LiveImpl::File,
        bindings::LiveImpl_kLive_LevelDb => LiveImpl::LevelDb,
        _ => return bindings::Result_kResult_InvalidArguments,
    };
    let archive_impl = match archive_impl {
        bindings::ArchiveImpl_kArchive_None => ArchiveImpl::None,
        bindings::ArchiveImpl_kArchive_LevelDb => ArchiveImpl::LevelDb,
        bindings::ArchiveImpl_kArchive_Sqlite => ArchiveImpl::Sqlite,
        _ => return bindings::Result_kResult_InvalidArguments,
    };
    // SAFETY:
    // - `directory` is a valid pointer to a byte array of length `length` (precondition)
    // - `directory` is valid for reads for the duration of the call (precondition)
    // - `directory` is not mutated for the duration of the call (precondition)
    let directory =
        unsafe { slice_from_raw_parts_scoped(directory as *const u8, length as usize, &token) };
    let db = open_carmen_db(schema, live_impl, archive_impl, directory);
    match db {
        Ok(db) => {
            // SAFETY:
            // - `out_database` is a valid pointer to a `*mut c_void` (precondition)
            // - `out_database` is valid for writes for the duration of the call (precondition)
            unsafe {
                std::ptr::write(
                    database_out,
                    Box::into_raw(Box::new(DbWrapper::from_db(db))) as *mut c_void,
                );
            }
            bindings::Result_kResult_Success
        }
        Err(err) => err.into(),
    }
}

/// Flushes all committed database information to disk to guarantee permanent storage. All
/// internally cached modifications are synced to disk.
///
/// # Safety
/// - `db` must be a valid pointer to a `DbWrapper` object which holds a pointer to a `dyn CarmenDb`
///   object
/// - `db` must be valid for reads for the duration of the call
/// - `db` must not be mutated for the duration of the call
/// - `db.inner` must be a valid pointer to a `dyn CarmenDb`
/// - `db.inner` must be valid for reads for the duration of the lifetime of `token`
/// - `db.inner` must not be mutated for the duration of the lifetime of `token`
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_Flush(db: *mut c_void) -> bindings::Result {
    let token = LifetimeToken;
    if db.is_null() {
        return bindings::Result_kResult_InvalidArguments;
    }
    // SAFETY:
    // - `db` is a valid pointer to a `DbWrapper` object (precondition)
    // - `db` is valid for reads for the duration of the call (precondition)
    // - `db` is not mutated for the duration of the call (precondition)
    let db = unsafe { ref_from_ptr_scoped(db as *const DbWrapper, &token) };
    // SAFETY:
    // - `db.inner` is a valid pointer to a `dyn CarmenDb` (precondition)
    // - `db.inner` is valid for reads for the duration of the lifetime of `token` (precondition)
    // - `db.inner` is not mutated for the duration of the lifetime of `token`(precondition)
    let db = unsafe { db.inner_to_ref_scoped(&token) };
    match db.flush() {
        Ok(_) => bindings::Result_kResult_Success,
        Err(err) => err.into(),
    }
}

/// Closes this database, releasing all IO handles and locks on external resources.
///
/// # Safety
/// - `db` must be a valid pointer to a `StateWrapper` object which holds a pointer to a `dyn
///   CarmenDb` object
/// - `db` must be valid for reads for the duration of the call
/// - `db` must not be mutated for the duration of the call
/// - `db.inner` must be a valid pointer to a `dyn CarmenDb`
/// - `db.inner` must be valid for reads for the duration of the lifetime of `token`
/// - `db.inner` must not be mutated for the duration of the lifetime of `token`
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_Close(db: *mut c_void) -> bindings::Result {
    let token = LifetimeToken;
    if db.is_null() {
        return bindings::Result_kResult_InvalidArguments;
    }
    // SAFETY:
    // - `db` is a valid pointer to a `StateWrapper` object (precondition)
    // - `db` is valid for reads for the duration of the call (precondition)
    // - `db` is not mutated for the duration of the call (precondition)
    let db = unsafe { ref_from_ptr_scoped(db as *const DbWrapper, &token) };
    // SAFETY:
    // - `db.inner` is a valid pointer to a `dyn CarmenDb` (precondition)
    // - `db.inner` is valid for reads for the duration of the lifetime of `token` (precondition)
    // - `db.inner` is not mutated for the duration of the lifetime of `token`(precondition)
    let db = unsafe { db.inner_to_ref_scoped(&token) };
    match db.close() {
        Ok(_) => bindings::Result_kResult_Success,
        Err(err) => err.into(),
    }
}

/// Releases a database object, thereby causing its destruction. After releasing it, no more
/// operations may be applied on it.
///
/// # Safety
/// - `db` must be a valid pointer to a `StateWrapper` object which holds a pointer to a `dyn
///   CarmenDb` object
/// - `db` must be valid for reads and writes for the duration of the call
/// - `db` must not be mutated for the duration of the call
/// - `db` must have been allocated using the global allocator
/// - `db` must not be used after this call
/// - `db.inner` must be a valid pointer to a `dyn CarmenDb`
/// - `db.inner` must be valid for reads and writes for the duration of the lifetime of `token`
/// - `db.inner` must not be mutated for the duration of the lifetime of `token`
/// - `db.inner` must have been allocated using the global allocator
/// - `db.inner` must not be used after this call
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_ReleaseDatabase(db: *mut c_void) -> bindings::Result {
    let token = LifetimeToken;
    if db.is_null() {
        return bindings::Result_kResult_InvalidArguments;
    }
    // SAFETY:
    // - `db` is a valid pointer to a `StateWrapper` object (precondition)
    // - `db` is valid for reads and writes for the duration of the call (precondition)
    // - `db` is not mutated for the duration of the call (precondition)
    let db = unsafe { ref_mut_from_ptr_scoped(db as *mut DbWrapper, &token) };
    // SAFETY:
    // - `db` was allocated using the global allocator (precondition)
    // - `db` is not used after this call (precondition)
    let db = unsafe { Box::from_raw(db) };
    // SAFETY:
    // - `db.inner` is a valid pointer to a `dyn CarmenDb` (precondition)
    // - `db.inner` is valid for reads and writes for the duration of the lifetime of `token`
    //   (precondition)
    // - `db.inner` is not mutated for the duration of the lifetime of `token`(precondition)
    let db_inner = unsafe { db.inner_to_ref_mut_scoped(&token) };
    // SAFETY:
    // - `db.inner` was allocated using the global allocator (precondition)
    // - `db.inner` is not used after this call (precondition)
    let _ = unsafe { Box::from_raw(db_inner) };
    bindings::Result_kResult_Success
}

/// Writes a handle to the live state of the database into the output parameter `out_state`. The
/// resulting state must be released and must not outlive the life time of the provided database.
///
/// # Safety
/// - `db` must be a valid pointer to a `StateWrapper` object which holds a pointer to a `dyn
///   CarmenDb` object
/// - `db` must be valid for reads for the duration of the call
/// - `db` must not be mutated for the duration of the call
/// - `db.inner` must be a valid pointer to a `dyn CarmenDb`
/// - `db.inner` must be valid for reads for the duration of the lifetime of `token`
/// - `db.inner` must not be mutated for the duration of the lifetime of `token`
/// - `out_state` must be a valid pointer to a `*mut c_void`
/// - `out_state` must be valid for writes for the duration of the call
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetLiveState(
    db: *mut c_void,
    out_state: *mut *mut c_void,
) -> bindings::Result {
    let token = LifetimeToken;
    if db.is_null() || out_state.is_null() {
        return bindings::Result_kResult_InvalidArguments;
    }
    // SAFETY:
    // - `db` is a valid pointer to a `StateWrapper` object (precondition)
    // - `db` is valid for reads for the duration of the call (precondition)
    // - `db` is not mutated for the duration of the call (precondition)
    let db = unsafe { ref_from_ptr_scoped(db as *const DbWrapper, &token) };
    // SAFETY:
    // - `db.inner` is a valid pointer to a `dyn CarmenDb` (precondition)
    // - `db.inner` is valid for reads for the duration of the lifetime of `token` (precondition)
    // - `db.inner` is not mutated for the duration of the lifetime of `token`(precondition)
    let db = unsafe { db.inner_to_ref_scoped(&token) };
    match db.get_live_state() {
        Ok(live_state) => {
            // SAFETY:
            // - `out_state` is a valid pointer to a `*mut c_void` (precondition)
            // - `out_state` is valid for writes for the duration of the call (precondition)
            unsafe {
                std::ptr::write(
                    out_state,
                    Box::into_raw(Box::new(StateWrapper::from_state(live_state))) as *mut c_void,
                );
            }
            bindings::Result_kResult_Success
        }
        Err(err) => err.into(),
    }
}

/// Writes a handle to an archive state reflecting the state at the given block height into the
/// output parameter `out_state`. The resulting state must be released and must not outlive the life
/// time of the provided database. This function will return an error if called with a database that
/// was opened with ArchiveImpl::kArchive_None.
///
/// # Safety
/// - `db` must be a valid pointer to a `StateWrapper` object which holds a pointer to a `dyn
///   CarmenDb` object
/// - `db` must be valid for reads for the duration of the call
/// - `db` must not be mutated for the duration of the call
/// - `db.inner` must be a valid pointer to a `dyn CarmenDb`
/// - `db.inner` must be valid for reads for the duration of the lifetime of `token`
/// - `db.inner` must not be mutated for the duration of the lifetime of `token`
/// - `out_state` must be a valid pointer to a `*mut c_void`
/// - `out_state` must be valid for writes for the duration of the call
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetArchiveState(
    db: *mut c_void,
    block: u64,
    out_state: *mut *mut c_void,
) -> bindings::Result {
    let token = LifetimeToken;
    if db.is_null() || out_state.is_null() {
        return bindings::Result_kResult_InvalidArguments;
    }
    // SAFETY:
    // - `db` is a valid pointer to a `StateWrapper` object (precondition)
    // - `db` is valid for reads for the duration of the call (precondition)
    // - `db` is not mutated for the duration of the call (precondition)
    let db = unsafe { ref_from_ptr_scoped(db as *const DbWrapper, &token) };
    // SAFETY:
    // - `db.inner` is a valid pointer to a `dyn CarmenDb` (precondition)
    // - `db.inner` is valid for reads for the duration of the lifetime of `token` (precondition)
    // - `db.inner` is not mutated for the duration of the lifetime of `token`(precondition)
    let db = unsafe { db.inner_to_ref_scoped(&token) };
    match db.get_archive_state(block) {
        Ok(archive_state) => {
            // SAFETY:
            // - `out_state` is a valid pointer to a `*mut c_void` (precondition)
            // - `out_state` is valid for writes for the duration of the call (precondition)
            unsafe {
                std::ptr::write(
                    out_state,
                    Box::into_raw(Box::new(StateWrapper::from_state(archive_state))) as *mut c_void,
                );
            }
            bindings::Result_kResult_Success
        }
        Err(err) => err.into(),
    }
}

/// Releases a state object, thereby causing its destruction. After releasing it, no more operations
/// may be applied on it.
///
/// # Safety
/// - `state` must be a valid pointer to a `StateWrapper` object which holds a pointer to a `dyn
///   CarmenState` object
/// - `state` must be valid for reads and writes for the duration of the call
/// - `state` must not be mutated for the duration of the call
/// - `state` must have been allocated using the global allocator
/// - `state` must not be used after this call
/// - `state.inner` must be a valid pointer to a `dyn CarmenState`
/// - `state.inner` must be valid for reads and writes for the duration of the lifetime of `token`
/// - `state.inner` must not be mutated for the duration of the lifetime of `token`
/// - `state.inner` must have been allocated using the global allocator
/// - `state.inner` must not be used after this call
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_ReleaseState(state: *mut c_void) -> bindings::Result {
    let token = LifetimeToken;
    if state.is_null() {
        return bindings::Result_kResult_InvalidArguments;
    }
    // SAFETY:
    // - `state` is a valid pointer to a `StateWrapper` object (precondition)
    // - `state` is valid for reads and writes for the duration of the call (precondition)
    // - `state` is not mutated for the duration of the call (precondition)
    let state = unsafe { ref_mut_from_ptr_scoped(state as *mut StateWrapper, &token) };
    // SAFETY:
    // - `state` was allocated using the global allocator (precondition)
    // - `state` is not used after this call (precondition)
    let state = unsafe { Box::from_raw(state) };
    // SAFETY:
    // - `state.inner` is a valid pointer to a `dyn CarmenState` (precondition)
    // - `state.inner` is valid for reads and writes for the duration of the lifetime of `token`
    //   (precondition)
    // - `state.inner` is not mutated for the duration of the lifetime of `token`(precondition)
    let state_inner = unsafe { state.inner_to_ref_mut_scoped(&token) };
    // SAFETY:
    // - `state.inner` was allocated using the global allocator (precondition)
    // - `state.inner` is not used after this call (precondition)
    let _ = unsafe { Box::from_raw(state_inner) };
    bindings::Result_kResult_Success
}

/// Checks if the given account exists.
///
/// # Safety
/// - `state` must be a valid pointer to a `StateWrapper` object which holds a pointer to a `dyn
///   CarmenState` object
/// - `state` must be valid for reads for the duration of the call
/// - `state` must not be mutated for the duration of the call
/// - `state.inner` must be a valid pointer to a `dyn CarmenState`
/// - `state.inner` must be valid for reads for the duration of the lifetime of `token`
/// - `state.inner` must not be mutated for the duration of the lifetime of `token`
/// - `addr` must be a valid pointer to a byte array of length 20
/// - `addr` must be valid for reads for the duration of the call
/// - `addr` must not be mutated for the duration of the call
/// - `out_state` must be a valid pointer to a `u8`
/// - `out_state` must be valid for writes for the duration of the call
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_AccountExists(
    state: *mut c_void,
    addr: *mut c_void,
    out_state: *mut c_void,
) -> bindings::Result {
    let token = LifetimeToken;
    if state.is_null() || addr.is_null() || out_state.is_null() {
        return bindings::Result_kResult_InvalidArguments;
    }
    // SAFETY:
    // - `state` is a valid pointer to a `StateWrapper` object (precondition)
    // - `state` is valid for reads for the duration of the call (precondition)
    // - `state` is not mutated for the duration of the call (precondition)
    let state = unsafe { ref_from_ptr_scoped(state as *const StateWrapper, &token) };
    // SAFETY:
    // - `state.inner` is a valid pointer to a `dyn CarmenState` (precondition)
    // - `state.inner` is valid for reads for the duration of the lifetime of `token` (precondition)
    // - `state.inner` is not mutated for the duration of the lifetime of `token`(precondition)
    let state = unsafe { state.inner_to_ref_scoped(&token) };
    // SAFETY:
    // - `addr` is a valid pointer to a byte array of length 20 (precondition)
    // - `addr` is valid for reads for the duration of the call (precondition)
    // - `addr` is not mutated for the duration of the call (precondition)
    let addr = unsafe { ref_from_ptr_scoped(addr as *mut Address, &token) };
    match state.account_exists(addr) {
        Ok(exists) => {
            // SAFETY:
            // - `out_state` is a valid pointer to a `u8` (precondition)
            // - `out_state` is valid for writes for the duration of the call (precondition)
            unsafe { std::ptr::write(out_state as *mut u8, exists as u8) };
            bindings::Result_kResult_Success
        }
        Err(err) => err.into(),
    }
}

/// Retrieves the balance of the given account.
///
/// # Safety
/// - `state` must be a valid pointer to a `StateWrapper` object which holds a pointer to a `dyn
///   CarmenState` object
/// - `state` must be valid for reads for the duration of the call
/// - `state` must not be mutated for the duration of the call
/// - `state.inner` must be a valid pointer to a `dyn CarmenState`
/// - `state.inner` must be valid for reads for the duration of the lifetime of `token`
/// - `state.inner` must not be mutated for the duration of the lifetime of `token`
/// - `addr` must be a valid pointer to a byte array of length 20
/// - `addr` must be valid for reads for the duration of the call
/// - `addr` must not be mutated for the duration of the call
/// - `out_balance` must be a valid pointer to a byte array of length 32
/// - `out_balance` must be valid for writes for the duration of the call
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetBalance(
    state: *mut c_void,
    addr: *mut c_void,
    out_balance: *mut c_void,
) -> bindings::Result {
    let token = LifetimeToken;
    if state.is_null() || addr.is_null() || out_balance.is_null() {
        return bindings::Result_kResult_InvalidArguments;
    }
    // SAFETY:
    // - `state` is a valid pointer to a `StateWrapper` object (precondition)
    // - `state` is valid for reads for the duration of the call (precondition)
    // - `state` is not mutated for the duration of the call (precondition)
    let state = unsafe { ref_from_ptr_scoped(state as *const StateWrapper, &token) };
    // SAFETY:
    // - `state.inner` is a valid pointer to a `dyn CarmenState` (precondition)
    // - `state.inner` is valid for reads for the duration of the lifetime of `token` (precondition)
    // - `state.inner` is not mutated for the duration of the lifetime of `token`(precondition)
    let state = unsafe { state.inner_to_ref_scoped(&token) };
    // SAFETY:
    // - `addr` is a valid pointer to a byte array of length 20 (precondition)
    // - `addr` is valid for reads for the duration of the call (precondition)
    // - `addr` is not mutated for the duration of the call (precondition)
    let addr = unsafe { ref_from_ptr_scoped(addr as *mut Address, &token) };
    match state.get_balance(addr) {
        Ok(balance) => {
            // SAFETY:
            // - `out_balance` is a valid pointer to a byte array of length 32 (precondition)
            // - `out_balance` is valid for writes for the duration of the call (precondition)
            unsafe { std::ptr::write(out_balance as *mut U256, balance) };
            bindings::Result_kResult_Success
        }
        Err(err) => err.into(),
    }
}

/// Retrieves the nonce of the given account.
///
/// # Safety
/// - `state` must be a valid pointer to a `StateWrapper` object which holds a pointer to a `dyn
///   CarmenState` object
/// - `state` must be valid for reads for the duration of the call
/// - `state` must not be mutated for the duration of the call
/// - `state.inner` must be a valid pointer to a `dyn CarmenState`
/// - `state.inner` must be valid for reads for the duration of the lifetime of `token`
/// - `state.inner` must not be mutated for the duration of the lifetime of `token`
/// - `addr` must be a valid pointer to a byte array of length 20
/// - `addr` must be valid for reads for the duration of the call
/// - `addr` must not be mutated for the duration of the call
/// - `out_nonce` must be a valid pointer to a `u64`
/// - `out_nonce` must be valid for writes for the duration of the call
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetNonce(
    state: *mut c_void,
    addr: *mut c_void,
    out_nonce: *mut c_void,
) -> bindings::Result {
    let token = LifetimeToken;
    if state.is_null() || addr.is_null() || out_nonce.is_null() {
        return bindings::Result_kResult_InvalidArguments;
    }
    // SAFETY:
    // - `state` is a valid pointer to a `StateWrapper` object (precondition)
    // - `state` is valid for reads for the duration of the call (precondition)
    // - `state` is not mutated for the duration of the call (precondition)
    let state = unsafe { ref_from_ptr_scoped(state as *const StateWrapper, &token) };
    // SAFETY:
    // - `state.inner` is a valid pointer to a `dyn CarmenState` (precondition)
    // - `state.inner` is valid for reads for the duration of the lifetime of `token` (precondition)
    // - `state.inner` is not mutated for the duration of the lifetime of `token`(precondition)
    let state = unsafe { state.inner_to_ref_scoped(&token) };
    // SAFETY:
    // - `addr` is a valid pointer to a byte array of length 20 (precondition)
    // - `addr` is valid for reads for the duration of the call (precondition)
    // - `addr` is not mutated for the duration of the call (pre
    let addr = unsafe { ref_from_ptr_scoped(addr as *mut Address, &token) };
    match state.get_nonce(addr) {
        Ok(nonce) => {
            // SAFETY:
            // - `out_nonce` is a valid pointer to a byte array of length 32 (precondition)
            // - `out_nonce` is valid for writes for the duration of the call (precondition)
            unsafe { std::ptr::write(out_nonce as _, nonce) };
            bindings::Result_kResult_Success
        }
        Err(err) => err.into(),
    }
}

/// Retrieves the value of storage location (addr,key) in the given state.
///
/// # Safety
/// - `state` must be a valid pointer to a `StateWrapper` object which holds a pointer to a `dyn
///   CarmenState` object
/// - `state` must be valid for reads for the duration of the call
/// - `state` must not be mutated for the duration of the call
/// - `state.inner` must be a valid pointer to a `dyn CarmenState`
/// - `state.inner` must be valid for reads for the duration of the lifetime of `token`
/// - `state.inner` must not be mutated for the duration of the lifetime of `token`
/// - `addr` must be a valid pointer to a byte array of length 20
/// - `addr` must be valid for reads for the duration of the call
/// - `addr` must not be mutated for the duration of the call
/// - `key` must be a valid pointer to a byte array of length 32
/// - `key` must be valid for reads for the duration of the call
/// - `key` must not be mutated for the duration of the call
/// - `out_value` must be a valid pointer to a byte array of length 32
/// - `out_value` must be valid for writes for the duration of the call
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetStorageValue(
    state: *mut c_void,
    addr: *mut c_void,
    key: *mut c_void,
    out_value: *mut c_void,
) -> bindings::Result {
    let token = LifetimeToken;
    if state.is_null() || addr.is_null() || key.is_null() || out_value.is_null() {
        return bindings::Result_kResult_InvalidArguments;
    }
    // SAFETY:
    // - `state` is a valid pointer to a `StateWrapper` object (precondition)
    // - `state` is valid for reads for the duration of the call (precondition)
    // - `state` is not mutated for the duration of the call (precondition)
    let state = unsafe { ref_from_ptr_scoped(state as *const StateWrapper, &token) };
    // SAFETY:
    // - `state.inner` is a valid pointer to a `dyn CarmenState` (precondition)
    // - `state.inner` is valid for reads for the duration of the lifetime of `token` (precondition)
    // - `state.inner` is not mutated for the duration of the lifetime of `token`(precondition)
    let state = unsafe { state.inner_to_ref_scoped(&token) };
    // SAFETY:
    // - `addr` is a valid pointer to a byte array of length 20 (precondition)
    // - `addr` is valid for reads for the duration of the call (precondition)
    // - `addr` is not mutated for the duration of the call (precondition)
    let addr = unsafe { ref_from_ptr_scoped(addr as *mut Address, &token) };
    // SAFETY:
    // - `key` is a valid pointer to a byte array of length 32 (precondition)
    // - `key` is valid for reads for the duration of the call (precondition)
    // - `key` is not mutated for the duration of the call (precondition)
    let key = unsafe { ref_from_ptr_scoped(key as *mut Key, &token) };
    match state.get_storage_value(addr, key) {
        Ok(value) => {
            // SAFETY:
            // - `out_value` is a valid pointer to a byte array of length 32 (precondition)
            // - `out_value` is valid for writes for the duration of the call (precondition)
            unsafe { std::ptr::write(out_value as *mut Value, value) };
            bindings::Result_kResult_Success
        }
        Err(err) => err.into(),
    }
}

/// Retrieves the code stored under the given address.
///
/// # Safety
/// - `state` must be a valid pointer to a `StateWrapper` object which holds a pointer to a `dyn
///   CarmenState` object
/// - `state` must be valid for reads for the duration of the call
/// - `state` must not be mutated for the duration of the call
/// - `state.inner` must be a valid pointer to a `dyn CarmenState`
/// - `state.inner` must be valid for reads for the duration of the lifetime of `token`
/// - `state.inner` must not be mutated for the duration of the lifetime of `token`
/// - `addr` must be a valid pointer to a byte array of length 20
/// - `addr` must be valid for reads for the duration of the call
/// - `addr` must not be mutated for the duration of the call
/// - `out_code` must be a valid pointer to a byte array of length `MAX_CODE_SIZE`
/// - `out_code` must be valid for writes for the duration of the call
/// - `out_code` must not be mutated for the duration of the call
/// - `out_length` must be a valid pointer to a `u32`
/// - `out_length` must be valid for writes for the duration of the call
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetCode(
    state: *mut c_void,
    addr: *mut c_void,
    out_code: *mut c_void,
    out_length: *mut u32,
) -> bindings::Result {
    let token = LifetimeToken;
    if state.is_null() || addr.is_null() || out_code.is_null() || out_length.is_null() {
        return bindings::Result_kResult_InvalidArguments;
    }
    // SAFETY:
    // - `state` is a valid pointer to a `StateWrapper` object (precondition)
    // - `state` is valid for reads for the duration of the call (precondition)
    // - `state` is not mutated for the duration of the call (precondition)
    let state = unsafe { ref_from_ptr_scoped(state as *const StateWrapper, &token) };
    // SAFETY:
    // - `state.inner` is a valid pointer to a `dyn CarmenState` (precondition)
    // - `state.inner` is valid for reads for the duration of the lifetime of `token` (precondition)
    // - `state.inner` is not mutated for the duration of the lifetime of `token`(precondition)
    let state = unsafe { state.inner_to_ref_scoped(&token) };
    // SAFETY:
    // - `addr` is a valid pointer to a byte array of length 20 (precondition)
    // - `addr` is valid for reads for the duration of the call (precondition)
    // - `addr` is not mutated for the duration of the call (precondition)
    let addr = unsafe { ref_from_ptr_scoped(addr as *mut Address, &token) };
    // SAFETY:
    // - `out_code` is a valid pointer to a byte array of length `MAX_CODE_SIZE` (precondition)
    // - `out_code` is valid for writes for the duration of the call (precondition)
    // - `out_code` is not mutated for the duration of the call (precondition)
    let out_code = unsafe {
        slice_from_raw_parts_mut_scoped(out_code as *mut MaybeUninit<u8>, MAX_CODE_SIZE, &token)
    };
    match state.get_code(addr, out_code) {
        Ok(len) => {
            // SAFETY:
            // - `out_length` is a valid pointer to a `u32` (precondition)
            // - `out_length` is valid for writes for the duration of the call (precondition)
            unsafe { std::ptr::write(out_length, len as u32) };
            bindings::Result_kResult_Success
        }
        Err(err) => err.into(),
    }
}

/// Retrieves the hash of the code stored under the given address.
///
/// # Safety
/// - `state` must be a valid pointer to a `StateWrapper` object which holds a pointer to a `dyn
///   CarmenState` object
/// - `state` must be valid for reads for the duration of the call
/// - `state` must not be mutated for the duration of the call
/// - `state.inner` must be a valid pointer to a `dyn CarmenState`
/// - `state.inner` must be valid for reads for the duration of the lifetime of `token`
/// - `state.inner` must not be mutated for the duration of the lifetime of `token`
/// - `addr` must be a valid pointer to a byte array of length 20
/// - `addr` must be valid for reads for the duration of the call
/// - `addr` must not be mutated for the duration of the call
/// - `out_hash` must be a valid pointer to a byte array of length 32
/// - `out_hash` must be valid for writes for the duration of the call
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetCodeHash(
    state: *mut c_void,
    addr: *mut c_void,
    out_hash: *mut c_void,
) -> bindings::Result {
    let token = LifetimeToken;
    if state.is_null() || addr.is_null() || out_hash.is_null() {
        return bindings::Result_kResult_InvalidArguments;
    }
    // SAFETY:
    // - `state` is a valid pointer to a `StateWrapper` object (precondition)
    // - `state` is valid for reads for the duration of the call (precondition)
    // - `state` is not mutated for the duration of the call (precondition)
    let state = unsafe { ref_from_ptr_scoped(state as *const StateWrapper, &token) };
    // SAFETY:
    // - `state.inner` is a valid pointer to a `dyn CarmenState` (precondition)
    // - `state.inner` is valid for reads for the duration of the lifetime of `token` (precondition)
    // - `state.inner` is not mutated for the duration of the lifetime of `token`(precondition)
    let state = unsafe { state.inner_to_ref_scoped(&token) };
    // SAFETY:
    // - `addr` is a valid pointer to a byte array of length 20 (precondition)
    // - `addr` is valid for reads for the duration of the call (precondition)
    // - `addr` is not mutated for the duration of the call (precondition)
    let addr = unsafe { ref_from_ptr_scoped(addr as *const Address, &token) };
    match state.get_code_hash(addr) {
        Ok(code_hash) => {
            // SAFETY:
            // - `out_hash` is a valid pointer to a `u32` (precondition)
            // - `out_hash` is valid for writes for the duration of the call (precondition)
            unsafe { std::ptr::write(out_hash as *mut Hash, code_hash) };
            bindings::Result_kResult_Success
        }
        Err(err) => err.into(),
    }
}

/// Retrieves the code length stored under the given address.
///
/// # Safety
/// - `state` must be a valid pointer to a `StateWrapper` object which holds a pointer to a `dyn
///   CarmenState` object
/// - `state` must be valid for reads for the duration of the call
/// - `state` must not be mutated for the duration of the call
/// - `state.inner` must be a valid pointer to a `dyn CarmenState`
/// - `state.inner` must be valid for reads for the duration of the lifetime of `token`
/// - `state.inner` must not be mutated for the duration of the lifetime of `token`
/// - `addr` must be a valid pointer to a byte array of length 20
/// - `addr` must be valid for reads for the duration of the call
/// - `addr` must not be mutated for the duration of the call
/// - `out_length` must be a valid pointer to a `u32`
/// - `out_length` must be valid for writes for the duration of the call
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetCodeSize(
    state: *mut c_void,
    addr: *mut c_void,
    out_length: *mut u32,
) -> bindings::Result {
    let token = LifetimeToken;
    if state.is_null() || addr.is_null() || out_length.is_null() {
        return bindings::Result_kResult_InvalidArguments;
    }
    // SAFETY:
    // - `state` is a valid pointer to a `StateWrapper` object (precondition)
    // - `state` is valid for reads for the duration of the call (precondition)
    // - `state` is not mutated for the duration of the call (precondition)
    let state = unsafe { ref_from_ptr_scoped(state as *const StateWrapper, &token) };
    // SAFETY:
    // - `state.inner` is a valid pointer to a `dyn CarmenState` (precondition)
    // - `state.inner` is valid for reads for the duration of the lifetime of `token` (precondition)
    // - `state.inner` is not mutated for the duration of the lifetime of `token`(precondition)
    let state = unsafe { state.inner_to_ref_scoped(&token) };
    // SAFETY:
    // - `addr` is a valid pointer to a byte array of length 20 (precondition)
    // - `addr` is valid for reads for the duration of the call (precondition)
    // - `addr` is not mutated for the duration of the call (precondition)
    let addr = unsafe { ref_from_ptr_scoped(addr as *const Address, &token) };
    match state.get_code_len(addr) {
        Ok(code_size) => {
            // SAFETY:
            // - `out_length` is a valid pointer to a `u32` (precondition)
            // - `out_length` is valid for writes for the duration of the call (precondition)
            unsafe { std::ptr::write(out_length, code_size) };
            bindings::Result_kResult_Success
        }
        Err(err) => err.into(),
    }
}

/// Applies the provided block update to the live state. This function will return an error if
/// called with an archive state.
///
/// # Safety
/// - `state` must be a valid pointer to a `StateWrapper` object which holds a pointer to a `dyn
///   CarmenState` object
/// - `state` must be valid for reads for the duration of the call
/// - `state` must not be mutated for the duration of the call
/// - `state.inner` must be a valid pointer to a `dyn CarmenState`
/// - `state.inner` must be valid for reads for the duration of the lifetime of `token`
/// - `state.inner` must not be mutated for the duration of the lifetime of `token`
/// - `update` must be a valid pointer to a byte array of length `length`
/// - `update` must be valid for reads for the duration of the call
/// - `update` must not be mutated for the duration of the call
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_Apply(
    state: *mut c_void,
    block: u64,
    update: *mut c_void,
    length: u64,
) -> bindings::Result {
    let token = LifetimeToken;
    if state.is_null() || update.is_null() || length == 0 {
        return bindings::Result_kResult_InvalidArguments;
    }
    // SAFETY:
    // - `state` is a valid pointer to a `StateWrapper` object (precondition)
    // - `state` is valid for reads for the duration of the call (precondition)
    // - `state` is not mutated for the duration of the call (precondition)
    let state = unsafe { ref_from_ptr_scoped(state as *const StateWrapper, &token) };
    // SAFETY:
    // - `state.inner` is a valid pointer to a `dyn CarmenState` (precondition)
    // - `state.inner` is valid for reads for the duration of the lifetime of `token` (precondition)
    // - `state.inner` is not mutated for the duration of the lifetime of `token`(precondition)
    let state = unsafe { state.inner_to_ref_scoped(&token) };
    // SAFETY:
    // - `update` is a valid pointer to a byte array of length `length` (precondition)
    // - `update` is valid for reads for the duration of the call
    // - `update` is not mutated for the duration of the call (precondition)
    let update_data =
        unsafe { slice_from_raw_parts_mut_scoped(update as *mut u8, length as usize, &token) };
    let Ok(update) = Update::from_encoded(update_data) else {
        return bindings::Result_kResult_InvalidArguments; // update parsing error
    };
    match state.apply_block_update(block, update) {
        Ok(_) => bindings::Result_kResult_Success,
        Err(err) => err.into(),
    }
}

/// Retrieves a global state hash of the given state.
///
/// # Safety
/// - `state` must be a valid pointer to a `StateWrapper` object which holds a pointer to a `dyn
///   CarmenState` object
/// - `state` must be valid for reads for the duration of the call
/// - `state` must not be mutated for the duration of the call
/// - `state.inner` must be a valid pointer to a `dyn CarmenState`
/// - `state.inner` must be valid for reads for the duration of the lifetime of `token`
/// - `state.inner` must not be mutated for the duration of the lifetime of `token`
/// - `out_hash` must be a valid pointer to a byte array of length 32
/// - `out_hash` must be valid for writes for the duration of the call
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetHash(
    state: *mut c_void,
    out_hash: *mut c_void,
) -> bindings::Result {
    let token = LifetimeToken;
    if state.is_null() || out_hash.is_null() {
        return bindings::Result_kResult_InvalidArguments;
    }
    // SAFETY:
    // - `state` is a valid pointer to a `StateWrapper` object (precondition)
    // - `state` is valid for reads for the duration of the call (precondition)
    // - `state` is not mutated for the duration of the call (precondition)
    let state = unsafe { ref_from_ptr_scoped(state as *const StateWrapper, &token) };
    // SAFETY:
    // - `state.inner` is a valid pointer to a `dyn CarmenState` (precondition)
    // - `state.inner` is valid for reads for the duration of the lifetime of `token` (precondition)
    // - `state.inner` is not mutated for the duration of the lifetime of `token`(precondition)
    let state = unsafe { state.inner_to_ref_scoped(&token) };
    match state.get_hash() {
        Ok(hash) => {
            // SAFETY:
            // - `out_hash` is a valid pointer to a byte array of length 32 (precondition)
            // - `out_hash` is valid for writes for the duration of the call (precondition)
            unsafe { std::ptr::write(out_hash as *mut Hash, hash) };
            bindings::Result_kResult_Success
        }
        Err(err) => err.into(),
    }
}

/// Retrieves a summary of the used memory. After the call the out variable will
/// point to a buffer with a serialized summary that needs to be freed by the
/// caller.
///
/// # Safety
/// - `db` must be a valid pointer to a `DbWrapper` object which holds a pointer to a `dyn CarmenDb`
///   object
/// - `db` must be valid for reads for the duration of the call
/// - `db` must not be mutated for the duration of the call
/// - `db.inner` must be a valid pointer to a `dyn CarmenDb`
/// - `db.inner` must be valid for reads for the duration of the lifetime of `token`
/// - `db.inner` must not be mutated for the duration of the lifetime of `token`
/// - `out` must be a valid pointer to a byte array of length `out_length`
/// - `out` must be valid for and writes for the duration of the call
/// - `out` must not be mutated for the duration of the call
/// - `out_length` must be a valid pointer to a `u64`
/// - `out_length` must be valid for writes for the duration of the call
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_GetMemoryFootprint(
    db: *mut c_void,
    out: *mut *mut c_char,
    out_length: *mut u64,
) -> bindings::Result {
    let token = LifetimeToken;
    if db.is_null() || out.is_null() || out_length.is_null() {
        return bindings::Result_kResult_InvalidArguments;
    }
    // SAFETY:
    // - `db` is a valid pointer to a `StateWrapper` object (precondition)
    // - `db` is valid for reads for the duration of the call (precondition)
    // - `db` is not mutated for the duration of the call (precondition)
    let db = unsafe { ref_from_ptr_scoped(db as *const DbWrapper, &token) };
    // SAFETY:
    // - `db.inner` is a valid pointer to a `dyn CarmenDb` (precondition)
    // - `db.inner` is valid for reads for the duration of the lifetime of `token` (precondition)
    // - `db.inner` is not mutated for the duration of the lifetime of `token`(precondition)
    let db = unsafe { db.inner_to_ref_scoped(&token) };
    match db.get_memory_footprint() {
        Ok(msg) => {
            // SAFETY:
            // - `out_length` is a valid pointer to a `u64` (precondition)
            // - `out_length` is valid for writes for the duration of the call (precondition)
            unsafe { std::ptr::write(out_length, msg.len() as u64) };
            // SAFETY:
            // - `out` is a valid pointer to a byte array of length `out_length` (precondition)
            // - `out` is valid for writes for the duration of the call (precondition)
            unsafe { std::ptr::write(out, Box::into_raw(msg) as *mut c_char) };
            bindings::Result_kResult_Success
        }
        Err(err) => err.into(),
    }
}

/// Releases the buffer returned by GetMemoryFootprint.
///
/// # Safety
/// - `buf` must be a valid pointer to a byte array of length `buf_length`
/// - `buf` must be valid for reads and writes for the duration of the call
/// - `buf` must not be mutated for the duration of the call
/// - `buf` must have been allocated using the global allocator
/// - `buf` must not be used after this call
#[unsafe(no_mangle)]
unsafe extern "C" fn Carmen_Rust_ReleaseMemoryFootprintBuffer(
    buf: *mut c_char,
    buf_length: u64,
) -> bindings::Result {
    if buf.is_null() || buf_length == 0 {
        return bindings::Result_kResult_InvalidArguments;
    }
    let buf = std::ptr::slice_from_raw_parts_mut(buf as *mut u8, buf_length as usize);
    // SAFETY:
    // - `buf` is a valid pointer to a byte array of length `buf_length` (precondition)
    // - `buf` is valid for reads and writes for the duration of the call (precondition)
    // - `buf` is not mutated for the duration of the call (precondition)
    // - `buf` has been allocated using the global allocator (precondition)
    // - `buf` is not used after this call (precondition)
    unsafe {
        let _ = Box::from_raw(buf);
    }
    bindings::Result_kResult_Success
}

/// A transparent wrapper around a pointer to a `dyn CarmenDb` object. Pointers to this wrapper
/// type are passed as `database` through the FFI interface.
#[repr(transparent)]
struct DbWrapper {
    inner: *mut dyn CarmenDb,
}

impl DbWrapper {
    /// # Safety
    /// - `self.inner` must be a valid pointer to a `dyn CarmenDb`
    /// - `self.inner` must be valid for reads for the duration of the lifetime of `_token`
    /// - `self.inner` must not be mutated for the duration of the lifetime of `_token`
    unsafe fn inner_to_ref_scoped<'db>(&self, _token: &'db LifetimeToken) -> &'db dyn CarmenDb {
        // SAFETY:
        // - `self.inner` is a valid pointer to a `dyn CarmenDb` (precondition)
        // - `self.inner` is valid for reads for the duration of the lifetime of `_token`
        // (precondition)
        // - `self.inner` is not mutated for the duration of the lifetime of `_token` (precondition)
        unsafe { &*self.inner }
    }

    /// # Safety
    /// - `self.inner` must be a valid pointer to a `dyn CarmenDb`
    /// - `self.inner` must be valid for reads for the duration of the lifetime of `_token`
    /// - `self.inner` must not be mutated for the duration of the lifetime of `_token`
    #[allow(clippy::mut_from_ref)] // false positive
    unsafe fn inner_to_ref_mut_scoped<'db>(
        &self,
        _token: &'db LifetimeToken,
    ) -> &'db mut dyn CarmenDb {
        // SAFETY:
        // - `self.inner` is a valid pointer to a `dyn CarmenDb` (precondition)
        // - `self.inner` is valid for reads and writes for the duration of the lifetime of `_token`
        // (precondition)
        // - `self.inner` is not mutated for the duration of the lifetime of `_token` (precondition)
        unsafe { &mut *self.inner }
    }

    fn from_db(state: Box<dyn CarmenDb>) -> Self {
        Self {
            inner: Box::into_raw(state),
        }
    }
}

/// A transparent wrapper around a pointer to a `dyn CarmenState` object. Pointers to this wrapper
/// type are passed as `state` through the FFI interface.
#[repr(transparent)]
struct StateWrapper {
    inner: *mut dyn CarmenState,
}

impl StateWrapper {
    /// # Safety
    /// - `self.inner` must be a valid pointer to a `dyn CarmenState`
    /// - `self.inner` must be valid for reads for the duration of the lifetime of `_token`
    /// - `self.inner` must not be mutated for the duration of the lifetime of `_token`
    unsafe fn inner_to_ref_scoped<'db>(&self, _token: &'db LifetimeToken) -> &'db dyn CarmenState {
        // SAFETY:
        // - `self.inner` is a valid pointer to a `dyn CarmenState` (precondition)
        // - `self.inner` is valid for reads for the duration of the lifetime of `_token`
        //   (precondition)
        // - `self.inner` is not mutated for the duration of the lifetime of `_token` (precondition)
        unsafe { &*self.inner }
    }

    /// # Safety
    /// - `self.inner` must be a valid pointer to a `dyn CarmenState`
    /// - `self.inner` must be valid for reads for the duration of the lifetime of `_token`
    /// - `self.inner` must not be mutated for the duration of the lifetime of `_token`
    #[allow(clippy::mut_from_ref)] // false positive
    unsafe fn inner_to_ref_mut_scoped<'db>(
        &self,
        _token: &'db LifetimeToken,
    ) -> &'db mut dyn CarmenState {
        // SAFETY:
        // - `self.inner` is a valid pointer to a `dyn CarmenState` (precondition)
        // - `self.inner` is valid for reads and writes for the duration of the lifetime of `_token`
        // (precondition)
        // - `self.inner` is not mutated for the duration of the lifetime of `_token` (precondition)
        unsafe { &mut *self.inner }
    }

    fn from_state(state: Box<dyn CarmenState>) -> Self {
        Self {
            inner: Box::into_raw(state),
        }
    }
}

/// A zero sized type, used to enforce the correct lifetime for references created from
/// pointers obtained via FFI. It is assumed that the lifetime of the pointer is bound by the
/// lifetime of the borrow of [`LifetimeToken`].
/// When the token is created at the beginning of a function call, it ensures that the
/// references created from pointers obtained via FFI are only valid for the duration of the
/// function call.
struct LifetimeToken;

/// # Safety
/// - `ptr` must be non-null
/// - `ptr` must be valid for reads for the lifetime of the borrow of `_token`
/// - `ptr` must not be mutated for the lifetime of the borrow of `_token`
#[allow(clippy::needless_lifetimes)] // use explicit lifetimes for easier understanding
unsafe fn ref_from_ptr_scoped<'s, T: ?Sized>(ptr: *const T, _token: &'s LifetimeToken) -> &'s T {
    // SAFETY:
    // - `ptr` is non-null (precondition)
    // - `ptr` is valid for reads for the lifetime of the borrow of `_token`(precondition)
    // - `ptr` is not mutated for the lifetime of the borrow of `_token` (precondition)
    unsafe { &*ptr }
}

/// # Safety
/// - `ptr` must be non-null
/// - `ptr` must be valid for reads and writes for the lifetime of the borrow of `_token`
/// - `ptr` must not be mutated for the lifetime of the borrow of `_token`
#[allow(clippy::needless_lifetimes)] // use explicit lifetimes for easier understanding
#[allow(clippy::mut_from_ref)] // false positive
unsafe fn ref_mut_from_ptr_scoped<'s, T: ?Sized>(
    ptr: *mut T,
    _token: &'s LifetimeToken,
) -> &'s mut T {
    // SAFETY:
    // - `ptr` is non-null (precondition)
    // - `ptr` is valid for reads and writes for the lifetime of the borrow of
    //   `_token`(precondition)
    // - `ptr` is not mutated for the lifetime of the borrow of `_token` (precondition)
    unsafe { &mut *ptr }
}

/// # Safety
/// - `ptr` must be non-null
/// - `ptr` must be valid for reads for `len * mem::size_of::<T>()` many bytes for the lifetime of
///   the borrow of `_token`
/// - `ptr` must not be mutated for the lifetime of the borrow of `_token`
#[allow(clippy::needless_lifetimes)] // use explicit lifetimes for easier understanding
unsafe fn slice_from_raw_parts_scoped<'s, T>(
    ptr: *const T,
    len: usize,
    _token: &'s LifetimeToken,
) -> &'s [T] {
    // SAFETY:
    // - `ptr` is non-null (precondition)
    // - `ptr` is valid for reads for `len * mem::size_of::<T>()` many bytes for the lifetime of the
    //   borrow of `_token`(precondition)
    // - `ptr` is not mutated for the lifetime of the borrow of `_token` (precondition)
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// # Safety
/// - `ptr` must be non-null
/// - `ptr` must be valid for reads and writes for `len * mem::size_of::<T>()` many bytes for the
///   lifetime of the borrow of `_token`
/// - `ptr` must not be mutated for the lifetime of the borrow of `_token`
#[allow(clippy::needless_lifetimes)] // use explicit lifetimes for easier understanding
#[allow(clippy::mut_from_ref)] // false positive
unsafe fn slice_from_raw_parts_mut_scoped<'s, T>(
    ptr: *mut T,
    len: usize,
    _token: &'s LifetimeToken,
) -> &'s mut [T] {
    // SAFETY:
    // - `ptr` is non-null (precondition)
    // - `ptr` is valid for reads and writes for `len * mem::size_of::<T>()` many bytes for the
    //   lifetime of the borrow of `_token`(precondition)
    // - `ptr` is not mutated for the lifetime of the borrow of `_token` (precondition)
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

/// Compile-time check that the signatures of the exported functions match the ones generated from
/// the C header file.
#[allow(unused)]
const COMPILE_TIME_CHECK_THAT_SIGNATURES_MATCH_SIGNATURES_GENERATED_FROM_C_HEADER: () = {
    assert_same_signature(
        Carmen_Rust_OpenDatabase as unsafe extern "C" fn(_, _, _, _, _, _) -> _,
        bindings::Carmen_Rust_OpenDatabase,
    );
    assert_same_signature(
        Carmen_Rust_Flush as unsafe extern "C" fn(_) -> _,
        bindings::Carmen_Rust_Flush,
    );
    assert_same_signature(
        Carmen_Rust_Close as unsafe extern "C" fn(_) -> _,
        bindings::Carmen_Rust_Close,
    );
    assert_same_signature(
        Carmen_Rust_ReleaseDatabase as unsafe extern "C" fn(_) -> _,
        bindings::Carmen_Rust_ReleaseDatabase,
    );
    assert_same_signature(
        Carmen_Rust_GetLiveState as unsafe extern "C" fn(_, _) -> _,
        bindings::Carmen_Rust_GetLiveState,
    );
    assert_same_signature(
        Carmen_Rust_GetArchiveState as unsafe extern "C" fn(_, _, _) -> _,
        bindings::Carmen_Rust_GetArchiveState,
    );
    assert_same_signature(
        Carmen_Rust_ReleaseState as unsafe extern "C" fn(_) -> _,
        bindings::Carmen_Rust_ReleaseState,
    );
    assert_same_signature(
        Carmen_Rust_AccountExists as unsafe extern "C" fn(_, _, _) -> _,
        bindings::Carmen_Rust_AccountExists,
    );
    assert_same_signature(
        Carmen_Rust_GetBalance as unsafe extern "C" fn(_, _, _) -> _,
        bindings::Carmen_Rust_GetBalance,
    );
    assert_same_signature(
        Carmen_Rust_GetNonce as unsafe extern "C" fn(_, _, _) -> _,
        bindings::Carmen_Rust_GetNonce,
    );
    assert_same_signature(
        Carmen_Rust_GetStorageValue as unsafe extern "C" fn(_, _, _, _) -> _,
        bindings::Carmen_Rust_GetStorageValue,
    );
    assert_same_signature(
        Carmen_Rust_GetCode as unsafe extern "C" fn(_, _, _, _) -> _,
        bindings::Carmen_Rust_GetCode,
    );
    assert_same_signature(
        Carmen_Rust_GetCodeHash as unsafe extern "C" fn(_, _, _) -> _,
        bindings::Carmen_Rust_GetCodeHash,
    );
    assert_same_signature(
        Carmen_Rust_GetCodeSize as unsafe extern "C" fn(_, _, _) -> _,
        bindings::Carmen_Rust_GetCodeSize,
    );
    assert_same_signature(
        Carmen_Rust_Apply as unsafe extern "C" fn(_, _, _, _) -> _,
        bindings::Carmen_Rust_Apply,
    );
    assert_same_signature(
        Carmen_Rust_GetHash as unsafe extern "C" fn(_, _) -> _,
        bindings::Carmen_Rust_GetHash,
    );
    assert_same_signature(
        Carmen_Rust_GetMemoryFootprint as unsafe extern "C" fn(_, _, _) -> _,
        bindings::Carmen_Rust_GetMemoryFootprint,
    );
    assert_same_signature(
        Carmen_Rust_ReleaseMemoryFootprintBuffer as unsafe extern "C" fn(_, _) -> _,
        bindings::Carmen_Rust_ReleaseMemoryFootprintBuffer,
    );
};

const fn assert_same_signature<T>(a: T, b: T) -> (T, T) {
    (a, b)
}

const _COMPILE_TIME_CHECK_THAT_WRAPPERS_HAVE_DOUBLE_THE_SIZE_OF_VOID_POINTERS: () = {
    assert!(size_of::<DbWrapper>() == 2 * size_of::<*mut c_void>());
    assert!(size_of::<StateWrapper>() == 2 * size_of::<*mut c_void>());
};

#[allow(
    clippy::multiple_unsafe_ops_per_block,
    clippy::undocumented_unsafe_blocks
)]
#[cfg(test)]
mod tests {
    use mockall::predicate::{always, eq};

    use super::*;
    use crate::{MockCarmenDb, MockCarmenState};

    #[test]
    fn carmen_rust_open_database_returns_non_null_pointers() {
        let live_impls = [LiveImpl::Memory, LiveImpl::File, LiveImpl::LevelDb];
        let archive_impls = [ArchiveImpl::None, ArchiveImpl::LevelDb, ArchiveImpl::Sqlite];
        for live_impl in live_impls {
            for archive_impl in archive_impls {
                unsafe {
                    let dir = "dir";
                    let mut out_database = std::ptr::null_mut();
                    let result = Carmen_Rust_OpenDatabase(
                        6,
                        live_impl as u8 as u32,
                        archive_impl as u8 as u32,
                        dir.as_ptr() as *const c_char,
                        dir.len() as i32,
                        &mut out_database,
                    );
                    assert_eq!(result, bindings::Result_kResult_Success);
                    assert!(!out_database.is_null());
                    let db_ref = &mut *(out_database as *mut DbWrapper);
                    assert!(!db_ref.inner.is_null());
                    Carmen_Rust_ReleaseDatabase(out_database);
                }
            }
        }
    }

    #[test]
    fn carmen_rust_open_database_checks_that_arguments_are_valid() {
        let dir = "dir";

        let dir_len = dir.len() as i32;
        let dir = dir.as_ptr() as *const c_char;
        let live_impl = 0; // bindings::LiveImpl_kLive_Memory
        let archive_impl = 0; // bindings::ArchiveImpl_kArchive_None
        let mut out_database = std::ptr::null_mut();

        unsafe {
            let result = Carmen_Rust_OpenDatabase(
                6,
                9999, // invalid
                archive_impl,
                dir,
                dir_len,
                &mut out_database,
            );
            assert_eq!(result, bindings::Result_kResult_InvalidArguments);

            let result = Carmen_Rust_OpenDatabase(
                6,
                live_impl,
                9999, // invalid
                dir,
                dir_len,
                &mut out_database,
            );
            assert_eq!(result, bindings::Result_kResult_InvalidArguments);

            let result = Carmen_Rust_OpenDatabase(
                6,
                live_impl,
                archive_impl,
                std::ptr::null(), // invalid
                dir_len,
                &mut out_database,
            );
            assert_eq!(result, bindings::Result_kResult_InvalidArguments);

            let result = Carmen_Rust_OpenDatabase(
                6,
                live_impl,
                archive_impl,
                dir,
                0, // invalid
                &mut out_database,
            );
            assert_eq!(result, bindings::Result_kResult_InvalidArguments);

            let result = Carmen_Rust_OpenDatabase(
                6,
                live_impl,
                archive_impl,
                dir,
                dir_len,
                std::ptr::null_mut(), // invalid
            );
            assert_eq!(result, bindings::Result_kResult_InvalidArguments);
        }
    }

    #[test]
    fn carmen_rust_open_database_returns_error_as_int() {
        unsafe {
            let dir = "dir";
            let mut out_database = std::ptr::null_mut();
            let result = Carmen_Rust_OpenDatabase(
                5, // unsupported schema version
                LiveImpl::Memory as u8 as u32,
                ArchiveImpl::None as u8 as u32,
                dir.as_ptr() as *const c_char,
                dir.len() as i32,
                &mut out_database,
            );
            assert_eq!(result, bindings::Result_kResult_UnsupportedSchema);
            assert!(out_database.is_null());
        }
    }

    #[test]
    fn carmen_rust_release_database_checks_that_arguments_are_valid() {
        let result = unsafe { Carmen_Rust_ReleaseDatabase(std::ptr::null_mut()) };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
    }

    #[test]
    fn carmen_rust_flush_calls_flush_on_carmen_db() {
        create_db_then_call_fn_then_release_db(
            |mock_db| {
                mock_db.expect_flush().returning(|| Ok(()));
            },
            |db| unsafe {
                Carmen_Rust_Flush(db);
            },
        );
    }

    #[test]
    fn carmen_rust_flush_checks_that_arguments_are_valid() {
        let result = unsafe { Carmen_Rust_Flush(std::ptr::null_mut()) };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
    }

    #[test]
    fn carmen_rust_flush_returns_error_as_int() {
        create_db_then_call_fn_then_release_db(
            |mock_db| {
                mock_db
                    .expect_flush()
                    .returning(|| Err(crate::Error::UnsupportedOperation("some error".into())));
            },
            |db| unsafe {
                let result = Carmen_Rust_Flush(db);
                assert_eq!(result, bindings::Result_kResult_UnsupportedOperation);
            },
        );
    }

    #[test]
    fn carmen_rust_close_calls_close_on_carmen_db() {
        create_db_then_call_fn_then_release_db(
            |mock_db| {
                mock_db.expect_close().returning(|| Ok(()));
            },
            |db| unsafe {
                Carmen_Rust_Close(db);
            },
        );
    }

    #[test]
    fn carmen_rust_close_checks_that_arguments_are_valid() {
        let result = unsafe { Carmen_Rust_Close(std::ptr::null_mut()) };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
    }

    #[test]
    fn carmen_rust_close_returns_error_as_int() {
        create_db_then_call_fn_then_release_db(
            |mock_db| {
                mock_db
                    .expect_close()
                    .returning(|| Err(crate::Error::UnsupportedOperation("some error".into())));
            },
            |db| unsafe {
                let result = Carmen_Rust_Close(db);
                assert_eq!(result, bindings::Result_kResult_UnsupportedOperation);
            },
        );
    }

    #[test]
    fn carmen_rust_get_live_state_returns_live_state_from_carmen_db() {
        create_db_then_call_fn_then_release_db(
            move |mock_db| {
                mock_db
                    .expect_get_live_state()
                    .returning(|| Ok(Box::new(MockCarmenState::new())));
            },
            move |db| {
                let mut out_state = std::ptr::null_mut();
                let result = unsafe { Carmen_Rust_GetLiveState(db, &mut out_state) };
                assert_eq!(result, bindings::Result_kResult_Success);
                assert!(!out_state.is_null());
                unsafe { Carmen_Rust_ReleaseState(out_state) };
            },
        );
    }

    #[test]
    fn carmen_rust_get_live_state_checks_that_arguments_are_valid() {
        let result = unsafe {
            Carmen_Rust_GetLiveState(
                std::ptr::null_mut(), // invalid
                &mut std::ptr::null_mut(),
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
        let result = unsafe {
            Carmen_Rust_GetLiveState(
                &mut 0u8 as *mut u8 as *mut c_void,
                std::ptr::null_mut(), // invalid
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
    }

    #[test]
    fn carmen_rust_get_live_state_returns_error_as_int() {
        create_db_then_call_fn_then_release_db(
            |mock_db| {
                mock_db
                    .expect_get_live_state()
                    .returning(|| Err(crate::Error::UnsupportedOperation("some error".into())));
            },
            |db| {
                let mut out_state = std::ptr::null_mut();
                let result = unsafe { Carmen_Rust_GetLiveState(db, &mut out_state) };
                assert_eq!(result, bindings::Result_kResult_UnsupportedOperation);
            },
        );
    }

    #[test]
    fn carmen_rust_get_archive_state_returns_archive_state_from_carmen_db() {
        let block = 1;
        create_db_then_call_fn_then_release_db(
            move |mock_db| {
                mock_db
                    .expect_get_archive_state()
                    .with(eq(block))
                    .returning(|_| Ok(Box::new(MockCarmenState::new())));
            },
            move |db| {
                let mut out_state = std::ptr::null_mut();
                let result = unsafe { Carmen_Rust_GetArchiveState(db, block, &mut out_state) };
                assert_eq!(result, bindings::Result_kResult_Success);
                assert!(!out_state.is_null());
                unsafe { Carmen_Rust_ReleaseState(out_state) };
            },
        );
    }

    #[test]
    fn carmen_rust_get_archive_state_checks_that_arguments_are_valid() {
        let result = unsafe {
            Carmen_Rust_GetArchiveState(
                std::ptr::null_mut(), // invalid
                0,
                &mut std::ptr::null_mut(),
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
        let result = unsafe {
            Carmen_Rust_GetArchiveState(
                &mut 0u8 as *mut u8 as *mut c_void,
                0,
                std::ptr::null_mut(), // invalid
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
    }

    #[test]
    fn carmen_rust_get_archive_state_returns_error_as_int() {
        let block = 1;
        create_db_then_call_fn_then_release_db(
            move |mock_db| {
                mock_db
                    .expect_get_archive_state()
                    .with(eq(block))
                    .returning(|_| Err(crate::Error::UnsupportedOperation("some error".into())));
            },
            move |db| {
                let mut out_state = std::ptr::null_mut();
                let result = unsafe { Carmen_Rust_GetArchiveState(db, block, &mut out_state) };
                assert_eq!(result, bindings::Result_kResult_UnsupportedOperation);
            },
        );
    }

    #[test]
    fn carmen_rust_release_state_checks_that_arguments_are_valid() {
        let result = unsafe { Carmen_Rust_ReleaseState(std::ptr::null_mut()) };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
    }

    #[test]
    fn carmen_rust_account_exists_returns_value_from_carmen_db() {
        let addr = [1; 20];
        let expected_account_state = true;
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_account_exists()
                    .with(eq(addr))
                    .returning(move |_| Ok(expected_account_state));
            },
            move |state| {
                let mut addr = addr;
                let mut out_state = 0u8;
                unsafe {
                    Carmen_Rust_AccountExists(
                        state,
                        &mut addr as *mut Address as *mut c_void,
                        &mut out_state as *mut u8 as *mut c_void,
                    );
                }
                assert_eq!(out_state, expected_account_state as u8);
            },
        );
    }

    #[test]
    fn carmen_rust_account_exists_checks_that_arguments_are_valid() {
        let addr = [1u8; 20];
        let mut out_state = 0u8;
        let result = unsafe {
            Carmen_Rust_AccountExists(
                std::ptr::null_mut(), // invalid
                &addr as *const Address as *mut c_void,
                &mut out_state as *mut u8 as *mut c_void,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_AccountExists(
                &mut 0u8 as *mut u8 as *mut c_void,
                std::ptr::null_mut(), // invalid
                &mut out_state as *mut u8 as *mut c_void,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_AccountExists(
                &mut 0u8 as *mut u8 as *mut c_void,
                &addr as *const Address as *mut c_void,
                std::ptr::null_mut(), // invalid
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
    }

    #[test]
    fn carmen_rust_account_exists_returns_error_as_int() {
        let addr = [1; 20];
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_account_exists()
                    .with(eq(addr))
                    .returning(|_| Err(crate::Error::UnsupportedOperation("some error".into())));
            },
            move |state| {
                let mut addr = addr;
                let mut out_state = 0u8;
                let result = unsafe {
                    Carmen_Rust_AccountExists(
                        state,
                        &mut addr as *mut Address as *mut c_void,
                        &mut out_state as *mut u8 as *mut c_void,
                    )
                };
                assert_eq!(result, bindings::Result_kResult_UnsupportedOperation);
            },
        );
    }

    #[test]
    fn carmen_rust_get_balance_returns_value_from_carmen_db() {
        let addr = [1; 20];
        let expected_balance = [2; 32];
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_get_balance()
                    .with(eq(addr))
                    .returning(move |_| Ok(expected_balance));
            },
            move |state| {
                let mut addr = addr;
                let mut out_balance = [0u8; 32];
                unsafe {
                    Carmen_Rust_GetBalance(
                        state,
                        &mut addr as *mut Address as *mut c_void,
                        &mut out_balance as *mut U256 as *mut c_void,
                    );
                }
                assert_eq!(out_balance, expected_balance);
            },
        );
    }

    #[test]
    fn carmen_rust_get_balance_checks_that_arguments_are_valid() {
        let addr = [0u8; 20];
        let mut out_balance = [0u8; 32];
        let result = unsafe {
            Carmen_Rust_GetBalance(
                std::ptr::null_mut(), // invalid state
                &addr as *const Address as *mut c_void,
                &mut out_balance as *mut U256 as *mut c_void,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_GetBalance(
                &mut 0u8 as *mut u8 as *mut c_void,
                std::ptr::null_mut(), // invalid addr
                &mut out_balance as *mut U256 as *mut c_void,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_GetBalance(
                &mut 0u8 as *mut u8 as *mut c_void,
                &addr as *const Address as *mut c_void,
                std::ptr::null_mut(), // invalid out_balance
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
    }

    #[test]
    fn carmen_rust_get_balance_returns_error_as_int() {
        let addr = [1; 20];
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_get_balance()
                    .with(eq(addr))
                    .returning(|_| Err(crate::Error::UnsupportedOperation("some error".into())));
            },
            move |state| {
                let mut addr = addr;
                let mut out_balance = [0u8; 32];
                let result = unsafe {
                    Carmen_Rust_GetBalance(
                        state,
                        &mut addr as *mut Address as *mut c_void,
                        &mut out_balance as *mut U256 as *mut c_void,
                    )
                };
                assert_eq!(result, bindings::Result_kResult_UnsupportedOperation);
            },
        );
    }

    #[test]
    fn carmen_rust_get_nonce_returns_value_from_carmen_db() {
        let addr = [1u8; 20];
        let expected_nonce = [2; 8];
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_get_nonce()
                    .with(eq(addr))
                    .returning(move |_| Ok(expected_nonce));
            },
            move |state| {
                let mut addr = addr;
                let mut out_nonce = [0; 8];
                unsafe {
                    Carmen_Rust_GetNonce(
                        state,
                        &mut addr as *mut Address as *mut c_void,
                        &mut out_nonce as *mut [u8; 8] as *mut c_void,
                    );
                }
                assert_eq!(out_nonce, expected_nonce);
            },
        );
    }

    #[test]
    fn carmen_rust_get_nonce_checks_that_arguments_are_valid() {
        let addr = [0u8; 20];
        let mut out_nonce: u64 = 0;
        let result = unsafe {
            Carmen_Rust_GetNonce(
                std::ptr::null_mut(), // invalid
                &addr as *const Address as *mut c_void,
                &mut out_nonce as *mut u64 as *mut c_void,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_GetNonce(
                &mut 0u8 as *mut u8 as *mut c_void,
                std::ptr::null_mut(), // invalid
                &mut out_nonce as *mut u64 as *mut c_void,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_GetNonce(
                &mut 0u8 as *mut u8 as *mut c_void,
                &addr as *const Address as *mut c_void,
                std::ptr::null_mut(), // invalid
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
    }

    #[test]
    fn carmen_rust_get_nonce_returns_error_as_int() {
        let addr = [1u8; 20];
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_get_nonce()
                    .with(eq(addr))
                    .returning(|_| Err(crate::Error::UnsupportedOperation("some error".into())));
            },
            move |state| {
                let mut addr = addr;
                let mut out_nonce: u64 = 0;
                let result = unsafe {
                    Carmen_Rust_GetNonce(
                        state,
                        &mut addr as *mut Address as *mut c_void,
                        &mut out_nonce as *mut u64 as *mut c_void,
                    )
                };
                assert_eq!(result, bindings::Result_kResult_UnsupportedOperation);
            },
        );
    }

    #[test]
    fn carmen_rust_get_storage_value_returns_value_from_carmen_db() {
        let addr = [1; 20];
        let key = [2; 32];
        let expected_value = [3; 32];
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_get_storage_value()
                    .with(eq(addr), eq(key))
                    .returning(move |_, _| Ok(expected_value));
            },
            move |state| {
                let mut addr = addr;
                let mut key = key;
                let mut out_value = [0u8; 32];
                unsafe {
                    Carmen_Rust_GetStorageValue(
                        state,
                        &mut addr as *mut Address as *mut c_void,
                        &mut key as *mut Key as *mut c_void,
                        &mut out_value as *mut Value as *mut c_void,
                    );
                }
                assert_eq!(out_value, expected_value);
            },
        );
    }

    #[test]
    fn carmen_rust_get_storage_value_checks_that_arguments_are_valid() {
        let addr = [0u8; 20];
        let key = [0u8; 32];
        let mut out_value = [0u8; 32];
        let result = unsafe {
            Carmen_Rust_GetStorageValue(
                std::ptr::null_mut(), // invalid
                &addr as *const Address as *mut c_void,
                &key as *const Key as *mut c_void,
                &mut out_value as *mut Value as *mut c_void,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_GetStorageValue(
                &mut 0u8 as *mut u8 as *mut c_void,
                std::ptr::null_mut(), // invalid
                &key as *const Key as *mut c_void,
                &mut out_value as *mut Value as *mut c_void,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_GetStorageValue(
                &mut 0u8 as *mut u8 as *mut c_void,
                &addr as *const Address as *mut c_void,
                std::ptr::null_mut(), // invalid
                &mut out_value as *mut Value as *mut c_void,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_GetStorageValue(
                &mut 0u8 as *mut u8 as *mut c_void,
                &addr as *const Address as *mut c_void,
                &key as *const Key as *mut c_void,
                std::ptr::null_mut(), // invalid
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
    }

    #[test]
    fn carmen_rust_get_storage_value_returns_error_as_int() {
        let addr = [1; 20];
        let key = [2; 32];
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_get_storage_value()
                    .with(eq(addr), eq(key))
                    .returning(|_, _| Err(crate::Error::UnsupportedOperation("some error".into())));
            },
            move |state| {
                let mut addr = addr;
                let mut key = key;
                let mut out_value = [0u8; 32];
                let result = unsafe {
                    Carmen_Rust_GetStorageValue(
                        state,
                        &mut addr as *mut Address as *mut c_void,
                        &mut key as *mut Key as *mut c_void,
                        &mut out_value as *mut Value as *mut c_void,
                    )
                };
                assert_eq!(result, bindings::Result_kResult_UnsupportedOperation);
            },
        );
    }

    #[test]
    fn carmen_rust_get_code_returns_code_from_carmen_db() {
        let addr = [1; 20];
        let expected_code = [2, 3, 4];
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_get_code()
                    .with(eq(addr), always())
                    .returning(move |_, code_buf| {
                        for i in 0..expected_code.len() {
                            code_buf[i].write(expected_code[i]);
                        }
                        Ok(expected_code.len())
                    });
            },
            move |state| {
                let mut addr = addr;
                let mut out_code = vec![MaybeUninit::uninit(); MAX_CODE_SIZE];
                let mut out_length = 0;
                unsafe {
                    Carmen_Rust_GetCode(
                        state,
                        &mut addr as *mut Address as *mut c_void,
                        out_code.as_mut_ptr() as *mut c_void,
                        &mut out_length,
                    );
                }
                assert_eq!(out_length, expected_code.len() as u32);
                let code = &out_code[..out_length as usize];
                let code = unsafe { std::mem::transmute::<&[MaybeUninit<u8>], &[u8]>(code) };
                assert_eq!(code, expected_code);
            },
        );
    }

    #[test]
    fn carmen_rust_get_code_checks_that_arguments_are_valid() {
        let addr = [0u8; 20];
        let mut out_code = vec![0u8; MAX_CODE_SIZE];
        let mut out_length = 0u32;
        let result = unsafe {
            Carmen_Rust_GetCode(
                std::ptr::null_mut(), // invalid
                &addr as *const Address as *mut c_void,
                out_code.as_mut_ptr() as *mut c_void,
                &mut out_length,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_GetCode(
                &mut 0u8 as *mut u8 as *mut c_void,
                std::ptr::null_mut(), // invalid
                out_code.as_mut_ptr() as *mut c_void,
                &mut out_length,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_GetCode(
                &mut 0u8 as *mut u8 as *mut c_void,
                &addr as *const Address as *mut c_void,
                std::ptr::null_mut(), // invalid
                &mut out_length,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_GetCode(
                &mut 0u8 as *mut u8 as *mut c_void,
                &addr as *const Address as *mut c_void,
                out_code.as_mut_ptr() as *mut c_void,
                std::ptr::null_mut(), // invalid
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
    }

    #[test]
    fn carmen_rust_get_code_returns_error_as_int() {
        let addr = [1; 20];
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_get_code()
                    .with(eq(addr), always())
                    .returning(|_, _| Err(crate::Error::UnsupportedOperation("some error".into())));
            },
            move |state| {
                let mut addr = addr;
                let mut out_code = vec![0u8; MAX_CODE_SIZE];
                let mut out_length = 0;
                let result = unsafe {
                    Carmen_Rust_GetCode(
                        state,
                        &mut addr as *mut Address as *mut c_void,
                        out_code.as_mut_ptr() as *mut c_void,
                        &mut out_length,
                    )
                };
                assert_eq!(result, bindings::Result_kResult_UnsupportedOperation);
            },
        );
    }

    #[test]
    fn carmen_rust_get_code_hash_returns_hash_from_carmen_db() {
        let addr = [1; 20];
        let expected_hash = [2; 32];
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_get_code_hash()
                    .with(eq(addr))
                    .returning(move |_| Ok(expected_hash));
            },
            move |state| {
                let mut addr = addr;
                let mut out_hash = [0; 32];
                unsafe {
                    Carmen_Rust_GetCodeHash(
                        state,
                        &mut addr as *mut Address as *mut c_void,
                        &mut out_hash as *mut Hash as *mut c_void,
                    );
                }
                assert_eq!(out_hash, expected_hash);
            },
        );
    }

    #[test]
    fn carmen_rust_get_code_hash_checks_that_arguments_are_valid() {
        let addr = [0u8; 20];
        let mut out_hash = [0u8; 32];
        let result = unsafe {
            Carmen_Rust_GetCodeHash(
                std::ptr::null_mut(), // invalid
                &addr as *const Address as *mut c_void,
                &mut out_hash as *mut Hash as *mut c_void,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_GetCodeHash(
                &mut 0u8 as *mut u8 as *mut c_void,
                std::ptr::null_mut(), // invalid
                &mut out_hash as *mut Hash as *mut c_void,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_GetCodeHash(
                &mut 0u8 as *mut u8 as *mut c_void,
                &addr as *const Address as *mut c_void,
                std::ptr::null_mut(), // invalid
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
    }

    #[test]
    fn carmen_rust_get_code_hash_returns_error_as_int() {
        let addr = [1; 20];
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_get_code_hash()
                    .with(eq(addr))
                    .returning(|_| Err(crate::Error::UnsupportedOperation("some error".into())));
            },
            move |state| {
                let mut addr = addr;
                let mut out_hash = [0u8; 32];
                let result = unsafe {
                    Carmen_Rust_GetCodeHash(
                        state,
                        &mut addr as *mut Address as *mut c_void,
                        &mut out_hash as *mut Hash as *mut c_void,
                    )
                };
                assert_eq!(result, bindings::Result_kResult_UnsupportedOperation);
            },
        );
    }

    #[test]
    fn carmen_rust_get_code_size_returns_size_from_carmen_db() {
        let addr = [1; 20];
        let expected_size = 2;
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_get_code_len()
                    .with(eq(addr))
                    .returning(move |_| Ok(expected_size));
            },
            move |state| {
                let mut addr = addr;
                let mut code_size = 0;
                unsafe {
                    Carmen_Rust_GetCodeSize(
                        state,
                        &mut addr as *mut Address as *mut c_void,
                        &mut code_size as *mut u32,
                    );
                }
                assert_eq!(code_size, expected_size);
            },
        );
    }

    #[test]
    fn carmen_rust_get_code_size_checks_that_arguments_are_valid() {
        let addr = [0u8; 20];
        let mut code_size = 0u32;
        let result = unsafe {
            Carmen_Rust_GetCodeSize(
                std::ptr::null_mut(), // invalid
                &addr as *const Address as *mut c_void,
                &mut code_size as *mut u32,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_GetCodeSize(
                &mut 0u8 as *mut u8 as *mut c_void,
                std::ptr::null_mut(), // invalid
                &mut code_size as *mut u32,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_GetCodeSize(
                &mut 0u8 as *mut u8 as *mut c_void,
                &addr as *const Address as *mut c_void,
                std::ptr::null_mut(), // invalid
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
    }

    #[test]
    fn carmen_rust_get_code_size_returns_error_as_int() {
        let addr = [1; 20];
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_get_code_len()
                    .with(eq(addr))
                    .returning(|_| Err(crate::Error::UnsupportedOperation("some error".into())));
            },
            move |state| {
                let mut addr = addr;
                let mut code_size = 0u32;
                let result = unsafe {
                    Carmen_Rust_GetCodeSize(
                        state,
                        &mut addr as *mut Address as *mut c_void,
                        &mut code_size as *mut u32,
                    )
                };
                assert_eq!(result, bindings::Result_kResult_UnsupportedOperation);
            },
        );
    }

    #[test]
    fn carmen_rust_apply_calls_apply_on_carmen_db() {
        let block: u64 = 1;
        let update_data = [0; 25]; // empty update
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_apply_block_update()
                    .with(eq(block), always())
                    .returning(|_, _| Ok(()));
            },
            move |state| {
                let mut update_data = update_data;
                unsafe {
                    Carmen_Rust_Apply(
                        state,
                        block,
                        update_data.as_mut_ptr() as *mut c_void,
                        update_data.len() as u64,
                    );
                }
            },
        );
    }

    #[test]
    fn carmen_rust_apply_checks_that_arguments_are_valid() {
        let block: u64 = 1;
        let mut update_data = [0; 25];
        let result = unsafe {
            Carmen_Rust_Apply(
                std::ptr::null_mut(), // invalid
                block,
                update_data.as_mut_ptr() as *mut c_void,
                update_data.len() as u64,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_Apply(
                &mut 0u8 as *mut u8 as *mut c_void,
                block,
                std::ptr::null_mut(), // invalid
                update_data.len() as u64,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_Apply(
                &mut 0u8 as *mut u8 as *mut c_void,
                block,
                update_data.as_mut_ptr() as *mut c_void,
                0, // invalid
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        create_state_then_call_fn_then_release_state(
            |_| {},
            move |state| {
                let mut update_data = [0; 24];
                let result = unsafe {
                    Carmen_Rust_Apply(
                        state,
                        block,
                        update_data.as_mut_ptr() as *mut c_void, // the update payload is invalid
                        update_data.len() as u64,                // the update payload is invalid
                    )
                };
                assert_eq!(result, bindings::Result_kResult_InvalidArguments);
            },
        );
    }

    #[test]
    fn carmen_rust_apply_returns_error_as_int() {
        let block: u64 = 1;
        let update_data = [0; 25];
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_apply_block_update()
                    .with(eq(block), always())
                    .returning(|_, _| Err(crate::Error::UnsupportedOperation("some error".into())));
            },
            move |state| {
                let mut update_data = update_data;
                let result = unsafe {
                    Carmen_Rust_Apply(
                        state,
                        block,
                        update_data.as_mut_ptr() as *mut c_void,
                        update_data.len() as u64,
                    )
                };
                assert_eq!(result, bindings::Result_kResult_UnsupportedOperation);
            },
        );
    }

    #[test]
    fn carmen_rust_get_hash_returns_hash_from_carmen_db() {
        let expected_hash = [1; 32];
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_get_hash()
                    .returning(move || Ok(expected_hash));
            },
            move |state| {
                let mut out_hash = [0u8; 32];
                unsafe {
                    Carmen_Rust_GetHash(state, &mut out_hash as *mut Hash as *mut c_void);
                }
                assert_eq!(out_hash, expected_hash);
            },
        );
    }

    #[test]
    fn carmen_rust_get_hash_checks_that_arguments_are_valid() {
        let mut out_hash = [0u8; 32];
        let result = unsafe {
            Carmen_Rust_GetHash(
                std::ptr::null_mut(), // invalid
                &mut out_hash as *mut Hash as *mut c_void,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_GetHash(
                &mut 0u8 as *mut u8 as *mut c_void,
                std::ptr::null_mut(), // invalid
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
    }

    #[test]
    fn carmen_rust_get_hash_returns_error_as_int() {
        create_state_then_call_fn_then_release_state(
            move |mock_db| {
                mock_db
                    .expect_get_hash()
                    .returning(|| Err(crate::Error::UnsupportedOperation("some error".into())));
            },
            move |state| {
                let mut out_hash = [0u8; 32];
                let result = unsafe {
                    Carmen_Rust_GetHash(state, &mut out_hash as *mut Hash as *mut c_void)
                };
                assert_eq!(result, bindings::Result_kResult_UnsupportedOperation);
            },
        );
    }

    #[test]
    fn carmen_rust_get_memory_footprint_returns_buffer_and_length() {
        let expected_str = "footprint";
        create_db_then_call_fn_then_release_db(
            move |mock_db| {
                mock_db
                    .expect_get_memory_footprint()
                    .returning(move || Ok(Box::from(expected_str)));
            },
            |db| {
                let mut out_ptr: *mut c_char = std::ptr::null_mut();
                let mut out_len: u64 = 0;
                unsafe {
                    Carmen_Rust_GetMemoryFootprint(db, &mut out_ptr, &mut out_len);
                }
                assert!(!out_ptr.is_null());
                assert_eq!(out_len, expected_str.len() as u64);
                // Clean up
                unsafe {
                    Carmen_Rust_ReleaseMemoryFootprintBuffer(out_ptr, out_len);
                }
            },
        );
    }

    #[test]
    fn carmen_rust_get_memory_footprint_checks_that_arguments_are_valid() {
        let mut out_ptr: *mut c_char = std::ptr::null_mut();
        let mut out_len: u64 = 0;
        let result = unsafe {
            Carmen_Rust_GetMemoryFootprint(
                std::ptr::null_mut(), // invalid
                &mut out_ptr,
                &mut out_len,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_GetMemoryFootprint(
                &mut 0u8 as *mut u8 as *mut c_void,
                std::ptr::null_mut(), // invalid
                &mut out_len,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_GetMemoryFootprint(
                &mut 0u8 as *mut u8 as *mut c_void,
                &mut out_ptr,
                std::ptr::null_mut(), // invalid
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
    }

    #[test]
    fn carmen_rust_get_memory_footprint_returns_error_as_int() {
        create_db_then_call_fn_then_release_db(
            move |mock_db| {
                mock_db
                    .expect_get_memory_footprint()
                    .returning(|| Err(crate::Error::UnsupportedOperation("some error".into())));
            },
            |db| {
                let mut out_ptr: *mut c_char = std::ptr::null_mut();
                let mut out_len: u64 = 0;
                let result =
                    unsafe { Carmen_Rust_GetMemoryFootprint(db, &mut out_ptr, &mut out_len) };
                assert_eq!(result, bindings::Result_kResult_UnsupportedOperation);
            },
        );
    }

    #[test]
    fn carmen_rust_release_memory_footprint_buffer_checks_that_arguments_are_valid() {
        let result = unsafe {
            Carmen_Rust_ReleaseMemoryFootprintBuffer(
                std::ptr::null_mut(), // invalid
                1,
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);

        let result = unsafe {
            Carmen_Rust_ReleaseMemoryFootprintBuffer(
                &mut 0, 0, // invalid
            )
        };
        assert_eq!(result, bindings::Result_kResult_InvalidArguments);
    }

    // `assert_same_signature` is only used at compile time to ensure that the signatures of the
    // exported Rust functions match those defined in the C header file. This tests just runs the
    // function once in tests, so that is shows up in coverage reports as covered.
    // Testing this function with arguments of different types would not compile (that is the whole
    // purpose of this check after all).
    #[test]
    fn assert_same_signature_compiles_if_arguments_have_same_type() {
        assert_same_signature(0, 0);
    }

    #[derive(Clone, Copy)]
    struct ThreadSafePtr(pub *mut c_void);
    unsafe impl Sync for ThreadSafePtr {}

    #[track_caller]
    fn create_db_then_call_fn_then_release_db(
        set_expectation: impl Fn(&mut MockCarmenDb) + 'static,
        call_ffi_fn: impl Fn(*mut c_void) + Sync + 'static,
    ) {
        unsafe {
            let mut mock_db = MockCarmenDb::new();

            set_expectation(&mut mock_db);

            let db_wrapper = DbWrapper::from_db(Box::new(mock_db));
            let db = Box::into_raw(Box::new(db_wrapper)) as *mut c_void;

            let thread_safe_db = ThreadSafePtr(db);
            std::thread::scope(|s| {
                s.spawn(|| {
                    // This is needed so ensure that a reference to ThreadSafePtr is captured and
                    // not a reference to *mut c_void which is not Sync
                    #[allow(clippy::redundant_locals)]
                    let thread_safe_db = thread_safe_db;
                    call_ffi_fn(thread_safe_db.0);
                });
                s.spawn(|| {
                    // This is needed so ensure that a reference to ThreadSafePtr is captured and
                    // not a reference to *mut c_void which is not Sync
                    #[allow(clippy::redundant_locals)]
                    let thread_safe_db = thread_safe_db;
                    call_ffi_fn(thread_safe_db.0);
                });
            });

            Carmen_Rust_ReleaseDatabase(db);
        }
    }

    #[track_caller]
    fn create_state_then_call_fn_then_release_state(
        set_expectation: impl Fn(&mut MockCarmenState) + 'static,
        call_ffi_fn: impl Fn(*mut c_void) + Sync + 'static,
    ) {
        unsafe {
            let mut mock_db = MockCarmenState::new();

            set_expectation(&mut mock_db);

            let state_wrapper = StateWrapper::from_state(Box::new(mock_db));
            let state = Box::into_raw(Box::new(state_wrapper)) as *mut c_void;

            let thread_safe_state = ThreadSafePtr(state);
            std::thread::scope(|s| {
                s.spawn(|| {
                    // This is needed so ensure that a reference to ThreadSafePtr is captured and
                    // not a reference to *mut c_void which is not Sync
                    #[allow(clippy::redundant_locals)]
                    let thread_safe_state = thread_safe_state;
                    call_ffi_fn(thread_safe_state.0);
                });
                s.spawn(|| {
                    // This is needed so ensure that a reference to ThreadSafePtr is captured and
                    // not a reference to *mut c_void which is not Sync
                    #[allow(clippy::redundant_locals)]
                    let thread_safe_state = thread_safe_state;
                    call_ffi_fn(thread_safe_state.0);
                });
            });

            Carmen_Rust_ReleaseState(state);
        }
    }
}
