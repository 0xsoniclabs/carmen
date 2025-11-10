// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

// This header file defines a C interface for manipulating the world state.
// It is intended to be used to bridge the Go/C++ boundary.

#include <stdint.h>

#if __cplusplus
extern "C" {
#endif

// Macro to duplicate function declarations for Rust and CPP
#define DUPLICATE_FOR_LANGS(ret_type, fn) \
  ret_type Carmen_Rust_##fn;              \
  ret_type Carmen_Cpp_##fn;

// The C interface for the storage system is designed to minimize overhead
// between Go and C. All data is passed as pointers and the memory management
// responsibility is generally left to the Go side. Parameters may serve as in
// or out parameters. Future extensions may utilize the return value as an error
// indicator.

// The following macro definitions provide syntactic sugar for type-erased
// pointers used in the interface definitions below. Their main purpose is to
// increase readability, not to enforce any type constraints.

#define C_Database void*
#define C_State void*
#define C_Schema uint8_t

#define C_bool uint8_t
#define C_Address void*
#define C_Key void*
#define C_Value void*
#define C_Balance void*
#define C_Nonce void*
#define C_Code void*
#define C_Update void*
#define C_Hash void*
#define C_AccountState void*

enum Result {
  kResult_Success = 0,
  kResult_CorruptedDatabase = 1,
  kResult_IOError = 2,
  kResult_UnsupportedOperation = 3,
  kResult_UnsupportedSchema = 4,
  kResult_UnsupportedImplementation = 5,
  kResult_InvalidArguments = 6,
  kResult_InternalError = 7,
};

// ------------------------------ Life Cycle ----------------------------------

// Opens a new database object based on the provided implementation maintaining
// its data in the given directory. If the directory does not exist, it is
// created. If it is empty, a new, empty database is initialized. If it contains
// information, the information is loaded.
//
// The function writes an opaque pointer to a database object into the output
// parameter `out_database`, that can be used with the remaining functions in
// this file. Ownership is transferred to the caller, which is required for
// releasing it eventually using ReleaseDatabase(). If for some reason the
// creation of the state instance failed, an error is returned and the output
// pointer is not written to.
DUPLICATE_FOR_LANGS(enum Result,
                    OpenDatabase(C_Schema schema, const char* live_impl,
                                 int live_impl_length, const char* archive_impl,
                                 int archive_impl_length, const char* directory,
                                 int directory_length,
                                 C_Database* out_database));

// Flushes all committed database information to disk to guarantee permanent
// storage. All internally cached modifications is synced to disk.
DUPLICATE_FOR_LANGS(enum Result, Flush(C_Database database));

// Closes this database, releasing all IO handles and locks on external
// resources.
DUPLICATE_FOR_LANGS(enum Result, Close(C_Database database));

// Releases a database object, thereby causing its destruction. After releasing
// it, no more operations may be applied on it.
DUPLICATE_FOR_LANGS(enum Result, ReleaseDatabase(C_Database database));

// ------------------------- Live and Archive State ---------------------------

// Releases a state object, thereby causing its destruction. After releasing it,
// no more operations may be applied on it.
DUPLICATE_FOR_LANGS(enum Result, ReleaseState(C_State state));

// Writes a handle to the live state of the database into the output parameter
// `out_state`. The resulting state must be released and must not outlive the
// life time of the provided database.
DUPLICATE_FOR_LANGS(enum Result,
                    GetLiveState(C_Database database, C_State* out_state));

// Writes a handle to an archive state reflecting the state at the given block
// height into the output parameter `out_state`. The resulting state must be
// released and must not outlive the life time of the provided database. This
// function will return an error if called with a database that was opened with
// ArchiveImpl "none".
DUPLICATE_FOR_LANGS(enum Result,
                    GetArchiveState(C_Database database, uint64_t block,
                                    C_State* out_state));

// ------------------------------- Accounts -----------------------------------

// Checks if the given account exists.
DUPLICATE_FOR_LANGS(enum Result, AccountExists(C_State state, C_Address addr,
                                               C_AccountState out_state));

// -------------------------------- Balance -----------------------------------

// Retrieves the balance of the given account.
DUPLICATE_FOR_LANGS(enum Result, GetBalance(C_State state, C_Address addr,
                                            C_Balance out_balance));

// --------------------------------- Nonce ------------------------------------

// Retrieves the nonce of the given account.
DUPLICATE_FOR_LANGS(enum Result,
                    GetNonce(C_State state, C_Address addr, C_Nonce out_nonce));

// -------------------------------- Storage -----------------------------------

// Retrieves the value of storage location (addr,key) in the given state.
DUPLICATE_FOR_LANGS(enum Result, GetStorageValue(C_State state, C_Address addr,
                                                 C_Key key, C_Value out_value));

// --------------------------------- Code -------------------------------------

// Retrieves the code stored under the given address.
DUPLICATE_FOR_LANGS(enum Result,
                    GetCode(C_State state, C_Address addr, C_Code out_code,
                            uint32_t* out_length));

// Retrieves the hash of the code stored under the given address.
DUPLICATE_FOR_LANGS(enum Result, GetCodeHash(C_State state, C_Address addr,
                                             C_Hash out_hash));

// Retrieves the code length stored under the given address.
DUPLICATE_FOR_LANGS(enum Result, GetCodeSize(C_State state, C_Address addr,
                                             uint32_t* out_length));

// -------------------------------- Update ------------------------------------

// Applies the provided block update to the live state. This function will
// return an error if called with an archive state.
DUPLICATE_FOR_LANGS(enum Result, Apply(C_State state, uint64_t block,
                                       C_Update update, uint64_t length));

// ------------------------------ Global Hash ---------------------------------

// Retrieves a global state hash of the given state.
DUPLICATE_FOR_LANGS(enum Result, GetHash(C_State state, C_Hash out_hash));

// --------------------------- Memory Footprint -------------------------------

// Retrieves a summary of the used memory. After the call the out variable will
// point to a buffer with a serialized summary that needs to be freed by the
// caller.
DUPLICATE_FOR_LANGS(enum Result, GetMemoryFootprint(C_Database db, char** out,
                                                    uint64_t* out_length));

// Releases the buffer returned by GetMemoryFootprint.
DUPLICATE_FOR_LANGS(enum Result,
                    ReleaseMemoryFootprintBuffer(char* buf,
                                                 uint64_t buf_length));

#undef DUPLICATE_FOR_LANGS

#if __cplusplus
}
#endif
