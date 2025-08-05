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

// An enumeration of supported live state implementations.
enum LiveImpl { kLive_Memory = 0, kLive_File = 1, kLive_LevelDb = 2 };

// An enumeration of supported archive state implementations.
enum ArchiveImpl {
  kArchive_None = 0,
  kArchive_LevelDb = 1,
  kArchive_Sqlite = 2
};

// ------------------------------ Life Cycle ----------------------------------

// Opens a new state object based on the provided implementation maintaining
// its data in the given directory. If the directory does not exist, it is
// created. If it is empty, a new, empty state is initialized. If it contains
// state information, the information is loaded.
//
// The function returns an opaque pointer to a state object that can be used
// with the remaining functions in this file. Ownership is transferred to the
// caller, which is required for releasing it eventually using Release().
// If for some reason the creation of the state instance failed, a nullptr is
// returned.
DUPLICATE_FOR_LANGS(C_State, OpenState(C_Schema schema, enum LiveImpl live_impl,
                                       enum ArchiveImpl archive_impl,
                                       const char* directory, int length));

// Flushes all committed state information to disk to guarantee permanent
// storage. All internally cached modifications is synced to disk.
DUPLICATE_FOR_LANGS(void, Flush(C_State state));

// Closes this state, releasing all IO handles and locks on external resources.
DUPLICATE_FOR_LANGS(void, Close(C_State state));

// Releases a state object, thereby causing its destruction. After releasing it,
// no more operations may be applied on it.
DUPLICATE_FOR_LANGS(void, ReleaseState(C_State state));

// ----------------------------- Archive State --------------------------------

// Creates a state snapshot reflecting the state at the given block height. The
// resulting state must be released and must not outlive the life time of the
// provided state.
DUPLICATE_FOR_LANGS(C_State, GetArchiveState(C_State state, uint64_t block));

// -------------------------------- Balance -----------------------------------

// Retrieves the balance of the given account.
DUPLICATE_FOR_LANGS(void, GetBalance(C_State state, C_Address addr,
                                     C_Balance out_balance));

// --------------------------------- Nonce ------------------------------------

// Retrieves the nonce of the given account.
DUPLICATE_FOR_LANGS(void,
                    GetNonce(C_State state, C_Address addr, C_Nonce out_nonce));

// -------------------------------- Storage -----------------------------------

// Retrieves the value of storage location (addr,key) in the given state.
DUPLICATE_FOR_LANGS(void, GetStorageValue(C_State state, C_Address addr,
                                          C_Key key, C_Value out_value));

// --------------------------------- Code -------------------------------------

// Retrieves the code stored under the given address.
DUPLICATE_FOR_LANGS(void, GetCode(C_State state, C_Address addr,
                                  C_Code out_code, uint32_t* out_length));

// Retrieves the hash of the code stored under the given address.
DUPLICATE_FOR_LANGS(void, GetCodeHash(C_State state, C_Address addr,
                                      C_Hash out_hash));

// Retrieves the code length stored under the given address.
DUPLICATE_FOR_LANGS(void, GetCodeSize(C_State state, C_Address addr,
                                      uint32_t* out_length));

// -------------------------------- Update ------------------------------------

// Applies the provided block update to the maintained state.
DUPLICATE_FOR_LANGS(void, Apply(C_State state, uint64_t block, C_Update update,
                                uint64_t length));

// ------------------------------ Global Hash ---------------------------------

// Retrieves a global state hash of the given state.
DUPLICATE_FOR_LANGS(void, GetHash(C_State state, C_Hash out_hash));

// --------------------------- Memory Footprint -------------------------------

// Retrieves a summary of the used memory. After the call the out variable will
// point to a buffer with a serialized summary that needs to be freed by the
// caller.
DUPLICATE_FOR_LANGS(void, GetMemoryFootprint(C_State state, char** out,
                                             uint64_t* out_length));

// Releases the buffer returned by GetMemoryFootprint.
DUPLICATE_FOR_LANGS(void, ReleaseMemoryFootprintBuffer(char* buf,
                                                       uint64_t buf_length));

#undef DUPLICATE_FOR_LANGS

#if __cplusplus
}
#endif
