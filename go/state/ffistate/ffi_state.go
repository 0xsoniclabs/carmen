// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package ffistate

//go:generate sh ../../lib/build_libcarmen.sh

/*
#cgo CFLAGS: -I${SRCDIR}/../../../cpp
#cgo LDFLAGS: -L${SRCDIR}/../../lib -lcarmen -L${SRCDIR}/../../../rust/target/release -lcarmen_rust
#cgo LDFLAGS: -Wl,-rpath,${SRCDIR}/../../lib -Wl,-rpath,${SRCDIR}/../../../rust/target/release
#include <stdlib.h>
#include "state/c_state.h"
*/
import "C"
import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/common/witness"

	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/state"

	"github.com/0xsoniclabs/carmen/go/backend"
	"github.com/0xsoniclabs/carmen/go/common"
)

const CodeCacheSize = 8_000 // ~ 200 MiB of memory for go-side code cache
const CodeMaxSize = 25000   // Contract limit is 24577

type FFIType int

const (
	FFITypeCpp FFIType = iota
	FFITypeRust
)

type FFIInterface interface {
	OpenDatabase(schema C.uint8_t, liveImpl C.enum_LiveImpl, archiveImpl C.enum_ArchiveImpl, dir *C.char, dirLen C.int, outDatabase *unsafe.Pointer) C.enum_Result
	Flush(database unsafe.Pointer) C.enum_Result
	Close(database unsafe.Pointer) C.enum_Result
	ReleaseDatabase(database unsafe.Pointer) C.enum_Result
	ReleaseState(state unsafe.Pointer) C.enum_Result
	GetLiveState(database unsafe.Pointer, outState *unsafe.Pointer) C.enum_Result
	GetArchiveState(database unsafe.Pointer, block C.uint64_t, outState *unsafe.Pointer) C.enum_Result
	AccountExists(state unsafe.Pointer, address unsafe.Pointer, outExists unsafe.Pointer) C.enum_Result
	GetBalance(state unsafe.Pointer, address unsafe.Pointer, outBalance unsafe.Pointer) C.enum_Result
	GetNonce(state unsafe.Pointer, address unsafe.Pointer, outNonce unsafe.Pointer) C.enum_Result
	GetStorageValue(state unsafe.Pointer, address unsafe.Pointer, key unsafe.Pointer, outValue unsafe.Pointer) C.enum_Result
	GetCode(state unsafe.Pointer, address unsafe.Pointer, outCode unsafe.Pointer, outSize *C.uint32_t) C.enum_Result
	GetCodeHash(state unsafe.Pointer, address unsafe.Pointer, outHash unsafe.Pointer) C.enum_Result
	GetCodeSize(state unsafe.Pointer, address unsafe.Pointer, outSize *C.uint32_t) C.enum_Result
	Apply(state unsafe.Pointer, block C.uint64_t, update unsafe.Pointer, updateLength C.uint64_t) C.enum_Result
	GetHash(state unsafe.Pointer, outHash unsafe.Pointer) C.enum_Result
	GetMemoryFootprint(database unsafe.Pointer, outBuffer **C.char, outSize *C.uint64_t) C.enum_Result
	ReleaseMemoryFootprintBuffer(buffer *C.char, size C.uint64_t) C.enum_Result
}

type RustInterface struct {
}

func (r RustInterface) OpenDatabase(schema C.uint8_t, liveImpl C.enum_LiveImpl, archiveImpl C.enum_ArchiveImpl, dir *C.char, dirLen C.int, outDatabase *unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_OpenDatabase(schema, liveImpl, archiveImpl, dir, dirLen, outDatabase)
}

func (r RustInterface) Flush(database unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_Flush(database)
}

func (r RustInterface) Close(database unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_Close(database)
}

func (r RustInterface) ReleaseDatabase(database unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_ReleaseDatabase(database)
}

func (r RustInterface) ReleaseState(state unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_ReleaseState(state)
}

func (r RustInterface) GetLiveState(database unsafe.Pointer, outState *unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_GetLiveState(database, outState)
}

func (r RustInterface) GetArchiveState(database unsafe.Pointer, block C.uint64_t, outState *unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_GetArchiveState(database, block, outState)
}

func (r RustInterface) AccountExists(state unsafe.Pointer, address unsafe.Pointer, outExists unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_AccountExists(state, address, outExists)
}

func (r RustInterface) GetBalance(state unsafe.Pointer, address unsafe.Pointer, outBalance unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_GetBalance(state, address, outBalance)
}

func (r RustInterface) GetNonce(state unsafe.Pointer, address unsafe.Pointer, outNonce unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_GetNonce(state, address, outNonce)
}

func (r RustInterface) GetStorageValue(state unsafe.Pointer, address unsafe.Pointer, key unsafe.Pointer, outValue unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_GetStorageValue(state, address, key, outValue)
}

func (r RustInterface) GetCode(state unsafe.Pointer, address unsafe.Pointer, outCode unsafe.Pointer, outSize *C.uint32_t) C.enum_Result {
	return C.Carmen_Rust_GetCode(state, address, outCode, outSize)
}

func (r RustInterface) GetCodeHash(state unsafe.Pointer, address unsafe.Pointer, outHash unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_GetCodeHash(state, address, outHash)
}

func (r RustInterface) GetCodeSize(state unsafe.Pointer, address unsafe.Pointer, outSize *C.uint32_t) C.enum_Result {
	return C.Carmen_Rust_GetCodeSize(state, address, outSize)
}

func (r RustInterface) Apply(state unsafe.Pointer, block C.uint64_t, update unsafe.Pointer, updateLength C.uint64_t) C.enum_Result {
	return C.Carmen_Rust_Apply(state, block, update, updateLength)
}

func (r RustInterface) GetHash(state unsafe.Pointer, outHash unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_GetHash(state, outHash)
}

func (r RustInterface) GetMemoryFootprint(database unsafe.Pointer, outBuffer **C.char, outSize *C.uint64_t) C.enum_Result {
	return C.Carmen_Rust_GetMemoryFootprint(database, outBuffer, outSize)
}

func (r RustInterface) ReleaseMemoryFootprintBuffer(buffer *C.char, size C.uint64_t) C.enum_Result {
	return C.Carmen_Rust_ReleaseMemoryFootprintBuffer(buffer, size)
}

type CppInterface struct {
}

func (c CppInterface) OpenDatabase(schema C.uint8_t, liveImpl C.enum_LiveImpl, archiveImpl C.enum_ArchiveImpl, dir *C.char, dirLen C.int, outDatabase *unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_OpenDatabase(schema, liveImpl, archiveImpl, dir, dirLen, outDatabase)
}

func (c CppInterface) Flush(database unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_Flush(database)
}

func (c CppInterface) Close(database unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_Close(database)
}

func (c CppInterface) ReleaseDatabase(database unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_ReleaseDatabase(database)
}

func (c CppInterface) ReleaseState(state unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_ReleaseState(state)
}

func (c CppInterface) GetLiveState(database unsafe.Pointer, outState *unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_GetLiveState(database, outState)
}

func (c CppInterface) GetArchiveState(database unsafe.Pointer, block C.uint64_t, outState *unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_GetArchiveState(database, block, outState)
}

func (c CppInterface) AccountExists(state unsafe.Pointer, address unsafe.Pointer, outExists unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_AccountExists(state, address, outExists)
}

func (c CppInterface) GetBalance(state unsafe.Pointer, address unsafe.Pointer, outBalance unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_GetBalance(state, address, outBalance)
}

func (c CppInterface) GetNonce(state unsafe.Pointer, address unsafe.Pointer, outNonce unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_GetNonce(state, address, outNonce)
}

func (c CppInterface) GetStorageValue(state unsafe.Pointer, address unsafe.Pointer, key unsafe.Pointer, outValue unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_GetStorageValue(state, address, key, outValue)
}

func (c CppInterface) GetCode(state unsafe.Pointer, address unsafe.Pointer, outCode unsafe.Pointer, outSize *C.uint32_t) C.enum_Result {
	return C.Carmen_Cpp_GetCode(state, address, outCode, outSize)
}

func (c CppInterface) GetCodeHash(state unsafe.Pointer, address unsafe.Pointer, outHash unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_GetCodeHash(state, address, outHash)
}

func (c CppInterface) GetCodeSize(state unsafe.Pointer, address unsafe.Pointer, outSize *C.uint32_t) C.enum_Result {
	return C.Carmen_Cpp_GetCodeSize(state, address, outSize)
}

func (c CppInterface) Apply(state unsafe.Pointer, block C.uint64_t, update unsafe.Pointer, updateLength C.uint64_t) C.enum_Result {
	return C.Carmen_Cpp_Apply(state, block, update, updateLength)
}

func (c CppInterface) GetHash(state unsafe.Pointer, outHash unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_GetHash(state, outHash)
}

func (c CppInterface) GetMemoryFootprint(database unsafe.Pointer, outBuffer **C.char, outSize *C.uint64_t) C.enum_Result {
	return C.Carmen_Cpp_GetMemoryFootprint(database, outBuffer, outSize)
}

func (c CppInterface) ReleaseMemoryFootprintBuffer(buffer *C.char, size C.uint64_t) C.enum_Result {
	return C.Carmen_Cpp_ReleaseMemoryFootprintBuffer(buffer, size)
}

// FFIState implements the state interface by forwarding all calls to a implementation
// in a foreign language.
type FFIState struct {
	// A pointer to an owned C++ object containing the actual state information.
	database unsafe.Pointer
	// A view on the state of the database (either live state or archive state).
	state unsafe.Pointer
	// cache of contract codes
	codeCache *common.LruCache[common.Address, []byte]
	// The foreign language implementation
	ffi FFIInterface
}

func newState(impl C.enum_LiveImpl, params state.Parameters, ffiType FFIType) (state.State, error) {
	if err := os.MkdirAll(filepath.Join(params.Directory, "live"), 0700); err != nil {
		return nil, err
	}
	dir := C.CString(params.Directory)
	defer C.free(unsafe.Pointer(dir))

	archive := 0
	switch params.Archive {
	case state.ArchiveType(""), state.NoArchive:
		archive = 0
	case state.LevelDbArchive:
		archive = 1
	case state.SqliteArchive:
		archive = 2
	default:
		return nil, fmt.Errorf("%w: unsupported archive type %v", state.UnsupportedConfiguration, params.Archive)
	}

	var ffi FFIInterface

	switch ffiType {
	case FFITypeCpp:
		ffi = CppInterface{}
	case FFITypeRust:
		ffi = RustInterface{}
	default:
		panic("unsupported FFI type")
	}

	db := unsafe.Pointer(nil)
	result := ffi.OpenDatabase(C.C_Schema(params.Schema), impl, C.enum_ArchiveImpl(archive), dir, C.int(len(params.Directory)), &db)
	if result != C.kResult_Success {
		return nil, fmt.Errorf("failed to create FFI database instance for parameters %v (error code %v)", params, result)
	}
	if db == unsafe.Pointer(nil) {
		return nil, fmt.Errorf("%w: failed to create FFI database instance for parameters %v", state.UnsupportedConfiguration, params)
	}

	live := unsafe.Pointer(nil)
	result = ffi.GetLiveState(db, &live)
	if result != C.kResult_Success {
		C.Carmen_Cpp_ReleaseDatabase(db)
		return nil, fmt.Errorf("failed to create FFI live state instance for parameters %v (error code %v)", params, result)
	}
	if live == unsafe.Pointer(nil) {
		ffi.ReleaseDatabase(db)
		return nil, fmt.Errorf("%w: failed to create FFI live state instance for parameters %v", state.UnsupportedConfiguration, params)
	}

	return state.WrapIntoSyncedState(&FFIState{
		database:  db,
		state:     live,
		codeCache: common.NewLruCache[common.Address, []byte](CodeCacheSize),
		ffi:       ffi,
	}), nil
}

func newRustInMemoryState(params state.Parameters) (state.State, error) {
	return newState(C.kLive_Memory, params, FFITypeRust)
}

func newCppInMemoryState(params state.Parameters) (state.State, error) {
	return newState(C.kLive_Memory, params, FFITypeCpp)
}

func newCppFileBasedState(params state.Parameters) (state.State, error) {
	return newState(C.kLive_File, params, FFITypeCpp)
}

func newCppLevelDbBasedState(params state.Parameters) (state.State, error) {
	return newState(C.kLive_LevelDb, params, FFITypeCpp)
}

func (s *FFIState) CreateAccount(address common.Address) error {
	update := common.Update{}
	return s.Apply(0, update)
}

func (s *FFIState) Exists(address common.Address) (bool, error) {
	var res common.AccountState
	result := s.ffi.AccountExists(s.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&res))
	if result != C.kResult_Success {
		return false, fmt.Errorf("failed to check if account exists (error code %v)", result)
	}
	return res == common.Exists, nil
}

func (s *FFIState) DeleteAccount(address common.Address) error {
	update := common.Update{}
	update.AppendDeleteAccount(address)
	return s.Apply(0, update)
}

func (s *FFIState) GetBalance(address common.Address) (amount.Amount, error) {
	var balance [amount.BytesLength]byte
	result := s.ffi.GetBalance(s.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&balance[0]))
	if result != C.kResult_Success {
		return amount.Amount{}, fmt.Errorf("failed to get balance for address %s (error code %v)", address, result)
	}
	return amount.NewFromBytes(balance[:]...), nil
}

func (s *FFIState) SetBalance(address common.Address, balance amount.Amount) error {
	update := common.Update{}
	update.AppendBalanceUpdate(address, balance)
	return s.Apply(0, update)
}

func (s *FFIState) GetNonce(address common.Address) (common.Nonce, error) {
	var nonce common.Nonce
	result := s.ffi.GetNonce(s.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&nonce[0]))
	if result != C.kResult_Success {
		return common.Nonce{}, fmt.Errorf("failed to get nonce for address %s (error code %v)", address, result)
	}
	return nonce, nil
}

func (s *FFIState) SetNonce(address common.Address, nonce common.Nonce) error {
	update := common.Update{}
	update.AppendNonceUpdate(address, nonce)
	return s.Apply(0, update)
}

func (s *FFIState) GetStorage(address common.Address, key common.Key) (common.Value, error) {
	var value common.Value
	result := s.ffi.GetStorageValue(s.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&key[0]), unsafe.Pointer(&value[0]))
	if result != C.kResult_Success {
		return common.Value{}, fmt.Errorf("failed to get storage value for address %s and key %s (error code %v)", address, key, result)
	}
	return value, nil
}

func (s *FFIState) SetStorage(address common.Address, key common.Key, value common.Value) error {
	update := common.Update{}
	update.AppendSlotUpdate(address, key, value)
	return s.Apply(0, update)
}

func (s *FFIState) GetCode(address common.Address) ([]byte, error) {
	// Try to obtain the code from the cache
	code, exists := s.codeCache.Get(address)
	if exists {
		return code, nil
	}

	// Load the code from C++
	code = make([]byte, CodeMaxSize)
	var size C.uint32_t = CodeMaxSize
	result := s.ffi.GetCode(s.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&code[0]), &size)
	if result != C.kResult_Success {
		return nil, fmt.Errorf("failed to get code for address %s (error code %v)", address, result)
	}
	if size >= CodeMaxSize {
		return nil, fmt.Errorf("unable to load contract exceeding maximum capacity of %d", CodeMaxSize)
	}
	if size > 0 {
		code = code[0:size]
	} else {
		code = nil
	}
	s.codeCache.Set(address, code)
	return code, nil
}

func (s *FFIState) SetCode(address common.Address, code []byte) error {
	update := common.Update{}
	update.AppendCodeUpdate(address, code)
	return s.Apply(0, update)
}

func (s *FFIState) GetCodeHash(address common.Address) (common.Hash, error) {
	var hash common.Hash
	result := s.ffi.GetCodeHash(s.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&hash[0]))
	if result != C.kResult_Success {
		return common.Hash{}, fmt.Errorf("failed to get code hash for address %s (error code %v)", address, result)
	}
	return hash, nil
}

func (s *FFIState) GetCodeSize(address common.Address) (int, error) {
	var size C.uint32_t
	result := s.ffi.GetCodeSize(s.state, unsafe.Pointer(&address[0]), &size)
	if result != C.kResult_Success {
		return 0, fmt.Errorf("failed to get code size for address %s (error code %v)", address, result)
	}
	return int(size), nil
}

func (s *FFIState) GetHash() (common.Hash, error) {
	var hash common.Hash
	result := s.ffi.GetHash(s.state, unsafe.Pointer(&hash[0]))
	if result != C.kResult_Success {
		return common.Hash{}, fmt.Errorf("failed to get state hash (error code %v)", result)
	}
	return hash, nil
}

func (s *FFIState) Apply(block uint64, update common.Update) error {
	if update.IsEmpty() {
		return nil
	}
	if err := update.Normalize(); err != nil {
		return err
	}
	data := update.ToBytes()
	dataPtr := unsafe.Pointer(&data[0])
	result := s.ffi.Apply(s.state, C.uint64_t(block), dataPtr, C.uint64_t(len(data)))
	if result != C.kResult_Success {
		return fmt.Errorf("failed to apply update at block %d (error code %v)", block, result)
	}
	// Apply code changes to Go-sided code cache.
	for _, change := range update.Codes {
		s.codeCache.Set(change.Account, change.Code)
	}
	return nil
}

func (s *FFIState) Flush() error {
	result := s.ffi.Flush(s.database)
	if result != C.kResult_Success {
		return fmt.Errorf("failed to flush state (error code %v)", result)
	}
	return nil
}

func (s *FFIState) Close() error {
	if s.state != nil {
		result := s.ffi.ReleaseState(s.state)
		if result != C.kResult_Success {
			return fmt.Errorf("failed to release C++ state (error code %v)", result)
		}
		s.state = nil
		result = s.ffi.Close(s.database)
		if result != C.kResult_Success {
			return fmt.Errorf("failed to close C++ database (error code %v)", result)
		}
		result = s.ffi.ReleaseDatabase(s.database)
		if result != C.kResult_Success {
			return fmt.Errorf("failed to release C++ database (error code %v)", result)
		}
		s.database = nil
	}
	return nil
}

func (s *FFIState) GetProof() (backend.Proof, error) {
	return nil, backend.ErrSnapshotNotSupported
}

func (s *FFIState) CreateSnapshot() (backend.Snapshot, error) {
	return nil, backend.ErrSnapshotNotSupported
}

func (s *FFIState) Restore(data backend.SnapshotData) error {
	return backend.ErrSnapshotNotSupported
}

func (s *FFIState) GetSnapshotVerifier(metadata []byte) (backend.SnapshotVerifier, error) {
	return nil, backend.ErrSnapshotNotSupported
}

func (s *FFIState) GetMemoryFootprint() *common.MemoryFootprint {
	if s.database == nil {
		return common.NewMemoryFootprint(unsafe.Sizeof(*s))
	}

	// Fetch footprint data from C++.
	var buffer *C.char
	var size C.uint64_t
	result := s.ffi.GetMemoryFootprint(s.database, &buffer, &size)
	if result != C.kResult_Success {
		res := common.NewMemoryFootprint(0)
		res.SetNote(fmt.Sprintf("failed to get C++ memory footprint (error code %v)", result))
		log.Printf("failed to get C++ memory footprint (error code %v)", result)
		return res
	}
	defer func() {
		result := s.ffi.ReleaseMemoryFootprintBuffer(buffer, size)
		if result != C.kResult_Success {
			log.Printf("failed to release memory footprint buffer (error code %v)", result)
		}
	}()

	data := C.GoBytes(unsafe.Pointer(buffer), C.int(size))

	// Use an index map mapping object IDs to memory footprints to facilitate
	// sharing of sub-structures.
	index := map[objectId]*common.MemoryFootprint{}
	res, unusedData := parseCMemoryFootprint(data, index)
	if len(unusedData) != 0 {
		panic("Failed to consume all of the provided footprint data")
	}

	res.AddChild("goCodeCache", s.codeCache.GetDynamicMemoryFootprint(func(code []byte) uintptr {
		return uintptr(cap(code)) // memory consumed by the code slice
	}))
	return res
}

func (s *FFIState) GetArchiveState(block uint64) (state.State, error) {
	state := unsafe.Pointer(nil)
	result := s.ffi.GetArchiveState(s.database, C.uint64_t(block), &state)
	if result != C.kResult_Success {
		return nil, fmt.Errorf("failed to get archive state for block %d (error code %v)", block, result)
	}
	return &FFIState{
		state:     state,
		codeCache: common.NewLruCache[common.Address, []byte](CodeCacheSize),
	}, nil
}

func (s *FFIState) GetArchiveBlockHeight() (uint64, bool, error) {
	return 0, false, state.NoArchiveError
}

func (s *FFIState) Check() error {
	// TODO: implement, see https://github.com/Fantom-foundation/Carmen/issues/313
	return nil
}

func (s *FFIState) CreateWitnessProof(address common.Address, keys ...common.Key) (witness.Proof, error) {
	panic("not implemented")
}

func (s *FFIState) HasEmptyStorage(addr common.Address) (bool, error) {
	// S3 schema is based on directly indexed files without ability to iterate
	// over a dataset. For this reason, this method is implemented as purely
	// returning a constant value all the time.
	return true, nil
}

func (s *FFIState) Export(context.Context, io.Writer) (common.Hash, error) {
	return common.Hash{}, state.ExportNotSupported
}

type objectId struct {
	obj_loc, obj_type uint64
}

func (o *objectId) isUnique() bool {
	return o.obj_loc == 0 && o.obj_type == 0
}

func readUint32(data []byte) (uint32, []byte) {
	return binary.LittleEndian.Uint32(data[:4]), data[4:]
}

func readUint64(data []byte) (uint64, []byte) {
	return binary.LittleEndian.Uint64(data[:8]), data[8:]
}

func readObjectId(data []byte) (objectId, []byte) {
	obj_loc, data := readUint64(data)
	obj_type, data := readUint64(data)
	return objectId{obj_loc, obj_type}, data
}

func readString(data []byte) (string, []byte) {
	length, data := readUint32(data)
	return string(data[:length]), data[length:]
}

func parseCMemoryFootprint(data []byte, index map[objectId]*common.MemoryFootprint) (*common.MemoryFootprint, []byte) {
	// 1) read object ID
	objId, data := readObjectId(data)

	// 2) read memory usage
	memUsage, data := readUint64(data)
	res := common.NewMemoryFootprint(uintptr(memUsage))

	// 3) read number of sub-components
	num_components, data := readUint32(data)

	// 4) read sub-components
	for i := 0; i < int(num_components); i++ {
		var label string
		label, data = readString(data)
		var child *common.MemoryFootprint
		child, data = parseCMemoryFootprint(data, index)
		res.AddChild(label, child)
	}

	// Unique objects are not cached since they shall not be reused.
	if objId.isUnique() {
		return res, data
	}

	// Return representative instance based on object ID.
	if represent, exists := index[objId]; exists {
		return represent, data
	}
	index[objId] = res
	return res, data
}
