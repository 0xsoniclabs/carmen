// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package cppstate

//go:generate sh ../../lib/build_libcarmen.sh

/*
#cgo CFLAGS: -I${SRCDIR}/../../../cpp
#cgo LDFLAGS: -L${SRCDIR}/../../lib -lcarmen
#cgo LDFLAGS: -Wl,-rpath,${SRCDIR}/../../lib
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

// CppState implements the state interface by forwarding all calls to a C++
// based implementation.
type CppState struct {
	// A pointer to an owned C++ object containing the actual state information.
	database unsafe.Pointer
	// A view on the state of the database (either live state or archive state).
	state unsafe.Pointer
	// cache of contract codes
	codeCache *common.LruCache[common.Address, []byte]
}

func newState(impl C.enum_LiveImpl, params state.Parameters) (state.State, error) {
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

	db := unsafe.Pointer(nil)
	result := C.Carmen_Cpp_OpenDatabase(C.C_Schema(params.Schema), impl, C.enum_ArchiveImpl(archive), dir, C.int(len(params.Directory)), &db)
	if result != 0 || db == unsafe.Pointer(nil) {
		return nil, fmt.Errorf("%w: failed to create C++ database instance for parameters %v (error code %v)", state.UnsupportedConfiguration, params, result)
	}

	live := unsafe.Pointer(nil)
	result = C.Carmen_Cpp_GetLiveState(db, &live)
	if result != 0 || live == unsafe.Pointer(nil) {
		C.Carmen_Cpp_ReleaseDatabase(db)
		return nil, fmt.Errorf("%w: failed to create C++ live state instance for parameters %v (error code %v)", state.UnsupportedConfiguration, params, result)
	}

	return state.WrapIntoSyncedState(&CppState{
		database:  db,
		state:     live,
		codeCache: common.NewLruCache[common.Address, []byte](CodeCacheSize),
	}), nil
}

func newInMemoryState(params state.Parameters) (state.State, error) {
	return newState(C.kLive_Memory, params)
}

func newFileBasedState(params state.Parameters) (state.State, error) {
	return newState(C.kLive_File, params)
}

func newLevelDbBasedState(params state.Parameters) (state.State, error) {
	return newState(C.kLive_LevelDb, params)
}

func (cs *CppState) CreateAccount(address common.Address) error {
	update := common.Update{}
	update.AppendCreateAccount(address)
	return cs.Apply(0, update)
}

func (cs *CppState) Exists(address common.Address) (bool, error) {
	var res common.AccountState
	result := C.Carmen_Cpp_AccountExists(cs.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&res))
	if result != 0 {
		return false, fmt.Errorf("failed to check if account exists (error code %v)", result)
	}
	return res == common.Exists, nil
}

func (cs *CppState) DeleteAccount(address common.Address) error {
	update := common.Update{}
	update.AppendDeleteAccount(address)
	return cs.Apply(0, update)
}

func (cs *CppState) GetBalance(address common.Address) (amount.Amount, error) {
	var balance [amount.BytesLength]byte
	result := C.Carmen_Cpp_GetBalance(cs.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&balance[0]))
	if result != 0 {
		return amount.Amount{}, fmt.Errorf("failed to get balance for address %s (error code %v)", address, result)
	}
	return amount.NewFromBytes(balance[:]...), nil
}

func (cs *CppState) SetBalance(address common.Address, balance amount.Amount) error {
	update := common.Update{}
	update.AppendBalanceUpdate(address, balance)
	return cs.Apply(0, update)
}

func (cs *CppState) GetNonce(address common.Address) (common.Nonce, error) {
	var nonce common.Nonce
	result := C.Carmen_Cpp_GetNonce(cs.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&nonce[0]))
	if result != 0 {
		return common.Nonce{}, fmt.Errorf("failed to get nonce for address %s (error code %v)", address, result)
	}
	return nonce, nil
}

func (cs *CppState) SetNonce(address common.Address, nonce common.Nonce) error {
	update := common.Update{}
	update.AppendNonceUpdate(address, nonce)
	return cs.Apply(0, update)
}

func (cs *CppState) GetStorage(address common.Address, key common.Key) (common.Value, error) {
	var value common.Value
	result := C.Carmen_Cpp_GetStorageValue(cs.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&key[0]), unsafe.Pointer(&value[0]))
	if result != 0 {

		return common.Value{}, fmt.Errorf("failed to get storage value for address %s and key %s (error code %v)", address, key, result)
	}
	return value, nil
}

func (cs *CppState) SetStorage(address common.Address, key common.Key, value common.Value) error {
	update := common.Update{}
	update.AppendSlotUpdate(address, key, value)
	return cs.Apply(0, update)
}

func (cs *CppState) GetCode(address common.Address) ([]byte, error) {
	// Try to obtain the code from the cache
	code, exists := cs.codeCache.Get(address)
	if exists {
		return code, nil
	}

	// Load the code from C++
	code = make([]byte, CodeMaxSize)
	var size C.uint32_t = CodeMaxSize
	result := C.Carmen_Cpp_GetCode(cs.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&code[0]), &size)
	if result != 0 {
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
	cs.codeCache.Set(address, code)
	return code, nil
}

func (cs *CppState) SetCode(address common.Address, code []byte) error {
	update := common.Update{}
	update.AppendCodeUpdate(address, code)
	return cs.Apply(0, update)
}

func (cs *CppState) GetCodeHash(address common.Address) (common.Hash, error) {
	var hash common.Hash
	result := C.Carmen_Cpp_GetCodeHash(cs.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&hash[0]))
	if result != 0 {
		return common.Hash{}, fmt.Errorf("failed to get code hash for address %s (error code %v)", address, result)
	}
	return hash, nil
}

func (cs *CppState) GetCodeSize(address common.Address) (int, error) {
	var size C.uint32_t
	result := C.Carmen_Cpp_GetCodeSize(cs.state, unsafe.Pointer(&address[0]), &size)
	if result != 0 {
		return 0, fmt.Errorf("failed to get code size for address %s (error code %v)", address, result)
	}
	return int(size), nil
}

func (cs *CppState) GetHash() (common.Hash, error) {
	var hash common.Hash
	result := C.Carmen_Cpp_GetHash(cs.state, unsafe.Pointer(&hash[0]))
	if result != 0 {
		return common.Hash{}, fmt.Errorf("failed to get state hash (error code %v)", result)
	}
	return hash, nil
}

func (cs *CppState) Apply(block uint64, update common.Update) error {
	if update.IsEmpty() {
		return nil
	}
	if err := update.Normalize(); err != nil {
		return err
	}
	data := update.ToBytes()
	dataPtr := unsafe.Pointer(&data[0])
	result := C.Carmen_Cpp_Apply(cs.state, C.uint64_t(block), dataPtr, C.uint64_t(len(data)))
	if result != 0 {
		return fmt.Errorf("failed to apply update at block %d (error code %v)", block, result)
	}
	// Apply code changes to Go-sided code cache.
	for _, change := range update.Codes {
		cs.codeCache.Set(change.Account, change.Code)
	}
	return nil
}

func (cs *CppState) Flush() error {
	result := C.Carmen_Cpp_Flush(cs.database)
	if result != 0 {
		return fmt.Errorf("failed to flush state (error code %v)", result)
	}
	return nil
}

func (cs *CppState) Close() error {
	if cs.state != nil {
		result := C.Carmen_Cpp_ReleaseState(cs.state)
		if result != 0 {
			return fmt.Errorf("failed to release C++ state (error code %v)", result)
		}
		cs.state = nil
		result = C.Carmen_Cpp_Close(cs.database)
		if result != 0 {
			return fmt.Errorf("failed to close C++ database (error code %v)", result)
		}
		result = C.Carmen_Cpp_ReleaseDatabase(cs.database)
		if result != 0 {
			return fmt.Errorf("failed to release C++ database (error code %v)", result)
		}
		cs.database = nil
	}
	return nil
}

func (s *CppState) GetProof() (backend.Proof, error) {
	return nil, backend.ErrSnapshotNotSupported
}

func (s *CppState) CreateSnapshot() (backend.Snapshot, error) {
	return nil, backend.ErrSnapshotNotSupported
}

func (s *CppState) Restore(data backend.SnapshotData) error {
	return backend.ErrSnapshotNotSupported
}

func (s *CppState) GetSnapshotVerifier(metadata []byte) (backend.SnapshotVerifier, error) {
	return nil, backend.ErrSnapshotNotSupported
}

func (cs *CppState) GetMemoryFootprint() *common.MemoryFootprint {
	if cs.database == nil {
		return common.NewMemoryFootprint(unsafe.Sizeof(*cs))
	}

	// Fetch footprint data from C++.
	var buffer *C.char
	var size C.uint64_t
	result := C.Carmen_Cpp_GetMemoryFootprint(cs.database, &buffer, &size)
	if result != 0 {
		res := common.NewMemoryFootprint(0)
		res.SetNote(fmt.Sprintf("Failed to get C++ memory footprint (error code %v)", result))
		log.Printf("Failed to get C++ memory footprint (error code %v)", result)
		return res
	}
	defer func() {
		result := C.Carmen_Cpp_ReleaseMemoryFootprintBuffer(buffer, size)
		if result != 0 {
			log.Printf("Failed to release memory footprint buffer (error code %v)", result)
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

	res.AddChild("goCodeCache", cs.codeCache.GetDynamicMemoryFootprint(func(code []byte) uintptr {
		return uintptr(cap(code)) // memory consumed by the code slice
	}))
	return res
}

func (cs *CppState) GetArchiveState(block uint64) (state.State, error) {
	state := unsafe.Pointer(nil)
	result := C.Carmen_Cpp_GetArchiveState(cs.database, C.uint64_t(block), &state)
	if result != 0 {
		return nil, fmt.Errorf("failed to get archive state for block %d (error code %v)", block, result)
	}
	return &CppState{
		state:     state,
		codeCache: common.NewLruCache[common.Address, []byte](CodeCacheSize),
	}, nil
}

func (cs *CppState) GetArchiveBlockHeight() (uint64, bool, error) {
	return 0, false, state.NoArchiveError
}

func (cs *CppState) Check() error {
	// TODO: implement, see https://github.com/Fantom-foundation/Carmen/issues/313
	return nil
}

func (cs *CppState) CreateWitnessProof(address common.Address, keys ...common.Key) (witness.Proof, error) {
	panic("not implemented")
}

func (cs *CppState) HasEmptyStorage(addr common.Address) (bool, error) {
	// S3 schema is based on directly indexed files without ability to iterate
	// over a dataset. For this reason, this method is implemented as purely
	// returning a constant value all the time.
	return true, nil
}

func (cs *CppState) Export(context.Context, io.Writer) (common.Hash, error) {
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
