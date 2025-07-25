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

// CppState implements the state interface by forwarding all calls to a C++ based implementation.
type CppState struct {
	// A pointer to an owned C++ object containing the actual state information.
	state unsafe.Pointer
	// cache of contract codes
	codeCache *common.LruCache[common.Address, []byte]
}

func newState(impl C.enum_StateImpl, params state.Parameters) (state.State, error) {
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

	st := C.Carmen_Cpp_OpenState(C.C_Schema(params.Schema), impl, C.enum_StateImpl(archive), dir, C.int(len(params.Directory)))
	if st == unsafe.Pointer(nil) {
		return nil, fmt.Errorf("%w: failed to create C++ state instance for parameters %v", state.UnsupportedConfiguration, params)
	}

	return state.WrapIntoSyncedState(&CppState{
		state:     st,
		codeCache: common.NewLruCache[common.Address, []byte](CodeCacheSize),
	}), nil
}

func newInMemoryState(params state.Parameters) (state.State, error) {
	return newState(C.kState_Memory, params)
}

func newFileBasedState(params state.Parameters) (state.State, error) {
	return newState(C.kState_File, params)
}

func newLevelDbBasedState(params state.Parameters) (state.State, error) {
	return newState(C.kState_LevelDb, params)
}

func (cs *CppState) CreateAccount(address common.Address) error {
	update := common.Update{}
	update.AppendCreateAccount(address)
	return cs.Apply(0, update)
}

func (cs *CppState) Exists(address common.Address) (bool, error) {
	var res common.AccountState
	C.Carmen_Cpp_GetAccountState(cs.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&res))
	return res == common.Exists, nil
}

func (cs *CppState) DeleteAccount(address common.Address) error {
	update := common.Update{}
	update.AppendDeleteAccount(address)
	return cs.Apply(0, update)
}

func (cs *CppState) GetBalance(address common.Address) (amount.Amount, error) {
	var balance [amount.BytesLength]byte
	C.Carmen_Cpp_GetBalance(cs.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&balance[0]))
	return amount.NewFromBytes(balance[:]...), nil
}

func (cs *CppState) SetBalance(address common.Address, balance amount.Amount) error {
	update := common.Update{}
	update.AppendBalanceUpdate(address, balance)
	return cs.Apply(0, update)
}

func (cs *CppState) GetNonce(address common.Address) (common.Nonce, error) {
	var nonce common.Nonce
	C.Carmen_Cpp_GetNonce(cs.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&nonce[0]))
	return nonce, nil
}

func (cs *CppState) SetNonce(address common.Address, nonce common.Nonce) error {
	update := common.Update{}
	update.AppendNonceUpdate(address, nonce)
	return cs.Apply(0, update)
}

func (cs *CppState) GetStorage(address common.Address, key common.Key) (common.Value, error) {
	var value common.Value
	C.Carmen_Cpp_GetStorageValue(cs.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&key[0]), unsafe.Pointer(&value[0]))
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
	C.Carmen_Cpp_GetCode(cs.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&code[0]), &size)
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
	C.Carmen_Cpp_GetCodeHash(cs.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&hash[0]))
	return hash, nil
}

func (cs *CppState) GetCodeSize(address common.Address) (int, error) {
	var size C.uint32_t
	C.Carmen_Cpp_GetCodeSize(cs.state, unsafe.Pointer(&address[0]), &size)
	return int(size), nil
}

func (cs *CppState) GetHash() (common.Hash, error) {
	var hash common.Hash
	C.Carmen_Cpp_GetHash(cs.state, unsafe.Pointer(&hash[0]))
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
	C.Carmen_Cpp_Apply(cs.state, C.uint64_t(block), dataPtr, C.uint64_t(len(data)))
	// Apply code changes to Go-sided code cache.
	for _, change := range update.Codes {
		cs.codeCache.Set(change.Account, change.Code)
	}
	return nil
}

func (cs *CppState) Flush() error {
	C.Carmen_Cpp_Flush(cs.state)
	return nil
}

func (cs *CppState) Close() error {
	if cs.state != nil {
		C.Carmen_Cpp_Close(cs.state)
		C.Carmen_Cpp_ReleaseState(cs.state)
		cs.state = nil
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
	if cs.state == nil {
		return common.NewMemoryFootprint(unsafe.Sizeof(*cs))
	}

	// Fetch footprint data from C++.
	var buffer *C.char
	var size C.uint64_t
	C.Carmen_Cpp_GetMemoryFootprint(cs.state, &buffer, &size)
	defer func() {
		C.Carmen_Cpp_ReleaseMemoryFootprintBuffer(buffer, size)
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
	return &CppState{
		state:     C.Carmen_Cpp_GetArchiveState(cs.state, C.uint64_t(block)),
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
