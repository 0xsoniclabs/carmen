// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

//go:build carmen_rust || carmen_cpp

package externalstate

//go:generate mockgen -source external_state.go -destination external_state_mocks.go -package externalstate

/*
#cgo CFLAGS: -I${SRCDIR}/../../../cpp
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

	"github.com/0xsoniclabs/carmen/go/common/future"
	"github.com/0xsoniclabs/carmen/go/common/result"
	"github.com/0xsoniclabs/carmen/go/common/witness"

	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/state"

	"github.com/0xsoniclabs/carmen/go/common"
)

const codeCacheSize = 8_000 // ~ 200 MiB of memory for go-side code cache
const codeMaxSize = 25000   // Contract limit is 24577

type externalImpl int

type externalBindings interface {
	OpenDatabase(schema C.uint8_t, liveImpl *C.char, liveImplLen C.int, archiveImpl *C.char, archiveImplLen C.int, dir *C.char, dirLen C.int, outDatabase *unsafe.Pointer) C.enum_Result
	Flush(database unsafe.Pointer) C.enum_Result
	Close(database unsafe.Pointer) C.enum_Result
	ReleaseState(state unsafe.Pointer) C.enum_Result
	GetLiveState(database unsafe.Pointer, outState *unsafe.Pointer) C.enum_Result
	GetArchiveState(database unsafe.Pointer, block C.uint64_t, outState *unsafe.Pointer) C.enum_Result
	GetArchiveBlockHeight(database unsafe.Pointer, outBlock *C.int64_t) C.enum_Result
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

// ExternalState implements the state interface by forwarding all calls to a implementation
// in a foreign language.
type ExternalState struct {
	// A pointer to an owned external object containing the actual state information.
	database unsafe.Pointer
	// A view on the state of the database (either live state or archive state).
	state unsafe.Pointer
	// cache of contract codes
	codeCache *common.LruCache[common.Address, []byte]
	// The foreign language implementation
	bindings externalBindings
}

func newState(impl string, params state.Parameters, bindings externalBindings) (state.State, error) {
	if err := os.MkdirAll(filepath.Join(params.Directory, "live"), 0700); err != nil {
		return nil, err
	}
	dir := C.CString(params.Directory)
	defer C.free(unsafe.Pointer(dir))

	implStr := C.CString(impl)
	defer C.free(unsafe.Pointer(implStr))

	archiveStr := C.CString(string(params.Archive)) // Ensure params.Archive is converted to string
	defer C.free(unsafe.Pointer(archiveStr))

	db := unsafe.Pointer(nil)
	result := bindings.OpenDatabase(C.C_Schema(params.Schema), implStr, C.int(len(impl)), archiveStr, C.int(len(params.Archive)), dir, C.int(len(params.Directory)), &db)
	if result != C.kResult_Success {
		return nil, fmt.Errorf("failed to create external database instance for parameters %v (error code %v)", params, result)
	}
	if db == unsafe.Pointer(nil) {
		return nil, fmt.Errorf("%w: failed to create external database instance for parameters %v", state.UnsupportedConfiguration, params)
	}

	live := unsafe.Pointer(nil)
	result = bindings.GetLiveState(db, &live)
	if result != C.kResult_Success {
		bindings.Close(db)
		return nil, fmt.Errorf("failed to create external live state instance for parameters %v (error code %v)", params, result)
	}
	if live == unsafe.Pointer(nil) {
		bindings.Close(db)
		return nil, fmt.Errorf("%w: failed to create external live state instance for parameters %v", state.UnsupportedConfiguration, params)
	}

	return state.WrapIntoSyncedState(&ExternalState{
		database:  db,
		state:     live,
		codeCache: common.NewLruCache[common.Address, []byte](codeCacheSize),
		bindings:  bindings,
	}), nil
}

func (s *ExternalState) CreateAccount(address common.Address) error {
	update := common.Update{}
	_, err := s.Apply(0, update)
	return err
}

func (s *ExternalState) Exists(address common.Address) (bool, error) {
	var res common.AccountState
	result := s.bindings.AccountExists(s.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&res))
	if result != C.kResult_Success {
		return false, fmt.Errorf("failed to check if account exists (error code %v)", result)
	}
	return res == common.Exists, nil
}

func (s *ExternalState) DeleteAccount(address common.Address) error {
	update := common.Update{}
	update.AppendDeleteAccount(address)
	_, err := s.Apply(0, update)
	return err
}

func (s *ExternalState) GetBalance(address common.Address) (amount.Amount, error) {
	var balance [amount.BytesLength]byte
	result := s.bindings.GetBalance(s.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&balance[0]))
	if result != C.kResult_Success {
		return amount.Amount{}, fmt.Errorf("failed to get balance for address %s (error code %v)", address, result)
	}
	return amount.NewFromBytes(balance[:]...), nil
}

func (s *ExternalState) SetBalance(address common.Address, balance amount.Amount) error {
	update := common.Update{}
	update.AppendBalanceUpdate(address, balance)
	_, err := s.Apply(0, update)
	return err
}

func (s *ExternalState) GetNonce(address common.Address) (common.Nonce, error) {
	var nonce common.Nonce
	result := s.bindings.GetNonce(s.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&nonce[0]))
	if result != C.kResult_Success {
		return common.Nonce{}, fmt.Errorf("failed to get nonce for address %s (error code %v)", address, result)
	}
	return nonce, nil
}

func (s *ExternalState) SetNonce(address common.Address, nonce common.Nonce) error {
	update := common.Update{}
	update.AppendNonceUpdate(address, nonce)
	_, err := s.Apply(0, update)
	return err
}

func (s *ExternalState) GetStorage(address common.Address, key common.Key) (common.Value, error) {
	var value common.Value
	result := s.bindings.GetStorageValue(s.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&key[0]), unsafe.Pointer(&value[0]))
	if result != C.kResult_Success {
		return common.Value{}, fmt.Errorf("failed to get storage value for address %s and key %s (error code %v)", address, key, result)
	}
	return value, nil
}

func (s *ExternalState) SetStorage(address common.Address, key common.Key, value common.Value) error {
	update := common.Update{}
	update.AppendSlotUpdate(address, key, value)
	_, err := s.Apply(0, update)
	return err
}

func (s *ExternalState) GetCode(address common.Address) ([]byte, error) {
	// Try to obtain the code from the cache
	code, exists := s.codeCache.Get(address)
	if exists {
		return code, nil
	}

	// Load the code from the external state
	code = make([]byte, codeMaxSize)
	var size C.uint32_t = codeMaxSize
	result := s.bindings.GetCode(s.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&code[0]), &size)
	if result != C.kResult_Success {
		return nil, fmt.Errorf("failed to get code for address %s (error code %v)", address, result)
	}
	if size >= codeMaxSize {
		return nil, fmt.Errorf("unable to load contract exceeding maximum capacity of %d", codeMaxSize)
	}
	if size > 0 {
		code = code[0:size]
	} else {
		code = nil
	}
	s.codeCache.Set(address, code)
	return code, nil
}

func (s *ExternalState) SetCode(address common.Address, code []byte) error {
	update := common.Update{}
	update.AppendCodeUpdate(address, code)
	_, err := s.Apply(0, update)
	return err
}

func (s *ExternalState) GetCodeHash(address common.Address) (common.Hash, error) {
	var hash common.Hash
	result := s.bindings.GetCodeHash(s.state, unsafe.Pointer(&address[0]), unsafe.Pointer(&hash[0]))
	if result != C.kResult_Success {
		return common.Hash{}, fmt.Errorf("failed to get code hash for address %s (error code %v)", address, result)
	}
	return hash, nil
}

func (s *ExternalState) GetCodeSize(address common.Address) (int, error) {
	var size C.uint32_t
	result := s.bindings.GetCodeSize(s.state, unsafe.Pointer(&address[0]), &size)
	if result != C.kResult_Success {
		return 0, fmt.Errorf("failed to get code size for address %s (error code %v)", address, result)
	}
	return int(size), nil
}

func (s *ExternalState) GetHash() (common.Hash, error) {
	return s.GetCommitment().Await().Get()
}

func (s *ExternalState) GetCommitment() future.Future[result.Result[common.Hash]] {
	var hash common.Hash
	res := s.bindings.GetHash(s.state, unsafe.Pointer(&hash[0]))
	if res != C.kResult_Success {
		return future.Immediate(result.Err[common.Hash](fmt.Errorf("failed to get commitment (error code %v)", res)))
	}
	return future.Immediate(result.Ok(hash))
}

func (s *ExternalState) Apply(block uint64, update common.Update) (<-chan error, error) {
	if err := update.Normalize(); err != nil {
		return nil, err
	}
	data := update.ToBytes()
	dataPtr := unsafe.Pointer(&data[0])
	result := s.bindings.Apply(s.state, C.uint64_t(block), dataPtr, C.uint64_t(len(data)))
	if result != C.kResult_Success {
		return nil, fmt.Errorf("failed to apply update at block %d (error code %v)", block, result)
	}
	// Apply code changes to Go-sided code cache.
	for _, change := range update.Codes {
		s.codeCache.Set(change.Account, change.Code)
	}
	return nil, nil
}

func (s *ExternalState) Flush() error {
	result := s.bindings.Flush(s.database)
	if result != C.kResult_Success {
		return fmt.Errorf("failed to flush state (error code %v)", result)
	}
	return nil
}

func (s *ExternalState) Close() error {
	if s.state != nil {
		result := s.bindings.ReleaseState(s.state)
		if result != C.kResult_Success {
			return fmt.Errorf("failed to release external state (error code %v)", result)
		}
		s.state = nil
	}
	if s.database != nil {
		result := s.bindings.Close(s.database)
		if result != C.kResult_Success {
			return fmt.Errorf("failed to close external database (error code %v)", result)
		}
		s.database = nil
	}
	return nil
}

func (s *ExternalState) GetMemoryFootprint() *common.MemoryFootprint {
	if s.database == nil {
		return common.NewMemoryFootprint(unsafe.Sizeof(*s))
	}

	// Fetch footprint data from the external implementation.
	var buffer *C.char
	var size C.uint64_t
	result := s.bindings.GetMemoryFootprint(s.database, &buffer, &size)
	if result != C.kResult_Success {
		res := common.NewMemoryFootprint(0)
		res.SetNote(fmt.Sprintf("failed to get external memory footprint (error code %v)", result))
		log.Printf("failed to get external memory footprint (error code %v)", result)
		return res
	}
	defer func() {
		result := s.bindings.ReleaseMemoryFootprintBuffer(buffer, size)
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

func (s *ExternalState) GetArchiveState(block uint64) (state.State, error) {
	state := unsafe.Pointer(nil)
	result := s.bindings.GetArchiveState(s.database, C.uint64_t(block), &state)
	if result != C.kResult_Success {
		return nil, fmt.Errorf("failed to get archive state for block %d (error code %v)", block, result)
	}
	return &ExternalState{
		state:     state,
		codeCache: common.NewLruCache[common.Address, []byte](codeCacheSize),
		bindings:  s.bindings,
	}, nil
}

func (s *ExternalState) GetArchiveBlockHeight() (uint64, bool, error) {
	var blockHeight C.int64_t
	result := s.bindings.GetArchiveBlockHeight(s.database, &blockHeight)
	if result != C.kResult_Success {
		return 0, false, state.NoArchiveError
	}
	if blockHeight < 0 {
		return 0, true, nil
	}
	return uint64(blockHeight), false, nil
}

func (s *ExternalState) Check() error {
	// TODO: implement, see https://github.com/Fantom-foundation/Carmen/issues/313
	return nil
}

func (s *ExternalState) CreateWitnessProof(address common.Address, keys ...common.Key) (witness.Proof, error) {
	panic("not implemented")
}

func (s *ExternalState) HasEmptyStorage(addr common.Address) (bool, error) {
	// S3 schema is based on directly indexed files without ability to iterate
	// over a dataset. For this reason, this method is implemented as purely
	// returning a constant value all the time.
	return true, nil
}

func (s *ExternalState) Export(context.Context, io.Writer) (common.Hash, error) {
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
