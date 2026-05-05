// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

//go:build carmen_cpp

package externalstate

//go:generate sh ../../lib/build_libcarmen.sh

/*
#cgo LDFLAGS: -L${SRCDIR}/../../lib -lcarmen
#cgo LDFLAGS: -Wl,-rpath,${SRCDIR}/../../lib
#cgo CFLAGS: -I${SRCDIR}/../../../cpp
#include <stdlib.h>
#include "state/c_state.h"
*/
import "C"
import (
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/database/flat"
	"github.com/0xsoniclabs/carmen/go/state"
)

const (
	VariantCppMemory  state.Variant = "cpp-memory"
	VariantCppFile    state.Variant = "cpp-file"
	VariantCppLevelDb state.Variant = "cpp-ldb"
)

type cppBindings struct {
}

func (c cppBindings) OpenDatabase(schema C.uint8_t, liveImpl *C.char, liveImplLen C.int, archiveImpl *C.char, archiveImplLen C.int, dir *C.char, dirLen C.int, outDatabase *unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_OpenDatabase(schema, liveImpl, liveImplLen, archiveImpl, archiveImplLen, dir, dirLen, outDatabase)
}

func (c cppBindings) Flush(database unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_Flush(database)
}

func (c cppBindings) Close(database unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_Close(database)
}

func (c cppBindings) ReleaseState(state unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_ReleaseState(state)
}

func (c cppBindings) GetLiveState(database unsafe.Pointer, outState *unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_GetLiveState(database, outState)
}

func (c cppBindings) GetArchiveState(database unsafe.Pointer, block C.uint64_t, outState *unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_GetArchiveState(database, block, outState)
}

func (c cppBindings) GetArchiveBlockHeight(database unsafe.Pointer, outHeight *C.int64_t) C.enum_Result {
	return C.kResult_UnsupportedOperation
}

func (c cppBindings) AccountExists(state unsafe.Pointer, address unsafe.Pointer, outExists unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_AccountExists(state, address, outExists)
}

func (c cppBindings) GetBalance(state unsafe.Pointer, address unsafe.Pointer, outBalance unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_GetBalance(state, address, outBalance)
}

func (c cppBindings) GetNonce(state unsafe.Pointer, address unsafe.Pointer, outNonce unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_GetNonce(state, address, outNonce)
}

func (c cppBindings) GetStorageValue(state unsafe.Pointer, address unsafe.Pointer, key unsafe.Pointer, outValue unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_GetStorageValue(state, address, key, outValue)
}

func (c cppBindings) GetCode(state unsafe.Pointer, address unsafe.Pointer, outCode unsafe.Pointer, outSize *C.uint32_t) C.enum_Result {
	return C.Carmen_Cpp_GetCode(state, address, outCode, outSize)
}

func (c cppBindings) GetCodeHash(state unsafe.Pointer, address unsafe.Pointer, outHash unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_GetCodeHash(state, address, outHash)
}

func (c cppBindings) GetCodeSize(state unsafe.Pointer, address unsafe.Pointer, outSize *C.uint32_t) C.enum_Result {
	return C.Carmen_Cpp_GetCodeSize(state, address, outSize)
}

func (c cppBindings) Apply(state unsafe.Pointer, block C.uint64_t, update unsafe.Pointer, updateLength C.uint64_t) C.enum_Result {
	return C.Carmen_Cpp_Apply(state, block, update, updateLength)
}

func (c cppBindings) GetHash(state unsafe.Pointer, outHash unsafe.Pointer) C.enum_Result {
	return C.Carmen_Cpp_GetHash(state, outHash)
}

func (c cppBindings) GetMemoryFootprint(database unsafe.Pointer, outBuffer **C.char, outSize *C.uint64_t) C.enum_Result {
	return C.Carmen_Cpp_GetMemoryFootprint(database, outBuffer, outSize)
}

func (c cppBindings) ReleaseMemoryFootprintBuffer(buffer *C.char, size C.uint64_t) C.enum_Result {
	return C.Carmen_Cpp_ReleaseMemoryFootprintBuffer(buffer, size)
}

func newCppInMemoryState(params state.Parameters) (state.State, error) {
	return newState("memory", params, cppBindings{})
}

func newCppFileBasedState(params state.Parameters) (state.State, error) {
	return newState("file", params, cppBindings{})
}

func newCppLevelDbBasedState(params state.Parameters) (state.State, error) {
	return newState("ldb", params, cppBindings{})
}

func initCpp() {
	factories := map[state.Configuration]state.StateFactory{}

	// Add all configuration options supported by the C++ implementation.
	supportedCppArchives := []state.ArchiveType{
		state.NoArchive,
		state.LevelDbArchive,
		state.SqliteArchive,
	}

	for schema := state.Schema(1); schema <= state.Schema(3); schema++ {
		for _, archive := range supportedCppArchives {
			factories[state.Configuration{
				Variant: VariantCppMemory,
				Schema:  schema,
				Archive: archive,
			}] = newCppInMemoryState
			factories[state.Configuration{
				Variant: VariantCppFile,
				Schema:  schema,
				Archive: archive,
			}] = newCppFileBasedState
			factories[state.Configuration{
				Variant: VariantCppLevelDb,
				Schema:  schema,
				Archive: archive,
			}] = newCppLevelDbBasedState
		}
	}

	// Register all experimental configurations.
	for config, factory := range factories {
		state.RegisterStateFactory(config, factory)

		// Also register flat database variants.
		config := config
		config.Variant += "-flat"
		state.RegisterStateFactory(config, flat.WrapFactory(factory))
	}
}
