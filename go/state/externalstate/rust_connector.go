// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

//go:build carmen_rust

package externalstate

/*
#cgo LDFLAGS: -L${SRCDIR}/../../../rust/target/release -lcarmen_rust
#cgo LDFLAGS: -Wl,-rpath,${SRCDIR}/../../../rust/target/release
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
	VariantRustMemory            state.Variant = "rust-memory"
	VariantRustCrateCryptoMemory state.Variant = "rust-crate-crypto-memory"
	VariantRustFile              state.Variant = "rust-file"
)

type rustBindings struct {
}

func (r rustBindings) OpenDatabase(schema C.uint8_t, liveImpl *C.char, liveImplLen C.int, archiveImpl *C.char, archiveImplLen C.int, dir *C.char, dirLen C.int, outDatabase *unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_OpenDatabase(schema, liveImpl, liveImplLen, archiveImpl, archiveImplLen, dir, dirLen, outDatabase)
}

func (r rustBindings) Flush(database unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_Flush(database)
}

func (r rustBindings) Close(database unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_Close(database)
}

func (r rustBindings) ReleaseState(state unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_ReleaseState(state)
}

func (r rustBindings) GetLiveState(database unsafe.Pointer, outState *unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_GetLiveState(database, outState)
}

func (r rustBindings) GetArchiveState(database unsafe.Pointer, block C.uint64_t, outState *unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_GetArchiveState(database, block, outState)
}

func (r rustBindings) GetArchiveBlockHeight(database unsafe.Pointer, outHeight *C.int64_t) C.enum_Result {
	return C.Carmen_Rust_GetArchiveBlockHeight(database, outHeight)
}

func (r rustBindings) AccountExists(state unsafe.Pointer, address unsafe.Pointer, outExists unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_AccountExists(state, address, outExists)
}

func (r rustBindings) GetBalance(state unsafe.Pointer, address unsafe.Pointer, outBalance unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_GetBalance(state, address, outBalance)
}

func (r rustBindings) GetNonce(state unsafe.Pointer, address unsafe.Pointer, outNonce unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_GetNonce(state, address, outNonce)
}

func (r rustBindings) GetStorageValue(state unsafe.Pointer, address unsafe.Pointer, key unsafe.Pointer, outValue unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_GetStorageValue(state, address, key, outValue)
}

func (r rustBindings) GetCode(state unsafe.Pointer, address unsafe.Pointer, outCode unsafe.Pointer, outSize *C.uint32_t) C.enum_Result {
	return C.Carmen_Rust_GetCode(state, address, outCode, outSize)
}

func (r rustBindings) GetCodeHash(state unsafe.Pointer, address unsafe.Pointer, outHash unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_GetCodeHash(state, address, outHash)
}

func (r rustBindings) GetCodeSize(state unsafe.Pointer, address unsafe.Pointer, outSize *C.uint32_t) C.enum_Result {
	return C.Carmen_Rust_GetCodeSize(state, address, outSize)
}

func (r rustBindings) Apply(state unsafe.Pointer, block C.uint64_t, update unsafe.Pointer, updateLength C.uint64_t) C.enum_Result {
	return C.Carmen_Rust_Apply(state, block, update, updateLength)
}

func (r rustBindings) GetHash(state unsafe.Pointer, outHash unsafe.Pointer) C.enum_Result {
	return C.Carmen_Rust_GetHash(state, outHash)
}

func (r rustBindings) GetMemoryFootprint(database unsafe.Pointer, outBuffer **C.char, outSize *C.uint64_t) C.enum_Result {
	return C.Carmen_Rust_GetMemoryFootprint(database, outBuffer, outSize)
}

func (r rustBindings) ReleaseMemoryFootprintBuffer(buffer *C.char, size C.uint64_t) C.enum_Result {
	return C.Carmen_Rust_ReleaseMemoryFootprintBuffer(buffer, size)
}

func newRustInMemoryState(params state.Parameters) (state.State, error) {
	return newState("memory", params, rustBindings{})
}

func newRustCrateCryptoInMemoryState(params state.Parameters) (state.State, error) {
	return newState("crate-crypto-memory", params, rustBindings{})
}

func newRustFileBasedState(params state.Parameters) (state.State, error) {
	return newState("file", params, rustBindings{})
}

func init() {
	factories := map[state.Configuration]state.StateFactory{}

	// Add all configuration options supported by the Rust implementation.
	factories[state.Configuration{
		Variant: VariantRustMemory,
		Schema:  6,
		Archive: state.NoArchive,
	}] = newRustInMemoryState

	factories[state.Configuration{
		Variant: VariantRustCrateCryptoMemory,
		Schema:  6,
		Archive: state.NoArchive,
	}] = newRustCrateCryptoInMemoryState

	factories[state.Configuration{
		Variant: VariantRustFile,
		Schema:  6,
		Archive: state.NoArchive,
	}] = newRustFileBasedState

	factories[state.Configuration{
		Variant: VariantRustFile,
		Schema:  6,
		Archive: "file",
	}] = newRustFileBasedState

	// Register all experimental configurations.
	for config, factory := range factories {
		state.RegisterStateFactory(config, factory)

		// Also register flat database variants.
		config := config
		config.Variant += "-flat"
		state.RegisterStateFactory(config, flat.WrapFactory(factory))
	}
}
