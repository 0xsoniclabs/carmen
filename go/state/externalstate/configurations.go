// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package externalstate

import (
	"github.com/0xsoniclabs/carmen/go/database/flat"
	"github.com/0xsoniclabs/carmen/go/state"
)

const (
	VariantCppMemory  state.Variant = "cpp-memory"
	VariantCppFile    state.Variant = "cpp-file"
	VariantCppLevelDb state.Variant = "cpp-ldb"

	VariantRustMemory            state.Variant = "rust-memory"
	VariantRustCrateCryptoMemory state.Variant = "rust-crate-crypto-memory"
	VariantRustFile              state.Variant = "rust-file"
)

func init() {
	supportedArchives := []state.ArchiveType{
		state.NoArchive,
		state.LevelDbArchive,
		state.SqliteArchive,
	}

	factories := map[state.Configuration]state.StateFactory{}

	// Register all configuration options supported by the C++ implementation.
	for schema := state.Schema(1); schema <= state.Schema(3); schema++ {
		for _, archive := range supportedArchives {
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

	// Register all experimental configurations.
	for config, factory := range factories {
		state.RegisterStateFactory(config, factory)

		// Also register flat database variants.
		config := config
		config.Variant += "-flat"
		state.RegisterStateFactory(config, flat.WrapFactory(factory))
	}
}
