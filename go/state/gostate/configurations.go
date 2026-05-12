// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package gostate

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/0xsoniclabs/carmen/go/database/flat"
	"github.com/0xsoniclabs/carmen/go/database/mpt"

	"github.com/0xsoniclabs/carmen/go/backend/archive"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/syndtr/goleveldb/leveldb/opt"
)

const HashTreeFactor = 32

// CacheCapacity is the size of the cache expressed as the number of cached keys
const CacheCapacity = 1 << 20 // 2 ^ 20 keys -> 32MB for 32-bytes keys

// TransactBufferMB is the size of buffer before the transaction is flushed expressed in MBs
const TransactBufferMB = 128 * opt.MiB

// PoolSize is the maximum amount of data pages loaded in memory for the paged file store
const PoolSize = 100000

// CodeHashGroupSize represents the number of codes grouped together in depots to form one leaf node of the hash tree.
const CodeHashGroupSize = 4

const defaultSchema = state.Schema(1)

const (
	VariantGoMemory         state.Variant = "go-memory"
	VariantGoFile           state.Variant = "go-file"
	VariantGoFileNoCache    state.Variant = "go-file-nocache"
	VariantGoLevelDb        state.Variant = "go-ldb"
	VariantGoLevelDbNoCache state.Variant = "go-ldb-nocache"
)

func init() {
	generallySupportedArchives := []state.ArchiveType{
		state.NoArchive,
	}

	// Register all configuration options supported by the Go implementation.
	// TODO [cleanup]: break this up on a per schema basis
	for schema := state.Schema(4); schema <= state.Schema(5); schema++ {
		for _, archive := range generallySupportedArchives {
			memoryConfig := state.Configuration{
				Variant: VariantGoMemory,
				Schema:  schema,
				Archive: archive,
			}
			fileConfig := state.Configuration{
				Variant: VariantGoFile,
				Schema:  schema,
				Archive: archive,
			}
			state.RegisterStateFactory(memoryConfig, newGoMemoryState)
			state.RegisterStateFactory(fileConfig, newGoCachedFileState)

			// Also register flat database variants.
			if schema == state.Schema(5) {
				config := memoryConfig
				config.Variant += "-flat"
				state.RegisterStateFactory(config, flat.WrapFactory(newGoMemoryState))

				config = fileConfig
				config.Variant += "-flat"
				state.RegisterStateFactory(config, flat.WrapFactory(newGoCachedFileState))
			}
		}
	}

	mptSetups := []struct {
		schema  state.Schema
		archive state.ArchiveType
	}{
		{4, state.S4Archive},
		{5, state.S5Archive},
	}

	for _, setup := range mptSetups {
		memoryConfig := state.Configuration{
			Variant: VariantGoMemory,
			Schema:  setup.schema,
			Archive: setup.archive,
		}
		fileConfig := state.Configuration{
			Variant: VariantGoFile,
			Schema:  setup.schema,
			Archive: setup.archive,
		}

		state.RegisterStateFactory(memoryConfig, newGoMemoryState)
		state.RegisterStateFactory(fileConfig, newGoFileState)

		// Also register flat database variants.
		if setup.schema == state.Schema(5) {
			config := memoryConfig
			config.Variant += "-flat"
			state.RegisterStateFactory(config, flat.WrapFactory(newGoMemoryState))

			config = fileConfig
			config.Variant += "-flat"
			state.RegisterStateFactory(config, flat.WrapFactory(newGoFileState))
		}
	}
}

// newGoMemoryState creates in memory implementation
// (path parameter for compatibility with other state factories, can be left empty)
func newGoMemoryState(params state.Parameters) (state.State, error) {
	_, err := prepareLiveDbDirectory(params)
	if err != nil {
		return nil, err
	}
	if params.Schema == 0 {
		params.Schema = defaultSchema
	}
	if params.Schema == 4 {
		return newGoMemoryS4State(params)
	}
	if params.Schema == 5 {
		return newGoMemoryS5State(params)
	}

	return nil, fmt.Errorf("%w: the go implementation only supports schemas 4-5, got %d", state.UnsupportedConfiguration, params.Schema)
}

// newGoFileState creates File based Index and Store implementations
func newGoFileState(params state.Parameters) (state.State, error) {
	_, err := prepareLiveDbDirectory(params)
	if err != nil {
		return nil, err
	}
	if params.Schema == 0 {
		params.Schema = defaultSchema
	}
	if params.Schema == 4 {
		return newGoFileS4State(params)
	}
	if params.Schema == 5 {
		return newGoFileS5State(params)
	}
	return nil, fmt.Errorf("%w: the go implementation only supports schemas 1-5, got %d", state.UnsupportedConfiguration, params.Schema)
}

// newGoCachedFileState creates File based Index and Store implementations
func newGoCachedFileState(params state.Parameters) (state.State, error) {
	_, err := prepareLiveDbDirectory(params)
	if err != nil {
		return nil, err
	}
	if params.Schema == 0 {
		params.Schema = defaultSchema
	}
	if params.Schema == 4 {
		return newGoFileS4State(params)
	}
	if params.Schema == 5 {
		return newGoFileS5State(params)
	}

	return nil, fmt.Errorf("%w: the go implementation only supports schemas 1-5, got %d", state.UnsupportedConfiguration, params.Schema)
}

func prepareLiveDbDirectory(params state.Parameters) (string, error) {
	path := filepath.Join(params.Directory, "live")
	return path, os.MkdirAll(path, 0700)
}

func getArchivePath(params state.Parameters) string {
	return filepath.Join(params.Directory, "archive")
}

func prepareArchiveDirectory(params state.Parameters) (string, error) {
	path := filepath.Join(params.Directory, "archive")
	return path, os.MkdirAll(path, 0700)
}

func openArchive(params state.Parameters) (archive archive.Archive, cleanup func(), err error) {
	switch params.Archive {

	case state.ArchiveType(""), state.NoArchive:
		// Check that the archive directory does not exist or is empty.
		path := getArchivePath(params)
		if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
			return nil, nil, nil
		}
		content, err := os.ReadDir(path)
		if err != nil {
			return nil, nil, err
		}
		if len(content) > 0 {
			return nil, nil, fmt.Errorf("opening DB with no archive would ignore existing archive at %s", path)
		}
		return nil, nil, nil

	case state.S4Archive:
		path, err := prepareArchiveDirectory(params)
		if err != nil {
			return nil, nil, err
		}
		arch, err := mpt.OpenArchiveTrie(path, mpt.S4ArchiveConfig, getNodeCacheConfig(params.ArchiveCache, params.BackgroundFlushPeriod), mpt.ArchiveConfig{
			CheckpointInterval: params.CheckpointInterval,
			CheckpointPeriod:   params.CheckpointPeriod,
		})
		return arch, nil, err

	case state.S5Archive:
		path, err := prepareArchiveDirectory(params)
		if err != nil {
			return nil, nil, err
		}
		arch, err := mpt.OpenArchiveTrie(path, mpt.S5ArchiveConfig, getNodeCacheConfig(params.ArchiveCache, params.BackgroundFlushPeriod), mpt.ArchiveConfig{
			CheckpointInterval: params.CheckpointInterval,
			CheckpointPeriod:   params.CheckpointPeriod,
		})
		return arch, nil, err
	}
	return nil, nil, fmt.Errorf("unknown archive type: %v", params.Archive)
}
