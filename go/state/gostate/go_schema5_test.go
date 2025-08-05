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
	"os"
	"testing"
	"time"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/database/mpt"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/stretchr/testify/require"
)

func TestScheme5_Archive_And_Live_Must_Be_InSync(t *testing.T) {
	dir := t.TempDir()

	archiveConfig := namedStateConfig{
		config: state.Configuration{
			Variant: VariantGoMemory,
			Schema:  5,
			Archive: state.S5Archive,
		},
		factory: newGoMemoryState,
	}

	addBlock := func(block uint64, db state.State) {
		update := common.Update{
			Balances: []common.BalanceUpdate{{common.Address{byte(block)}, amount.New(100)}},
		}
		if err := db.Apply(block, update); err != nil {
			t.Fatalf("cannot add block: %v", err)
		}
	}

	// open and create some blocks including archive
	db, err := archiveConfig.createState(dir)
	if err != nil {
		t.Fatalf("failed to open database: %v", err)
	}
	const blocks = 10
	for i := 0; i < blocks; i++ {
		addBlock(uint64(i), db)
	}

	if err := db.Close(); err != nil {
		t.Fatalf("cannot close database: %v", err)
	}

	// move the archive to a backup file
	archivePath := getArchivePath(state.Parameters{Directory: dir})
	backup := archivePath + ".backup"
	require.NoError(t, os.Rename(archivePath, backup))

	// open as non-archive
	noArchiveConfig := namedStateConfig{
		config: state.Configuration{
			Variant: archiveConfig.config.Variant,
			Schema:  archiveConfig.config.Schema,
			Archive: state.NoArchive,
		},
		factory: archiveConfig.factory,
	}

	// continue adding without the archive
	db, err = noArchiveConfig.createState(dir)
	if err != nil {
		t.Fatalf("failed to open database: %v", err)
	}
	for i := 0; i < blocks; i++ {
		addBlock(uint64(i+blocks), db)
	}

	if err := db.Close(); err != nil {
		t.Fatalf("cannot close database: %v", err)
	}

	// restore the backup
	require.NoError(t, os.Rename(backup, archivePath))

	// opening archive should fail as archive and non-archive is not in-sync
	if _, err := archiveConfig.createState(dir); err == nil {
		t.Errorf("opening database should fail")
	}
}

func TestCarmen_Empty_Archive_And_Live_Must_Be_InSync(t *testing.T) {

	dir := t.TempDir()

	archiveConfig := namedStateConfig{
		config: state.Configuration{
			Variant: VariantGoMemory,
			Schema:  5,
			Archive: state.S5Archive,
		},
		factory: newGoMemoryState,
	}

	noArchiveConfig := namedStateConfig{
		config: state.Configuration{
			Variant: archiveConfig.config.Variant,
			Schema:  archiveConfig.config.Schema,
			Archive: state.NoArchive,
		},
		factory: archiveConfig.factory,
	}

	// open and start some blocks as non archive
	db, err := noArchiveConfig.createState(dir)
	if err != nil {
		t.Fatalf("failed to open database: %v", err)
	}
	const blocks = 10
	for i := 0; i < blocks; i++ {
		block := uint64(i)
		update := common.Update{
			Balances: []common.BalanceUpdate{{common.Address{byte(block)}, amount.New(100)}},
		}
		if err := db.Apply(block, update); err != nil {
			t.Fatalf("cannot add block: %v", err)
		}
	}

	if err := db.Close(); err != nil {
		t.Fatalf("cannot close database: %v", err)
	}

	// opening archive should fail as archive and non-archive is not in-sync
	if _, err := archiveConfig.createState(dir); err == nil {
		t.Errorf("opening database should fail")
	}
}

func TestGetNodeCacheConfig(t *testing.T) {
	tests := map[string]struct {
		cacheSize int64 // in bytes
		capacity  int   // in number of nodes
	}{
		"zero": {
			cacheSize: 0,
			capacity:  0,
		},
		"none-zero": {
			cacheSize: 1, // < if the cache size is greater than 0,
			capacity:  1, // < than the capacity should be greater than 0 to trigger
			// the usage of the minimum cache size instead of the default
			// cache size
		},
		"one node": {
			cacheSize: int64(mpt.EstimatePerNodeMemoryUsage()),
			capacity:  1,
		},
		"ten nodes": {
			cacheSize: 10 * int64(mpt.EstimatePerNodeMemoryUsage()),
			capacity:  10,
		},
		"negative": {
			cacheSize: -1,
			capacity:  0,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			cfg := getNodeCacheConfig(test.cacheSize, 0)
			if cfg.Capacity != test.capacity {
				t.Errorf("unexpected capacity: %d != %d", cfg.Capacity, test.capacity)
			}
		})
	}
}

func TestGetNodeCacheConfig_BackgroundFlushPeriod(t *testing.T) {
	const expected = 123 * time.Second

	if got, want := getNodeCacheConfig(0, expected).BackgroundFlushPeriod, expected; got != want {
		t.Errorf("unexpected background flush period: %v != %v", got, want)
	}
}
