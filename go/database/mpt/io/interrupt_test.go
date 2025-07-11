// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package io

import (
	"context"
	"errors"
	"io"
	"syscall"
	"testing"
	"time"

	"github.com/0xsoniclabs/carmen/go/common/interrupt"
	"github.com/0xsoniclabs/carmen/go/database/mpt"
)

func TestExport_CanBeInterrupted(t *testing.T) {
	type testFuncs struct {
		// export is the tested export func
		export func(context.Context, *Log, string, io.Writer) error
		// createDB is an init of the database
		createDB func(t *testing.T, sourceDir string)
		// check that the interrupted did not corrupt the db by re-opening it
		check func(t *testing.T, sourceDir string)
	}

	exportBlockFromArchive := func(ctx context.Context, _ *Log, dir string, out io.Writer) error {
		return ExportBlockFromArchive(ctx, NewLog(), dir, out, 3)
	}

	tests := map[string]testFuncs{
		"live": {
			export:   Export,
			createDB: createTestLive,
			check:    checkCanOpenLiveDB,
		},
		"archive": {
			export:   ExportArchive,
			createDB: createTestArchive,
			check:    checkCanOpenArchive,
		},
		"live-from-archive": {
			export:   exportBlockFromArchive,
			createDB: createTestArchive,
			check:    checkCanOpenArchive,
		},
	}

	for name, tf := range tests {
		t.Run(name, func(t *testing.T) {
			// Create a small db to be exported.
			sourceDir := t.TempDir()
			tf.createDB(t, sourceDir)

			countWriter := &interruptSendingWriter{signalInterrupt: false}
			// first find number of writes
			if err := tf.export(context.Background(), NewLog(), sourceDir, countWriter); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// save max count and reset number of writes
			maxCount := countWriter.numOfWrites

			ctx := interrupt.CancelOnInterrupt(context.Background())

			writer := &interruptSendingWriter{}
			writer.signalInterrupt = true
			err := tf.export(ctx, NewLog(), sourceDir, writer)
			if got, want := err, interrupt.ErrCanceled; !errors.Is(got, want) {
				t.Errorf("unexpected error: got: %v, want: %v", got, want)
			}

			if maxCount == writer.numOfWrites || writer.numOfWrites == 0 {
				t.Error("export was not interrupted")
			}

			// lastly check that the database is not corrupted
			tf.check(t, sourceDir)
		})
	}
}

func createTestLive(t *testing.T, sourceDir string) {
	t.Helper()
	db := createExampleLiveDB(t, sourceDir)
	if err := db.Close(); err != nil {
		t.Fatalf("failed to close example live db: %v", err)
	}
}

func createTestArchive(t *testing.T, sourceDir string) {
	t.Helper()
	source, err := mpt.OpenArchiveTrie(sourceDir, mpt.S5ArchiveConfig, mpt.NodeCacheConfig{Capacity: 1024}, mpt.ArchiveConfig{})
	if err != nil {
		t.Fatalf("failed to create archive: %v", err)
	}
	fillTestBlocksIntoArchive(t, source)
	if err = source.Close(); err != nil {
		t.Fatalf("failed to close test DB: %v", err)
	}
}

// checkCanOpenLiveDB makes sure LiveDB is not corrupted and can be opened (and closed)
func checkCanOpenLiveDB(t *testing.T, sourceDir string) {
	db, err := mpt.OpenGoFileState(sourceDir, mpt.S5LiveConfig, mpt.NodeCacheConfig{})
	if err != nil {
		t.Fatalf("failed to open db: %v", err)
	}
	err = db.Close()
	if err != nil {
		t.Fatalf("failed to close db: %v", err)
	}
}

// checkCanOpenLiveDB makes sure Archive is not corrupted and can be opened (and closed)
func checkCanOpenArchive(t *testing.T, sourceDir string) {
	archive, err := mpt.OpenArchiveTrie(sourceDir, mpt.S5ArchiveConfig, mpt.NodeCacheConfig{}, mpt.ArchiveConfig{})
	if err != nil {
		t.Fatalf("failed to open archive: %v", err)
	}
	err = archive.Close()
	if err != nil {
		t.Fatalf("failed to close archive: %v", err)
	}
}

type interruptSendingWriter struct {
	numOfWrites     int
	signalInterrupt bool
}

func (m *interruptSendingWriter) Write([]byte) (n int, err error) {
	m.numOfWrites++
	// inform the test that first write has happened
	if m.numOfWrites > 0 && m.signalInterrupt {
		m.signalInterrupt = false
		err = syscall.Kill(syscall.Getpid(), syscall.SIGINT)
		if err != nil {
			return 0, errors.New("failed to create a SIGINT signal")
		}
		time.Sleep(100 * time.Millisecond)
	}

	return 0, nil
}
