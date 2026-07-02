// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package mpt

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/backend/utils"
	"github.com/0xsoniclabs/carmen/go/backend/utils/checkpoint"
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
)

func TestCodes_OpenCodes(t *testing.T) {
	codes, err := openCodes(t.TempDir())
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}

	if want, got := 0, len(codes.codes); want != got {
		t.Fatalf("expected codes to be empty, got %d", got)
	}
}

func TestCodes_OpenCodes_IOErrorsAreHandled(t *testing.T) {
	tests := map[string]func(t *testing.T) string{
		"invalid directory": func(t *testing.T) string {
			dir := t.TempDir()
			file := filepath.Join(dir, "file")
			if err := os.WriteFile(file, []byte{}, 0600); err != nil {
				t.Fatalf("failed to create file: %v", err)
			}
			return file //< passing a file instead of a directory
		},
		"missing directory permissions": func(t *testing.T) string {
			dir := t.TempDir()
			stat, err := os.Stat(dir)
			if err != nil {
				t.Fatalf("failed to stat directory: %v", err)
			}
			if err := os.Chmod(dir, 0500); err != nil {
				t.Fatalf("failed to change directory permissions: %v", err)
			}
			t.Cleanup(func() {
				require.NoError(t, os.Chmod(dir, stat.Mode()))
			})
			return dir
		},
		"missing permissions to create code file": func(t *testing.T) string {
			dir := t.TempDir()
			// the code directory must exist to reach the code file creation
			if err := os.MkdirAll(filepath.Join(dir, fileNameCodesCheckpointDirectory), 0700); err != nil {
				t.Fatalf("failed to create codes directory: %v", err)
			}
			stat, err := os.Stat(dir)
			if err != nil {
				t.Fatalf("failed to stat directory: %v", err)
			}
			if err := os.Chmod(dir, 0500); err != nil {
				t.Fatalf("failed to change directory permissions: %v", err)
			}
			t.Cleanup(func() {
				require.NoError(t, os.Chmod(dir, stat.Mode()))
			})
			return dir
		},
		"missing permissions to read code file": func(t *testing.T) string {
			dir := t.TempDir()
			file := filepath.Join(dir, fileNameCodes)
			if err := os.WriteFile(file, []byte{}, 0600); err != nil {
				t.Fatalf("failed to create file: %v", err)
			}
			if err := os.Chmod(file, 0200); err != nil {
				t.Fatalf("failed to change file permissions: %v", err)
			}
			t.Cleanup(func() {
				require.NoError(t, os.Chmod(file, 0600))
			})
			return dir
		},
		"missing permissions to read checkpoint data": func(t *testing.T) string {
			dir := t.TempDir()
			nested := filepath.Join(dir, fileNameCodesCheckpointDirectory)
			if err := os.MkdirAll(nested, 0700); err != nil {
				t.Fatalf("failed to create codes directory: %v", err)
			}
			file := filepath.Join(nested, fileNameCodesCommittedCheckpoint)
			if err := os.WriteFile(file, []byte{}, 0600); err != nil {
				t.Fatalf("failed to create file: %v", err)
			}
			if err := os.Chmod(file, 0200); err != nil {
				t.Fatalf("failed to change file permissions: %v", err)
			}
			t.Cleanup(func() {
				require.NoError(t, os.Chmod(file, 0600))
			})
			return dir
		},
	}

	for name, prepare := range tests {
		t.Run(name, func(t *testing.T) {
			dir := prepare(t)
			_, err := openCodes(dir)
			if err == nil {
				t.Fatalf("expected error, got nil")
			}
		})
	}
}

func TestCodes_CodesCanBeAddedAndRetrieved(t *testing.T) {
	codes, err := openCodes(t.TempDir())
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}

	code1 := []byte("code1")
	code2 := []byte("code2")

	hash1 := codes.add(code1)
	hash2 := codes.add(code2)

	if want, got := 2, codeSize(t, codes); want != got {
		t.Fatalf("expected codes to have 2 entries, got %d", got)
	}

	if want, got := code1, codes.getCodeForHash(hash1); string(want) != string(got) {
		t.Fatalf("expected code1, got %s", got)
	}

	if want, got := code2, codes.getCodeForHash(hash2); string(want) != string(got) {
		t.Fatalf("expected code2, got %s", got)
	}
}

func TestCodes_Flush_EmptyCodesCanBeFlushed(t *testing.T) {
	codes, err := openCodes(t.TempDir())
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}

	if err := codes.Flush(); err != nil {
		t.Fatalf("failed to flush: %v", err)
	}

	stats, err := os.Stat(codes.file)
	if err != nil {
		t.Fatalf("failed to stat file: %v", err)
	}
	if want, got := int64(0), stats.Size(); want != got {
		t.Fatalf("expected file size to be %d, got %d", want, got)
	}
}

func TestCodes_Flush_CodesAreWrittenIncrementally(t *testing.T) {
	codes, err := openCodes(t.TempDir())
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}

	code1 := []byte("code1")
	code2 := []byte("code2")
	code3 := []byte("code3")

	codes.add(code1)
	codes.add(code2)

	// After add, codes are in cache (not pending) until flush or eviction.
	if want, got := 0, len(codes.pending); want != got {
		t.Fatalf("expected %d pending codes, got %d", want, got)
	}

	if err := codes.Flush(); err != nil {
		t.Fatalf("failed to flush: %v", err)
	}

	// After flush, everything is on disk.
	if want, got := 2, len(codes.codes); want != got {
		t.Fatalf("expected %d codes on disk, got %d", want, got)
	}
	if want, got := 0, len(codes.pending); want != got {
		t.Fatalf("expected %d pending codes, got %d", want, got)
	}

	snapshot1, err := os.ReadFile(codes.file)
	if err != nil {
		t.Fatalf("failed to read file: %v", err)
	}

	if codes.fileSize != uint64(len(snapshot1)) {
		t.Fatalf("expected file size to be %d, got %d", len(snapshot1), codes.fileSize)
	}

	// The next step is incremental.
	codes.add(code3)

	if err := codes.Flush(); err != nil {
		t.Fatalf("failed to flush: %v", err)
	}

	if want, got := 3, len(codes.codes); want != got {
		t.Fatalf("expected %d codes on disk, got %d", want, got)
	}
	if want, got := 0, len(codes.pending); want != got {
		t.Fatalf("expected %d pending codes, got %d", want, got)
	}

	snapshot2, err := os.ReadFile(codes.file)
	if err != nil {
		t.Fatalf("failed to read file: %v", err)
	}

	if codes.fileSize != uint64(len(snapshot2)) {
		t.Fatalf("expected file size to be %d, got %d", len(snapshot2), codes.fileSize)
	}

	if !bytes.HasPrefix(snapshot2, snapshot1) {
		t.Fatalf("expected snapshot2 to be a continuation of snapshot1")
	}
}

func TestCodes_getCodes_ReturnsAllCodes(t *testing.T) {
	require := require.New(t)
	c, err := openCodes(t.TempDir())
	require.NoError(err)

	// Use a small cache to force eviction into pending.
	c.cache = common.NewLruCache[common.Hash, []byte](2)

	code1 := []byte("code1")
	code2 := []byte("code2")
	code3 := []byte("code3")

	hash1 := c.add(code1)
	require.NoError(c.Flush()) // flush to disk
	hash2 := c.add(code2)
	hash3 := c.add(code3) // evicts hash1 from cache (already on disk, no pending)

	// Evict hash2 into pending by adding another code.
	code4 := []byte("code4")
	hash4 := c.add(code4) // evicts hash2 from cache into pending
	_ = hash4

	// Check code positions:
	// - hash1 is on disk
	_, onDisk := c.codes[hash1]
	require.True(onDisk)
	// - hash2 is in pending (evicted from cache, not on disk)
	_, inPending := c.pending[hash2]
	require.True(inPending)
	// - hash3, hash4 are in cache
	_, inCache := c.cache.Get(hash3)
	require.True(inCache)
	_, inCache = c.cache.Get(hash4)
	require.True(inCache)

	got, err := c.getCodes()
	require.NoError(err)

	require.Equal(4, len(got))
	require.Equal(code1, got[hash1])
	require.Equal(code2, got[hash2])
	require.Equal(code3, got[hash3])
	require.Equal(code4, got[hash4])
}

func TestCodes_GetMemoryFootprint_ReturnsProperSize(t *testing.T) {
	codes, err := openCodes(t.TempDir())
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}

	code1 := []byte("short")
	code2 := []byte("something longer")

	codes.add(code1)
	codes.add(code2)

	footprint := codes.GetMemoryFootprint()

	cacheFootprint := codes.cache.GetDynamicMemoryFootprint(func(v []byte) uintptr {
		return uintptr(len(v))
	})
	// After add, codes are in cache only (no pending, no disk offsets).
	want := unsafe.Sizeof(*codes) + cacheFootprint.Total()

	got := footprint.Total()
	if want != got {
		t.Fatalf("expected %d, got %d", want, got)
	}
}

func TestCodes_GuaranteeCheckpoint_PendingCheckpointIsCommitted(t *testing.T) {
	codes, err := openCodes(t.TempDir())
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}

	cp0 := checkpoint.Checkpoint(0)

	if err := codes.GuaranteeCheckpoint(cp0); err != nil {
		t.Fatalf("failed to guarantee initial checkpoint: %v", err)
	}

	cp1 := checkpoint.Checkpoint(1)
	if err := codes.Prepare(cp1); err != nil {
		t.Fatalf("failed to prepare checkpoint: %v", err)
	}

	if want, got := cp0, codes.checkpoint; want != got {
		t.Fatalf("expected checkpoint to be %d, got %d", want, got)
	}

	if err := codes.GuaranteeCheckpoint(cp1); err != nil {
		t.Fatalf("failed to guarantee pending checkpoint: %v", err)
	}

	if want, got := cp1, codes.checkpoint; want != got {
		t.Fatalf("expected checkpoint to be %d, got %d", want, got)
	}

	if err := codes.GuaranteeCheckpoint(cp0); err == nil {
		t.Fatalf("expected error, got nil")
	}
}

func TestCodes_GuaranteeCheckpoint_IoErrorsAreHandled(t *testing.T) {
	codes, err := openCodes(t.TempDir())
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}
	cp1 := checkpoint.Checkpoint(1)
	if err := codes.Prepare(cp1); err != nil {
		t.Fatalf("failed to prepare checkpoint: %v", err)
	}

	pendingFile := filepath.Join(codes.directory, fileNameCodesPrepareCheckpoint)
	if err := os.WriteFile(pendingFile, []byte("invalid json"), 0600); err != nil {
		t.Fatalf("failed to write file: %v", err)
	}

	if err := codes.GuaranteeCheckpoint(cp1); err == nil {
		t.Fatalf("expected error, got nil")
	}
}

func TestCodes_Prepare_CheckpointIsIncremental(t *testing.T) {
	codes, err := openCodes(t.TempDir())
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}

	cp1 := checkpoint.Checkpoint(1)
	if err := codes.Prepare(cp1); err != nil {
		t.Fatalf("failed to prepare initial checkpoint: %v", err)
	}

	cp2 := checkpoint.Checkpoint(2)
	if err := codes.Prepare(cp2); err == nil {
		t.Fatalf("expected error, got nil")
	}
}

func TestCodes_Prepare_FailsIfFlushFails(t *testing.T) {
	codes, err := openCodes(t.TempDir())
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}

	codes.add([]byte("code1"))

	require.NoError(t, os.Chmod(codes.file, 0400)) // make the file read-only
	defer func() { require.NoError(t, os.Chmod(codes.file, 0600)) }()

	cp1 := checkpoint.Checkpoint(1)
	if err := codes.Prepare(cp1); err == nil {
		t.Fatalf("expected error, got nil")
	}
}

func TestCodes_Commit_HandlesIoIssues(t *testing.T) {
	tests := map[string]func(*testing.T, string) error{
		"missing prepare file": func(t *testing.T, dir string) error {
			return os.Remove(filepath.Join(dir, fileNameCodesCheckpointDirectory, fileNameCodesPrepareCheckpoint))
		},
		"invalid prepare file": func(t *testing.T, dir string) error {
			return os.WriteFile(filepath.Join(dir, fileNameCodesCheckpointDirectory, fileNameCodesPrepareCheckpoint), []byte("invalid json"), 0600)
		},
		"missing rename permissions": func(t *testing.T, dir string) error {
			subDir := filepath.Join(dir, fileNameCodesCheckpointDirectory)
			if err := os.Chmod(subDir, 0500); err != nil {
				return err
			}
			t.Cleanup(func() {
				require.NoError(t, os.Chmod(subDir, 0700))
			})
			return nil
		},
	}

	for name, temper := range tests {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			codes, err := openCodes(dir)
			if err != nil {
				t.Fatalf("failed to open codes: %v", err)
			}

			codes.add([]byte("code1"))

			cp1 := checkpoint.Checkpoint(1)
			if err := codes.Prepare(cp1); err != nil {
				t.Fatalf("failed to prepare test: %v", err)
			}

			if err := temper(t, dir); err != nil {
				t.Fatalf("failed to prepare test: %v", err)
			}

			if err := codes.Commit(cp1); err == nil {
				t.Fatalf("expected error, got nil")
			}
		})
	}
}

func TestCodes_Restore_CanRestoreCommittedAndPendingCheckpoint(t *testing.T) {
	for _, name := range []string{"committed", "pending"} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()

			codes, err := openCodes(dir)
			if err != nil {
				t.Fatalf("failed to open codes: %v", err)
			}
			codes.add([]byte("code1"))

			cp1 := checkpoint.Checkpoint(1)
			if err := codes.Prepare(cp1); err != nil {
				t.Fatalf("failed to prepare checkpoint: %v", err)
			}
			if name == "committed" {
				if err := codes.Commit(cp1); err != nil {
					t.Fatalf("failed to commit checkpoint: %v", err)
				}
			}

			codes.add([]byte("code2"))
			if err := codes.Flush(); err != nil {
				t.Fatalf("failed to flush: %v", err)
			}

			codes, err = openCodes(dir)
			if err != nil {
				t.Fatalf("failed to re-open original codes: %v", err)
			}

			if want, got := 2, len(codes.codes); want != got {
				t.Fatalf("expected codes to have %d entries, got %d", want, got)
			}

			if err := getCodeRestorer(dir).Restore(cp1); err != nil {
				t.Fatalf("failed to restore checkpoint: %v", err)
			}

			codes, err = openCodes(dir)
			if err != nil {
				t.Fatalf("failed to re-open recovered codes: %v", err)
			}

			if want, got := 1, len(codes.codes); want != got {
				t.Fatalf("expected codes to have %d entries, got %d", want, got)
			}
		})
	}
}

func TestCodes_Restore_InvalidCheckpointMetaDataIsDetected(t *testing.T) {
	dir := t.TempDir()
	restorer := getCodeRestorer(dir)

	subDir := filepath.Join(dir, fileNameCodesCheckpointDirectory)
	if err := os.MkdirAll(subDir, 0700); err != nil {
		t.Fatalf("failed to create codes directory: %v", err)
	}

	if err := os.WriteFile(filepath.Join(subDir, fileNameCodesCommittedCheckpoint), []byte("invalid json"), 0600); err != nil {
		t.Fatalf("failed to write file: %v", err)
	}

	cp := checkpoint.Checkpoint(0)
	if err := restorer.Restore(cp); err == nil {
		t.Fatalf("expected error, got nil")
	}
}

func TestCodes_Restore_InvalidCheckpointDataIsDetected(t *testing.T) {
	dir := t.TempDir()
	restorer := getCodeRestorer(dir)

	cp := checkpoint.Checkpoint(42) // < non-existing checkpoint
	if err := restorer.Restore(cp); err == nil {
		t.Fatalf("expected error, got nil")
	}
}

func TestCodes_Restore_CanHandleErrorCorruptedData(t *testing.T) {
	tests := map[string]func(dir string) error{
		"no corruption": func(string) error {
			return nil
		},
		"extra data in code file": func(dir string) error {
			file, _ := getCodePaths(dir)
			data, err := os.ReadFile(file)
			if err != nil {
				return err
			}
			data = append(data, []byte("extra")...)
			return os.WriteFile(file, data, 0600)
		},
	}

	for name, temper := range tests {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()

			// Prepare a valid code state.
			codes, err := openCodes(dir)
			if err != nil {
				t.Fatalf("failed to open codes: %v", err)
			}

			codes.add([]byte("code1"))
			codes.add([]byte("code2"))

			cp := checkpoint.Checkpoint(1)
			if err := codes.Prepare(cp); err != nil {
				t.Fatalf("failed to prepare checkpoint: %v", err)
			}
			if err := codes.Commit(cp); err != nil {
				t.Fatalf("failed to commit checkpoint: %v", err)
			}

			backup, err := os.ReadFile(codes.file)
			if err != nil {
				t.Fatalf("failed to read file: %v", err)
			}
			if len(backup) == 0 {
				t.Fatalf("expected file to be non-empty")
			}

			// Corrupt the code state.
			if err := temper(dir); err != nil {
				t.Fatalf("failed to corrupt codes: %v", err)
			}

			// Attempt to restore the code state.
			restorer := getCodeRestorer(dir)
			if err := restorer.Restore(cp); err != nil {
				t.Fatalf("failed to restore checkpoint: %v", err)
			}

			// Verify the restored state.
			restored, err := os.ReadFile(codes.file)
			if err != nil {
				t.Fatalf("failed to read file: %v", err)
			}

			if !bytes.Equal(backup, restored) {
				t.Fatalf("expected file to be equal after restore")
			}
		})
	}
}

func TestCodes_CheckpointsCanBeRestored(t *testing.T) {
	dir := t.TempDir()
	file, _ := getCodePaths(dir)
	codes, err := openCodes(dir)
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}

	codes.add([]byte("code1"))
	codes.add([]byte("code2"))

	checkpoint := checkpoint.Checkpoint(1)
	if err := codes.Prepare(checkpoint); err != nil {
		t.Fatalf("failed to prepare checkpoint: %v", err)
	}

	if err := codes.Commit(checkpoint); err != nil {
		t.Fatalf("failed to commit checkpoint: %v", err)
	}

	backup, err := os.Stat(file)
	if err != nil {
		t.Fatalf("failed to stat file: %v", err)
	}

	codes.add([]byte("code3"))
	if want, got := 2, len(codes.codes); want != got {
		t.Fatalf("expected codes to have %d entries, got %d", want, got)
	}
	codeSize := 0
	codes.cache.Iterate(func(h common.Hash, b []byte) bool {
		codeSize++
		return true
	})
	if want, got := 3, codeSize; want != got {
		t.Fatalf("expected cache to have %d entries, got %d", want, got)
	}

	if err := codes.Flush(); err != nil {
		t.Fatalf("failed to flush: %v", err)
	}

	modified, err := os.Stat(file)
	if err != nil {
		t.Fatalf("failed to stat file: %v", err)
	}

	if modified.Size() <= backup.Size() {
		t.Fatalf("expected file to be larger after flush")
	}

	if err := getCodeRestorer(dir).Restore(checkpoint); err != nil {
		t.Fatalf("failed to restore checkpoint: %v", err)
	}

	restored, err := os.Stat(file)
	if err != nil {
		t.Fatalf("failed to stat file: %v", err)
	}

	if restored.Size() != backup.Size() {
		t.Fatalf("expected file to be same size after restore")
	}

	codes, err = openCodes(dir)
	if err != nil {
		t.Fatalf("failed to re-open recovered codes: %v", err)
	}

	if want, got := 2, len(codes.codes); want != got {
		t.Fatalf("expected codes to have %d entries, got %d", want, got)
	}
}

func TestCodes_CheckpointsCanBeAborted(t *testing.T) {
	dir := t.TempDir()
	codes, err := openCodes(dir)
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}

	codes.add([]byte("code1"))
	codes.add([]byte("code2"))

	cp := checkpoint.Checkpoint(1)
	if err := codes.Prepare(cp); err != nil {
		t.Fatalf("failed to prepare checkpoint: %v", err)
	}

	if err := codes.Abort(cp); err != nil {
		t.Fatalf("failed to commit checkpoint: %v", err)
	}

	if want, got := 2, len(codes.codes); want != got {
		t.Fatalf("expected codes to have %d entries, got %d", want, got)
	}

	cp = checkpoint.Checkpoint(0)
	if err := getCodeRestorer(dir).Restore(cp); err != nil {
		t.Fatalf("failed to restore checkpoint: %v", err)
	}

	codes, err = openCodes(dir)
	if err != nil {
		t.Fatalf("failed to re-open recovered codes: %v", err)
	}

	if want, got := 0, len(codes.codes); want != got {
		t.Fatalf("expected codes to have %d entries, got %d", want, got)
	}
}

func TestCodes_CanBeHandledByCheckpointCoordinator(t *testing.T) {
	dir := t.TempDir()
	codes, err := openCodes(dir)
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}

	coordinator, err := checkpoint.NewCoordinator(t.TempDir(), codes)
	if err != nil {
		t.Fatalf("failed to create coordinator: %v", err)
	}

	codes.add([]byte("code1"))

	if _, err := coordinator.CreateCheckpoint(); err != nil {
		t.Fatalf("failed to create checkpoint: %v", err)
	}

	codes.add([]byte("code2"))

	if err := getCodeRestorer(dir).Restore(coordinator.GetCurrentCheckpoint()); err != nil {
		t.Fatalf("failed to restore checkpoint: %v", err)
	}

	codes, err = openCodes(dir)
	if err != nil {
		t.Fatalf("failed to re-open recovered codes: %v", err)
	}

	if want, got := 1, len(codes.codes); want != got {
		t.Fatalf("expected codes to have %d entries, got %d", want, got)
	}

}

func TestCodes_writeCodes_WritesCodesToFile(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, fileNameCodes)

	codes := map[common.Hash][]byte{
		{1}: {5},
		{2}: {7, 8},
	}

	if err := writeCodes(codes, file); err != nil {
		t.Fatalf("failed to write codes: %v", err)
	}

	readCodes, err := readCodes(file)
	if err != nil {
		t.Fatalf("failed to read codes: %v", err)
	}

	if want, got := 2, len(readCodes); want != got {
		t.Fatalf("expected codes to have %d entries, got %d", want, got)
	}
}

func TestCodes_writeCode_WriteFailures(t *testing.T) {
	codes := map[common.Hash][]byte{
		{1}: {5, 6, 7, 8, 9},
		{2}: {10, 11},
	}

	// Dry-run: count total Write calls needed to write all codes.
	var count int
	{
		ctrl := gomock.NewController(t)
		osfile := utils.NewMockOsFile(ctrl)
		osfile.EXPECT().Write(gomock.Any()).AnyTimes().DoAndReturn(func(data []byte) (int, error) {
			count++
			return len(data), nil
		})
		for hash, code := range codes {
			require.NoError(t, writeCode(hash, code, osfile))
		}
	}

	// For each Write call, inject an error at that position and verify propagation.
	injectedErr := errors.New("write error")
	for i := 0; i < count; i++ {
		t.Run(fmt.Sprintf("error_on_write_%d", i), func(t *testing.T) {
			ctrl := gomock.NewController(t)
			osfile := utils.NewMockOsFile(ctrl)

			calls := make([]*gomock.Call, 0, i+1)
			for j := 0; j < i; j++ {
				calls = append(calls, osfile.EXPECT().Write(gomock.Any()).DoAndReturn(func(data []byte) (int, error) {
					return len(data), nil
				}))
			}
			calls = append(calls, osfile.EXPECT().Write(gomock.Any()).Return(0, injectedErr))
			gomock.InOrder(calls...)

			for hash, code := range codes {
				if err := writeCode(hash, code, osfile); err != nil {
					require.ErrorIs(t, err, injectedErr)
					return
				}
			}
			t.Fatal("expected an error but none occurred")
		})
	}
}

func TestCodes_writeCodes_CannotCreateTheOutputFile(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, fileNameCodesCheckpointDirectory)
	if err := os.Mkdir(file, os.FileMode(0644)); err != nil {
		t.Fatalf("cannot create dir: %s", err)
	}
	if err := writeCodes(make(map[common.Hash][]byte, 1), file); err == nil {
		t.Errorf("writing roots should fail")
	}
}

func TestCodes_writeCode_ForwardWriteError(t *testing.T) {
	ctrl := gomock.NewController(t)

	hash := common.Hash{3}
	code := []byte{9, 10, 11}

	file := utils.NewMockOsFile(ctrl)
	injectedError := errors.New("injected error")
	file.EXPECT().Write(gomock.Any()).Return(0, injectedError)

	require.Equal(t, injectedError, writeCode(hash, code, file))
}

func TestCodes_readCodeOffsetsAndSize_ReadsValuesCorrectly(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, fileNameCodes)

	codes := map[common.Hash][]byte{
		{1}: {5},
		{2}: {7, 8},
	}

	require.NoError(writeCodes(codes, path))

	readCodes, size, err := readCodeOffsetsAndSize(path)
	require.NoError(err)
	require.Len(readCodes, len(codes))

	for hash, code := range codes {
		offset, ok := readCodes[hash]
		require.True(ok)
		file, err := os.Open(path)
		require.NoError(err)
		_, err = file.Seek(int64(offset), 0)
		require.NoError(err)
		gotHash, gotCode, err := readCode(file)
		require.NoError(err)
		require.Equal(hash, gotHash)
		require.Equal(code, gotCode)
		require.NoError(file.Close())
	}

	require.Greater(size, uint64(0))
}

func TestCodes_readCodeOffsetsAndSize_ReadingNonExistingFileReturnsEmptyCodeMap(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, fileNameCodes)
	codes, size, err := readCodeOffsetsAndSize(path)
	if err != nil {
		t.Fatalf("failed to read codes: %v", err)
	}
	if want, got := 0, len(codes); want != got {
		t.Fatalf("expected codes to be empty, got %d", got)
	}
	if want, got := uint64(0), size; want != got {
		t.Fatalf("expected code file-size to be 0, got %d", got)
	}
}

func TestCodes_readCodeOffsetsAndSize_ReadingIssuesAreReported(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, fileNameCodes)

	if err := os.WriteFile(path, []byte("invalid"), 0600); err != nil {
		t.Fatalf("failed to prepare invalid code file: %v", err)
	}

	_, _, err := readCodeOffsetsAndSize(path)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
}

func TestCodes_readCodeOffsetsAndSize_PermissionErrorsAreDetected(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, fileNameCodes)

	if err := os.Chmod(dir, 0000); err != nil {
		t.Fatalf("failed to change directory permissions: %v", err)
	}
	defer func() { require.NoError(t, os.Chmod(dir, 0700)) }()

	_, _, err := readCodeOffsetsAndSize(path)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
}

func TestCodes_readCodes_Cannot_Read(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "dir")
	if err := os.Mkdir(file, os.FileMode(0)); err != nil {
		t.Fatalf("cannot create dir: %s", err)
	}
	if _, err := readCodes(file); err == nil {
		t.Errorf("reading codes should fail")
	}
}

func TestCodes_parseCodes_ReadFailures(t *testing.T) {
	var injectedErr = errors.New("read error")
	ctrl := gomock.NewController(t)
	osfile := utils.NewMockOsFile(ctrl)

	var h common.Hash
	sizes := []int{len(h), 4, 100}
	// execute three times - parseCode calls io.Reader three times to get [<key>, <length>, <code>]
	for i := 0; i < 3; i++ {
		calls := make([]*gomock.Call, 0, i+1)
		seekCall := osfile.EXPECT().Seek(gomock.Any(), gomock.Any()).Return(int64(0), nil)
		calls = append(calls, seekCall)

		for j := 0; j < i; j++ {
			pos := j
			call := osfile.EXPECT().Read(gomock.Any()).DoAndReturn(func(buf []byte) (int, error) {
				buf[0] = 1             // fill in an non-zero value not to return an empty array
				return sizes[pos], nil // returning expected size causes this io.Reader is called exactly once
			})

			calls = append(calls, call)
		}
		calls = append(calls, osfile.EXPECT().Read(gomock.Any()).Return(1, injectedErr))
		gomock.InOrder(calls...)

		if _, err := parseCodes(osfile); !errors.Is(err, injectedErr) {
			t.Errorf("reading codes should fail")
		}

	}
}

func TestCodes_readCodeFromDisk_ReadsValueCorrectly(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, fileNameCodes)

	codesToWrite := map[common.Hash][]byte{
		{1}: {5},
		{2}: {7, 8},
	}

	require.NoError(writeCodes(codesToWrite, path))

	codes, err := openCodes(dir)
	require.NoError(err)
	for hash, offset := range codes.codes {
		gotCode, err := codes.readCodeFromDisk(offset)
		require.NoError(err)
		wantCode := codesToWrite[hash]
		require.Equal(wantCode, gotCode)
	}
}

func TestCodes_getCodeForHash(t *testing.T) {
	tests := map[string]struct {
		setup func(t *testing.T, c *codes) (hash common.Hash, want []byte)
	}{
		"returns from cache": {
			setup: func(t *testing.T, c *codes) (common.Hash, []byte) {
				want := []byte("cache-code")
				h := c.add(want)
				return h, want
			},
		},
		"returns from pending when evicted from cache": {
			setup: func(t *testing.T, c *codes) (common.Hash, []byte) {
				// Use a tiny cache so we can force eviction.
				c.cache = common.NewLruCache[common.Hash, []byte](2)
				want := []byte("pending-code")
				h := c.add(want)
				// Fill the cache to evict the target into pending.
				c.add([]byte("filler-1"))
				c.add([]byte("filler-2"))
				return h, want
			},
		},
		"returns from disk": {
			setup: func(t *testing.T, c *codes) (common.Hash, []byte) {
				want := []byte("disk-code")
				h := c.add(want)
				require.NoError(t, c.Flush())
				c.cache.Clear()
				return h, want
			},
		},
		"returns nil when hash is unknown": {
			setup: func(t *testing.T, c *codes) (common.Hash, []byte) {
				return common.Hash{0xFF}, nil
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			c, err := openCodes(t.TempDir())
			require.NoError(err)
			hash, want := test.setup(t, c)
			got := c.getCodeForHash(hash)
			require.Equal(want, got)
		})
	}
}

func TestCodes_readCodeFromDisk_ErrorCases(t *testing.T) {
	tests := map[string]struct {
		prepare func(t *testing.T) (*codes, uint64)
	}{
		"file does not exist": {
			prepare: func(t *testing.T) (*codes, uint64) {
				return &codes{file: filepath.Join(t.TempDir(), "missing.dat")}, 0
			},
		},
		"truncated length field": {
			prepare: func(t *testing.T) (*codes, uint64) {
				dir := t.TempDir()
				file := filepath.Join(dir, fileNameCodes)
				require.NoError(t, os.WriteFile(file, make([]byte, 33), 0600))
				return &codes{file: file}, 0
			},
		},
		"declared code length larger than available bytes": {
			prepare: func(t *testing.T) (*codes, uint64) {
				dir := t.TempDir()
				file := filepath.Join(dir, fileNameCodes)
				content := make([]byte, 0, 40)
				content = append(content, make([]byte, 32)...)
				content = append(content, []byte{0, 0, 0, 10}...)
				content = append(content, []byte{1, 2}...)
				require.NoError(t, os.WriteFile(file, content, 0600))
				return &codes{file: file}, 0
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			c, offset := test.prepare(t)
			_, err := c.readCodeFromDisk(offset)
			require.Error(err)
		})
	}
}

func TestCodes_getCodeForHash_ReturnsNilOnDiskReadError(t *testing.T) {
	require := require.New(t)
	c, err := openCodes(t.TempDir())
	require.NoError(err)

	// Add a code and flush it to disk.
	code := []byte("some-code")
	h := c.add(code)
	require.NoError(c.Flush())
	c.cache.Clear()

	// Remove the backing file to make readCodeFromDisk fail.
	require.NoError(os.Remove(c.file))

	got := c.getCodeForHash(h)
	require.Nil(got)
}

func TestCodes_getCodeForHash_PendingPromotionEvictsIntoPending(t *testing.T) {
	require := require.New(t)
	c, err := openCodes(t.TempDir())
	require.NoError(err)

	// Use a cache of size 2 so promoting from pending causes an eviction.
	c.cache = common.NewLruCache[common.Hash, []byte](2)

	// Add 3 codes: first two stay in cache, third evicts the first into pending.
	code1 := []byte("code-one")
	code2 := []byte("code-two")
	code3 := []byte("code-three")

	h1 := c.add(code1)
	h2 := c.add(code2)
	h3 := c.add(code3) // evicts code1 into pending

	// Verify state: h1 is in pending, h2 and h3 are in cache.
	_, inPending := c.pending[h1]
	require.True(inPending)
	_, inCache := c.cache.Get(h2)
	require.True(inCache)
	_, inCache = c.cache.Get(h3)
	require.True(inCache)

	// Now request h1: it will be promoted from pending back to cache,
	// which must evict another entry (h2) into pending.
	got := c.getCodeForHash(h1)
	require.Equal(code1, got)

	// h1 should no longer be in pending (was deleted before handleCacheSet).
	_, inPending = c.pending[h1]
	require.False(inPending)

	// h2 should have been evicted into pending by the promotion.
	_, inPending = c.pending[h2]
	require.True(inPending)

	// h2 is still retrievable.
	got2 := c.getCodeForHash(h2)
	require.Equal(code2, got2)
}

func TestCodes_appendCodes(t *testing.T) {
	tests := map[string]struct {
		// existing codes already on disk before calling appendCodes
		existing map[common.Hash][]byte
		// new codes to append
		toAppend map[common.Hash][]byte
	}{
		"empty map appends nothing": {
			toAppend: map[common.Hash][]byte{},
		},
		"single code on empty file": {
			toAppend: map[common.Hash][]byte{
				{1}: {0xAA, 0xBB},
			},
		},
		"multiple codes on empty file": {
			toAppend: map[common.Hash][]byte{
				{1}: {0xAA},
				{2}: {0xBB, 0xCC},
				{3}: {0xDD, 0xEE, 0xFF},
			},
		},
		"append to existing file preserves previous data": {
			existing: map[common.Hash][]byte{
				{10}: {1, 2, 3},
			},
			toAppend: map[common.Hash][]byte{
				{20}: {4, 5},
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			dir := t.TempDir()
			file := filepath.Join(dir, fileNameCodes)

			// Write existing codes to establish a non-empty file.
			if len(test.existing) > 0 {
				require.NoError(writeCodes(test.existing, file))
			}

			offsets := make(map[common.Hash]uint64)
			fileSize, err := appendCodes(test.toAppend, file, offsets)
			require.NoError(err)

			// Verify offsets map is populated for all appended codes.
			require.Equal(len(test.toAppend), len(offsets))
			for h := range test.toAppend {
				_, exists := offsets[h]
				require.True(exists, "offset missing for hash %v", h)
			}

			// Verify fileSize matches actual file size.
			info, err := os.Stat(file)
			require.NoError(err)
			require.Equal(uint64(info.Size()), fileSize)

			// Verify all appended codes can be read back from their offsets.
			c := &codes{file: file}
			for h, want := range test.toAppend {
				got, err := c.readCodeFromDisk(offsets[h])
				require.NoError(err)
				require.Equal(want, got, "mismatch for hash %v", h)
			}
		})
	}
}

func TestCodes_appendCodes_ErrorCases(t *testing.T) {
	tests := map[string]struct {
		prepare func(t *testing.T) string
	}{
		"cannot open directory as file": {
			prepare: func(t *testing.T) string {
				return t.TempDir()
			},
		},
		"path in non-existent directory": {
			prepare: func(t *testing.T) string {
				return filepath.Join(t.TempDir(), "no", "such", "dir", "codes.dat")
			},
		},
		"read-only file causes flush error": {
			prepare: func(t *testing.T) string {
				dir := t.TempDir()
				file := filepath.Join(dir, fileNameCodes)
				require.NoError(t, os.WriteFile(file, nil, 0600))
				require.NoError(t, os.Chmod(file, 0444))
				return file
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			file := test.prepare(t)
			offsets := make(map[common.Hash]uint64)
			_, err := appendCodes(map[common.Hash][]byte{{1}: {2}}, file, offsets)
			require.Error(t, err)
		})
	}
}

func TestCodes_handleCacheSet_InsertsIntoCache(t *testing.T) {
	require := require.New(t)
	c, err := openCodes(t.TempDir())
	require.NoError(err)
	c.cache = common.NewLruCache[common.Hash, []byte](5)

	key := common.Hash{1}
	value := []byte{0xAA}
	c.handleCacheSet(key, value)

	got, found := c.cache.Get(key)
	require.True(found)
	require.Equal(value, got)
	require.Empty(c.pending)
}

func TestCodes_handleCacheSet_EvictedEntryGoesToPending(t *testing.T) {
	require := require.New(t)
	c, err := openCodes(t.TempDir())
	require.NoError(err)
	c.cache = common.NewLruCache[common.Hash, []byte](2)

	// Fill cache: {10} is LRU, {11} is MRU.
	c.cache.Set(common.Hash{10}, []byte{1})
	c.cache.Set(common.Hash{11}, []byte{2})

	// Insert a new key — evicts {10} into pending.
	c.handleCacheSet(common.Hash{99}, []byte{0xFF})

	_, found := c.pending[common.Hash{10}]
	require.True(found, "evicted entry should be in pending")
}

func TestCodes_handleCacheSet_EvictedEntryDiscardedWhenOnDisk(t *testing.T) {
	require := require.New(t)
	c, err := openCodes(t.TempDir())
	require.NoError(err)
	c.cache = common.NewLruCache[common.Hash, []byte](2)

	// Fill cache and mark {10} as already on disk.
	c.cache.Set(common.Hash{10}, []byte{1})
	c.cache.Set(common.Hash{11}, []byte{2})
	c.codes[common.Hash{10}] = 0

	// Insert a new key — evicts {10}, but it's on disk so not added to pending.
	c.handleCacheSet(common.Hash{99}, []byte{0xFF})

	_, found := c.pending[common.Hash{10}]
	require.False(found, "on-disk entry should not go to pending")
	require.Empty(c.pending)
}

func TestCodes_handleCacheSet_FlushesWhenPendingReachesThreshold(t *testing.T) {
	require := require.New(t)
	c, err := openCodes(t.TempDir())
	require.NoError(err)
	c.cache = common.NewLruCache[common.Hash, []byte](2)

	// Fill pending to threshold - 1.
	for i := 0; i < pendingFlushThreshold-1; i++ {
		h := common.Hash{byte(i), byte(i >> 8), byte(i >> 16), 0xAA}
		c.pending[h] = []byte{byte(i)}
	}
	require.Equal(pendingFlushThreshold-1, len(c.pending))

	// Fill cache so next insert evicts into pending, reaching threshold.
	c.cache.Set(common.Hash{0xFF, 0x01}, []byte{1})
	c.cache.Set(common.Hash{0xFF, 0x02}, []byte{2})

	c.handleCacheSet(common.Hash{0xFF, 0x03}, []byte{3})

	// Flush should have been triggered, clearing pending.
	require.Empty(c.pending)
	// All flushed entries should now be in c.codes (on disk).
	require.Equal(pendingFlushThreshold, len(c.codes))
}

func TestCodes_writeCode_ReturnsError(t *testing.T) {
	injectedErr := errors.New("injected write error")

	tests := map[string]struct {
		// failOnCall is the 1-based index of the Write call that should fail.
		// writeCode makes 3 Write calls: hash, length, code.
		failOnCall int
	}{
		"error writing hash":   {failOnCall: 1},
		"error writing length": {failOnCall: 2},
		"error writing code":   {failOnCall: 3},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			ctrl := gomock.NewController(t)
			file := utils.NewMockOsFile(ctrl)

			// Succeed for writes before the failing one.
			for i := 1; i < test.failOnCall; i++ {
				file.EXPECT().Write(gomock.Any()).Return(0, nil)
			}
			// Fail on the target call.
			file.EXPECT().Write(gomock.Any()).Return(0, injectedErr)

			h := common.Hash{7}
			code := []byte{1, 2, 3}
			err := writeCode(h, code, file)
			require.ErrorIs(t, err, injectedErr)
		})
	}
}

func codeSize(t *testing.T, c *codes) int {
	t.Helper()
	codes, err := c.getCodes()
	require.NoError(t, err)
	return len(codes)
}
