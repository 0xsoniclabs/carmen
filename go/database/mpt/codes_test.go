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
	"maps"
	"os"
	"path/filepath"
	"testing"

	"github.com/0xsoniclabs/carmen/go/backend/kv_file"
	"github.com/0xsoniclabs/carmen/go/backend/utils/checkpoint"
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
	"golang.org/x/crypto/sha3"
)

func TestCodes_OpenCodes(t *testing.T) {
	require := require.New(t)
	codes, err := openCodes(t.TempDir())
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}
	defer func() { require.NoError(codes.Close()) }()

	got, err := codes.getCodesForTesting()
	require.NoError(err)
	if want, got := 0, len(got); want != got {
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
	require := require.New(t)
	codes, err := openCodes(t.TempDir())
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}
	defer func() { require.NoError(codes.Close()) }()

	code1 := []byte("code1")
	code2 := []byte("code2")

	hash1, err := codes.add(code1)
	require.NoError(err)
	hash2, err := codes.add(code2)
	require.NoError(err)

	all, err := codes.getCodesForTesting()
	require.NoError(err)
	if want, got := 2, len(all); want != got {
		t.Fatalf("expected codes to have 2 entries, got %d", got)
	}

	got, err := codes.getCodeForHash(hash1)
	require.NoError(err)
	if want, got := code1, got; string(want) != string(got) {
		t.Fatalf("expected code1, got %s", got)
	}

	got, err = codes.getCodeForHash(hash2)
	require.NoError(err)
	if want, got := code2, got; string(want) != string(got) {
		t.Fatalf("expected code2, got %s", got)
	}
}

func TestCodes_Flush_EmptyCodesCanBeFlushed(t *testing.T) {
	dir := t.TempDir()
	codes, err := openCodes(dir)
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}
	defer func() { require.NoError(t, codes.Close()) }()

	if err := codes.Flush(); err != nil {
		t.Fatalf("failed to flush: %v", err)
	}

	file, _ := getCodePaths(dir)
	stats, err := os.Stat(file)
	if err != nil {
		t.Fatalf("failed to stat file: %v", err)
	}
	if want, got := int64(0), stats.Size(); want != got {
		t.Fatalf("expected file size to be %d, got %d", want, got)
	}
}

func TestCodes_Flush_CodesAreWrittenIncrementally(t *testing.T) {
	dir := t.TempDir()
	codes, err := openCodes(dir)
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}
	defer func() { require.NoError(t, codes.Close()) }()
	file, _ := getCodePaths(dir)

	code1 := []byte("code1")
	code2 := []byte("code2")
	code3 := []byte("code3")

	_, err = codes.add(code1)
	require.NoError(t, err)
	_, err = codes.add(code2)
	require.NoError(t, err)

	if err := codes.Flush(); err != nil {
		t.Fatalf("failed to flush: %v", err)
	}

	snapshot1, err := os.ReadFile(file)
	if err != nil {
		t.Fatalf("failed to read file: %v", err)
	}

	// The next step is incremental.
	_, err = codes.add(code3)
	require.NoError(t, err)

	if err := codes.Flush(); err != nil {
		t.Fatalf("failed to flush: %v", err)
	}

	snapshot2, err := os.ReadFile(file)
	if err != nil {
		t.Fatalf("failed to read file: %v", err)
	}

	if !bytes.HasPrefix(snapshot2, snapshot1) {
		t.Fatalf("expected snapshot2 to be a continuation of snapshot1")
	}
}

// TestCodes_Add_KnownCodesAreNotRewritten verifies that re-adding a code that
// is already stored does not append a duplicate record to the code file —
// neither within a session nor after a reopen.
func TestCodes_Add_KnownCodesAreNotRewritten(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	codes, err := openCodes(dir)
	require.NoError(err)
	file, _ := getCodePaths(dir)

	code := []byte("code1")
	hash1, err := codes.add(code)
	require.NoError(err)
	require.NoError(codes.Flush())

	info, err := os.Stat(file)
	require.NoError(err)
	sizeAfterFirstAdd := info.Size()

	// Re-adding the same code within the session must not grow the file.
	hash2, err := codes.add(code)
	require.NoError(err)
	require.Equal(hash1, hash2)
	require.NoError(codes.Flush())

	info, err = os.Stat(file)
	require.NoError(err)
	require.Equal(sizeAfterFirstAdd, info.Size())

	// The same must hold for codes loaded from disk after a reopen.
	require.NoError(codes.Close())
	codes, err = openCodes(dir)
	require.NoError(err)
	defer func() { require.NoError(codes.Close()) }()

	hash3, err := codes.add(code)
	require.NoError(err)
	require.Equal(hash1, hash3)
	require.NoError(codes.Flush())

	info, err = os.Stat(file)
	require.NoError(err)
	require.Equal(sizeAfterFirstAdd, info.Size())
}

func TestCodes_getCodes_ReturnsAllCodes(t *testing.T) {
	require := require.New(t)
	codes, err := openCodes(t.TempDir())
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}
	defer func() { require.NoError(codes.Close()) }()

	code1 := []byte("code1")
	code2 := []byte("code2")

	hash1, err := codes.add(code1)
	require.NoError(err)
	hash2, err := codes.add(code2)
	require.NoError(err)

	it, err := codes.getCodes()
	require.NoError(err)
	got := maps.Collect(it)

	if want, got := 2, len(got); want != got {
		t.Fatalf("expected %d codes, got %d", want, got)
	}

	if want, got := code1, got[hash1]; !bytes.Equal(want, got) {
		t.Fatalf("expected %x, got %x", want, got)
	}

	if want, got := code2, got[hash2]; !bytes.Equal(want, got) {
		t.Fatalf("expected %x, got %x", want, got)
	}
}

func TestCodes_getCodes_ReturnsErrorOnIterateError(t *testing.T) {
	require := require.New(t)
	ctrl := gomock.NewController(t)

	codeFile := kv_file.NewMockKVFileWithMemoryFootprint[common.Hash, []byte](ctrl)
	injected := errors.New("iterate failed")
	codeFile.EXPECT().Iterate().Return(nil, injected)

	cached, err := kv_file.OpenKVCachedFile[common.Hash, []byte](codeFile, 1, 1)
	require.NoError(err)

	c := &codes{codes: cached}
	got, err := c.getCodes()
	require.ErrorIs(err, injected)
	require.Nil(got)
}

func TestCodes_getCodes_IteratorTerminatesGracefullyOnClose(t *testing.T) {
	require := require.New(t)
	codes, err := openCodes(t.TempDir())
	require.NoError(err)

	_, err = codes.add([]byte("code1"))
	require.NoError(err)
	_, err = codes.add([]byte("code2"))
	require.NoError(err)

	it, err := codes.getCodes()
	require.NoError(err)

	require.NoError(codes.Close())

	// After the store has been closed, iterating the previously obtained
	// sequence must not panic; it may yield fewer entries than the store
	// contained but must terminate gracefully.
	for range it {
	}
}

func TestCodes_GetMemoryFootprint_ReturnsProperSize(t *testing.T) {
	codes, err := openCodes(t.TempDir())
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}
	defer func() { require.NoError(t, codes.Close()) }()

	code1 := []byte("short")
	code2 := []byte("something longer")

	_, err = codes.add(code1)
	require.NoError(t, err)
	_, err = codes.add(code2)
	require.NoError(t, err)

	footprint := codes.GetMemoryFootprint()
	if got := footprint.Total(); got == 0 {
		t.Fatalf("expected non-zero footprint, got %d", got)
	}
}

func TestCodes_GuaranteeCheckpoint_PendingCheckpointIsCommitted(t *testing.T) {
	codes, err := openCodes(t.TempDir())
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}
	defer func() { require.NoError(t, codes.Close()) }()

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
	defer func() { require.NoError(t, codes.Close()) }()
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
	defer func() { require.NoError(t, codes.Close()) }()

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
	require := require.New(t)
	ctrl := gomock.NewController(t)

	// The code file retains its file descriptor for its whole lifetime, so
	// permission changes on the file cannot make a flush fail. The failure is
	// therefore injected through a mock of the underlying file.
	injected := errors.New("injected flush failure")
	file := kv_file.NewMockKVFileWithMemoryFootprint[common.Hash, []byte](ctrl)
	file.EXPECT().Has(gomock.Any()).Return(false, nil)
	file.EXPECT().SetBatch(gomock.Any()).Return(injected)

	cached, err := kv_file.OpenKVCachedFile[common.Hash, []byte](file, 10, 10)
	require.NoError(err)

	c := &codes{
		codes:     cached,
		hasher:    sha3.NewLegacyKeccak256(),
		directory: t.TempDir(),
	}
	_, err = c.add([]byte("code1"))
	require.NoError(err)

	cp1 := checkpoint.Checkpoint(1)
	require.ErrorIs(c.Prepare(cp1), injected)
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
			defer func() { require.NoError(t, codes.Close()) }()

			_, err = codes.add([]byte("code1"))
			require.NoError(t, err)

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
			_, err = codes.add([]byte("code1"))
			require.NoError(t, err)

			cp1 := checkpoint.Checkpoint(1)
			if err := codes.Prepare(cp1); err != nil {
				t.Fatalf("failed to prepare checkpoint: %v", err)
			}
			if name == "committed" {
				if err := codes.Commit(cp1); err != nil {
					t.Fatalf("failed to commit checkpoint: %v", err)
				}
			}

			_, err = codes.add([]byte("code2"))
			require.NoError(t, err)
			if err := codes.Flush(); err != nil {
				t.Fatalf("failed to flush: %v", err)
			}
			require.NoError(t, codes.Close())

			codes, err = openCodes(dir)
			if err != nil {
				t.Fatalf("failed to re-open original codes: %v", err)
			}

			stored, err := codes.getCodesForTesting()
			require.NoError(t, err)
			if want, got := 2, len(stored); want != got {
				t.Fatalf("expected codes to have %d entries, got %d", want, got)
			}
			require.NoError(t, codes.Close())

			if err := getCodeRestorer(dir).Restore(cp1); err != nil {
				t.Fatalf("failed to restore checkpoint: %v", err)
			}

			codes, err = openCodes(dir)
			if err != nil {
				t.Fatalf("failed to re-open recovered codes: %v", err)
			}
			defer func() { require.NoError(t, codes.Close()) }()

			stored, err = codes.getCodesForTesting()
			require.NoError(t, err)
			if want, got := 1, len(stored); want != got {
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
			file, _ := getCodePaths(dir)

			// Prepare a valid code state.
			codes, err := openCodes(dir)
			if err != nil {
				t.Fatalf("failed to open codes: %v", err)
			}
			defer func() { require.NoError(t, codes.Close()) }()

			_, err = codes.add([]byte("code1"))
			require.NoError(t, err)
			_, err = codes.add([]byte("code2"))
			require.NoError(t, err)

			cp := checkpoint.Checkpoint(1)
			if err := codes.Prepare(cp); err != nil {
				t.Fatalf("failed to prepare checkpoint: %v", err)
			}
			if err := codes.Commit(cp); err != nil {
				t.Fatalf("failed to commit checkpoint: %v", err)
			}

			backup, err := os.ReadFile(file)
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
			restored, err := os.ReadFile(file)
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

	_, err = codes.add([]byte("code1"))
	require.NoError(t, err)
	_, err = codes.add([]byte("code2"))
	require.NoError(t, err)

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

	_, err = codes.add([]byte("code3"))
	require.NoError(t, err)
	stored, err := codes.getCodesForTesting()
	require.NoError(t, err)
	if want, got := 3, len(stored); want != got {
		t.Fatalf("expected codes to have %d entries, got %d", want, got)
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
	require.NoError(t, codes.Close())

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
	defer func() { require.NoError(t, codes.Close()) }()

	restoredCodes, err := codes.getCodesForTesting()
	require.NoError(t, err)
	if want, got := 2, len(restoredCodes); want != got {
		t.Fatalf("expected codes to have %d entries, got %d", want, got)
	}
}

func TestCodes_CheckpointsCanBeAborted(t *testing.T) {
	dir := t.TempDir()
	codes, err := openCodes(dir)
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}

	_, err = codes.add([]byte("code1"))
	require.NoError(t, err)
	_, err = codes.add([]byte("code2"))
	require.NoError(t, err)

	cp := checkpoint.Checkpoint(1)
	if err := codes.Prepare(cp); err != nil {
		t.Fatalf("failed to prepare checkpoint: %v", err)
	}

	if err := codes.Abort(cp); err != nil {
		t.Fatalf("failed to commit checkpoint: %v", err)
	}

	stored, err := codes.getCodesForTesting()
	require.NoError(t, err)
	if want, got := 2, len(stored); want != got {
		t.Fatalf("expected codes to have %d entries, got %d", want, got)
	}
	require.NoError(t, codes.Close())

	cp = checkpoint.Checkpoint(0)
	if err := getCodeRestorer(dir).Restore(cp); err != nil {
		t.Fatalf("failed to restore checkpoint: %v", err)
	}

	codes, err = openCodes(dir)
	if err != nil {
		t.Fatalf("failed to re-open recovered codes: %v", err)
	}
	defer func() { require.NoError(t, codes.Close()) }()

	stored, err = codes.getCodesForTesting()
	require.NoError(t, err)
	if want, got := 0, len(stored); want != got {
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

	_, err = codes.add([]byte("code1"))
	require.NoError(t, err)

	if _, err := coordinator.CreateCheckpoint(); err != nil {
		t.Fatalf("failed to create checkpoint: %v", err)
	}

	_, err = codes.add([]byte("code2"))
	require.NoError(t, err)
	require.NoError(t, codes.Close())

	if err := getCodeRestorer(dir).Restore(coordinator.GetCurrentCheckpoint()); err != nil {
		t.Fatalf("failed to restore checkpoint: %v", err)
	}

	codes, err = openCodes(dir)
	if err != nil {
		t.Fatalf("failed to re-open recovered codes: %v", err)
	}
	defer func() { require.NoError(t, codes.Close()) }()

	stored, err := codes.getCodesForTesting()
	require.NoError(t, err)
	if want, got := 1, len(stored); want != got {
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

	if err := writeCodesForTesting(codes, file); err != nil {
		t.Fatalf("failed to write codes: %v", err)
	}

	readCodes, err := readCodesForTesting(file)
	if err != nil {
		t.Fatalf("failed to read codes: %v", err)
	}

	if want, got := 2, len(readCodes); want != got {
		t.Fatalf("expected codes to have %d entries, got %d", want, got)
	}
}

func TestCodes_writeCodes_TruncatesFileIfItExists(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, fileNameCodes)

	codes1 := map[common.Hash][]byte{
		{1}: {5},
		{2}: {7, 8},
	}

	if err := writeCodesForTesting(codes1, file); err != nil {
		t.Fatalf("failed to write codes: %v", err)
	}

	codes2 := map[common.Hash][]byte{
		{3}: {9},
	}

	if err := writeCodesForTesting(codes2, file); err != nil {
		t.Fatalf("failed to write codes: %v", err)
	}

	readCodes, err := readCodesForTesting(file)
	if err != nil {
		t.Fatalf("failed to read codes: %v", err)
	}

	if want, got := 1, len(readCodes); want != got {
		t.Fatalf("expected codes to have %d entries, got %d", want, got)
	}
}

func TestCodes_writeCodes_CannotCreateTheOutputFile(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, fileNameCodesCheckpointDirectory)
	if err := os.Mkdir(file, os.FileMode(0400)); err != nil {
		t.Fatalf("cannot create dir: %s", err)
	}
	require.NoError(t, os.Chmod(dir, 0500))
	defer func() { require.NoError(t, os.Chmod(dir, 0777)) }()
	if err := writeCodesForTesting(make(map[common.Hash][]byte, 1), file); err == nil {
		t.Errorf("writing roots should fail")
	}
}

func TestCodes_readCodesAndSize_ReadingNonExistingFileReturnsEmptyCodeMap(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, fileNameCodes)
	codes, err := readCodesForTesting(path)
	if err != nil {
		t.Fatalf("failed to read codes: %v", err)
	}
	if want, got := 0, len(codes); want != got {
		t.Fatalf("expected codes to be empty, got %d", got)
	}
}

func TestCodes_readCodesAndSize_ReadingIssuesAreReported(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, fileNameCodes)

	if err := os.WriteFile(path, []byte("invalid"), 0600); err != nil {
		t.Fatalf("failed to prepare invalid code file: %v", err)
	}

	_, err := readCodesForTesting(path)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
}

func TestCodes_readCodesAndSize_PermissionErrorsAreDetected(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, fileNameCodes)

	if err := os.Chmod(dir, 0000); err != nil {
		t.Fatalf("failed to change directory permissions: %v", err)
	}
	defer func() { require.NoError(t, os.Chmod(dir, 0700)) }()

	_, err := readCodesForTesting(path)
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
	if _, err := readCodesForTesting(file); err == nil {
		t.Errorf("reading codes should fail")
	}
}

func TestCodes_add_ReturnsHashOfCodeAndStoresIt(t *testing.T) {
	tests := map[string][]byte{
		"empty":  {},
		"short":  []byte("code1"),
		"binary": {0x00, 0xff, 0x10, 0x20},
		"long":   bytes.Repeat([]byte("x"), 4096),
	}

	for name, code := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			codes, err := openCodes(t.TempDir())
			require.NoError(err)
			defer func() { require.NoError(codes.Close()) }()

			hash, err := codes.add(code)
			require.NoError(err)
			require.Equal(common.GetKeccak256Hash(code), hash,
				"add must return the keccak256 hash of the code")
			got, err := codes.getCodeForHash(hash)
			require.NoError(err)
			require.Equal(code, got,
				"the code must be retrievable via the returned hash")
		})
	}
}

func TestCodes_getCodeForHash_ReturnsCodeForKnownHashAndNilOtherwise(t *testing.T) {
	dir := t.TempDir()
	codes, err := openCodes(dir)
	require.NoError(t, err)
	defer func() { require.NoError(t, codes.Close()) }()

	code1 := []byte("code1")
	code2 := []byte("another code")
	hash1, err := codes.add(code1)
	require.NoError(t, err)
	hash2, err := codes.add(code2)
	require.NoError(t, err)

	tests := map[string]struct {
		hash common.Hash
		want []byte
	}{
		"known hash returns code":        {hash1, code1},
		"second known hash returns code": {hash2, code2},
		"unknown hash returns nil":       {common.Hash{0xde, 0xad}, nil},
		"zero hash returns nil":          {common.Hash{}, nil},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := codes.getCodeForHash(tc.hash)
			require.NoError(t, err)
			require.Equal(t, tc.want, got)
		})
	}

	// Flushing to disk and re-opening must not change the lookup behaviour.
	require.NoError(t, codes.Flush())
	reopened, err := openCodes(dir)
	require.NoError(t, err)
	defer func() { require.NoError(t, reopened.Close()) }()
	got, err := reopened.getCodeForHash(hash1)
	require.NoError(t, err)
	require.Equal(t, code1, got,
		"code must still be retrievable after flush + reopen")
	got, err = reopened.getCodeForHash(hash2)
	require.NoError(t, err)
	require.Equal(t, code2, got,
		"code must still be retrievable after flush + reopen")
	got, err = reopened.getCodeForHash(common.Hash{0xde, 0xad})
	require.NoError(t, err)
	require.Nil(t, got,
		"unknown hash must still return nil after flush + reopen")
}

func TestCodes_Flush_CodesArePersistedOnDisk(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()

	codes, err := openCodes(dir)
	require.NoError(err)
	defer func() { require.NoError(codes.Close()) }()

	want := map[common.Hash][]byte{}
	for _, code := range [][]byte{
		[]byte("code1"),
		[]byte("another code"),
		{0x00, 0xff, 0x10, 0x20},
		bytes.Repeat([]byte("x"), 1024),
	} {
		hash, err := codes.add(code)
		require.NoError(err)
		want[hash] = code
	}

	require.NoError(codes.Flush())

	// Re-opening the store must yield the exact same set of codes read back
	// from disk.
	reopened, err := openCodes(dir)
	require.NoError(err)
	defer func() { require.NoError(reopened.Close()) }()

	got, err := reopened.getCodesForTesting()
	require.NoError(err)
	require.Equal(len(want), len(got))
	for hash, code := range want {
		require.Equal(code, got[hash],
			"code for hash %x must survive flush + reopen unchanged", hash)
	}
}
