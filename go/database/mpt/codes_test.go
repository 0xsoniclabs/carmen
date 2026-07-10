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

import "testing"

func TestCodes_OpenCodes(t *testing.T) {
	codes, err := openCodes(t.TempDir())
	if err != nil {
		t.Fatalf("failed to open codes: %v", err)
	}

	if want, got := uint64(0), codes.codes.FileSize(); want != got {
		t.Fatalf("expected codes to be empty, got %d", got)
	}
}

// func TestCodes_OpenCodes_IOErrorsAreHandled(t *testing.T) {
// 	tests := map[string]func(t *testing.T) string{
// 		"invalid directory": func(t *testing.T) string {
// 			dir := t.TempDir()
// 			file := filepath.Join(dir, "file")
// 			if err := os.WriteFile(file, []byte{}, 0600); err != nil {
// 				t.Fatalf("failed to create file: %v", err)
// 			}
// 			return file //< passing a file instead of a directory
// 		},
// 		"missing directory permissions": func(t *testing.T) string {
// 			dir := t.TempDir()
// 			stat, err := os.Stat(dir)
// 			if err != nil {
// 				t.Fatalf("failed to stat directory: %v", err)
// 			}
// 			if err := os.Chmod(dir, 0500); err != nil {
// 				t.Fatalf("failed to change directory permissions: %v", err)
// 			}
// 			t.Cleanup(func() {
// 				os.Chmod(dir, stat.Mode())
// 			})
// 			return dir
// 		},
// 		"missing permissions to create code file": func(t *testing.T) string {
// 			dir := t.TempDir()
// 			// the code directory must exist to reach the code file creation
// 			if err := os.MkdirAll(filepath.Join(dir, fileNameCodesCheckpointDirectory), 0700); err != nil {
// 				t.Fatalf("failed to create codes directory: %v", err)
// 			}
// 			stat, err := os.Stat(dir)
// 			if err != nil {
// 				t.Fatalf("failed to stat directory: %v", err)
// 			}
// 			if err := os.Chmod(dir, 0500); err != nil {
// 				t.Fatalf("failed to change directory permissions: %v", err)
// 			}
// 			t.Cleanup(func() {
// 				os.Chmod(dir, stat.Mode())
// 			})
// 			return dir
// 		},
// 		"missing permissions to read code file": func(t *testing.T) string {
// 			dir := t.TempDir()
// 			file := filepath.Join(dir, fileNameCodes)
// 			if err := os.WriteFile(file, []byte{}, 0600); err != nil {
// 				t.Fatalf("failed to create file: %v", err)
// 			}
// 			if err := os.Chmod(file, 0200); err != nil {
// 				t.Fatalf("failed to change file permissions: %v", err)
// 			}
// 			t.Cleanup(func() {
// 				os.Chmod(file, 0600)
// 			})
// 			return dir
// 		},
// 		"missing permissions to read checkpoint data": func(t *testing.T) string {
// 			dir := t.TempDir()
// 			nested := filepath.Join(dir, fileNameCodesCheckpointDirectory)
// 			if err := os.MkdirAll(nested, 0700); err != nil {
// 				t.Fatalf("failed to create codes directory: %v", err)
// 			}
// 			file := filepath.Join(nested, fileNameCodesCommittedCheckpoint)
// 			if err := os.WriteFile(file, []byte{}, 0600); err != nil {
// 				t.Fatalf("failed to create file: %v", err)
// 			}
// 			if err := os.Chmod(file, 0200); err != nil {
// 				t.Fatalf("failed to change file permissions: %v", err)
// 			}
// 			t.Cleanup(func() {
// 				os.Chmod(file, 0600)
// 			})
// 			return dir
// 		},
// 	}

// 	for name, prepare := range tests {
// 		t.Run(name, func(t *testing.T) {
// 			dir := prepare(t)
// 			_, err := openCodes(dir)
// 			if err == nil {
// 				t.Fatalf("expected error, got nil")
// 			}
// 		})
// 	}
// }

// func TestCodes_CodesCanBeAddedAndRetrieved(t *testing.T) {
// 	codes, err := openCodes(t.TempDir())
// 	if err != nil {
// 		t.Fatalf("failed to open codes: %v", err)
// 	}

// 	code1 := []byte("code1")
// 	code2 := []byte("code2")

// 	hash1 := codes.add(code1)
// 	hash2 := codes.add(code2)

// 	if want, got := 2, len(codes.offsets); want != got {
// 		t.Fatalf("expected codes to have 2 entries, got %d", got)
// 	}

// 	if want, got := code1, codes.getCodeForHash(hash1); string(want) != string(got) {
// 		t.Fatalf("expected code1, got %s", got)
// 	}

// 	if want, got := code2, codes.getCodeForHash(hash2); string(want) != string(got) {
// 		t.Fatalf("expected code2, got %s", got)
// 	}
// }

// func TestCodes_Flush_EmptyCodesCanBeFlushed(t *testing.T) {
// 	codes, err := openCodes(t.TempDir())
// 	if err != nil {
// 		t.Fatalf("failed to open codes: %v", err)
// 	}

// 	if err := codes.Flush(); err != nil {
// 		t.Fatalf("failed to flush: %v", err)
// 	}

// 	stats, err := os.Stat(codes.file)
// 	if err != nil {
// 		t.Fatalf("failed to stat file: %v", err)
// 	}
// 	if want, got := int64(0), stats.Size(); want != got {
// 		t.Fatalf("expected file size to be %d, got %d", want, got)
// 	}
// }

// // func TestCodes_Flush_CodesAreWrittenIncrementally(t *testing.T) {
// // 	codes, err := openCodes(t.TempDir())
// // 	if err != nil {
// // 		t.Fatalf("failed to open codes: %v", err)
// // 	}

// // 	code1 := []byte("code1")
// // 	code2 := []byte("code2")
// // 	code3 := []byte("code3")

// // 	codes.add(code1)
// // 	codes.add(code2)

// // 	if want, got := 2, len(codes.pending); want != got {
// // 		t.Fatalf("expected %d pending codes, got %d", want, got)
// // 	}

// // 	if err := codes.Flush(); err != nil {
// // 		t.Fatalf("failed to flush: %v", err)
// // 	}

// // 	if want, got := 0, len(codes.pending); want != got {
// // 		t.Fatalf("expected %d pending codes, got %d", want, got)
// // 	}

// // 	snapshot1, err := os.ReadFile(codes.file)
// // 	if err != nil {
// // 		t.Fatalf("failed to read file: %v", err)
// // 	}

// // 	if codes.fileSize != uint64(len(snapshot1)) {
// // 		t.Fatalf("expected file size to be %d, got %d", len(snapshot1), codes.fileSize)
// // 	}

// // 	// The next step is incremental.
// // 	codes.add(code3)

// // 	if want, got := 1, len(codes.pending); want != got {
// // 		t.Fatalf("expected %d pending codes, got %d", want, got)
// // 	}

// // 	if err := codes.Flush(); err != nil {
// // 		t.Fatalf("failed to flush: %v", err)
// // 	}

// // 	if want, got := 0, len(codes.pending); want != got {
// // 		t.Fatalf("expected %d pending codes, got %d", want, got)
// // 	}

// // 	snapshot2, err := os.ReadFile(codes.file)
// // 	if err != nil {
// // 		t.Fatalf("failed to read file: %v", err)
// // 	}

// // 	if codes.fileSize != uint64(len(snapshot2)) {
// // 		t.Fatalf("expected file size to be %d, got %d", len(snapshot2), codes.fileSize)
// // 	}

// // 	if !bytes.HasPrefix(snapshot2, snapshot1) {
// // 		t.Fatalf("expected snapshot2 to be a continuation of snapshot1")
// // 	}
// // }

// func TestCodes_getCodes_ReturnsAllCodes(t *testing.T) {
// 	codes, err := openCodes(t.TempDir())
// 	if err != nil {
// 		t.Fatalf("failed to open codes: %v", err)
// 	}

// 	code1 := []byte("code1")
// 	code2 := []byte("code2")

// 	hash1 := codes.add(code1)
// 	hash2 := codes.add(code2)

// 	got := codes.getCodes()

// 	if want, got := 2, len(got); want != got {
// 		t.Fatalf("expected %d codes, got %d", want, got)
// 	}

// 	if want, got := code1, got[hash1]; !bytes.Equal(want, got) {
// 		t.Fatalf("expected %x, got %x", want, got)
// 	}

// 	if want, got := code2, got[hash2]; !bytes.Equal(want, got) {
// 		t.Fatalf("expected %x, got %x", want, got)
// 	}
// }

// func TestCodes_GetMemoryFootprint_ReturnsProperSize(t *testing.T) {
// 	codes, err := openCodes(t.TempDir())
// 	if err != nil {
// 		t.Fatalf("failed to open codes: %v", err)
// 	}

// 	code1 := []byte("short")
// 	code2 := []byte("something longer")

// 	codes.add(code1)
// 	codes.add(code2)

// 	footprint := codes.GetMemoryFootprint()
// 	want := unsafe.Sizeof(*codes) + uintptr(len(code1)+len(code2)+2*32)
// 	got := footprint.Total()
// 	if want != got {
// 		t.Fatalf("expected %d, got %d", want, got)
// 	}
// }

// func TestCodes_GuaranteeCheckpoint_PendingCheckpointIsCommitted(t *testing.T) {
// 	codes, err := openCodes(t.TempDir())
// 	if err != nil {
// 		t.Fatalf("failed to open codes: %v", err)
// 	}

// 	cp0 := checkpoint.Checkpoint(0)

// 	if err := codes.GuaranteeCheckpoint(cp0); err != nil {
// 		t.Fatalf("failed to guarantee initial checkpoint: %v", err)
// 	}

// 	cp1 := checkpoint.Checkpoint(1)
// 	if err := codes.Prepare(cp1); err != nil {
// 		t.Fatalf("failed to prepare checkpoint: %v", err)
// 	}

// 	if want, got := cp0, codes.checkpoint; want != got {
// 		t.Fatalf("expected checkpoint to be %d, got %d", want, got)
// 	}

// 	if err := codes.GuaranteeCheckpoint(cp1); err != nil {
// 		t.Fatalf("failed to guarantee pending checkpoint: %v", err)
// 	}

// 	if want, got := cp1, codes.checkpoint; want != got {
// 		t.Fatalf("expected checkpoint to be %d, got %d", want, got)
// 	}

// 	if err := codes.GuaranteeCheckpoint(cp0); err == nil {
// 		t.Fatalf("expected error, got nil")
// 	}
// }

// func TestCodes_GuaranteeCheckpoint_IoErrorsAreHandled(t *testing.T) {
// 	codes, err := openCodes(t.TempDir())
// 	if err != nil {
// 		t.Fatalf("failed to open codes: %v", err)
// 	}
// 	cp1 := checkpoint.Checkpoint(1)
// 	if err := codes.Prepare(cp1); err != nil {
// 		t.Fatalf("failed to prepare checkpoint: %v", err)
// 	}

// 	pendingFile := filepath.Join(codes.directory, fileNameCodesPrepareCheckpoint)
// 	if err := os.WriteFile(pendingFile, []byte("invalid json"), 0600); err != nil {
// 		t.Fatalf("failed to write file: %v", err)
// 	}

// 	if err := codes.GuaranteeCheckpoint(cp1); err == nil {
// 		t.Fatalf("expected error, got nil")
// 	}
// }

// func TestCodes_Prepare_CheckpointIsIncremental(t *testing.T) {
// 	codes, err := openCodes(t.TempDir())
// 	if err != nil {
// 		t.Fatalf("failed to open codes: %v", err)
// 	}

// 	cp1 := checkpoint.Checkpoint(1)
// 	if err := codes.Prepare(cp1); err != nil {
// 		t.Fatalf("failed to prepare initial checkpoint: %v", err)
// 	}

// 	cp2 := checkpoint.Checkpoint(2)
// 	if err := codes.Prepare(cp2); err == nil {
// 		t.Fatalf("expected error, got nil")
// 	}
// }

// func TestCodes_Prepare_FailsIfFlushFails(t *testing.T) {
// 	codes, err := openCodes(t.TempDir())
// 	if err != nil {
// 		t.Fatalf("failed to open codes: %v", err)
// 	}

// 	codes.add([]byte("code1"))

// 	os.Chmod(codes.file, 0400) // make the file read-only
// 	defer os.Chmod(codes.file, 0600)

// 	cp1 := checkpoint.Checkpoint(1)
// 	if err := codes.Prepare(cp1); err == nil {
// 		t.Fatalf("expected error, got nil")
// 	}
// }

// func TestCodes_Commit_HandlesIoIssues(t *testing.T) {
// 	tests := map[string]func(*testing.T, string) error{
// 		"missing prepare file": func(t *testing.T, dir string) error {
// 			return os.Remove(filepath.Join(dir, fileNameCodesCheckpointDirectory, fileNameCodesPrepareCheckpoint))
// 		},
// 		"invalid prepare file": func(t *testing.T, dir string) error {
// 			return os.WriteFile(filepath.Join(dir, fileNameCodesCheckpointDirectory, fileNameCodesPrepareCheckpoint), []byte("invalid json"), 0600)
// 		},
// 		"missing rename permissions": func(t *testing.T, dir string) error {
// 			subDir := filepath.Join(dir, fileNameCodesCheckpointDirectory)
// 			if err := os.Chmod(subDir, 0500); err != nil {
// 				return err
// 			}
// 			t.Cleanup(func() {
// 				os.Chmod(subDir, 0700)
// 			})
// 			return nil
// 		},
// 	}

// 	for name, temper := range tests {
// 		t.Run(name, func(t *testing.T) {
// 			dir := t.TempDir()
// 			codes, err := openCodes(dir)
// 			if err != nil {
// 				t.Fatalf("failed to open codes: %v", err)
// 			}

// 			codes.add([]byte("code1"))

// 			cp1 := checkpoint.Checkpoint(1)
// 			if err := codes.Prepare(cp1); err != nil {
// 				t.Fatalf("failed to prepare test: %v", err)
// 			}

// 			if err := temper(t, dir); err != nil {
// 				t.Fatalf("failed to prepare test: %v", err)
// 			}

// 			if err := codes.Commit(cp1); err == nil {
// 				t.Fatalf("expected error, got nil")
// 			}
// 		})
// 	}
// }

// func TestCodes_Restore_CanRestoreCommittedAndPendingCheckpoint(t *testing.T) {
// 	for _, name := range []string{"committed", "pending"} {
// 		t.Run(name, func(t *testing.T) {
// 			dir := t.TempDir()

// 			codes, err := openCodes(dir)
// 			if err != nil {
// 				t.Fatalf("failed to open codes: %v", err)
// 			}
// 			codes.add([]byte("code1"))

// 			cp1 := checkpoint.Checkpoint(1)
// 			if err := codes.Prepare(cp1); err != nil {
// 				t.Fatalf("failed to prepare checkpoint: %v", err)
// 			}
// 			if name == "committed" {
// 				if err := codes.Commit(cp1); err != nil {
// 					t.Fatalf("failed to commit checkpoint: %v", err)
// 				}
// 			}

// 			codes.add([]byte("code2"))
// 			if err := codes.Flush(); err != nil {
// 				t.Fatalf("failed to flush: %v", err)
// 			}

// 			codes, err = openCodes(dir)
// 			if err != nil {
// 				t.Fatalf("failed to re-open original codes: %v", err)
// 			}

// 			if want, got := 2, len(codes.offsets); want != got {
// 				t.Fatalf("expected codes to have %d entries, got %d", want, got)
// 			}

// 			if err := getCodeRestorer(dir).Restore(cp1); err != nil {
// 				t.Fatalf("failed to restore checkpoint: %v", err)
// 			}

// 			codes, err = openCodes(dir)
// 			if err != nil {
// 				t.Fatalf("failed to re-open recovered codes: %v", err)
// 			}

// 			if want, got := 1, len(codes.offsets); want != got {
// 				t.Fatalf("expected codes to have %d entries, got %d", want, got)
// 			}
// 		})
// 	}
// }

// func TestCodes_Restore_InvalidCheckpointMetaDataIsDetected(t *testing.T) {
// 	dir := t.TempDir()
// 	restorer := getCodeRestorer(dir)

// 	subDir := filepath.Join(dir, fileNameCodesCheckpointDirectory)
// 	if err := os.MkdirAll(subDir, 0700); err != nil {
// 		t.Fatalf("failed to create codes directory: %v", err)
// 	}

// 	if err := os.WriteFile(filepath.Join(subDir, fileNameCodesCommittedCheckpoint), []byte("invalid json"), 0600); err != nil {
// 		t.Fatalf("failed to write file: %v", err)
// 	}

// 	cp := checkpoint.Checkpoint(0)
// 	if err := restorer.Restore(cp); err == nil {
// 		t.Fatalf("expected error, got nil")
// 	}
// }

// func TestCodes_Restore_InvalidCheckpointDataIsDetected(t *testing.T) {
// 	dir := t.TempDir()
// 	restorer := getCodeRestorer(dir)

// 	cp := checkpoint.Checkpoint(42) // < non-existing checkpoint
// 	if err := restorer.Restore(cp); err == nil {
// 		t.Fatalf("expected error, got nil")
// 	}
// }

// func TestCodes_Restore_CanHandleErrorCorruptedData(t *testing.T) {
// 	tests := map[string]func(dir string) error{
// 		"no corruption": func(string) error {
// 			return nil
// 		},
// 		"extra data in code file": func(dir string) error {
// 			file, _ := getCodePaths(dir)
// 			data, err := os.ReadFile(file)
// 			if err != nil {
// 				return err
// 			}
// 			data = append(data, []byte("extra")...)
// 			return os.WriteFile(file, data, 0600)
// 		},
// 	}

// 	for name, temper := range tests {
// 		t.Run(name, func(t *testing.T) {
// 			dir := t.TempDir()

// 			// Prepare a valid code state.
// 			codes, err := openCodes(dir)
// 			if err != nil {
// 				t.Fatalf("failed to open codes: %v", err)
// 			}

// 			codes.add([]byte("code1"))
// 			codes.add([]byte("code2"))

// 			cp := checkpoint.Checkpoint(1)
// 			if err := codes.Prepare(cp); err != nil {
// 				t.Fatalf("failed to prepare checkpoint: %v", err)
// 			}
// 			if err := codes.Commit(cp); err != nil {
// 				t.Fatalf("failed to commit checkpoint: %v", err)
// 			}

// 			backup, err := os.ReadFile(codes.file)
// 			if err != nil {
// 				t.Fatalf("failed to read file: %v", err)
// 			}
// 			if len(backup) == 0 {
// 				t.Fatalf("expected file to be non-empty")
// 			}

// 			// Corrupt the code state.
// 			if err := temper(dir); err != nil {
// 				t.Fatalf("failed to corrupt codes: %v", err)
// 			}

// 			// Attempt to restore the code state.
// 			restorer := getCodeRestorer(dir)
// 			if err := restorer.Restore(cp); err != nil {
// 				t.Fatalf("failed to restore checkpoint: %v", err)
// 			}

// 			// Verify the restored state.
// 			restored, err := os.ReadFile(codes.file)
// 			if err != nil {
// 				t.Fatalf("failed to read file: %v", err)
// 			}

// 			if !bytes.Equal(backup, restored) {
// 				t.Fatalf("expected file to be equal after restore")
// 			}
// 		})
// 	}
// }

// func TestCodes_CheckpointsCanBeRestored(t *testing.T) {
// 	dir := t.TempDir()
// 	file, _ := getCodePaths(dir)
// 	codes, err := openCodes(dir)
// 	if err != nil {
// 		t.Fatalf("failed to open codes: %v", err)
// 	}

// 	codes.add([]byte("code1"))
// 	codes.add([]byte("code2"))

// 	checkpoint := checkpoint.Checkpoint(1)
// 	if err := codes.Prepare(checkpoint); err != nil {
// 		t.Fatalf("failed to prepare checkpoint: %v", err)
// 	}

// 	if err := codes.Commit(checkpoint); err != nil {
// 		t.Fatalf("failed to commit checkpoint: %v", err)
// 	}

// 	backup, err := os.Stat(file)
// 	if err != nil {
// 		t.Fatalf("failed to stat file: %v", err)
// 	}

// 	codes.add([]byte("code3"))
// 	if want, got := 3, len(codes.offsets); want != got {
// 		t.Fatalf("expected codes to have %d entries, got %d", want, got)
// 	}

// 	if err := codes.Flush(); err != nil {
// 		t.Fatalf("failed to flush: %v", err)
// 	}

// 	modified, err := os.Stat(file)
// 	if err != nil {
// 		t.Fatalf("failed to stat file: %v", err)
// 	}

// 	if modified.Size() <= backup.Size() {
// 		t.Fatalf("expected file to be larger after flush")
// 	}

// 	if err := getCodeRestorer(dir).Restore(checkpoint); err != nil {
// 		t.Fatalf("failed to restore checkpoint: %v", err)
// 	}

// 	restored, err := os.Stat(file)
// 	if err != nil {
// 		t.Fatalf("failed to stat file: %v", err)
// 	}

// 	if restored.Size() != backup.Size() {
// 		t.Fatalf("expected file to be same size after restore")
// 	}

// 	codes, err = openCodes(dir)
// 	if err != nil {
// 		t.Fatalf("failed to re-open recovered codes: %v", err)
// 	}

// 	if want, got := 2, len(codes.offsets); want != got {
// 		t.Fatalf("expected codes to have %d entries, got %d", want, got)
// 	}
// }

// func TestCodes_CheckpointsCanBeAborted(t *testing.T) {
// 	dir := t.TempDir()
// 	codes, err := openCodes(dir)
// 	if err != nil {
// 		t.Fatalf("failed to open codes: %v", err)
// 	}

// 	codes.add([]byte("code1"))
// 	codes.add([]byte("code2"))

// 	cp := checkpoint.Checkpoint(1)
// 	if err := codes.Prepare(cp); err != nil {
// 		t.Fatalf("failed to prepare checkpoint: %v", err)
// 	}

// 	if err := codes.Abort(cp); err != nil {
// 		t.Fatalf("failed to commit checkpoint: %v", err)
// 	}

// 	if want, got := 2, len(codes.offsets); want != got {
// 		t.Fatalf("expected codes to have %d entries, got %d", want, got)
// 	}

// 	cp = checkpoint.Checkpoint(0)
// 	if err := getCodeRestorer(dir).Restore(cp); err != nil {
// 		t.Fatalf("failed to restore checkpoint: %v", err)
// 	}

// 	codes, err = openCodes(dir)
// 	if err != nil {
// 		t.Fatalf("failed to re-open recovered codes: %v", err)
// 	}

// 	if want, got := 0, len(codes.offsets); want != got {
// 		t.Fatalf("expected codes to have %d entries, got %d", want, got)
// 	}
// }

// func TestCodes_CanBeHandledByCheckpointCoordinator(t *testing.T) {
// 	dir := t.TempDir()
// 	codes, err := openCodes(dir)
// 	if err != nil {
// 		t.Fatalf("failed to open codes: %v", err)
// 	}

// 	coordinator, err := checkpoint.NewCoordinator(t.TempDir(), codes)
// 	if err != nil {
// 		t.Fatalf("failed to create coordinator: %v", err)
// 	}

// 	codes.add([]byte("code1"))

// 	if _, err := coordinator.CreateCheckpoint(); err != nil {
// 		t.Fatalf("failed to create checkpoint: %v", err)
// 	}

// 	codes.add([]byte("code2"))

// 	if err := getCodeRestorer(dir).Restore(coordinator.GetCurrentCheckpoint()); err != nil {
// 		t.Fatalf("failed to restore checkpoint: %v", err)
// 	}

// 	codes, err = openCodes(dir)
// 	if err != nil {
// 		t.Fatalf("failed to re-open recovered codes: %v", err)
// 	}

// 	if want, got := 1, len(codes.offsets); want != got {
// 		t.Fatalf("expected codes to have %d entries, got %d", want, got)
// 	}

// }

// func TestCodes_writeCodes_WritesCodesToFile(t *testing.T) {
// 	dir := t.TempDir()
// 	file := filepath.Join(dir, fileNameCodes)

// 	codes := map[common.Hash][]byte{
// 		{1}: {5},
// 		{2}: {7, 8},
// 	}

// 	if err := writeCodes(codes, file); err != nil {
// 		t.Fatalf("failed to write codes: %v", err)
// 	}

// 	readCodes, _, err := readCodesAndSize(file)
// 	if err != nil {
// 		t.Fatalf("failed to read codes: %v", err)
// 	}

// 	if want, got := 2, len(readCodes); want != got {
// 		t.Fatalf("expected codes to have %d entries, got %d", want, got)
// 	}
// }

// // func TestCodes_writeCodes_WriteFailures(t *testing.T) {
// // 	codes := make(map[common.Hash][]byte, 1)
// // 	var h common.Hash
// // 	code := make([]byte, 5)
// // 	h[0] = byte(1)
// // 	code[0] = byte(5)
// // 	codes[h] = code

// // 	// execute dry-run to compute the number of calls to io.Writer
// // 	var count int
// // 	{
// // 		ctrl := gomock.NewController(t)
// // 		osfile := utils.NewMockOsFile(ctrl)

// // 		osfile.EXPECT().Write(gomock.Any()).AnyTimes().DoAndReturn(func(data []byte) (int, error) {
// // 			count++
// // 			return len(data), nil
// // 		})
// // 		if err := writeCodesTo(codes, osfile); err != nil {
// // 			t.Fatalf("cannot execute writeCodesTo: %s", err)
// // 		}
// // 	}

// // 	var injectedErr = errors.New("write error")
// // 	ctrl := gomock.NewController(t)
// // 	osfile := utils.NewMockOsFile(ctrl)

// // 	// execute the computed number of loops and mock calls to io.Writer so that
// // 	// the last one is failing.
// // 	// This way all branches are exercised.
// // 	for i := 0; i < count; i++ {
// // 		t.Run(fmt.Sprintf("io_error_%d", i), func(t *testing.T) {
// // 			calls := make([]*gomock.Call, 0, i+1)
// // 			for j := 0; j < i; j++ {
// // 				calls = append(calls, osfile.EXPECT().Write(gomock.Any()).Return(0, nil))
// // 			}
// // 			calls = append(calls, osfile.EXPECT().Write(gomock.Any()).Return(0, injectedErr))
// // 			gomock.InOrder(calls...)

// // 			if err := writeCodesTo(codes, osfile); !errors.Is(err, injectedErr) {
// // 				t.Errorf("writing roots should fail")
// // 			}
// // 		})

// // 	}
// // }

// func TestCodes_writeCodes_CannotCreateTheOutputFile(t *testing.T) {
// 	dir := t.TempDir()
// 	file := filepath.Join(dir, fileNameCodesCheckpointDirectory)
// 	if err := os.Mkdir(file, os.FileMode(0644)); err != nil {
// 		t.Fatalf("cannot create dir: %s", err)
// 	}
// 	if err := writeCodes(make(map[common.Hash][]byte, 1), file); err == nil {
// 		t.Errorf("writing roots should fail")
// 	}
// }

// // func TestCodes_writeCodesTo_ForwardWriteErrors(t *testing.T) {
// // 	ctrl := gomock.NewController(t)

// // 	codes := map[common.Hash][]byte{
// // 		{1}: {5},
// // 		{2}: {7, 8},
// // 	}

// // 	// count number of writing steps
// // 	counter := 0
// // 	file := utils.NewMockOsFile(ctrl)
// // 	file.EXPECT().Write(gomock.Any()).AnyTimes().DoAndReturn(func(data []byte) (int, error) {
// // 		counter++
// // 		return len(data), nil
// // 	})

// // 	if err := writeCodesTo(codes, file); err != nil {
// // 		t.Fatalf("cannot execute writeCodesTo: %s", err)
// // 	}
// // 	if counter == 0 {
// // 		t.Fatalf("expected at least one write operation")
// // 	}

// // 	for i := 0; i < counter; i++ {
// // 		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
// // 			ctrl := gomock.NewController(t)
// // 			file := utils.NewMockOsFile(ctrl)
// // 			injectedError := errors.New("injected error")
// // 			gomock.InOrder(
// // 				file.EXPECT().Write(gomock.Any()).Times(i).DoAndReturn(func(data []byte) (int, error) {
// // 					return len(data), nil
// // 				}),
// // 				file.EXPECT().Write(gomock.Any()).Return(0, injectedError),
// // 			)
// // 			err := writeCodesTo(codes, file)
// // 			if !errors.Is(err, injectedError) {
// // 				t.Fatalf("expected error, got %v", err)
// // 			}
// // 		})
// // 	}
// // }

// func TestCodes_readCodesAndSize_ReadingNonExistingFileReturnsEmptyCodeMap(t *testing.T) {
// 	dir := t.TempDir()
// 	path := filepath.Join(dir, fileNameCodes)
// 	codes, size, err := readCodesAndSize(path)
// 	if err != nil {
// 		t.Fatalf("failed to read codes: %v", err)
// 	}
// 	if want, got := 0, len(codes); want != got {
// 		t.Fatalf("expected codes to be empty, got %d", got)
// 	}
// 	if want, got := uint64(0), size; want != got {
// 		t.Fatalf("expected code file-size to be 0, got %d", got)
// 	}
// }

// func TestCodes_readCodesAndSize_ReadingIssuesAreReported(t *testing.T) {
// 	dir := t.TempDir()
// 	path := filepath.Join(dir, fileNameCodes)

// 	if err := os.WriteFile(path, []byte("invalid"), 0600); err != nil {
// 		t.Fatalf("failed to prepare invalid code file: %v", err)
// 	}

// 	_, _, err := readCodesAndSize(path)
// 	if err == nil {
// 		t.Fatalf("expected error, got nil")
// 	}
// }

// func TestCodes_readCodesAndSize_PermissionErrorsAreDetected(t *testing.T) {
// 	dir := t.TempDir()
// 	path := filepath.Join(dir, fileNameCodes)

// 	if err := os.Chmod(dir, 0000); err != nil {
// 		t.Fatalf("failed to change directory permissions: %v", err)
// 	}
// 	defer os.Chmod(dir, 0700)

// 	_, _, err := readCodesAndSize(path)
// 	if err == nil {
// 		t.Fatalf("expected error, got nil")
// 	}
// }

// func TestCodes_readCodes_Cannot_Read(t *testing.T) {
// 	dir := t.TempDir()
// 	file := filepath.Join(dir, "dir")
// 	if err := os.Mkdir(file, os.FileMode(0)); err != nil {
// 		t.Fatalf("cannot create dir: %s", err)
// 	}
// 	if _, err := readCodes(file); err == nil {
// 		t.Errorf("reading codes should fail")
// 	}
// }

// func TestCodes_parseCodes_ReadFailures(t *testing.T) {
// 	var injectedErr = errors.New("read error")
// 	ctrl := gomock.NewController(t)
// 	osfile := utils.NewMockOsFile(ctrl)

// 	var h common.Hash
// 	sizes := []int{len(h), 4, 100}
// 	// execute three times - parseCode calls io.Reader three times to get [<key>, <length>, <code>]
// 	for i := 0; i < 3; i++ {
// 		calls := make([]*gomock.Call, 0, i+1)
// 		for j := 0; j < i; j++ {
// 			pos := j
// 			call := osfile.EXPECT().Read(gomock.Any()).DoAndReturn(func(buf []byte) (int, error) {
// 				buf[0] = 1             // fill in an non-zero value not to return an empty array
// 				return sizes[pos], nil // returning expected size causes this io.Reader is called exactly once
// 			})
// 			calls = append(calls, call)
// 		}
// 		calls = append(calls, osfile.EXPECT().Read(gomock.Any()).Return(1, injectedErr))
// 		gomock.InOrder(calls...)

// 		if _, err := parseCodes(osfile); !errors.Is(err, injectedErr) {
// 			t.Errorf("reading codes should fail")
// 		}

// 	}
// }

// func TestCodes_addToCache_CacheIsUpdated(t *testing.T) {
// 	require := require.New(t)
// 	codes, err := openCodes(t.TempDir())
// 	if err != nil {
// 		t.Fatalf("failed to open codes: %v", err)
// 	}

// 	code := []byte("code1")
// 	hash := common.GetHash(codes.hasher, code)
// 	err = codes.handleCacheSet(hash, code)
// 	require.NoError(err)

// 	require.Equal(1, getCacheSize(codes.cache))
// 	require.Equal(code, codes.getCodeForHash(hash))
// }

// func TestCodes_addToCache_WritesToBufferOnEviction(t *testing.T) {
// 	codes, err := openCodes(t.TempDir())
// 	if err != nil {
// 		t.Fatalf("failed to open codes: %v", err)
// 	}

// 	// Fill the cache with codes until it reaches the eviction threshold.
// 	hashes := make([]common.Hash, cacheSize+1)
// 	for i := range cacheSize + 1 {
// 		code := []byte(fmt.Sprintf("code%d", i))
// 		hashes[i] = codes.add(code)
// 	}

// 	require := require.New(t)
// 	require.Equal(1, len(codes.flushBuffer))
// 	code, found := codes.flushBuffer[hashes[0]]
// 	require.True(found)
// 	require.Equal([]byte("code0"), code)
// }

// func TestCodes_addToCache_WritesToDiskWhenBufferIsFull(t *testing.T) {
// 	codes, err := openCodes(t.TempDir())
// 	if err != nil {
// 		t.Fatalf("failed to open codes: %v", err)
// 	}

// 	// Fill the cache with codes until it reaches the eviction threshold.
// 	hashes := make([]common.Hash, cacheSize+flushBufferThreshold-1)
// 	for i := range cacheSize + flushBufferThreshold - 1 {
// 		code := fmt.Appendf(nil, "code%d", i)
// 		hashes[i] = codes.add(code)
// 	}

// 	// Check the first flushBufferThreshold-1 entries are in the flush buffer.
// 	require := require.New(t)
// 	for i := range flushBufferThreshold - 1 {
// 		code, found := codes.flushBuffer[hashes[i]]
// 		require.True(found)
// 		require.Equal([]byte(fmt.Sprintf("code%d", i)), code)
// 	}

// 	// Check that the other ones are in the cache
// 	for i := flushBufferThreshold - 1; i < len(hashes); i++ {
// 		code, found := codes.cache.Get(hashes[i])
// 		require.True(found)
// 		require.Equal(fmt.Appendf(nil, "code%d", i), code)
// 	}

// 	// Add one more code to trigger flush to disk.
// 	lastCode := fmt.Appendf(nil, "code%d", len(hashes))
// 	lastHash := codes.add(lastCode)

// 	// Check that the last code is in the cache.
// 	code, found := codes.cache.Get(lastHash)
// 	require.True(found)
// 	require.Equal(lastCode, code)

// 	// FLush buffer is now empty
// 	require.Equal(0, len(codes.flushBuffer))
// 	// Flush buffer codes are on disk now
// 	require.Equal(flushBufferThreshold, len(codes.offsets))

// 	codeRead, err := readCodes(codes.file)
// 	require.NoError(err)
// 	require.Equal(flushBufferThreshold, len(codeRead))
// 	for i := range flushBufferThreshold {
// 		code, found := codeRead[hashes[i]]
// 		require.True(found)
// 		require.Equal([]byte(fmt.Sprintf("code%d", i)), code)
// 	}
// }

// func TestCodes_getCodeForHash_ReturnsCode(t *testing.T) {
// 	require := require.New(t)
// 	dir := t.TempDir()
// 	codes, err := openCodes(dir)
// 	require.NoError(err)

// 	codeOnDisk := []byte("code1")
// 	hashOnDisk := codes.add(codeOnDisk)
// 	codes.Flush()

// 	codes, err = openCodes(dir)
// 	require.NoError(err)
// 	size := 0
// 	codes.cache.Iterate(func(h common.Hash, b []byte) bool {
// 		size += 1
// 		return true
// 	})
// 	require.Equal(size, 0)

// 	readCode := codes.getCodeForHash(hashOnDisk)
// 	require.Equal(codeOnDisk, readCode)

// 	codeInCache := []byte("code2")
// 	hashInCache := codes.add(codeInCache)
// 	readCode = codes.getCodeForHash(hashInCache)
// 	require.Equal(codeInCache, readCode)
// }

// func TestCodes_getCodeForHash_PromotesCodeFromFlushBufferToCache(t *testing.T) {
// 	require := require.New(t)
// 	dir := t.TempDir()
// 	codes, err := openCodes(dir)
// 	require.NoError(err)

// 	codeInFlush := []byte("code1")
// 	hashInFlush := common.GetHash(codes.hasher, codeInFlush)
// 	codes.flushBuffer[hashInFlush] = codeInFlush

// 	readCode := codes.getCodeForHash(hashInFlush)
// 	require.Equal(codeInFlush, readCode)

// 	// Check that the code is now in the cache and not in the flush buffer
// 	_, foundInCache := codes.cache.Get(hashInFlush)
// 	require.True(foundInCache)

// 	_, foundInFlush := codes.flushBuffer[hashInFlush]
// 	require.False(foundInFlush)
// }

// func TestCodes_add_ignoresAlreadyExistingEntries(t *testing.T) {
// 	require := require.New(t)
// 	dir := t.TempDir()
// 	codes, err := openCodes(dir)
// 	require.NoError(err)

// 	codeInCache := []byte("code1")
// 	hashInCache := common.GetHash(codes.hasher, codeInCache)
// 	codes.cache.Set(hashInCache, codeInCache)
// 	codeInBuffer := []byte("code2")
// 	hashInBuffer := common.GetHash(codes.hasher, codeInBuffer)
// 	codes.flushBuffer[hashInBuffer] = codeInBuffer
// 	codesOnDisk := []byte("code2")
// 	codes.offsets[common.GetHash(codes.hasher, codesOnDisk)] = 0 // Simulate on disk

// 	hash := codes.add(codeInCache)
// 	require.Equal(hashInCache, hash)
// 	require.Equal(1, getCacheSize(codes.cache))
// 	require.Equal(1, len(codes.offsets))

// 	hash = codes.add(codeInBuffer)
// 	require.Equal(hashInBuffer, hash)
// 	require.Equal(1, getCacheSize(codes.cache))
// 	require.Equal(1, len(codes.offsets))

// 	hash = codes.add(codesOnDisk)
// 	require.Equal(common.GetHash(codes.hasher, codesOnDisk), hash)
// 	require.Equal(1, getCacheSize(codes.cache))
// 	require.Equal(1, len(codes.offsets))
// }

// func TestCodes_openCodes_InitializeFilesCorrectly(t *testing.T) {
// 	require := require.New(t)
// 	dir := t.TempDir()
// 	c, err := openCodes(dir)
// 	require.NoError(err)

// 	// Check that the codes file exists and is empty
// 	codesFile := filepath.Join(dir, fileNameCodes)
// 	info, err := os.Stat(codesFile)
// 	require.NoError(err)
// 	require.Equal(int64(0), info.Size())

// 	// Write some codes inside
// 	codes := map[common.Hash][]byte{
// 		{1}: {5},
// 		{2}: {7, 8},
// 	}
// 	err = writeCodes(codes, codesFile)
// 	require.NoError(err)

// 	// Re-open codes and check that the codes are loaded correctly
// 	c, err = openCodes(dir)
// 	require.NoError(err)
// 	require.Equal(2, len(c.offsets))
// 	for h, code := range codes {
// 		offset, exists := c.offsets[h]
// 		require.True(exists)
// 		readCode, err := readCodeAtOffset(c.file, offset)
// 		require.NoError(err)
// 		require.Equal(code, readCode)
// 	}
// }

// func TestCodes_Flush_WritesToDisk(t *testing.T) {
// 	require := require.New(t)
// 	dir := t.TempDir()
// 	codes, err := openCodes(dir)
// 	require.NoError(err)

// 	// Simulate something on disk
// 	codesOnDisk := []byte("codeOnDisk")
// 	hashOnDisk := common.GetHash(codes.hasher, codesOnDisk)
// 	codes.offsets[hashOnDisk] = 0 // Simulate on disk

// 	codeInFlush := []byte("codeInFlush")
// 	hashInFlush := common.GetHash(codes.hasher, codeInFlush)
// 	codes.flushBuffer[hashInFlush] = codeInFlush

// 	code1 := []byte("code1")
// 	code2 := []byte("code2")
// 	hash1 := common.GetHash(codes.hasher, code1)
// 	hash2 := common.GetHash(codes.hasher, code2)

// 	codes.cache.Set(hash1, code1)
// 	codes.cache.Set(hash2, code2)
// 	codes.cache.Set(hashOnDisk, codesOnDisk) // This should be ignored during flush

// 	err = codes.Flush()
// 	require.NoError(err)

// 	readCodes, err := readCodes(codes.file)
// 	require.NoError(err)
// 	require.Equal(3, len(readCodes)) // The CodeOnDisk is skipped
// 	require.Equal(code1, readCodes[hash1])
// 	require.Equal(code2, readCodes[hash2])
// 	require.Equal(codeInFlush, readCodes[hashInFlush])
// }

// func TestCodes_flushPending_WritesPendingCodesToDiskAndUpdatesOffsets(t *testing.T) {
// 	require := require.New(t)
// 	dir := t.TempDir()
// 	codes, err := openCodes(dir)
// 	require.NoError(err)

// 	// Add some codes to the flush buffer
// 	code1 := []byte("code1")
// 	code2 := []byte("code2")
// 	hash1 := common.GetHash(codes.hasher, code1)
// 	hash2 := common.GetHash(codes.hasher, code2)

// 	codes.flushBuffer[hash1] = code1
// 	codes.flushBuffer[hash2] = code2

// 	// Flush pending codes
// 	err = codes.flushPending()
// 	require.NoError(err)

// 	// Check that the flush buffer is empty
// 	require.Equal(0, len(codes.flushBuffer))

// 	// Check that the codes are written to disk and offsets are updated
// 	readCodes, err := readCodes(codes.file)
// 	require.NoError(err)
// 	require.Equal(2, len(readCodes))

// 	// Read from file using offsets
// 	offset1, exists1 := codes.offsets[hash1]
// 	require.True(exists1)
// 	readCode1, err := readCodeAtOffset(codes.file, offset1)
// 	require.NoError(err)
// 	require.Equal(code1, readCode1)

// 	offset2, exists2 := codes.offsets[hash2]
// 	require.True(exists2)
// 	readCode2, err := readCodeAtOffset(codes.file, offset2)
// 	require.NoError(err)
// 	require.Equal(code2, readCode2)
// }

// func getCacheSize[K comparable, V any](cache *common.LruCache[K, V]) int {
// 	size := 0
// 	cache.Iterate(func(h K, b V) bool {
// 		size += 1
// 		return true
// 	})
// 	return size
// }
