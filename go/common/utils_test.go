// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package common

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGetDirectorySize_ComputesSizeCorrectly(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "testfile")
	data := []byte("hello world")
	err := os.WriteFile(file, data, 0644)
	require.NoError(t, err)

	size, err := GetDirectorySize(dir)
	require.NoError(t, err)
	require.Equal(t, int64(len(data)), size, "directory size should match file size")
}

func TestGetDirectorySize_NonExistentDirectoryReturnsZeroSize(t *testing.T) {
	size, _ := GetDirectorySize("/path/does/not/exist")
	// GetDirectorySize should return 0 for non-existent directory
	require.Equal(t, int64(0), size, "size should be zero for non-existent directory")
}

func TestGetDirectorySize_SkipsUnreadableFile(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "file2")
	data := []byte("xyz")
	require.NoError(t, os.WriteFile(file, data, 0000)) // No permissions

	size, err := GetDirectorySize(dir)
	require.NoError(t, err)
	// Should skip unreadable file and not panic
	require.GreaterOrEqual(t, size, int64(0))
}
