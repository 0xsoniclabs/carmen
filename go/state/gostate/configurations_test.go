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

	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/stretchr/testify/require"
)

func TestOpenArchive_FailsForNoArchiveIfThereIsAnArchive(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()

	params := state.Parameters{
		Directory: dir,
		Archive:   state.S5Archive,
	}

	// Create an archive in the directory.
	archive, cleanup, err := openArchive(params)
	require.NoError(err)
	require.Nil(cleanup)
	require.NoError(archive.Close())

	// Reopening the archive with the NoArchive option should fail.
	params.Archive = state.NoArchive
	_, _, err = openArchive(params)
	require.Error(err)
}

func TestOpenArchive_FailsForNoArchiveIfTheArchiveDirectoryCanNotBeAccessed(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()

	params := state.Parameters{
		Directory: dir,
	}

	path, err := prepareArchiveDirectory(params)
	require.NoError(err)

	// An empty archive can be opened as a No-Archive.
	archive, cleanup, err := openArchive(params)
	require.NoError(err)
	require.Nil(archive)
	require.Nil(cleanup)

	// If the empty archive can not be accessed, an error should be returned.
	os.Chmod(path, 0000)
	defer os.Chmod(path, 0755)

	_, _, err = openArchive(params)
	require.Error(err)
}
