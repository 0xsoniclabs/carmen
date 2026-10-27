// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package main

import (
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/database/mpt"
	"github.com/stretchr/testify/require"
	"github.com/urfave/cli/v2"
)

func TestDiff_Run_SucceedsOnArchiveTrie(t *testing.T) {
	require := require.New(t)

	dir := t.TempDir()
	archive, err := mpt.OpenArchiveTrie(dir, mpt.S5ArchiveConfig, mpt.NodeCacheConfig{}, mpt.ArchiveConfig{})
	require.NoError(err)
	require.NoError(archive.Add(0, common.Update{
		Balances: []common.BalanceUpdate{{Account: common.Address{1}, Balance: amount.New(1)}},
	}, nil))
	require.NoError(archive.Add(1, common.Update{
		Balances: []common.BalanceUpdate{{Account: common.Address{2}, Balance: amount.New(2)}},
	}, nil))
	require.NoError(archive.Close())

	require.NoError(runDiff("--block-from", "0", "--block-to", "1", dir))
}

func TestDiff_Run_FailsOnLiveTrie(t *testing.T) {
	require := require.New(t)

	dir := t.TempDir()
	trie, err := mpt.OpenFileLiveTrie(dir, mpt.S5LiveConfig, mpt.NodeCacheConfig{})
	require.NoError(err)
	require.NoError(trie.Close())

	require.ErrorContains(runDiff(dir), "requires an archive")
}

func TestDiff_Run_FailsOnMissingBlock(t *testing.T) {
	require := require.New(t)

	dir := t.TempDir()
	archive, err := mpt.OpenArchiveTrie(dir, mpt.S5ArchiveConfig, mpt.NodeCacheConfig{}, mpt.ArchiveConfig{})
	require.NoError(err)
	require.NoError(archive.Close())

	require.ErrorContains(runDiff("--block-to", "5", dir), "not present in archive")
}

func runDiff(args ...string) error {
	app := &cli.App{Commands: []*cli.Command{&Diff}}
	return app.Run(append([]string{"tool", Diff.Name}, args...))
}
