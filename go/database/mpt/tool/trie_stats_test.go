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

func TestTrieStatistics_Visit_CountsNodesByType(t *testing.T) {
	require := require.New(t)

	trie, err := mpt.OpenInMemoryLiveTrie(t.TempDir(), mpt.S5LiveConfig, mpt.NodeCacheConfig{})
	require.NoError(err)
	defer func() {
		require.NoError(trie.Close())
	}()

	const numAccounts = 100
	for i := range numAccounts {
		addr := common.Address{byte(i)}
		require.NoError(trie.SetAccountInfo(addr, mpt.AccountInfo{Nonce: common.ToNonce(1)}))
	}
	require.NoError(trie.SetValue(common.Address{1}, common.Key{1}, common.Value{1}))
	require.NoError(trie.SetValue(common.Address{1}, common.Key{2}, common.Value{2}))

	stats := &trieStatistics{}
	require.NoError(trie.VisitTrie(mpt.ReadAccess{}, stats))

	require.Equal(uint64(numAccounts), stats.accounts)
	require.Equal(uint64(2), stats.values)
	require.Zero(stats.empty)
	require.NotZero(stats.branches)

	branches := uint64(0)
	for _, count := range stats.branchFill {
		branches += count
	}
	require.Equal(stats.branches, branches)
	require.Zero(stats.branchFill[0])
	require.Zero(stats.branchFill[1])
}

func TestTrieStatistics_Visit_CountsEmptyTrie(t *testing.T) {
	require := require.New(t)

	trie, err := mpt.OpenInMemoryLiveTrie(t.TempDir(), mpt.S5LiveConfig, mpt.NodeCacheConfig{})
	require.NoError(err)
	defer func() {
		require.NoError(trie.Close())
	}()

	stats := &trieStatistics{}
	require.NoError(trie.VisitTrie(mpt.ReadAccess{}, stats))

	require.Equal(trieStatistics{empty: 1}, *stats)
}

func TestTrieStatistics_String_PrintsCountsAndBranchFill(t *testing.T) {
	require := require.New(t)

	stats := &trieStatistics{accounts: 2, branches: 2, values: 1}
	stats.branchFill[2] = 1
	stats.branchFill[16] = 1

	require.Equal(
		"Node types:\n"+
			"\tAccounts:              2 ( 40.00%)\n"+
			"\tBranches:              2 ( 40.00%)\n"+
			"\tExtensions:            0 (  0.00%)\n"+
			"\tValues:                1 ( 20.00%)\n"+
			"\tEmpty:                 0 (  0.00%)\n"+
			"\tTotal:                 5\n"+
			"Branch node fill (non-empty children out of 16):\n"+
			"\t 2:            1 ( 50.00%)\n"+
			"\t16:            1 ( 50.00%)\n"+
			"\tAverage: 9.00 children (56.25% filled)\n",
		stats.String(),
	)
}

func TestTrieStats_Run_SucceedsOnLiveTrie(t *testing.T) {
	require := require.New(t)

	dir := t.TempDir()
	trie, err := mpt.OpenFileLiveTrie(dir, mpt.S5LiveConfig, mpt.NodeCacheConfig{})
	require.NoError(err)
	require.NoError(trie.SetAccountInfo(common.Address{1}, mpt.AccountInfo{Nonce: common.ToNonce(1)}))
	require.NoError(trie.Close())

	require.NoError(runTrieStats(dir))
}

func TestTrieStats_Run_SucceedsOnArchiveTrie(t *testing.T) {
	require := require.New(t)

	dir := t.TempDir()
	archive, err := mpt.OpenArchiveTrie(dir, mpt.S5ArchiveConfig, mpt.NodeCacheConfig{}, mpt.ArchiveConfig{})
	require.NoError(err)
	require.NoError(archive.Add(0, common.Update{
		Balances: []common.BalanceUpdate{{Account: common.Address{1}, Balance: amount.New(1)}},
	}, nil))
	require.NoError(archive.Close())

	require.NoError(runTrieStats("--block-height", "0", dir))
}

func TestTrieStats_Run_FailsWithoutDirectory(t *testing.T) {
	require := require.New(t)
	require.ErrorContains(runTrieStats(), "expected 1 positional argument")
}

func runTrieStats(args ...string) error {
	app := &cli.App{Commands: []*cli.Command{&TrieStats}}
	return app.Run(append([]string{"tool", TrieStats.Name}, args...))
}
