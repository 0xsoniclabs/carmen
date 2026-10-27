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

func TestDeltaStatistics_Record_AccumulatesChangesSinceLastFullNode(t *testing.T) {
	require := require.New(t)

	stats := newDeltaStatistics()
	// Three clones of the same path, each changing one new child.
	stats.record(mpt.BranchId(0), mpt.BranchId(1), 0b001)
	stats.record(mpt.BranchId(1), mpt.BranchId(2), 0b010)
	stats.record(mpt.BranchId(2), mpt.BranchId(3), 0b100)

	require.Equal(uint64(3), stats.changedChildren[1])
	// One slot: the second clone overflows and is written in full, the third
	// clone is a delta of that fresh base.
	require.Equal(uint64(2), stats.delta[0])
	require.Equal(uint64(1), stats.full[0])
	// Two slots: the third clone no longer fits.
	require.Equal(uint64(2), stats.delta[1])
	require.Equal(uint64(1), stats.full[1])
	// Three slots: all clones are deltas of the first node.
	require.Equal(uint64(3), stats.delta[2])
	require.Equal(uint64(0), stats.full[2])
	require.Equal(map[mpt.NodeId]deltaEntry{
		mpt.BranchId(3): {masks: [maxDeltaSlots]uint16{0b100, 0, 0b111, 0b111, 0b111, 0b111, 0b111, 0b111}},
	}, stats.nodes)
}

func TestDeltaStatistics_Record_IgnoresUnchangedClones(t *testing.T) {
	require := require.New(t)

	stats := newDeltaStatistics()
	stats.record(mpt.BranchId(0), mpt.BranchId(1), 0)

	require.Equal(uint64(1), stats.changedChildren[0])
	for k := range maxDeltaSlots {
		require.Equal(uint64(1), stats.full[k])
		require.Equal(uint64(0), stats.delta[k])
	}
	require.Empty(stats.nodes)
}

func TestDeltaStatistics_String_PrintsHistogramAndSavings(t *testing.T) {
	require := require.New(t)

	stats := newDeltaStatistics()
	stats.record(mpt.BranchId(0), mpt.BranchId(1), 0b11)

	out := stats.String()
	require.Contains(out, "Branch clones: 1\n")
	require.Contains(out, "\t 2:            1 (100.00%)\n")
	require.Contains(out, "All full: 610 bytes\n")
	require.Contains(out, "\t1            1            0          610   0.00%\n")
	require.Contains(out, "\t2            0            1           86  85.90%\n")
}

func TestDeltaStats_Run_SucceedsOnArchiveTrie(t *testing.T) {
	require := require.New(t)

	dir := t.TempDir()
	archive, err := mpt.OpenArchiveTrie(dir, mpt.S5ArchiveConfig, mpt.NodeCacheConfig{}, mpt.ArchiveConfig{})
	require.NoError(err)
	for i := range 3 {
		require.NoError(archive.Add(uint64(i), common.Update{
			Balances: []common.BalanceUpdate{{Account: common.Address{byte(i)}, Balance: amount.New(1)}},
		}, nil))
	}
	require.NoError(archive.Close())

	require.NoError(runDeltaStats(dir))
}

func TestDeltaStats_Run_FailsOnLiveTrie(t *testing.T) {
	require := require.New(t)

	dir := t.TempDir()
	trie, err := mpt.OpenFileLiveTrie(dir, mpt.S5LiveConfig, mpt.NodeCacheConfig{})
	require.NoError(err)
	require.NoError(trie.Close())

	require.ErrorContains(runDeltaStats(dir), "requires an archive")
}

func runDeltaStats(args ...string) error {
	app := &cli.App{Commands: []*cli.Command{&DeltaStats}}
	return app.Run(append([]string{"tool", DeltaStats.Name}, args...))
}
