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
	"errors"
	"fmt"
	"math/bits"
	"strings"
	"sync"

	"github.com/0xsoniclabs/carmen/go/common/diagnostics"
	"github.com/0xsoniclabs/carmen/go/database/mpt"
	mptio "github.com/0xsoniclabs/carmen/go/database/mpt/io"
	"github.com/urfave/cli/v2"
)

var DeltaStats = cli.Command{
	Action:    diagnostics.AddPerformanceDiagnosticsAction(deltaStats, &diagnosticsFlag, &cpuProfileFlag, &traceFlag),
	Name:      "delta-stats",
	Flags:     []cli.Flag{&blockFrom, &blockTo},
	Usage:     "estimates the disk space saved by storing cloned branch nodes of an archive as deltas of their predecessors",
	ArgsUsage: "[--block-from <height>] [--block-to <height>] <directory>",
}

// maxDeltaSlots is the largest delta capacity that is simulated.
const maxDeltaSlots = 8

// deltaEntry is the simulation state of a branch node that was written as a
// delta for at least one slot capacity. masks[k-1] is the set of children
// changed since the last full node when k slots are used, 0 if the node was
// written in full.
type deltaEntry struct {
	masks [maxDeltaSlots]uint16
}

// deltaStatistics simulates storing cloned branch nodes as deltas of the last
// full node on their path, for every slot capacity up to maxDeltaSlots.
type deltaStatistics struct {
	mutex sync.Mutex
	// changedChildren[i] is the number of clones differing from their
	// predecessor in i children.
	changedChildren [17]uint64
	full            [maxDeltaSlots]uint64
	delta           [maxDeltaSlots]uint64
	nodes           map[mpt.NodeId]deltaEntry
}

func newDeltaStatistics() *deltaStatistics {
	return &deltaStatistics{nodes: map[mpt.NodeId]deltaEntry{}}
}

// Visit registers that after replaced before at the same trie position.
func (s *deltaStatistics) Visit(beforeId, afterId mpt.NodeId, before, after *mpt.BranchNode) {
	s.record(beforeId, afterId, changedChildren(before, after))
}

func (s *deltaStatistics) record(before, after mpt.NodeId, changed uint16) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.changedChildren[bits.OnesCount16(changed)]++
	// A node is the predecessor of at most one clone, so its entry is no
	// longer needed. Absent entries are full nodes.
	entry := s.nodes[before]
	delete(s.nodes, before)
	var next deltaEntry
	store := false
	for k := range maxDeltaSlots {
		mask := entry.masks[k] | changed
		if changed != 0 && bits.OnesCount16(mask) <= k+1 {
			next.masks[k] = mask
			s.delta[k]++
			store = true
		} else {
			s.full[k]++
		}
	}
	if store {
		s.nodes[after] = next
	}
}

// String returns the statistics in human-readable form.
func (s *deltaStatistics) String() string {
	out := &strings.Builder{}
	clones := uint64(0)
	for _, count := range s.changedChildren {
		clones += count
	}
	fmt.Fprintf(out, "Branch clones: %d\n", clones)
	fmt.Fprintf(out, "Changed children per clone:\n")
	for i, count := range s.changedChildren {
		if count != 0 {
			fmt.Fprintf(out, "\t%2d: %12d (%6.2f%%)\n", i, count, percent(count, clones))
		}
	}
	fullSize := uint64(mpt.BranchNodeEncoderWithChildHashes{}.GetEncodedSize())
	fmt.Fprintf(out, "All full: %d bytes\n", clones*fullSize)
	fmt.Fprintf(out, "Slots         full        delta        bytes   saved\n")
	for k := range maxDeltaSlots {
		deltaSize := uint64(deltaNodeSize(k + 1))
		size := s.full[k]*fullSize + s.delta[k]*deltaSize
		fmt.Fprintf(out, "\t%d %12d %12d %12d %6.2f%%\n", k+1, s.full[k], s.delta[k], size, 100-percent(size, clones*fullSize))
	}
	return out.String()
}

// deltaNodeSize is the on-disk size of a delta branch node with the given
// number of slots: base id, changed and embedded masks, and per slot a child
// id and hash.
func deltaNodeSize(slots int) int {
	idSize := mpt.NodeIdEncoder{}.GetEncodedSize()
	return idSize + 2 + 2 + slots*(idSize+32)
}

// changedChildren returns the mask of children that differ between the nodes.
func changedChildren(before, after *mpt.BranchNode) uint16 {
	mask := uint16(0)
	lhs, rhs := before.GetChildren(), after.GetChildren()
	for i := range 16 {
		if lhs[i].Id() != rhs[i].Id() {
			mask |= 1 << i
		}
	}
	return mask
}

func deltaStats(context *cli.Context) (retErr error) {
	if context.Args().Len() != 1 {
		return fmt.Errorf("expected 1 positional argument, got %d; usage: %s",
			context.Args().Len(), context.Command.ArgsUsage)
	}
	dir := context.Args().Get(0)

	mptInfo, err := mptio.CheckMptDirectoryAndGetInfo(dir)
	if err != nil {
		return err
	}
	if mptInfo.Mode != mpt.Immutable {
		return fmt.Errorf("delta-stats requires an archive, found a live MPT at %q", dir)
	}

	archive, err := mpt.OpenArchiveTrie(dir, mptInfo.Config, mpt.NodeCacheConfig{}, mpt.ArchiveConfig{})
	if err != nil {
		return fmt.Errorf("failed to open archive: %w", err)
	}
	defer func() {
		retErr = errors.Join(retErr, archive.Close())
	}()

	from := max(uint64(context.Int(blockFrom.Name)), 1)
	to := uint64(context.Int(blockTo.Name))
	if !context.IsSet(blockTo.Name) {
		height, empty, err := archive.GetBlockHeight()
		if err != nil {
			return err
		}
		if empty {
			return fmt.Errorf("archive is empty")
		}
		to = height
	}

	stats := newDeltaStatistics()
	for block := from; block <= to; block++ {
		if err := archive.VisitBranchClones(block-1, block, stats.Visit); err != nil {
			return fmt.Errorf("failed to visit block %d: %w", block, err)
		}
		if block%10000 == 0 {
			fmt.Printf("Visited block %d of %d, tracking %d delta nodes\n", block, to, len(stats.nodes))
		}
	}
	fmt.Print(stats.String())
	return nil
}
