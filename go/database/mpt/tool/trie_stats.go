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
	"strings"

	"github.com/0xsoniclabs/carmen/go/common/diagnostics"
	"github.com/0xsoniclabs/carmen/go/database/mpt"
	mptio "github.com/0xsoniclabs/carmen/go/database/mpt/io"
	"github.com/urfave/cli/v2"
)

var TrieStats = cli.Command{
	Action:    diagnostics.AddPerformanceDiagnosticsAction(trieStats, &diagnosticsFlag, &cpuProfileFlag, &traceFlag),
	Name:      "trie-stats",
	Flags:     []cli.Flag{&blockHeightFlag},
	Usage:     "prints node type counts and branch node fill levels of the live or archive trie",
	ArgsUsage: "[--block-height <height>] <directory>",
}

// trieStatistics holds node counts of a trie and the fill level of its branch nodes.
type trieStatistics struct {
	accounts   uint64
	branches   uint64
	extensions uint64
	values     uint64
	empty      uint64

	// branchFill[i] is the number of branch nodes with i non-empty children.
	branchFill         [17]uint64
	loggingBlockHeight uint64
	blockSeen          uint64
}

// Visit registers the given node in the statistics.
func (s *trieStatistics) Visit(node mpt.Node, _ mpt.NodeInfo) mpt.VisitResponse {
	s.blockSeen += 1
	if s.blockSeen == s.loggingBlockHeight {
		fmt.Printf("Visited %d nodes at block height %d\n", s.accounts+s.branches+s.extensions+s.values+s.empty, s.loggingBlockHeight)
		s.blockSeen = 0
	}
	switch n := node.(type) {
	case *mpt.AccountNode:
		s.accounts++
	case *mpt.BranchNode:
		s.branches++
		children := 0
		for _, child := range n.GetChildren() {
			if !child.Id().IsEmpty() {
				children++
			}
		}
		s.branchFill[children]++
	case *mpt.ExtensionNode:
		s.extensions++
	case *mpt.ValueNode:
		s.values++
	case mpt.EmptyNode:
		s.empty++
	}
	return mpt.VisitResponseContinue
}

// String returns the statistics in human-readable form.
func (s *trieStatistics) String() string {
	out := &strings.Builder{}
	total := s.accounts + s.branches + s.extensions + s.values + s.empty
	fmt.Fprintf(out, "Node types:\n")
	fmt.Fprintf(out, "\tAccounts:   %12d (%6.2f%%)\n", s.accounts, percent(s.accounts, total))
	fmt.Fprintf(out, "\tBranches:   %12d (%6.2f%%)\n", s.branches, percent(s.branches, total))
	fmt.Fprintf(out, "\tExtensions: %12d (%6.2f%%)\n", s.extensions, percent(s.extensions, total))
	fmt.Fprintf(out, "\tValues:     %12d (%6.2f%%)\n", s.values, percent(s.values, total))
	fmt.Fprintf(out, "\tEmpty:      %12d (%6.2f%%)\n", s.empty, percent(s.empty, total))
	fmt.Fprintf(out, "\tTotal:      %12d\n", total)

	fmt.Fprintf(out, "Branch node fill (non-empty children out of 16):\n")
	sum := uint64(0)
	for i, count := range s.branchFill {
		sum += uint64(i) * count
		if count == 0 {
			continue
		}
		fmt.Fprintf(out, "\t%2d: %12d (%6.2f%%)\n", i, count, percent(count, s.branches))
	}
	if s.branches > 0 {
		avg := float64(sum) / float64(s.branches)
		fmt.Fprintf(out, "\tAverage: %.2f children (%.2f%% filled)\n", avg, avg/16*100)
	}
	return out.String()
}

func trieStats(context *cli.Context) (retErr error) {
	if context.Args().Len() != 1 {
		return fmt.Errorf("expected 1 positional argument, got %d; usage: %s",
			context.Args().Len(), context.Command.ArgsUsage)
	}
	dir := context.Args().Get(0)

	mptInfo, err := mptio.CheckMptDirectoryAndGetInfo(dir)
	if err != nil {
		return err
	}

	trie, err := openTrie(context, dir, mptInfo)
	if err != nil {
		return err
	}
	defer func() {
		retErr = errors.Join(retErr, trie.Close())
	}()

	stats := &trieStatistics{loggingBlockHeight: 10000}
	if err := trie.VisitTrie(mpt.ReadAccess{}, stats); err != nil {
		return fmt.Errorf("failed to visit trie: %w", err)
	}
	fmt.Print(stats.String())
	return nil
}

func percent(part, total uint64) float64 {
	if total == 0 {
		return 0
	}
	return float64(part) / float64(total) * 100
}
