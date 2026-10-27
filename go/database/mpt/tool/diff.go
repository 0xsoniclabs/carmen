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
	"time"

	"github.com/0xsoniclabs/carmen/go/common/diagnostics"
	"github.com/0xsoniclabs/carmen/go/database/mpt"
	mptio "github.com/0xsoniclabs/carmen/go/database/mpt/io"
	"github.com/urfave/cli/v2"
)

var Diff = cli.Command{
	Action:    diagnostics.AddPerformanceDiagnosticsAction(diff, &diagnosticsFlag, &cpuProfileFlag, &traceFlag),
	Name:      "diff",
	Flags:     []cli.Flag{&blockFrom, &blockTo},
	Usage:     "prints the state difference between two blocks of an archive",
	ArgsUsage: "--block-from <height> --block-to <height> <directory>",
}

func diff(context *cli.Context) (retErr error) {
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
		return fmt.Errorf("diff requires an archive, found a live MPT at %q", dir)
	}

	archive, err := mpt.OpenArchiveTrie(dir, mptInfo.Config, mpt.NodeCacheConfig{}, mpt.ArchiveConfig{})
	if err != nil {
		return fmt.Errorf("failed to open archive: %w", err)
	}
	defer func() {
		retErr = errors.Join(retErr, archive.Close())
	}()

	from := uint64(context.Int(blockFrom.Name))
	to := uint64(context.Int(blockTo.Name))
	start := time.Now()
	result, err := archive.GetDiff(from, to)
	if err != nil {
		return fmt.Errorf("failed to compute diff: %w", err)
	}
	fmt.Println(result.String())
	fmt.Printf("Diff of blocks %d -> %d: %d accounts changed, took %v\n", from, to, len(result), time.Since(start))
	return nil
}
