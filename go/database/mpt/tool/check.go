// Copyright (c) 2024 Fantom Foundation
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at fantom.foundation/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package main

import (
	"errors"
	"fmt"

	"github.com/0xsoniclabs/carmen/go/database/mpt"
	"github.com/0xsoniclabs/carmen/go/database/mpt/io"
	"github.com/urfave/cli/v2"
)

var Check = cli.Command{
	Action:    addPerformanceDiagnoses(check),
	Name:      "check",
	Usage:     "performs extensive invariants checks",
	ArgsUsage: "<director>",
}

func check(context *cli.Context) error {
	// parse the directory argument
	if context.Args().Len() != 1 {
		return fmt.Errorf("missing directory storing state")
	}

	dir := context.Args().Get(0)

	// try to obtain information of the contained MPT
	info, err := io.CheckMptDirectoryAndGetInfo(dir)
	if err != nil {
		return err
	}

	if info.Mode == mpt.Immutable {
		fmt.Printf("Checking archive in %s ...\n", dir)
		err = checkArchive(dir, info)
	} else {
		fmt.Printf("Checking live DB in %s ...\n", dir)
		err = checkLiveDB(dir, info)
	}
	if err == nil {
		fmt.Printf("All checks passed!\n")
	}
	return err
}

func checkLiveDB(dir string, info io.MptInfo) error {
	live, err := mpt.OpenFileLiveTrie(dir, info.Config, mpt.NodeCacheConfig{})
	if err != nil {
		return err
	}
	return errors.Join(
		live.Check(), // check must be called before closing
		live.Close(),
	)
}

func checkArchive(dir string, info io.MptInfo) error {
	archive, err := mpt.OpenArchiveTrie(dir, info.Config, mpt.NodeCacheConfig{}, mpt.ArchiveConfig{})
	if err != nil {
		return err
	}
	return errors.Join(
		archive.Check(), // check must be called before closing
		archive.Close(),
	)
}
