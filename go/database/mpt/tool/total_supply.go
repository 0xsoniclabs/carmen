package main

import (
	"fmt"
	"github.com/0xsoniclabs/carmen/go/common/interrupt"
	"github.com/0xsoniclabs/carmen/go/database/mpt"
	"github.com/0xsoniclabs/carmen/go/database/mpt/io"
	"github.com/urfave/cli/v2"
)

var TotalSupplyCmd = cli.Command{
	Action:    doTotalSupplyCalc,
	Name:      "total-supply",
	Usage:     "calculate total supply of native tokens in the state",
	ArgsUsage: "<db director>",
	Flags: []cli.Flag{
		&targetBlockFlag,
	},
}

func doTotalSupplyCalc(context *cli.Context) error {
	if context.Args().Len() != 1 {
		return fmt.Errorf("missing state directory parameter")
	}
	dir := context.Args().Get(0)

	// check the type of target database
	mptInfo, err := io.CheckMptDirectoryAndGetInfo(dir)
	if err != nil {
		return err
	}

	logger := io.NewLog()

	ctx := interrupt.CancelOnInterrupt(context.Context)

	if mptInfo.Mode == mpt.Immutable {
		if !context.IsSet(targetBlockFlag.Name) {
			return fmt.Errorf("you need to specify --%s for archive", targetBlockFlag.Name)
		}
		// Passed Archive and chosen block
		blkNumber := context.Uint64(targetBlockFlag.Name)
		return io.CalculateArchiveTotalSupply(ctx, logger, dir, blkNumber)
	} else {
		// Passed LiveDB
		return io.CalculateLiveTotalSupply(ctx, logger, dir)
	}
}
