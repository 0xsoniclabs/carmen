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
	"fmt"
	"os"

	"github.com/urfave/cli/v2"
)

// Run using
//  go run ./database/mpt/tool <command> <flags>

var (
	diagnosticsFlag = cli.IntFlag{
		Name:  "diagnostic-port",
		Usage: "enable hosting of a realtime diagnostic server by providing a port",
		Value: 0,
	}
	cpuProfileFlag = cli.StringFlag{
		Name:  "cpuprofile",
		Usage: "sets the target file for storing CPU profiles to, disabled if empty",
		Value: "",
	}
	traceFlag = cli.StringFlag{
		Name:  "tracefile",
		Usage: "sets the target file for traces to, disabled if empty",
		Value: "",
	}
)

func main() {
	app := &cli.App{
		Name:      "tool",
		Usage:     "Carmen MPT toolbox",
		Copyright: "(c) 2022-25 Sonic Operations Ltd",
		Flags: []cli.Flag{
			&diagnosticsFlag,
			&cpuProfileFlag,
			&traceFlag,
		},
		Commands: []*cli.Command{
			&Check,
			&ExportCmd,
			&ImportLiveDbCmd,
			&ImportArchiveCmd,
			&ImportLiveAndArchiveCmd,
			&Info,
			&InitArchive,
			&Verify,
			&VerifyProof,
			&Benchmark,
			&Block,
			&StressTestCmd,
			&Reset,
		},
	}

	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
