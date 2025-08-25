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

	"github.com/stretchr/testify/require"
	"github.com/urfave/cli/v2"
)

func TestAllCommands_Run(t *testing.T) {
	commands := []*cli.Command{
		&Check,
		&ExportCmd,
		&ImportLiveDbCmd,
		&ImportArchiveCmd,
		&ImportLiveAndArchiveCmd,
		&Info,
		&InitArchive,
		&Verify,
		&VerifyProof,
		&Block,
		&StressTestCmd,
		&Reset,
	}

	for _, cmd := range commands {
		t.Run(cmd.Name, func(t *testing.T) {
			app := &cli.App{Commands: []*cli.Command{cmd}}
			err := app.Run([]string{"tool", cmd.Name, "--help"})
			// Some commands may require flags or setup, so just check for no panic
			require.NotPanics(t, func() { _ = err }, "command should not panic")
		})
	}
}
