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
	"os"
	"os/exec"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAllCommands_Run(t *testing.T) {
	for _, cmd := range commands {
		t.Run(cmd.Name, func(t *testing.T) {
			os.Args = []string{"tool", cmd.Name, "--help"}
			main() // ensure commands can be invoked without error
		})
	}
}

func TestMain_ErrorArgument(t *testing.T) {
	cmd := exec.Command("go", "run", "./database/mpt/tool", "--nonexistent-command")
	err := cmd.Run()
	exitErr, ok := err.(*exec.ExitError)
	require.True(t, ok, "expected process to exit with error")
	require.Equal(t, 1, exitErr.ExitCode(), "expected exit code 1")
}
