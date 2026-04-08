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

// func TestMain_RunNoArgs(t *testing.T) {
// 	// we test only that it runs without crashing
// 	os.Args = []string{"tool", "--help"}
// 	main()
// }

// func TestMain_ErrorArgument(t *testing.T) {
// 	cmd := exec.Command("go", "run", "./database/state/tool", "--nonexistent-command")
// 	err := cmd.Run()
// 	exitErr, ok := err.(*exec.ExitError)
// 	require.True(t, ok, "expected process to exit with error")
// 	require.Equal(t, 1, exitErr.ExitCode(), "expected exit code 1")
// }
