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
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/urfave/cli/v2"
)

func TestGetMemoryUsage(t *testing.T) {
	mem := getMemoryUsage()
	require.Greater(t, mem, uint64(0), "memory usage should be greater than zero")
}

func TestStressTest_BasicRun(t *testing.T) {
	app := &cli.App{
		Commands: []*cli.Command{&StressTestCmd},
	}
	// Use a temp dir and minimal flags
	err := app.Run([]string{
		"tool",
		"stress-test",
		"--num-blocks=1",
		"--report-period=10s",
		"--flush-period=10ms",
	})
	require.NoError(t, err, "stressTest should run without error for minimal input")
}

func TestStressTest_InvalidTmpDir(t *testing.T) {
	app := &cli.App{
		Commands: []*cli.Command{&StressTestCmd},
	}
	// Provide an invalid tmp-dir to trigger error
	err := app.Run([]string{
		"tool",
		"stress-test",
		"--tmp-dir=/invalid/path/does/not/exist",
		"--num-blocks=1",
	})
	require.Error(t, err, "should error with invalid tmp-dir")
}

func TestStressTest_ZeroBlocks(t *testing.T) {
	app := &cli.App{
		Commands: []*cli.Command{&StressTestCmd},
	}

	// Simulate interrupt signal after test assertions
	// it prevents running the test for 1000 blocks, which is the default value
	go func() {
		time.Sleep(5 * time.Second)
		_ = syscall.Kill(syscall.Getpid(), syscall.SIGINT)
	}()

	// Zero blocks should default to 1000, but should not error
	err := app.Run([]string{
		"tool",
		"stress-test",
		"--num-blocks=0",
	})
	require.NoError(t, err, "should not error with zero blocks")
}

func TestGetFreeSpace_ValidPath(t *testing.T) {
	dir := t.TempDir()
	free, err := getFreeSpace(dir)
	require.NoError(t, err, "should not error for valid path")
	require.Greater(t, free, int64(0), "free space should be greater than zero")
}

func TestGetFreeSpace_InvalidPath(t *testing.T) {
	free, err := getFreeSpace("/path/does/not/exist")
	require.Error(t, err, "should error for non-existent path")
	require.Equal(t, int64(0), free, "free space should be zero on error")
}
