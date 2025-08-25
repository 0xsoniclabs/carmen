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
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/urfave/cli/v2"
)

func TestGetMemoryUsage(t *testing.T) {
	mem := getMemoryUsage()
	require.Greater(t, mem, uint64(0), "memory usage should be greater than zero")
}

func TestGetDirectorySize(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "testfile")
	data := []byte("hello world")
	err := os.WriteFile(file, data, 0644)
	require.NoError(t, err)

	size := getDirectorySize(dir)
	require.Equal(t, int64(len(data)), size, "directory size should match file size")
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
