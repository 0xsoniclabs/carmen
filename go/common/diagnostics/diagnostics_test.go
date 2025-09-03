// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package diagnostics

import (
	"net/http"
	_ "net/http/pprof"
	"path"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/urfave/cli/v2"
)

func TestAddPerformanceDiagnosticsAction(t *testing.T) {
	dir := t.TempDir()
	called := false
	action := func(ctx *cli.Context) error {
		// profile file created
		require.FileExists(t, path.Join(dir, "cpu.profile"))
		require.FileExists(t, path.Join(dir, "tracer.out"))

		// server started
		var statusCode int
		var counter int
		const loops = 10
		var lastHttpGetErr error
		wait := 100 * time.Millisecond
		for statusCode != http.StatusOK && counter < loops {
			resp, err := http.Get("http://localhost:6060/debug/pprof/")
			lastHttpGetErr = err
			if resp != nil {
				statusCode = resp.StatusCode
			}
			counter++
			time.Sleep(wait)
			wait *= 2
		}

		require.NoError(t, lastHttpGetErr)
		require.Equal(t, http.StatusOK, statusCode)

		called = true
		return nil
	}

	diagnosticsFlag := cli.IntFlag{Name: "diagnostics"}
	cpuProfileFlag := cli.StringFlag{Name: "cpu-profile"}
	traceFlag := cli.StringFlag{Name: "trace"}

	app := &cli.App{
		Action: AddPerformanceDiagnosticsAction(action, &diagnosticsFlag, &cpuProfileFlag, &traceFlag),
		Flags:  []cli.Flag{&diagnosticsFlag, &cpuProfileFlag, &traceFlag},
	}

	set := []string{"cmd", "--diagnostics", "6060", "--cpu-profile", path.Join(dir, "cpu.profile"), "--trace", path.Join(dir, "tracer.out")}
	err := app.RunContext(
		nil,
		set,
	)
	require.NoError(t, err)

	require.True(t, called, "action should be called")

}
