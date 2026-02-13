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
	"bytes"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/common/future"
	"github.com/0xsoniclabs/carmen/go/common/result"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/stretchr/testify/require"
	"github.com/urfave/cli/v2"
	"go.uber.org/mock/gomock"
)

func TestBenchmark_RunExampleBenchmark(t *testing.T) {
	dir := t.TempDir()
	start := time.Now()
	result, err := runBenchmark(benchmarkParams{
		numBlocks:          300,
		numInsertsPerBlock: 10,
		tmpDir:             dir,
		reportInterval:     100,
		keepState:          false,
		schema:             5,
	}, func(string, ...any) {})
	end := time.Now()

	if err != nil {
		t.Fatalf("failed to run benchmark: %v", err)
	}

	limit := end.Sub(start)
	if result.insertTime < 0 || result.insertTime > limit {
		t.Errorf("invalid insert time: %v not in interval [0,%v]", result.insertTime, limit)
	}
	if result.reportTime < 0 || result.reportTime > limit {
		t.Errorf("invalid report time: %v not in interval [0,%v]", result.insertTime, limit)
	}
	total := result.insertTime + result.reportTime
	if total < 0 || total > limit {
		t.Errorf("invalid total time: %v not in interval [0,%v]", result.insertTime, limit)
	}

	if got, want := result.numInserts, int64(300*10); got != want {
		t.Fatalf("unexpected number of completed inserts, wanted %d, got %d", want, got)
	}

	if got, want := len(result.intervals), 3; got != want {
		t.Fatalf("unexpected size of result, wanted %d, got %d", want, got)
	}

	for i, cur := range result.intervals {
		if got, want := cur.endOfBlock, (i+1)*100; got != want {
			t.Errorf("invalid block in result line %d, wanted %d, got %d", i, want, got)
		}
		if cur.memory <= 0 {
			t.Errorf("invalid value for memory usage: %d", cur.memory)
		}
		if cur.disk <= 0 {
			t.Errorf("invalid value for disk usage: %d", cur.disk)
		}
		if cur.throughput <= 0 {
			t.Errorf("invalid value for throughput: %f", cur.throughput)
		}
	}

	filepath.Walk(dir, func(path string, info fs.FileInfo, err error) error {
		if strings.HasPrefix(info.Name(), "mpt_") {
			t.Errorf("temporary DB was not deleted")
		}
		return nil
	})
}

func TestBenchmark_KeepStateRetainsState(t *testing.T) {
	dir := t.TempDir()
	_, err := runBenchmark(benchmarkParams{
		numBlocks:          300,
		numInsertsPerBlock: 10,
		tmpDir:             dir,
		reportInterval:     100,
		keepState:          true,
		schema:             5,
	}, func(string, ...any) {})

	if err != nil {
		t.Fatalf("failed to run benchmark: %v", err)
	}

	found := false
	filepath.Walk(dir, func(path string, info fs.FileInfo, err error) error {
		if strings.HasPrefix(info.Name(), "state_") {
			found = true
		}
		return nil
	})

	if !found {
		t.Errorf("temporary MPT was not retained")
	}
}

func TestBenchmark_SupportsDifferentModes(t *testing.T) {
	cases := []bool{false, true}

	for _, mode := range cases {
		t.Run(fmt.Sprintf("with_archive=%t", mode), func(t *testing.T) {
			dir := t.TempDir()
			_, err := runBenchmark(benchmarkParams{
				archive:            mode,
				numBlocks:          300,
				numInsertsPerBlock: 10,
				tmpDir:             dir,
				reportInterval:     100,
				keepState:          true,
				schema:             5,
			}, func(string, ...any) {})

			if err != nil {
				t.Fatalf("failed to run benchmark: %v", err)
			}

			found := false
			filepath.Walk(dir, func(path string, info fs.FileInfo, err error) error {
				if strings.HasPrefix(info.Name(), "archive") {
					found = true
				}
				return nil
			})

			if found != mode {
				t.Errorf("unexpected presence of archive, wanted %t, got %t", mode, found)
			}
		})
	}
}

func TestGetDirectorySize_Normal(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "file1")
	data := []byte("abcde")
	require.NoError(t, os.WriteFile(file, data, 0644))

	size := getDirectorySize(dir)
	require.Equal(t, int64(len(data)), size)
}

func TestGetDirectorySize_NonExistentDir(t *testing.T) {
	size := getDirectorySize("/path/does/not/exist")
	require.Equal(t, int64(0), size)
}

func TestGetDirectorySize_UnreadableFile(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "file2")
	data := []byte("xyz")
	require.NoError(t, os.WriteFile(file, data, 0000)) // No permissions

	size := getDirectorySize(dir)
	// Should skip unreadable file and not panic
	require.GreaterOrEqual(t, size, int64(0))
}

func TestBenchmark_CLIAction(t *testing.T) {
	tmpDirs := map[string]struct {
		tmpdir     func() string
		shouldFail bool
	}{
		"tempdir":     {func() string { return t.TempDir() }, false},
		"nonexistent": {func() string { return t.TempDir() + "/nonexistent" }, true},
		"empty":       {func() string { return "" }, false},
	}

	for name, tc := range tmpDirs {
		t.Run(name, func(t *testing.T) {
			app := cli.NewApp()
			set := flag.NewFlagSet("test", 0)
			set.Bool("archive", false, "")
			set.Int("num-blocks", 10, "")
			set.Int("reads-per-block", 2, "")
			set.Int("inserts-per-block", 2, "")
			set.Int("report-interval", 5, "")
			set.String("tmp-dir", tc.tmpdir(), "")
			set.Bool("keep-state", false, "")
			set.Int("schema", 5, "")
			set.String("variant", "go-file", "")

			ctx := cli.NewContext(app, set, nil)

			// Capture stdout
			oldStdout := os.Stdout
			r, w, _ := os.Pipe()
			os.Stdout = w

			err := benchmark(ctx)

			// Restore stdout
			require.NoError(t, w.Close())
			os.Stdout = oldStdout

			var buf bytes.Buffer
			_, _ = buf.ReadFrom(r)

			if tc.shouldFail {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				require.Contains(t, buf.String(), "Overall time")
			}
		})
	}
}

func TestRunBenchmarkState_ApplyError(t *testing.T) {
	ctrl := gomock.NewController(t)

	injectedError := fmt.Errorf("injected error")

	getError := func(threshold, current int) error {
		if current >= threshold {
			return injectedError
		}
		return nil
	}

	const methods = 3
	for i := 0; i <= methods; i++ {
		state := state.NewMockState(ctrl)
		state.EXPECT().GetBalance(gomock.Any()).Return(amount.New(), getError(i, 1)).AnyTimes()
		state.EXPECT().Apply(gomock.Any(), gomock.Any()).Return(nil, getError(i, 2)).AnyTimes()
		state.EXPECT().GetCommitment().DoAndReturn(func() future.Future[result.Result[common.Hash]] {
			return future.Immediate(result.Err[common.Hash](getError(i, 3)))
		}).AnyTimes()

		_, err := runBenchmarkState(state, "/tmp", benchmarkParams{
			numBlocks:          1,
			numReadsPerBlock:   1,
			numInsertsPerBlock: 1,
			reportInterval:     1,
		}, func(string, ...any) {})

		require.ErrorAs(t, err, &injectedError)
	}
}
