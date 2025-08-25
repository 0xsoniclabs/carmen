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
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
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
