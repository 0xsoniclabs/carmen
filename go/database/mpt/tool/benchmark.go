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
	_ "net/http/pprof"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"time"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/urfave/cli/v2"

	"github.com/0xsoniclabs/carmen/go/state/gostate"
)

var Benchmark = cli.Command{
	Action: benchmark,
	Name:   "benchmark",
	Usage:  "benchmarks MPT performance by filling data into a fresh instance",
	Flags: []cli.Flag{
		&archiveFlag,
		&numBlocksFlag,
		&numReadsPerBlockFlag,
		&numInsertsPerBlockFlag,
		&reportIntervalFlag,
		&tmpDirFlag,
		&keepStateFlag,
		&cpuProfileFlag,
		&schemaFlag,
		&diagnosticsFlag,
	},
}

var (
	archiveFlag = cli.BoolFlag{
		Name:  "archive",
		Usage: "enables archive mode",
	}
	numBlocksFlag = cli.IntFlag{
		Name:  "num-blocks",
		Usage: "the number of blocks to be filled in",
		Value: 10_000,
	}
	numReadsPerBlockFlag = cli.IntFlag{
		Name:  "reads-per-block",
		Usage: "the number of reads per block",
		Value: 0,
	}
	numInsertsPerBlockFlag = cli.IntFlag{
		Name:  "inserts-per-block",
		Usage: "the number of inserts per block",
		Value: 1_000,
	}
	reportIntervalFlag = cli.IntFlag{
		Name:  "report-interval",
		Usage: "the size of a reporting interval in number of blocks",
		Value: 1000,
	}
	tmpDirFlag = cli.StringFlag{
		Name:  "tmp-dir",
		Usage: "the directory to place the state for running benchmarks on",
	}
	keepStateFlag = cli.BoolFlag{
		Name:  "keep-state",
		Usage: "disables the deletion of temporary data at the end of the benchmark",
	}
	schemaFlag = cli.IntFlag{
		Name:  "schema",
		Usage: "database scheme to use represented by its number [1..N]",
		Value: 5,
	}
)

func benchmark(context *cli.Context) error {

	tmpDir := context.String(tmpDirFlag.Name)
	if len(tmpDir) == 0 {
		tmpDir = os.TempDir()
	}

	startDiagnosticServer(context.Int(diagnosticsFlag.Name))

	start := time.Now()
	results, err := runBenchmark(
		benchmarkParams{
			archive:            context.Bool(archiveFlag.Name),
			numBlocks:          context.Int(numBlocksFlag.Name),
			numReadsPerBlock:   context.Int(numReadsPerBlockFlag.Name),
			numInsertsPerBlock: context.Int(numInsertsPerBlockFlag.Name),
			tmpDir:             tmpDir,
			keepState:          context.Bool(keepStateFlag.Name),
			cpuProfilePrefix:   context.String(cpuProfileFlag.Name),
			traceFilePrefix:    context.String(traceFlag.Name),
			reportInterval:     context.Int(reportIntervalFlag.Name),
			schema:             context.Int(schemaFlag.Name),
		},
		func(msg string, args ...any) {
			delta := uint64(time.Since(start).Round(time.Second).Seconds())
			fmt.Printf("[t=%3d:%02d:%02d]: ", delta/3600, (delta/60)%60, delta%60)
			fmt.Printf(msg+"\n", args...)
		},
	)
	if err != nil {
		return err
	}

	fmt.Printf("block, memory, disk, throughput\n")
	for _, cur := range results.intervals {
		fmt.Printf("%d, %d, %d, %.2f\n", cur.endOfBlock, cur.memory, cur.disk, cur.throughput)
	}
	fmt.Printf("Overall time: %v (+%v for reporting)\n", results.insertTime, results.reportTime)
	fmt.Printf("Overall throughput: %.2f inserts/second\n", float64(results.numInserts)/results.insertTime.Seconds())
	return nil
}

type benchmarkParams struct {
	archive            bool
	numBlocks          int
	numReadsPerBlock   int
	numInsertsPerBlock int
	tmpDir             string
	keepState          bool
	cpuProfilePrefix   string
	traceFilePrefix    string
	reportInterval     int
	schema             int
}

type benchmarkRecord struct {
	endOfBlock int
	memory     int64
	disk       int64
	throughput float64
}

type benchmarkResult struct {
	intervals  []benchmarkRecord
	reportTime time.Duration
	insertTime time.Duration
	numInserts int64
}

func runBenchmark(
	params benchmarkParams,
	observer func(string, ...any),
) (benchmarkResult, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	res := benchmarkResult{}

	profilingEnabled := len(params.cpuProfilePrefix) > 0
	tracingEnabled := len(params.traceFilePrefix) > 0

	// Start profiling ...
	if profilingEnabled {
		if err := startCpuProfiler(fmt.Sprintf("%s_%06d", params.cpuProfilePrefix, 1)); err != nil {
			return res, err
		}
		defer stopCpuProfiler()
	}

	// Start tracing ...
	if tracingEnabled {
		if err := startTracer(fmt.Sprintf("%s_%06d", params.traceFilePrefix, 1)); err != nil {
			return res, err
		}
		defer stopTracer()
	}

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)
	go func() {
		for range c {
			pprof.Lookup("goroutine").WriteTo(os.Stdout, 1)
			fmt.Printf("signal: interrupt")
			os.Exit(1)
		}
	}()

	// Create the target state.
	path := fmt.Sprintf(params.tmpDir+string(os.PathSeparator)+"state_%d", time.Now().Unix())

	if params.archive {
		observer("Creating state with archive in %s ..", path)
	} else {
		observer("Creating state without archive in %s ..", path)
	}
	if err := os.Mkdir(path, 0700); err != nil {
		return res, fmt.Errorf("failed to create temporary state directory: %v", err)
	}
	if params.keepState {
		observer("state in %s will not be removed at the end of the run", path)
	} else {
		observer("state in %s will be removed at the end of the run", path)
		defer func() {
			observer("Cleaning up state in %s ..", path)
			if err := os.RemoveAll(path); err != nil {
				observer("Cleanup failed: %v", err)
			}
		}()
	}

	// Open the state to be tested.
	archive := state.NoArchive
	if params.archive {
		archive = state.S5Archive
	}
	state, err := state.NewState(state.Parameters{
		Directory: path,
		Variant:   gostate.VariantGoFile,
		Schema:    state.Schema(params.schema),
		Archive:   archive,
	})
	if err != nil {
		return res, err
	}
	defer func() {
		start := time.Now()
		if err := state.Close(); err != nil {
			observer("Failed to close state: %v", err)
		}
		observer("Closing state took %v", time.Since(start))
		observer("Final disk usage: %d", getDirectorySize(path))
	}()

	// Progress tracking.
	reportingInterval := params.reportInterval
	lastReportTime := time.Now()

	// Record results.
	res.intervals = make([]benchmarkRecord, 0, params.numBlocks/reportingInterval+1)

	benchmarkStart := time.Now()
	reportingTime := 0 * time.Second

	// Simulate insertions.
	numBlocks := params.numBlocks
	numReadsPerBlock := params.numReadsPerBlock
	numInsertsPerBlock := params.numInsertsPerBlock
	counter := uint64(0)
	observer(
		"Simulating %d blocks with %d reads and %d inserts each",
		numBlocks, numReadsPerBlock, numInsertsPerBlock,
	)
	for i := 0; i < numBlocks; i++ {
		for j := 0; j < numReadsPerBlock; j++ {
			addr := common.Address{byte(counter), byte(counter >> 8), byte(counter >> 16), byte(counter >> 24), byte(counter >> 32)}
			state.GetBalance(addr)
			counter++
		}
		update := common.Update{}
		update.CreatedAccounts = make([]common.Address, 0, numInsertsPerBlock)
		for j := 0; j < numInsertsPerBlock; j++ {
			addr := common.Address{byte(counter), byte(counter >> 8), byte(counter >> 16), byte(counter >> 24), byte(counter >> 32)}
			update.CreatedAccounts = append(update.CreatedAccounts, addr)
			update.Nonces = append(update.Nonces, common.NonceUpdate{Account: addr, Nonce: common.ToNonce(1)})
			counter++
		}
		if err := state.Apply(uint64(i), update); err != nil {
			return res, fmt.Errorf("error applying block %d: %v", i, err)
		}

		if (i+1)%reportingInterval == 0 {
			if tracingEnabled {
				stopTracer()
			}
			if profilingEnabled {
				stopCpuProfiler()
			}
			startReporting := time.Now()

			throughput := float64(reportingInterval*numInsertsPerBlock) / startReporting.Sub(lastReportTime).Seconds()
			memory := state.GetMemoryFootprint().Total()
			disk := getDirectorySize(path)
			observer(
				"Reached block %d, memory %.2f GB, disk %.2f GB, %.2f inserts/second",
				i+1,
				float64(memory)/float64(1<<30),
				float64(disk)/float64(1<<30),
				throughput,
			)

			res.intervals = append(res.intervals, benchmarkRecord{
				endOfBlock: i + 1,
				memory:     int64(memory),
				disk:       disk,
				throughput: throughput,
			})

			endReporting := time.Now()
			reportingTime += endReporting.Sub(startReporting)
			lastReportTime = endReporting

			intervalNumber := ((i + 1) / reportingInterval) + 1
			if profilingEnabled {
				startCpuProfiler(fmt.Sprintf("%s_%06d", params.cpuProfilePrefix, intervalNumber))
			}
			if tracingEnabled {
				startTracer(fmt.Sprintf("%s_%06d", params.traceFilePrefix, intervalNumber))
			}
		}
	}
	observer(
		"Finished %.2e blocks with %.2e reads and %.2e inserts",
		float64(numBlocks),
		float64(numBlocks*numReadsPerBlock),
		float64(numBlocks*numInsertsPerBlock),
	)

	benchmarkTime := time.Since(benchmarkStart)
	res.numInserts = int64(counter)
	res.insertTime = benchmarkTime - reportingTime
	res.reportTime = reportingTime

	return res, nil
}

// GetDirectorySize computes the size of all files in the given directory in bytes.
func getDirectorySize(directory string) int64 {
	var sum int64 = 0
	filepath.Walk(directory, func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if !info.IsDir() {
			sum += info.Size()
		}
		return nil
	})
	return sum
}
