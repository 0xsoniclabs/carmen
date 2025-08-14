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
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"strings"

	"github.com/urfave/cli/v2"
)

// AddPerformanceDiagnosticsAction wraps an action function to add performance diagnostics
// such as CPU profiling, tracing, and a diagnostic server.
// It takes the action function and flags for diagnostics, CPU profiling, and tracing.
// The diagnosticsFlag must be an integer, and it starts diagnostic server at the port parsed from this flag,
// cpuProfileFlag is a string that starts CPU profiling giving the file name from this flag,
// and traceFlag is a string that starts tracing giving the file name from this flag.
func AddPerformanceDiagnosticsAction(action cli.ActionFunc, diagnosticsFlag *cli.IntFlag, cpuProfileFlag, traceFlag *cli.StringFlag) cli.ActionFunc {
	return func(context *cli.Context) error {

		// Start the diagnostic service if requested.
		diagnosticPort := context.Int(diagnosticsFlag.Names()[0])
		startDiagnosticServer(diagnosticPort)

		// Start CPU profiling.
		cpuProfileFileName := context.String(cpuProfileFlag.Names()[0])
		if strings.TrimSpace(cpuProfileFileName) != "" {
			if err := startCpuProfiler(cpuProfileFileName); err != nil {
				return err
			}
			defer stopCpuProfiler()
		}

		// Start recording a trace.
		traceFileName := context.String(traceFlag.Names()[0])
		if strings.TrimSpace(traceFileName) != "" {
			if err := startTracer(traceFileName); err != nil {
				return err
			}
			defer stopTracer()
		}

		return action(context)
	}
}

func startDiagnosticServer(port int) {
	if port <= 0 || port >= (1<<16) {
		return
	}
	fmt.Printf("Starting diagnostic server at port http://localhost:%d\n", port)
	fmt.Printf("(see https://pkg.go.dev/net/http/pprof#hdr-Usage_examples for usage examples)\n")
	fmt.Printf("Block and mutex sampling rate is set to 100%% for diagnostics, which may impact overall performance\n")
	go func() {
		addr := fmt.Sprintf("localhost:%d", port)
		log.Println(http.ListenAndServe(addr, nil))
	}()
	runtime.SetBlockProfileRate(1)
	runtime.SetMutexProfileFraction(1)
}

func startCpuProfiler(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("could not create CPU profile: %s", err)
	}
	if err := pprof.StartCPUProfile(f); err != nil {
		return fmt.Errorf("could not start CPU profile: %s", err)
	}
	return nil
}

func stopCpuProfiler() {
	pprof.StopCPUProfile()
}

func startTracer(filename string) error {
	traceFile, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create trace file: %v", err)
	}
	if err := trace.Start(traceFile); err != nil {
		return fmt.Errorf("failed to start trace: %v", err)
	}
	return nil
}

func stopTracer() {
	trace.Stop()
}
