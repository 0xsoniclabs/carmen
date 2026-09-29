// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package mpt

import (
	"bytes"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/0xsoniclabs/carmen/go/backend/stock"
	"github.com/0xsoniclabs/carmen/go/database/mpt/shared"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
)

// setUpDiagnosticsTest redirects the log and dump files and resets counters.
func setUpDiagnosticsTest(t *testing.T) (logs *bytes.Buffer, dumpDir string) {
	t.Helper()
	dumpDir = t.TempDir()
	t.Setenv("CARMEN_DIAG_DIR", dumpDir)
	diagIncidentCount.Store(0)
	diagDumpCount.Store(0)
	logs = &bytes.Buffer{}
	log.SetOutput(logs)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })
	return logs, dumpDir
}

func newMockStockForForest[N any](ctrl *gomock.Controller) *stock.MockStock[uint64, N] {
	res := stock.NewMockStock[uint64, N](ctrl)
	res.EXPECT().Flush().AnyTimes()
	res.EXPECT().Close().AnyTimes()
	return res
}

func TestDiagnostics_FlusherSkipsAndReportsNilNodeValue(t *testing.T) {
	logs, dumpDir := setUpDiagnosticsTest(t)

	cache := NewNodeCache(100)
	healthyId, brokenId := ValueId(1), ValueId(2)
	healthyRef, brokenRef := NewNodeReference(healthyId), NewNodeReference(brokenId)
	healthy := shared.MakeShared[Node](&ValueNode{})
	broken := shared.MakeShared[Node](nil) // < the situation observed in production
	cache.GetOrSet(&healthyRef, healthy)
	cache.GetOrSet(&brokenRef, broken)

	ctrl := gomock.NewController(t)
	sink := NewMockNodeSink(ctrl)
	sink.EXPECT().Write(healthyId, gomock.Any()).Return(nil) // only the healthy node is written

	// Without the diagnostics this call crashed with a nil pointer dereference.
	require.NoError(t, tryFlushDirtyNodes(cache, sink))

	require.Equal(t, uint64(1), diagIncidentCount.Load())
	out := logs.String()
	require.Contains(t, out, "CARMEN-DIAG: broken node detected")
	require.Contains(t, out, "node flusher (collecting dirty nodes)")
	require.Contains(t, out, "Shared[Node] holds a nil Node interface")
	require.Contains(t, out, "value[2]")
	require.Contains(t, out, "shared raw:")
	require.Contains(t, out, "capacity=100")
	require.Contains(t, out, "slot=")

	files, err := filepath.Glob(filepath.Join(dumpDir, "carmen-diag-nil-node-*.txt"))
	require.NoError(t, err)
	require.Len(t, files, 1)
	content, err := os.ReadFile(files[0])
	require.NoError(t, err)
	require.Contains(t, string(content), "==== all goroutines ====")
	require.Contains(t, string(content), "tryFlushDirtyNodes")
}

func TestDiagnostics_FlusherSkipsAndReportsNilSharedPointer(t *testing.T) {
	logs, _ := setUpDiagnosticsTest(t)

	ctrl := gomock.NewController(t)
	cache := NewMockNodeCache(ctrl)
	sink := NewMockNodeSink(ctrl)
	cache.EXPECT().ForEach(gomock.Any()).Do(func(f func(NodeId, *shared.Shared[Node])) {
		f(ValueId(7), nil)
	})

	require.NoError(t, tryFlushDirtyNodes(cache, sink))
	require.Equal(t, uint64(1), diagIncidentCount.Load())
	require.Contains(t, logs.String(), "nil *Shared[Node] provided by cache")
}

func TestDiagnostics_FlusherStillFlushesHealthyDirtyNodes(t *testing.T) {
	logs, _ := setUpDiagnosticsTest(t)

	cache := NewNodeCache(100)
	id := ValueId(1)
	ref := NewNodeReference(id)
	node := shared.MakeShared[Node](&ValueNode{})
	cache.GetOrSet(&ref, node)

	ctrl := gomock.NewController(t)
	sink := NewMockNodeSink(ctrl)
	sink.EXPECT().Write(id, gomock.Any()).Return(nil)

	require.NoError(t, tryFlushDirtyNodes(cache, sink))
	require.Equal(t, uint64(0), diagIncidentCount.Load())
	require.NotContains(t, logs.String(), "CARMEN-DIAG")

	handle := node.GetReadHandle()
	defer handle.Release()
	require.False(t, handle.Get().IsDirty())
}

func TestDiagnostics_InsertingBrokenNodeIsReportedWithCreatorStack(t *testing.T) {
	logs, _ := setUpDiagnosticsTest(t)

	ctrl := gomock.NewController(t)
	forest, err := makeForest(
		MptConfig{Hashing: DirectHashing},
		newMockStockForForest[BranchNode](ctrl),
		newMockStockForForest[ExtensionNode](ctrl),
		newMockStockForForest[AccountNode](ctrl),
		newMockStockForForest[ValueNode](ctrl),
		ForestConfig{NodeCacheConfig: NodeCacheConfig{BackgroundFlushPeriod: -1}},
	)
	require.NoError(t, err)

	ref := NewNodeReference(ValueId(3))
	forest.addToCache(&ref, shared.MakeShared[Node](nil))

	require.Equal(t, uint64(1), diagIncidentCount.Load())
	out := logs.String()
	require.Contains(t, out, "Forest.addToCache (node to be inserted)")
	require.Contains(t, out, "TestDiagnostics_InsertingBrokenNodeIsReportedWithCreatorStack") // < stack shows the caller

	// Healthy nodes are not reported.
	ref2 := NewNodeReference(ValueId(4))
	forest.addToCache(&ref2, shared.MakeShared[Node](&ValueNode{}))
	require.Equal(t, uint64(1), diagIncidentCount.Load())
}

func TestDiagnostics_NodesLockedExclusivelyAreNotReported(t *testing.T) {
	setUpDiagnosticsTest(t)
	node := shared.MakeShared[Node](&ValueNode{})
	handle := node.GetWriteHandle()
	defer handle.Release()
	require.Empty(t, isBrokenSharedNode(node))
}

func TestDiagnostics_FlusherPanicIsReportedAndReraised(t *testing.T) {
	logs, dumpDir := setUpDiagnosticsTest(t)

	ctrl := gomock.NewController(t)
	cache := NewMockNodeCache(ctrl)
	panicked := make(chan struct{})
	cache.EXPECT().ForEach(gomock.Any()).DoAndReturn(func(func(NodeId, *shared.Shared[Node])) {
		defer close(panicked)
		panic("injected test panic")
	})

	// The re-raised panic would terminate the test binary if raised from the
	// flusher goroutine, so the deferred function is exercised directly.
	func() {
		defer func() {
			r := recover()
			require.Equal(t, "injected test panic", r)
		}()
		defer func() {
			if r := recover(); r != nil {
				reportPanic("node flusher", r)
				panic(r)
			}
		}()
		_ = tryFlushDirtyNodes(cache, nil)
	}()
	<-panicked

	require.Contains(t, logs.String(), "panic in node flusher: injected test panic")
	files, err := filepath.Glob(filepath.Join(dumpDir, "carmen-diag-panic-*.txt"))
	require.NoError(t, err)
	require.Len(t, files, 1)
}

func TestDiagnostics_DumpFilesAreLimited(t *testing.T) {
	_, dumpDir := setUpDiagnosticsTest(t)
	for i := 0; i < diagMaxDumpFiles+3; i++ {
		reportNilNode("test", "test", ValueId(1), nil, nil)
	}
	files, err := filepath.Glob(filepath.Join(dumpDir, "carmen-diag-*.txt"))
	require.NoError(t, err)
	require.Len(t, files, diagMaxDumpFiles)
	require.True(t, strings.Contains(os.Getenv("CARMEN_DIAG_DIR"), dumpDir))
}
