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

// TEMPORARY DIAGNOSTICS -- NOT INTENDED FOR MERGING.
//
// This file contains helpers to investigate a production crash where the node
// flusher dereferenced a nil Node interface stored in a shared.Shared[Node]
// obtained from the node cache:
//
//	panic: runtime error: invalid memory address or nil pointer dereference
//	[signal SIGSEGV: segmentation violation code=0x1 addr=0x58 ...]
//	  mpt.tryFlushDirtyNodes.func1 (node_flusher.go:102)
//
// Instead of crashing, the checks added around the code base detect the
// broken cache entry, skip it, and write as much information as possible to
// the log and to a dump file. Additionally, a panic in the background flusher
// is reported (including a dump of all goroutines) before it is re-raised, so
// that the information is not lost if the journal drops log lines.
//
// Environment variables:
//   CARMEN_DIAG_DIR  directory for the dump files (default: os.TempDir())

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/database/mpt/shared"
)

const (
	diagLogPrefix = "CARMEN-DIAG: "
	// Only the first few incidents write a full dump file, to bound the cost
	// and the amount of data in case the problem repeats in a tight loop.
	diagMaxDumpFiles = 5
	// Upper limit for the size of the all-goroutine stack dump.
	diagMaxStackDumpBytes = 64 << 20
)

var (
	diagIncidentCount atomic.Uint64
	diagDumpCount     atomic.Uint64
)

// isBrokenSharedNode checks the given shared node. It returns a non-empty
// reason if the node is nil or holds a nil Node. If the node is currently
// locked exclusively, its content can not be inspected and it is assumed to be
// healthy.
func isBrokenSharedNode(node *shared.Shared[Node]) string {
	if node == nil {
		return "nil *Shared[Node]"
	}
	handle, ok := node.TryGetReadHandle()
	if !ok {
		return ""
	}
	value := handle.Get()
	handle.Release()
	if value == nil {
		return "Shared[Node] holds a nil Node interface"
	}
	return ""
}

// checkSharedNode reports an incident if the given node is broken. It returns
// true if the node is healthy (or could not be inspected).
func checkSharedNode(where string, id NodeId, node *shared.Shared[Node], cache NodeCache) bool {
	if reason := isBrokenSharedNode(node); reason != "" {
		reportNilNode(where, reason, id, node, cache)
		return false
	}
	return true
}

// reportNilNode logs everything known about a broken cache entry. It must not
// be called while holding the mutex of the node cache.
func reportNilNode(where, reason string, id NodeId, node *shared.Shared[Node], cache NodeCache) {
	n := diagIncidentCount.Add(1)

	var sb strings.Builder
	fmt.Fprintf(&sb, "%sbroken node detected (incident #%d) at %s: %s\n", diagLogPrefix, n, where, reason)
	fmt.Fprintf(&sb, "  time:       %s\n", time.Now().Format(time.RFC3339Nano))
	fmt.Fprintf(&sb, "  node id:    %v (%d, %#x)\n", id, uint64(id), uint64(id))
	fmt.Fprintf(&sb, "  node kind:  %s\n", describeNodeIdKind(id))
	fmt.Fprintf(&sb, "  shared ptr: %p\n", node)
	if node != nil {
		fmt.Fprintf(&sb, "  shared raw: %s\n", rawWords(unsafe.Pointer(node), unsafe.Sizeof(*node)))
		fmt.Fprintf(&sb, "              (words 0,1 = Node interface itab,data; the rest are the two RWMutex)\n")
	}
	if c, ok := cache.(*nodeCache); ok && c != nil {
		fmt.Fprintf(&sb, "  cache:      %s\n", c.describeForDiagnostics(id))
	}
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	fmt.Fprintf(&sb, "  runtime:    goroutines=%d heapAlloc=%dMiB heapSys=%dMiB numGC=%d lastGC=%s go=%s\n",
		runtime.NumGoroutine(), ms.HeapAlloc>>20, ms.HeapSys>>20, ms.NumGC,
		time.Unix(0, int64(ms.LastGC)).Format(time.RFC3339), runtime.Version())
	fmt.Fprintf(&sb, "  stack of detecting goroutine:\n%s", indent(string(debug.Stack()), "    "))
	if path := writeDumpFile("nil-node", sb.String()); path != "" {
		fmt.Fprintf(&sb, "  full dump (all goroutines): %s\n", path)
	} else {
		fmt.Fprintf(&sb, "  full dump (all goroutines): not written (limit reached or error)\n")
	}
	log.Print(sb.String())
}

// reportPanic logs a panic observed in a background goroutine, including the
// stacks of all goroutines. It is meant to be called from a deferred recover
// that re-panics afterwards.
func reportPanic(where string, value any) {
	var sb strings.Builder
	fmt.Fprintf(&sb, "%spanic in %s: %v\n", diagLogPrefix, where, value)
	fmt.Fprintf(&sb, "  time: %s\n", time.Now().Format(time.RFC3339Nano))
	fmt.Fprintf(&sb, "  stack of panicking goroutine:\n%s", indent(string(debug.Stack()), "    "))
	if path := writeDumpFile("panic", sb.String()); path != "" {
		fmt.Fprintf(&sb, "  full dump (all goroutines): %s\n", path)
	}
	log.Print(sb.String())
}

// writeDumpFile writes the given header followed by the stacks of all
// goroutines to a new file and returns its path, or an empty string if no file
// was written.
func writeDumpFile(kind, header string) string {
	seq := diagDumpCount.Add(1)
	if seq > diagMaxDumpFiles {
		return ""
	}
	dir := os.Getenv("CARMEN_DIAG_DIR")
	if dir == "" {
		dir = os.TempDir()
	}
	name := fmt.Sprintf("carmen-diag-%s-%s-pid%d-%d.txt", kind, time.Now().UTC().Format("20060102T150405.000"), os.Getpid(), seq)
	path := filepath.Join(dir, name)

	buf := make([]byte, 1<<20)
	for {
		n := runtime.Stack(buf, true)
		if n < len(buf) || len(buf) >= diagMaxStackDumpBytes {
			buf = buf[:n]
			break
		}
		buf = make([]byte, 2*len(buf))
	}
	content := header + "\n==== all goroutines ====\n" + string(buf)
	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		log.Printf("%sfailed to write dump file %s: %v", diagLogPrefix, path, err)
		return ""
	}
	return path
}

func describeNodeIdKind(id NodeId) string {
	switch {
	case id.IsEmpty():
		return "empty"
	case id.IsValue():
		return fmt.Sprintf("value[%d]", id.Index())
	case id.IsAccount():
		return fmt.Sprintf("account[%d]", id.Index())
	case id.IsBranch():
		return fmt.Sprintf("branch[%d]", id.Index())
	case id.IsExtension():
		return fmt.Sprintf("extension[%d]", id.Index())
	}
	return "unknown"
}

// rawWords renders size bytes at the given address as hexadecimal 8-byte words.
func rawWords(p unsafe.Pointer, size uintptr) string {
	var parts []string
	for off := uintptr(0); off+8 <= size; off += 8 {
		parts = append(parts, fmt.Sprintf("%016x", *(*uint64)(unsafe.Add(p, off))))
	}
	return strings.Join(parts, " ")
}

func indent(s, prefix string) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	for i := range lines {
		lines[i] = prefix + lines[i]
	}
	return strings.Join(lines, "\n") + "\n"
}

// describeForDiagnostics summarizes the cache state and the slot owning the
// given id. It acquires the cache mutex.
func (c *nodeCache) describeForDiagnostics(id NodeId) string {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	res := fmt.Sprintf("capacity=%d indexSize=%d tagCounter=%d head=%d tail=%d",
		len(c.owners), len(c.index), c.tagCounter, c.head, c.tail)
	pos, found := c.index[id]
	if !found {
		return res + " | id not in index (evicted meanwhile?)"
	}
	describe := func(p ownerPosition) string {
		o := &c.owners[p]
		return fmt.Sprintf("slot=%d tag=%d id=%v nodePtr=%p prev=%d next=%d",
			p, o.tag.Load(), NodeId(o.id.Load()), o.node.Load(), o.prev, o.next)
	}
	res += " | index -> " + describe(pos)
	// Also describe the neighbors in the LRU list.
	if int(c.owners[pos].prev) < len(c.owners) {
		res += " | prev: " + describe(c.owners[pos].prev)
	}
	if int(c.owners[pos].next) < len(c.owners) {
		res += " | next: " + describe(c.owners[pos].next)
	}
	return res
}
