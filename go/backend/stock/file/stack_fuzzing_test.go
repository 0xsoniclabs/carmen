// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package file

import (
	"encoding/binary"
	"github.com/0xsoniclabs/carmen/go/backend/stock"
	"github.com/0xsoniclabs/carmen/go/fuzzing"
	"slices"
	"strings"
	"testing"
)

func FuzzStack_RandomOps(f *testing.F) {
	var opPush = func(_ opType, value int, t fuzzing.TestingT, c *stackFuzzingContext) {
		err := c.stack.Push(value)
		if err != nil {
			t.Errorf("error to push value: %s", err)
		}
		c.shadow.Push(value)
	}

	var opPop = func(_ opType, t fuzzing.TestingT, c *stackFuzzingContext) {
		if got, err := c.stack.Pop(); err != nil {
			// error when the shadow is empty is OK state
			if c.shadow.Empty() && strings.HasPrefix(err.Error(), "cannot pop from empty stack") {
				return
			}

			t.Errorf("error to pop value: %s", err)
		} else {
			want := c.shadow.Pop()
			if got != want {
				t.Errorf("stack does not match expected value: %v != %v", got, want)
			}
		}
	}

	var opGetAll = func(_ opType, t fuzzing.TestingT, c *stackFuzzingContext) {
		got, err := c.stack.GetAll()
		if err != nil {
			t.Errorf("error to get all values: %s", err)
		}
		want := c.shadow.GetAll()
		if !slices.Equal(got, want) {
			t.Errorf("stack does not match expected value: %v != %v", got, want)
		}
	}

	var opSize = func(_ opType, t fuzzing.TestingT, c *stackFuzzingContext) {
		if got, want := c.stack.Size(), c.shadow.Size(); got != want {
			t.Errorf("stack does not match expected value: %v != %v", got, want)
		}
	}

	var opEmpty = func(_ opType, t fuzzing.TestingT, c *stackFuzzingContext) {
		if got, want := c.stack.Empty(), c.shadow.Empty(); got != want {
			t.Errorf("stack does not match expected value: %v != %v", got, want)
		}
	}

	var opClose = func(_ opType, t fuzzing.TestingT, c *stackFuzzingContext) {
		if err := c.stack.Close(); err != nil {
			t.Errorf("error to flush stack: %s", err)
		}
		stack, err := openFileBasedStack[int](c.path)
		if err != nil {
			t.Fatalf("failed to open buffered file: %v", err)
		}
		c.stack = stack
	}

	var opFlush = func(_ opType, t fuzzing.TestingT, c *stackFuzzingContext) {
		if err := c.stack.Flush(); err != nil {
			t.Errorf("error to flush stack: %s", err)
		}
	}

	serialise := func(payload int) []byte {
		return binary.BigEndian.AppendUint32(make([]byte, 0, 4), uint32(payload))
	}
	deserialise := func(b *[]byte) int {
		if len(*b) >= 4 {
			res := int(binary.BigEndian.Uint32((*b)[0:4]))
			*b = (*b)[4:]
			return res
		} else {
			*b = (*b)[:]
			return 0
		}
	}

	registry := fuzzing.NewRegistry[opType, stackFuzzingContext]()
	fuzzing.RegisterDataOp(registry, push, serialise, deserialise, opPush)
	fuzzing.RegisterNoDataOp(registry, pop, opPop)
	fuzzing.RegisterNoDataOp(registry, getAll, opGetAll)
	fuzzing.RegisterNoDataOp(registry, size, opSize)
	fuzzing.RegisterNoDataOp(registry, empty, opEmpty)
	fuzzing.RegisterNoDataOp(registry, close, opClose)
	fuzzing.RegisterNoDataOp(registry, flush, opFlush)

	fuzzing.Fuzz[stackFuzzingContext](f, &stackFuzzingCampaign{registry: registry})
}

// opType is operation type to be applied to a stack.
type opType byte

const (
	push opType = iota
	pop
	getAll
	size
	empty
	close
	flush
)

type stackFuzzingCampaign struct {
	registry fuzzing.OpsFactoryRegistry[opType, stackFuzzingContext]
}

type stackFuzzingContext struct {
	path   string
	stack  *fileBasedStack[int]
	shadow stack[int]
}

func (c *stackFuzzingCampaign) Init() []fuzzing.OperationSequence[stackFuzzingContext] {

	push1 := c.registry.CreateDataOp(push, 99)
	push2 := c.registry.CreateDataOp(push, ^99)

	popOp := c.registry.CreateNoDataOp(pop)
	flushOp := c.registry.CreateNoDataOp(flush)
	closeOp := c.registry.CreateNoDataOp(close)
	emptyOp := c.registry.CreateNoDataOp(empty)
	getAllOp := c.registry.CreateNoDataOp(getAll)
	sizeOp := c.registry.CreateNoDataOp(size)

	// generate some adhoc sequences of operations
	data := []fuzzing.OperationSequence[stackFuzzingContext]{
		{push1, popOp, flushOp, closeOp},
		{push1, popOp, sizeOp, emptyOp,
			flushOp, closeOp},
		{push1, push2, getAllOp, closeOp},
		{popOp, push2, getAllOp, closeOp},
		{closeOp, push2, getAllOp},
	}

	return data
}

func (c *stackFuzzingCampaign) CreateContext(t fuzzing.TestingT) *stackFuzzingContext {
	path := t.TempDir() + "/test.dat"
	fileStack, err := openFileBasedStack[int](path)
	if err != nil {
		t.Fatalf("failed to open file stack: %v", err)
	}
	shadow := stack[int]{}
	return &stackFuzzingContext{path, fileStack, shadow}
}

func (c *stackFuzzingCampaign) Deserialize(rawData []byte) []fuzzing.Operation[stackFuzzingContext] {
	return parseOperations(c.registry, rawData)
}

func (c *stackFuzzingCampaign) Cleanup(t fuzzing.TestingT, context *stackFuzzingContext) {
	if err := context.stack.Close(); err != nil {
		t.Fatalf("cannot close file: %s", err)
	}
}

// parseOperations converts the input byte array
// to the list of operations.
// It is converted from the format:<opType><value>
// Value part is parsed only when the opType equals to push.
// This method tries to parse as many those tuples as possible, terminating when no more
// elements are available.
// This method recognizes expensive operations, which is flush, close and getAll, and it caps
// the number of these operations to 20 in total and 3 in a row.
// The fuzzing mechanism requires one campaign does not run more than 1s, which is quickly
// broken when an expensive operation is triggered extensively.
func parseOperations(registry fuzzing.OpsFactoryRegistry[opType, stackFuzzingContext], b []byte) []fuzzing.Operation[stackFuzzingContext] {
	var ops []fuzzing.Operation[stackFuzzingContext]
	var expensiveOpTotal, expensiveOpRow int
	for len(b) >= 1 {
		opType, op := registry.ReadNextOp(&b)
		if opType == flush || opType == close || opType == getAll {
			expensiveOpRow++
			expensiveOpTotal++
			if expensiveOpRow > 3 || expensiveOpTotal > 20 {
				continue
			}
		} else {
			expensiveOpRow = 0
		}

		ops = append(ops, op)
	}
	return ops
}

// stack used as a shadow implementation for testing.
type stack[I stock.Index] []I

func (s *stack[I]) GetAll() []I {
	return *s
}

func (s *stack[I]) Empty() bool {
	return len(*s) == 0
}

func (s *stack[I]) Size() int {
	return len(*s)
}

func (s *stack[I]) Push(v I) {
	*s = append(*s, v)
}

func (s *stack[I]) Pop() I {
	res := (*s)[len(*s)-1]
	*s = (*s)[:len(*s)-1]
	return res
}
