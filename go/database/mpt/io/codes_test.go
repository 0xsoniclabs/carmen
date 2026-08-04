// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package io

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/stretchr/testify/require"
)

func TestWriteCodes_WritesCodesOrderedByHash(t *testing.T) {
	require := require.New(t)

	codes := map[common.Hash][]byte{
		{3}: {6, 7, 8},
		{1}: {1, 2, 3},
		{2}: {4, 5},
	}

	var buf bytes.Buffer
	require.NoError(writeCodes(maps.All(codes), &buf, t.TempDir()))

	// Output must be ordered by hash regardless of iteration order.
	want := [][]byte{
		{1, 2, 3},
		{4, 5},
		{6, 7, 8},
	}
	var got [][]byte
	in := bytes.NewReader(buf.Bytes())
	for range codes {
		tag, err := in.ReadByte()
		require.NoError(err)
		require.Equal(byte('C'), tag)
		code, err := readCode(in)
		require.NoError(err)
		got = append(got, code)
	}
	_, err := in.ReadByte()
	require.ErrorIs(err, io.EOF)

	require.Equal(want, got)
}

func TestWriteCodes_WritesNothingForEmptyIterator(t *testing.T) {
	require := require.New(t)

	var buf bytes.Buffer
	require.NoError(writeCodes(maps.All(map[common.Hash][]byte{}), &buf, t.TempDir()))
	require.Zero(buf.Len())
}

func TestWriteCodes_PropagatesWriteError(t *testing.T) {
	injected := errors.New("write failed")
	tests := map[string]struct {
		failAfter int
	}{
		"on header write": {failAfter: 0},
		"on body write":   {failAfter: 1},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)

			w := &failingWriter{failAfter: test.failAfter, err: injected}
			codes := map[common.Hash][]byte{{1}: {1, 2, 3}}

			err := writeCodes(maps.All(codes), w, t.TempDir())
			require.ErrorContains(err, injected.Error())
		})
	}
}

func TestWriteCodes_RemovesStagingStore(t *testing.T) {
	tests := map[string]struct {
		out             io.Writer
		emptyScratchDir bool
	}{
		"on success":           {out: &bytes.Buffer{}},
		"on writer failure":    {out: &failingWriter{err: errors.New("boom")}},
		"in fallback temp dir": {out: &bytes.Buffer{}, emptyScratchDir: true},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)

			tempRoot := t.TempDir()
			scratchDir := tempRoot
			if test.emptyScratchDir {
				// An empty scratch dir falls back to the system temp directory,
				// redirected here so the test can observe the cleanup.
				t.Setenv("TMPDIR", tempRoot)
				scratchDir = ""
			}

			codes := map[common.Hash][]byte{{1}: {1, 2, 3}}
			_ = writeCodes(maps.All(codes), test.out, scratchDir)

			entries, err := os.ReadDir(tempRoot)
			require.NoError(err)
			require.Empty(entries, "writeCodes must remove its staging store")
		})
	}
}

func TestReadCode_ReturnsCodeWrittenByWriteCodes(t *testing.T) {
	require := require.New(t)

	want := []byte{9, 8, 7, 6}
	codes := map[common.Hash][]byte{{1}: want}

	var buf bytes.Buffer
	require.NoError(writeCodes(maps.All(codes), &buf, t.TempDir()))

	tag, err := buf.ReadByte()
	require.NoError(err)
	require.Equal(byte('C'), tag)

	got, err := readCode(&buf)
	require.NoError(err)
	require.Equal(want, got)
}

func TestReadCode_ReturnsEmptyCodeForZeroLength(t *testing.T) {
	require := require.New(t)

	code, err := readCode(bytes.NewReader([]byte{0, 0}))
	require.NoError(err)
	require.Empty(code)
}

func TestReadCode_ReturnsErrorOnTruncatedInput(t *testing.T) {
	// Header claims 4 bytes of body but only 2 are provided.
	truncatedBody := []byte{0, 0, 1, 2}
	binary.BigEndian.PutUint16(truncatedBody, 4)

	tests := map[string]struct {
		input []byte
	}{
		"short length header": {input: []byte{0}},
		"short code body":     {input: truncatedBody},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)

			_, err := readCode(bytes.NewReader(test.input))
			require.ErrorIs(err, io.ErrUnexpectedEOF)
		})
	}
}

func TestNewCodeSortStore_CreatesUniqueDirectoriesUnderScratchDir(t *testing.T) {
	require := require.New(t)

	scratchDir := t.TempDir()
	a, err := newCodeSortStore(scratchDir)
	require.NoError(err)
	b, err := newCodeSortStore(scratchDir)
	require.NoError(err)

	require.NotEqual(a.dir, b.dir)
	require.Equal(scratchDir, filepath.Dir(a.dir))
	require.Equal(scratchDir, filepath.Dir(b.dir))

	require.NoError(a.close())
	require.NoError(b.close())
	entries, err := os.ReadDir(scratchDir)
	require.NoError(err)
	require.Empty(entries, "close must remove the staging directories")
}

func TestCodeSortStore_AddAndWriteTo_DeduplicatesRepeatedHashes(t *testing.T) {
	require := require.New(t)

	store, err := newCodeSortStore(t.TempDir())
	require.NoError(err)
	t.Cleanup(func() { require.NoError(store.close()) })

	code := []byte{1, 2, 3}
	require.NoError(store.add(common.Hash{1}, code))
	require.NoError(store.add(common.Hash{1}, code))

	var buf bytes.Buffer
	require.NoError(store.writeTo(&buf))

	tag, err := buf.ReadByte()
	require.NoError(err)
	require.Equal(byte('C'), tag)
	got, err := readCode(&buf)
	require.NoError(err)
	require.Equal(code, got)
	require.Zero(buf.Len(), "a repeatedly added code must be written only once")
}

func BenchmarkCodeSortStore(b *testing.B) {
	sizes := []int{100_000, 1_000_000, 10_000_000}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			require := require.New(b)
			var total time.Duration
			for range b.N {
				store, err := newCodeSortStore(b.TempDir())
				if err != nil {
					b.Fatal(err)
				}
				start := time.Now()
				for i := range n {
					code := [200]byte{byte(i >> 24), byte(i >> 16), byte(i >> 8), byte(i)}
					// Mix i to avoid sequential key insertion.
					u := uint32(i) * 2654435761
					var hash common.Hash
					binary.BigEndian.PutUint32(hash[0:4], u)
					if err := store.add(hash, code[:]); err != nil {
						b.Fatal(err)
					}
				}
				require.NoError(store.writeTo(io.Discard))
				total += time.Since(start)
				require.NoError(store.close())
			}
			perOp := total.Seconds() / float64(b.N)
			b.ReportMetric(perOp, "Add+WriteTo/s")
		})
	}
}

// failingWriter is an io.Writer that succeeds for the first failAfter calls
// and then returns err on every subsequent call.
type failingWriter struct {
	failAfter int
	calls     int
	err       error
}

func (w *failingWriter) Write(p []byte) (int, error) {
	if w.calls >= w.failAfter {
		return 0, w.err
	}
	w.calls++
	return len(p), nil
}
