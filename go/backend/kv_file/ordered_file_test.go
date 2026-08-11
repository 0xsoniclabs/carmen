// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package kv_file

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"iter"
	"os"
	"path/filepath"
	"testing"

	"github.com/0xsoniclabs/carmen/go/backend/utils"
	"github.com/0xsoniclabs/carmen/go/common/iter_utils"
	"github.com/stretchr/testify/require"
)

const orderedItemSize uint64 = 8

func TestOrderedFile_Open_CreatesFileWhenMissing(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "new.dat")

	_, err := os.Stat(path)
	require.True(os.IsNotExist(err))

	f, err := OpenOrderedFile[uint64](path, orderedItemSize, orderedReadValue, orderedWriteValue)
	require.NoError(err)
	defer func() { require.NoError(f.Close()) }()

	info, err := os.Stat(path)
	require.NoError(err)
	require.EqualValues(0, info.Size())
}

func TestOrderedFile_Open_OpensExistingFile(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "existing.dat")

	// Pre-populate two entries.
	var buf [16]byte
	binary.LittleEndian.PutUint64(buf[0:8], 111)
	binary.LittleEndian.PutUint64(buf[8:16], 222)
	require.NoError(os.WriteFile(path, buf[:], 0600))

	f, err := OpenOrderedFile[uint64](path, orderedItemSize, orderedReadValue, orderedWriteValue)
	require.NoError(err)
	defer func() { require.NoError(f.Close()) }()

	size, err := f.FileSize()
	require.NoError(err)
	require.EqualValues(2*orderedItemSize, size)
}

func TestOrderedFile_Open_ReturnsErrorIfDirectoryDoesNotExist(t *testing.T) {
	require := require.New(t)
	// A path under a non-existent directory cannot be created.
	path := filepath.Join(t.TempDir(), "missing-dir", "file.dat")
	_, err := OpenOrderedFile[uint64](path, orderedItemSize, orderedReadValue, orderedWriteValue)
	require.Error(err)
}

func TestOrderedFile_Open_ReturnsErrorIfFileIsNotMultipleOfEntrySize(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "invalid.dat")

	// Write 10 bytes, which is not a multiple of the 8-byte entry size.
	require.NoError(os.WriteFile(path, []byte("1234567890"), 0600))

	_, err := OpenOrderedFile[uint64](path, orderedItemSize, orderedReadValue, orderedWriteValue)
	require.Error(err)
}

func TestOrderedFile_Get_ReadsPreviouslyWrittenValue(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	require.NoError(f.Set(3, 12345))

	got, err := f.Get(3)
	require.NoError(err)
	require.NotNil(got)
	require.EqualValues(12345, *got)
}

func TestOrderedFile_Get_ReturnsNilForKeyBeyondFile(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	// Nothing was written yet; per the KVFile contract, Get must return
	// (nil, nil) for keys that do not exist rather than an error.
	got, err := f.Get(0)
	require.NoError(err)
	require.Nil(got)
}

func TestOrderedFile_Get_PropagatesReadError(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "err.dat")

	injected := errors.New("injected read failure")
	readFn := func(r io.Reader) (uint64, uint64, error) { return 0, 0, injected }

	// Write enough bytes so that Seek succeeds.
	require.NoError(os.WriteFile(path, []byte("12345678"), 0600))

	f, err := OpenOrderedFile[uint64](path, orderedItemSize, readFn, orderedWriteValue)
	require.NoError(err)
	defer func() { require.NoError(f.Close()) }()

	_, err = f.Get(0)
	require.ErrorIs(err, injected)
}

// TestOrderedFile_Get_ReadIsBoundedToSingleRecord verifies that a value
// decoder cannot read beyond its own record: attempting to consume more than
// itemSize bytes fails instead of silently returning a neighbouring record's
// data.
func TestOrderedFile_Get_ReadIsBoundedToSingleRecord(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "ordered.dat")

	// Two full records on disk.
	require.NoError(os.WriteFile(path, []byte("1234567812345678"), 0600))

	// A faulty decoder that tries to read two records worth of data.
	greedyRead := func(r io.Reader) (uint64, uint64, error) {
		buf := make([]byte, 2*orderedItemSize)
		if _, err := io.ReadFull(r, buf); err != nil {
			return 0, 0, err
		}
		return 0, binary.LittleEndian.Uint64(buf[:8]), nil
	}

	f, err := OpenOrderedFile[uint64](path, orderedItemSize, greedyRead, orderedWriteValue)
	require.NoError(err)
	defer func() { require.NoError(f.Close()) }()

	_, err = f.Get(0)
	require.ErrorIs(err, io.ErrUnexpectedEOF)
}

func TestOrderedFile_Get_ReturnsZeroForNonInitializedKeys(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	require.NoError(f.Set(1000, 42))

	for key := range uint64(1000) {
		got, err := f.Get(key)
		require.NoError(err)
		require.NotNil(got)
		require.EqualValues(0, *got, "key %d", key)
	}
}

func TestOrderedFile_Has_ReportsKeysWithinAllocatedRange(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	has, err := f.Has(0)
	require.NoError(err)
	require.False(has)

	// Writing key 2 allocates slots 0..2, so all of them exist.
	require.NoError(f.Set(2, 42))
	for key := uint64(0); key <= 2; key++ {
		has, err = f.Has(key)
		require.NoError(err)
		require.True(has, "key %d", key)
	}

	has, err = f.Has(3)
	require.NoError(err)
	require.False(has)
}

func TestOrderedFile_Set_WritesValueAtCorrectOffset(t *testing.T) {
	require := require.New(t)
	f, path := openTestOrderedFile(t)

	require.NoError(f.Set(0, 42))
	require.NoError(f.Set(1, 43))
	require.NoError(f.Set(2, 44))

	require.NoError(f.Close())

	data, err := os.ReadFile(path)
	require.NoError(err)
	require.Len(data, 24)
	require.EqualValues(42, binary.LittleEndian.Uint64(data[0:8]))
	require.EqualValues(43, binary.LittleEndian.Uint64(data[8:16]))
	require.EqualValues(44, binary.LittleEndian.Uint64(data[16:24]))
}

func TestOrderedFile_Set_OverwritesExistingValue(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	require.NoError(f.Set(0, 1))
	require.NoError(f.Set(0, 99))

	got, err := f.Get(0)
	require.NoError(err)
	require.NotNil(got)
	require.EqualValues(99, *got)
}

func TestOrderedFile_Set_CanWriteSparseKeys(t *testing.T) {
	require := require.New(t)
	f, path := openTestOrderedFile(t)

	// Writing at key 5 extends the file with an intermediate hole.
	require.NoError(f.Set(5, 500))
	require.NoError(f.Close())

	info, err := os.Stat(path)
	require.NoError(err)
	// Bytes 0..40 correspond to keys 0..4 (unwritten), bytes 40..48 to key 5.
	require.EqualValues(48, info.Size())
}

func TestOrderedFile_Set_PropagatesWriteError(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "err.dat")

	injected := errors.New("injected write failure")
	writeFn := func(w io.Writer, _ uint64, _ uint64) error { return injected }

	f, err := OpenOrderedFile[uint64](path, orderedItemSize, orderedReadValue, writeFn)
	require.NoError(err)
	defer func() { require.NoError(f.Close()) }()

	err = f.Set(0, 1)
	require.ErrorIs(err, injected)
}

func TestOrderedFile_SetBatch_WritesAllEntries(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	batch := map[uint64]uint64{
		0: 10,
		1: 20,
		2: 30,
	}
	require.NoError(f.SetBatch(batch))

	for k, want := range batch {
		got, err := f.Get(k)
		require.NoError(err)
		require.NotNil(got)
		require.EqualValues(want, *got, "key %d", k)
	}
}

func TestOrderedFile_SetBatch_EmptyBatchIsNoop(t *testing.T) {
	require := require.New(t)
	f, path := openTestOrderedFile(t)

	require.NoError(f.SetBatch(map[uint64]uint64{}))

	info, err := os.Stat(path)
	require.NoError(err)
	require.EqualValues(0, info.Size())
}

func TestOrderedFile_Flush_IsNoop(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	require.NoError(f.Flush())
	require.NoError(f.Flush())
}

func TestOrderedFile_Iterate_ReturnsAllStoredValuesIndexedByPosition(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	require.NoError(f.Set(0, 100))
	require.NoError(f.Set(1, 101))
	require.NoError(f.Set(2, 102))

	seq, err := f.Iterate()
	require.NoError(err)

	res, err := iter_utils.CollectOk2(seq)
	require.NoError(err)
	require.Equal(map[uint64]uint64{0: 100, 1: 101, 2: 102}, res)
}

func TestOrderedFile_Iterate_ReturnsEmptyIteratorForEmptyFile(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	seq, err := f.Iterate()
	require.NoError(err)

	res, err := iter_utils.CollectOk2(seq)
	require.NoError(err)
	require.Empty(res)
}

func TestOrderedFile_Iterate_ReportsReadError(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "err.dat")

	// Two 8-byte entries: the iterator would visit both if reads succeeded.
	require.NoError(os.WriteFile(path, []byte("1234567812345678"), 0600))

	injected := errors.New("injected read failure")
	readFn := func(r io.Reader) (uint64, uint64, error) { return 0, 0, injected }

	f, err := OpenOrderedFile[uint64](path, orderedItemSize, readFn, orderedWriteValue)
	require.NoError(err)
	defer func() { require.NoError(f.Close()) }()

	// The iterator is lazy, so Iterate itself does not read the file and
	// therefore cannot surface the read error. It is reported by the sequence
	// instead, which aborts the iteration at the failing read.
	seq, err := f.Iterate()
	require.NoError(err)

	entries, seqErr := iter_utils.CollectOk2(seq)
	require.Empty(entries, "read errors must abort the iterator before yielding")
	require.ErrorIs(seqErr, injected)
}

func TestOrderedFile_Iterate_ReportsErrorWhenFileIsClosed(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	require.NoError(f.Set(0, 1))
	require.NoError(f.Set(1, 2))
	require.NoError(f.Set(2, 3))

	seq, err := f.Iterate()
	require.NoError(err)

	next, stop := iter.Pull(seq)
	defer stop()
	res, ok := next()
	require.True(ok)
	pair, err := res.Get()
	require.NoError(err)
	require.EqualValues(0, pair.Key)
	require.EqualValues(1, pair.Value)

	// Close the file before consuming the iterator.
	require.NoError(f.Close())

	res, ok = next()
	require.True(ok)
	_, err = res.Get()
	require.ErrorIs(err, os.ErrClosed)
}

func TestOrderedFile_Iterate_HandlesChunkedReader(t *testing.T) {
	require := require.New(t)

	values := []uint64{100, 200, 300, 400}

	for _, chunkSize := range []int{1, 2, 4, 7, 16, 1024} {
		t.Run(fmt.Sprintf("chunk=%d", chunkSize), func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "ordered.dat")

			// A readValueFn that first pulls the raw entry bytes from the
			// underlying reader and then streams them through utils.NewChunkReader
			// to simulate short reads. This exercises the pattern that value
			// decoders should follow (use io.ReadFull to tolerate short reads).
			chunkedRead := func(r io.Reader) (uint64, uint64, error) {
				raw := make([]byte, orderedItemSize)
				if _, err := io.ReadFull(r, raw); err != nil {
					return 0, 0, err
				}
				chunked := utils.NewChunkReader(raw, chunkSize)
				var buf [8]byte
				if _, err := io.ReadFull(chunked, buf[:]); err != nil {
					return 0, 0, err
				}
				return 0, binary.LittleEndian.Uint64(buf[:]), nil
			}

			f, err := OpenOrderedFile[uint64](path, orderedItemSize, chunkedRead, orderedWriteValue)
			require.NoError(err)
			defer func() { require.NoError(f.Close()) }()

			for i, v := range values {
				require.NoError(f.Set(uint64(i), v))
			}

			seq, err := f.Iterate()
			require.NoError(err)

			all, err := iter_utils.CollectOk2(seq)
			require.NoError(err)
			require.Len(all, len(values))
			for i, v := range values {
				require.EqualValues(v, all[uint64(i)], "position %d", i)
			}
		})
	}
}

func TestOrderedFile_Close_ReleasesFileHandle(t *testing.T) {
	require := require.New(t)
	f, path := openTestOrderedFile(t)

	require.NoError(f.Set(0, 7))
	require.NoError(f.Close())

	// After Close, we should still be able to read the file directly and see
	// what was written.
	data, err := os.ReadFile(path)
	require.NoError(err)
	require.EqualValues(7, binary.LittleEndian.Uint64(data))
}

func TestOrderedFile_Close_IsIdempotent(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	require.NoError(f.Close())
	// A second Close must not attempt to close the underlying file again.
	require.NoError(f.Close())
}

func TestOrderedFile_GetMemoryFootprint_IsNonZero(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	mf := f.GetMemoryFootprint()
	require.NotNil(mf)
	require.Greater(mf.Total(), uintptr(0))
}

// TestOrderedFile_OperationsFailAfterClose pins down that every operation
// touching the file reports os.ErrClosed once the file has been closed; there
// is no explicit close-state tracking, the guarantee comes from os.File.
func TestOrderedFile_OperationsFailAfterClose(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	require.NoError(f.Set(0, 1))
	require.NoError(f.Close())

	require.ErrorIs(f.Set(0, 1), os.ErrClosed)
	require.ErrorIs(f.SetBatch(map[uint64]uint64{0: 1}), os.ErrClosed)
	_, err := f.Get(0)
	require.ErrorIs(err, os.ErrClosed)
	_, err = f.Has(0)
	require.ErrorIs(err, os.ErrClosed)
	_, err = f.FileSize()
	require.ErrorIs(err, os.ErrClosed)
	_, err = f.Iterate()
	require.ErrorIs(err, os.ErrClosed)
}

// TestOrderedFile_ConcurrentReadsAndWritesAreSafe stresses concurrent Set,
// Get, and Iterate calls. Records are overwritten in place, so without proper
// synchronisation a reader racing a writer on the same record could observe a
// torn value. Every written value consists of 8 identical bytes, so any
// observed value with mixed bytes proves a torn read.
// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

// The OrderedFile under test stores 8-byte uint64 values indexed by their
// position. The key argument to readValueFn / writeValueFn is ignored by the
// implementation (the position is derived from the seek offset), so the
// helpers simply pass 0.

func orderedReadValue(reader io.Reader) (uint64, uint64, error) {
	var buf [8]byte
	if _, err := io.ReadFull(reader, buf[:]); err != nil {
		return 0, 0, err
	}
	return 0, binary.LittleEndian.Uint64(buf[:]), nil
}

func orderedWriteValue(writer io.Writer, _ uint64, value uint64) error {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], value)
	_, err := writer.Write(buf[:])
	return err
}

func openTestOrderedFile(t *testing.T) (*OrderedFile[uint64], string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "ordered.dat")
	f, err := OpenOrderedFile[uint64](path, orderedItemSize, orderedReadValue, orderedWriteValue)
	require.NoError(t, err)
	t.Cleanup(func() {
		// Best-effort close; some tests close explicitly.
		_ = f.Close()
	})
	return f, path
}
