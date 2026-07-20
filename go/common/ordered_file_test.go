// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package common

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// The OrderedFile under test stores 8-byte uint64 values indexed by their
// position. The key argument to readValueFn / writeValueFn is ignored by the
// implementation (the position is derived from the seek offset), so the
// helpers simply pass 0.

const orderedItemSize uint64 = 8

func orderedReadValue(reader io.ReadSeeker) (uint64, uint64, error) {
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

func TestOrderedFile_Open_CreatesFileWhenMissing(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "new.dat")

	_, err := os.Stat(path)
	require.True(os.IsNotExist(err))

	f, err := OpenOrderedFile[uint64](path, orderedItemSize, orderedReadValue, orderedWriteValue)
	require.NoError(err)
	defer f.Close()

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
	defer f.Close()

	size, err := f.Size()
	require.NoError(err)
	require.EqualValues(2, size)
}

func TestOrderedFile_Open_ReturnsErrorIfDirectoryDoesNotExist(t *testing.T) {
	require := require.New(t)
	// A path under a non-existent directory cannot be created.
	path := filepath.Join(t.TempDir(), "missing-dir", "file.dat")
	_, err := OpenOrderedFile[uint64](path, orderedItemSize, orderedReadValue, orderedWriteValue)
	require.Error(err)
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

func TestOrderedFile_Get_ReadsPreviouslyWrittenValue(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	require.NoError(f.Set(3, 12345))

	got, err := f.Get(3)
	require.NoError(err)
	require.NotNil(got)
	require.EqualValues(12345, *got)
}

func TestOrderedFile_Get_ReturnsErrorForKeyBeyondFile(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	// Nothing was written yet, so reading any key must fail with EOF.
	_, err := f.Get(0)
	require.Error(err)
	require.ErrorIs(err, io.EOF)
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

func TestOrderedFile_GetAll_ReturnsAllStoredValuesIndexedByPosition(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	require.NoError(f.Set(0, 100))
	require.NoError(f.Set(1, 101))
	require.NoError(f.Set(2, 102))

	all, err := f.GetAll()
	require.NoError(err)
	require.Equal(map[uint64]uint64{0: 100, 1: 101, 2: 102}, all)
}

func TestOrderedFile_GetAll_ReturnsEmptyMapForEmptyFile(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	all, err := f.GetAll()
	require.NoError(err)
	require.Empty(all)
}

func TestOrderedFile_Size_ReflectsFileSizeDividedByItemSize(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	size, err := f.Size()
	require.NoError(err)
	require.EqualValues(0, size)

	require.NoError(f.Set(0, 1))
	size, err = f.Size()
	require.NoError(err)
	require.EqualValues(1, size)

	require.NoError(f.Set(1, 2))
	size, err = f.Size()
	require.NoError(err)
	require.EqualValues(2, size)
}

func TestOrderedFile_Size_TruncatesPartiallyWrittenTrailingEntry(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "ordered.dat")

	// One full item plus 3 spurious bytes → Size must report 1.
	buf := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
	require.NoError(os.WriteFile(path, buf, 0600))

	f, err := OpenOrderedFile[uint64](path, orderedItemSize, orderedReadValue, orderedWriteValue)
	require.NoError(err)
	defer f.Close()

	size, err := f.Size()
	require.NoError(err)
	require.EqualValues(1, size)
}

func TestOrderedFile_Flush_IsNoop(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	require.NoError(f.Flush())
	require.NoError(f.Flush())
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

func TestOrderedFile_GetMemoryFootprint_IsNonZero(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOrderedFile(t)

	mf := f.GetMemoryFootprint()
	require.NotNil(mf)
	require.Greater(mf.Total(), uintptr(0))
}

func TestOrderedFile_Set_PropagatesWriteError(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "err.dat")

	injected := errors.New("injected write failure")
	writeFn := func(w io.Writer, _ uint64, _ uint64) error { return injected }

	f, err := OpenOrderedFile[uint64](path, orderedItemSize, orderedReadValue, writeFn)
	require.NoError(err)
	defer f.Close()

	err = f.Set(0, 1)
	require.ErrorIs(err, injected)
}

func TestOrderedFile_Get_PropagatesReadError(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "err.dat")

	injected := errors.New("injected read failure")
	readFn := func(r io.ReadSeeker) (uint64, uint64, error) { return 0, 0, injected }

	// Write enough bytes so that Seek succeeds.
	require.NoError(os.WriteFile(path, []byte("12345678"), 0600))

	f, err := OpenOrderedFile[uint64](path, orderedItemSize, readFn, orderedWriteValue)
	require.NoError(err)
	defer f.Close()

	_, err = f.Get(0)
	require.ErrorIs(err, injected)
}

func TestOrderedFile_GetAll_PropagatesReadError(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "err.dat")

	require.NoError(os.WriteFile(path, []byte("12345678"), 0600))

	injected := errors.New("injected read failure")
	readFn := func(r io.ReadSeeker) (uint64, uint64, error) { return 0, 0, injected }

	f, err := OpenOrderedFile[uint64](path, orderedItemSize, readFn, orderedWriteValue)
	require.NoError(err)
	defer f.Close()

	_, err = f.GetAll()
	require.ErrorIs(err, injected)
}

// TestOrderedFile_GetAll_HandlesChunkedReader verifies that values can be
// loaded correctly even when the readValueFn reads from a source that
// returns data in small chunks. This ports the concept of the previous
// TestArchiveTrie_CanLoadRootsFromJunkySource test to the OrderedFile
// abstraction: the file wrapper hands the underlying reader to
// readValueFn, and a well-behaved readValueFn (using io.ReadFull) must
// still assemble each record even in the presence of short reads.
func TestOrderedFile_GetAll_HandlesChunkedReader(t *testing.T) {
	require := require.New(t)

	values := []uint64{100, 200, 300, 400}

	for _, chunkSize := range []int{1, 2, 4, 7, 16, 1024} {
		t.Run(fmt.Sprintf("chunk=%d", chunkSize), func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "ordered.dat")

			// A readValueFn that wraps the underlying reader in a
			// chunk-splitting decorator to simulate short reads. It uses
			// io.ReadFull to demonstrate the pattern that value decoders
			// should follow.
			chunkedRead := func(r io.ReadSeeker) (uint64, uint64, error) {
				var buf [8]byte
				chunked := &chunkedReader{src: r, chunkSize: chunkSize}
				if _, err := io.ReadFull(chunked, buf[:]); err != nil {
					return 0, 0, err
				}
				return 0, binary.LittleEndian.Uint64(buf[:]), nil
			}

			f, err := OpenOrderedFile[uint64](path, orderedItemSize, chunkedRead, orderedWriteValue)
			require.NoError(err)
			defer f.Close()

			for i, v := range values {
				require.NoError(f.Set(uint64(i), v))
			}

			all, err := f.GetAll()
			require.NoError(err)
			require.Len(all, len(values))
			for i, v := range values {
				require.EqualValues(v, all[uint64(i)], "position %d", i)
			}
		})
	}
}

// chunkedReader is an io.Reader that returns data from the underlying reader
// in fixed-size chunks. It is used to verify that value decoders that use
// io.ReadFull correctly reassemble records from short reads.
type chunkedReader struct {
	src       io.Reader
	chunkSize int
}

func (c *chunkedReader) Read(p []byte) (int, error) {
	if c.chunkSize > 0 && len(p) > c.chunkSize {
		p = p[:c.chunkSize]
	}
	return c.src.Read(p)
}
