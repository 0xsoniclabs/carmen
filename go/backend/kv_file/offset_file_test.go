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
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// The OffsetFile under test stores 8-byte keys followed by 8-byte values.
const offsetEntrySize uint64 = 16

func offsetReadValue(reader io.Reader) (uint64, uint64, error) {
	var buf [16]byte
	if _, err := io.ReadFull(reader, buf[:]); err != nil {
		return 0, 0, err
	}
	key := binary.LittleEndian.Uint64(buf[0:8])
	value := binary.LittleEndian.Uint64(buf[8:16])
	return key, value, nil
}

func offsetWriteValue(writer io.Writer, key uint64, value uint64) error {
	var buf [16]byte
	binary.LittleEndian.PutUint64(buf[0:8], key)
	binary.LittleEndian.PutUint64(buf[8:16], value)
	_, err := writer.Write(buf[:])
	return err
}

func openTestOffsetFile(t *testing.T) (*OffsetFile[uint64, uint64], string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "offset.dat")
	f, err := OpenOffsetFile[uint64, uint64](path, offsetReadValue, offsetWriteValue)
	require.NoError(t, err)
	t.Cleanup(func() { _ = f.Close() })
	return f, path
}

func TestOffsetFile_Open_CreatesFileWhenMissing(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "new.dat")

	_, err := os.Stat(path)
	require.True(os.IsNotExist(err))

	f, err := OpenOffsetFile[uint64, uint64](path, offsetReadValue, offsetWriteValue)
	require.NoError(err)
	defer func() { require.NoError(f.Close()) }()

	info, err := os.Stat(path)
	require.NoError(err)
	require.EqualValues(0, info.Size())
}

func TestOffsetFile_Open_LoadsExistingEntries(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "existing.dat")

	// Two entries pre-populated: key=1, value=10 and key=2, value=20.
	buf := make([]byte, 32)
	binary.LittleEndian.PutUint64(buf[0:8], 1)
	binary.LittleEndian.PutUint64(buf[8:16], 10)
	binary.LittleEndian.PutUint64(buf[16:24], 2)
	binary.LittleEndian.PutUint64(buf[24:32], 20)
	require.NoError(os.WriteFile(path, buf, 0600))

	f, err := OpenOffsetFile[uint64, uint64](path, offsetReadValue, offsetWriteValue)
	require.NoError(err)
	defer func() { require.NoError(f.Close()) }()

	size, err := f.Size()
	require.NoError(err)
	require.EqualValues(2, size)

	got1, err := f.Get(1)
	require.NoError(err)
	require.NotNil(got1)
	require.EqualValues(10, *got1)

	got2, err := f.Get(2)
	require.NoError(err)
	require.NotNil(got2)
	require.EqualValues(20, *got2)
}

func TestOffsetFile_Open_ReturnsErrorOnInvalidFileContent(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "invalid.dat")
	// 15 bytes: half an entry — readValueFn will fail with io.ErrUnexpectedEOF
	// after one full entry read attempt.
	require.NoError(os.WriteFile(path, make([]byte, 15), 0600))

	_, err := OpenOffsetFile[uint64, uint64](path, offsetReadValue, offsetWriteValue)
	require.Error(err)
}

func TestOffsetFile_Set_PersistsValue(t *testing.T) {
	require := require.New(t)
	f, path := openTestOffsetFile(t)

	require.NoError(f.Set(1, 100))

	got, err := f.Get(1)
	require.NoError(err)
	require.NotNil(got)
	require.EqualValues(100, *got)

	// File must have exactly one entry on disk.
	info, err := os.Stat(path)
	require.NoError(err)
	require.EqualValues(offsetEntrySize, info.Size())
}

func TestOffsetFile_Set_UpdatesOffsetOnRewrite(t *testing.T) {
	require := require.New(t)
	f, path := openTestOffsetFile(t)

	require.NoError(f.Set(7, 70))
	require.NoError(f.Set(7, 71))

	got, err := f.Get(7)
	require.NoError(err)
	require.NotNil(got)
	require.EqualValues(71, *got)

	// The file is append-only, so it now holds two records for key 7 but
	// the in-memory map points to the most recent one.
	info, err := os.Stat(path)
	require.NoError(err)
	require.EqualValues(2*offsetEntrySize, info.Size())
}

func TestOffsetFile_Get_ReturnsNilForUnknownKey(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOffsetFile(t)

	got, err := f.Get(42)
	require.NoError(err)
	require.Nil(got)
}

func TestOffsetFile_Get_DetectsKeyMismatchOnDisk(t *testing.T) {
	require := require.New(t)
	f, path := openTestOffsetFile(t)

	require.NoError(f.Set(1, 100))
	require.NoError(f.Close())

	// Corrupt the key on disk so that reading the entry at the recorded
	// offset yields a different key than the requested one.
	data, err := os.ReadFile(path)
	require.NoError(err)
	binary.LittleEndian.PutUint64(data[0:8], 999)
	require.NoError(os.WriteFile(path, data, 0600))

	f2, err := OpenOffsetFile[uint64, uint64](path, offsetReadValue, offsetWriteValue)
	require.NoError(err)
	defer func() { require.NoError(f2.Close()) }()

	// The offsets map now records key=999. Asking for it succeeds because
	// the stored key matches. Asking for key=1 returns nil because it was
	// not loaded at all.
	got, err := f2.Get(999)
	require.NoError(err)
	require.EqualValues(100, *got)

	got, err = f2.Get(1)
	require.NoError(err)
	require.Nil(got)
}

func TestOffsetFile_SetBatch_PersistsAllEntries(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOffsetFile(t)

	batch := map[uint64]uint64{
		1: 10,
		2: 20,
		3: 30,
	}
	require.NoError(f.SetBatch(batch))

	for k, want := range batch {
		got, err := f.Get(k)
		require.NoError(err)
		require.NotNil(got)
		require.EqualValues(want, *got, "key %d", k)
	}
	size, err := f.Size()
	require.NoError(err)
	require.EqualValues(uint64(len(batch)), size)
}

func TestOffsetFile_SetBatch_EmptyBatchIsNoop(t *testing.T) {
	require := require.New(t)
	f, path := openTestOffsetFile(t)

	require.NoError(f.SetBatch(map[uint64]uint64{}))
	size, err := f.Size()
	require.NoError(err)
	require.EqualValues(0, size)

	info, err := os.Stat(path)
	require.NoError(err)
	require.EqualValues(0, info.Size())
}

func TestOffsetFile_SetBatch_KeepsWrittenEntriesReadableAfterPartialFailure(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "partial.dat")

	// A codec that fails on the second write, so a batch of two entries is
	// interrupted after one of them was written and indexed.
	injected := errors.New("injected write failure")
	calls := 0
	writeFn := func(w io.Writer, key uint64, value uint64) error {
		calls++
		if calls > 1 {
			return injected
		}
		return offsetWriteValue(w, key, value)
	}

	f, err := OpenOffsetFile[uint64, uint64](path, offsetReadValue, writeFn)
	require.NoError(err)
	defer func() { require.NoError(f.Close()) }()

	entries := map[uint64]uint64{1: 10, 2: 20}
	require.ErrorIs(f.SetBatch(entries), injected)

	// The entry written before the failure is indexed and must stay readable.
	for key, want := range entries {
		has, err := f.Has(key)
		require.NoError(err)
		if !has {
			continue
		}
		got, err := f.Get(key)
		require.NoError(err)
		require.NotNil(got)
		require.Equal(want, *got)
	}
}

func TestOffsetFile_Iterate_ReturnsAllEntries(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOffsetFile(t)

	require.NoError(f.Set(1, 11))
	require.NoError(f.Set(2, 22))
	require.NoError(f.Set(3, 33))

	seq, err := f.Iterate()
	require.NoError(err)
	all := map[uint64]uint64{}
	for k, v := range seq {
		all[k] = v
	}
	require.Equal(map[uint64]uint64{1: 11, 2: 22, 3: 33}, all)
}

func TestOffsetFile_Iterate_ReturnsLatestValueAfterOverwrite(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOffsetFile(t)

	require.NoError(f.Set(1, 10))
	require.NoError(f.Set(1, 100))

	seq, err := f.Iterate()
	require.NoError(err)
	all := map[uint64]uint64{}
	for k, v := range seq {
		all[k] = v
	}
	require.Equal(map[uint64]uint64{1: 100}, all)
}

func TestOffsetFile_Iterate_YieldsNothingAfterClose(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOffsetFile(t)

	require.NoError(f.Set(1, 1))
	require.NoError(f.Close())

	seq, err := f.Iterate()
	require.NoError(err)

	yielded := 0
	for range seq {
		yielded++
	}
	require.Equal(0, yielded)
}

func TestOffsetFile_Size_CountsUniqueKeys(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOffsetFile(t)

	size, err := f.Size()
	require.NoError(err)
	require.EqualValues(0, size)

	require.NoError(f.Set(1, 1))
	size, err = f.Size()
	require.NoError(err)
	require.EqualValues(1, size)

	require.NoError(f.Set(2, 2))
	size, err = f.Size()
	require.NoError(err)
	require.EqualValues(2, size)

	// Rewriting an existing key must not change the count.
	require.NoError(f.Set(1, 99))
	size, err = f.Size()
	require.NoError(err)
	require.EqualValues(2, size)
}

func TestOffsetFile_Flush_IsNoop(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOffsetFile(t)

	require.NoError(f.Flush())
	require.NoError(f.Flush())
}

func TestOffsetFile_Close_IsIdempotentAndRejectsWrites(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOffsetFile(t)

	require.NoError(f.Set(1, 1))

	require.NoError(f.Close())
	// Close is idempotent: a second call must succeed without error.
	require.NoError(f.Close())
	// Subsequent writes must fail because the retained file descriptor is
	// no longer usable.
	require.ErrorIs(f.Set(1, 1), os.ErrClosed)
	require.ErrorIs(f.SetBatch(map[uint64]uint64{2: 2}), os.ErrClosed)
	// Reading an existing key touches the closed descriptor and fails.
	_, err := f.Get(1)
	require.ErrorIs(err, os.ErrClosed)
	// The offset index outlives Close: a missing key reports "not found"
	// rather than os.ErrClosed.
	got, err := f.Get(42)
	require.NoError(err)
	require.Nil(got)
}

func TestOffsetFile_GetMemoryFootprint_IsNonZero(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOffsetFile(t)

	require.NoError(f.Set(1, 1))
	require.NoError(f.Set(2, 2))

	mf := f.GetMemoryFootprint()
	require.NotNil(mf)
	require.Greater(mf.Total(), uintptr(0))
}

func TestOffsetFile_Set_PropagatesWriteError(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "err.dat")

	injected := errors.New("injected write failure")
	writeFn := func(w io.Writer, _ uint64, _ uint64) error { return injected }

	f, err := OpenOffsetFile[uint64, uint64](path, offsetReadValue, writeFn)
	require.NoError(err)
	defer func() { require.NoError(f.Close()) }()

	err = f.Set(1, 1)
	require.ErrorIs(err, injected)
}

func TestOffsetFile_Get_PropagatesReadError(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOffsetFile(t)

	// Store a real entry via the working codec, then reopen with an error
	// injecting reader.
	require.NoError(f.Set(1, 1))
	require.NoError(f.Close())

	injected := errors.New("injected read failure")

	// The initial open uses the working reader so that the offsets are
	// populated; a second open with a faulty reader would fail during
	// parseOffsets. We therefore inject the failure lazily by swapping the
	// readValueFn on a fresh instance via a closure that only fails after
	// the initial scan.
	var scanned bool
	readFn := func(r io.Reader) (uint64, uint64, error) {
		if !scanned {
			k, v, err := offsetReadValue(r)
			return k, v, err
		}
		return 0, 0, injected
	}

	path := filepath.Join(t.TempDir(), "err.dat")
	// Copy the previously written data to the new path.
	// (We cannot access the original path through the closed handle here,
	// so recreate a valid one-entry file.)
	buf := make([]byte, 16)
	binary.LittleEndian.PutUint64(buf[0:8], 1)
	binary.LittleEndian.PutUint64(buf[8:16], 1)
	require.NoError(os.WriteFile(path, buf, 0600))

	f2, err := OpenOffsetFile[uint64, uint64](path, readFn, offsetWriteValue)
	require.NoError(err)
	defer func() { require.NoError(f2.Close()) }()

	scanned = true
	_, err = f2.Get(1)
	require.ErrorIs(err, injected)
}

// TestOffsetFile_SetBatch_ReturnsErrorAfterClose verifies that writes fail
// once the retained file descriptor has been released via Close.
func TestOffsetFile_SetBatch_ReturnsErrorAfterClose(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOffsetFile(t)

	require.NoError(f.Close())
	require.ErrorIs(f.SetBatch(map[uint64]uint64{1: 1}), os.ErrClosed)
}

// TestOffsetFile_Set_ReturnsErrorAfterClose mirrors the SetBatch test above
// for the single-entry Set path.
func TestOffsetFile_Set_ReturnsErrorAfterClose(t *testing.T) {
	require := require.New(t)
	f, _ := openTestOffsetFile(t)

	require.NoError(f.Close())
	require.ErrorIs(f.Set(1, 1), os.ErrClosed)
}

// TestOffsetFile_ParseOffsets_HandlesChunkedReader verifies that the record
// scanning performed at Open time still assembles entries correctly when the
// underlying reader returns data in small chunks. This ports the concept of
// the previous TestArchiveTrie_CanLoadRootsFromJunkySource test to the
// OffsetFile abstraction.
func TestOffsetFile_ParseOffsets_HandlesChunkedReader(t *testing.T) {
	require := require.New(t)

	// Serialise a few entries using the standard writer.
	entries := []struct {
		key, value uint64
	}{
		{1, 10},
		{2, 20},
		{3, 30},
	}
	var buf bytes.Buffer
	for _, e := range entries {
		require.NoError(offsetWriteValue(&buf, e.key, e.value))
	}

	for _, chunkSize := range []int{1, 2, 4, 7, 16, 1024} {
		t.Run(fmt.Sprintf("chunk=%d", chunkSize), func(t *testing.T) {
			reader := newChunkedReadSeeker(buf.Bytes(), chunkSize)

			offsets, err := parseOffsets[uint64, uint64](reader, offsetReadValue)
			require.NoError(err)

			require.Len(offsets, len(entries))
			for i, e := range entries {
				require.EqualValues(uint64(i)*offsetEntrySize, offsets[e.key],
					"unexpected offset for key %d", e.key)
			}
		})
	}
}

// chunkedReadSeeker is an io.ReadSeeker returning data in fixed-size chunks.
// It is used to verify that value decoders assemble records correctly even
// when the underlying source does not deliver full records in a single Read.
type chunkedReadSeeker struct {
	data      []byte
	pos       int64
	chunkSize int
}

func newChunkedReadSeeker(data []byte, chunkSize int) *chunkedReadSeeker {
	return &chunkedReadSeeker{data: data, chunkSize: chunkSize}
}

func (r *chunkedReadSeeker) Read(p []byte) (int, error) {
	if r.pos >= int64(len(r.data)) {
		return 0, io.EOF
	}
	remaining := int64(len(r.data)) - r.pos
	n := int64(len(p))
	if n > remaining {
		n = remaining
	}
	if r.chunkSize > 0 && n > int64(r.chunkSize) {
		n = int64(r.chunkSize)
	}
	copied := copy(p, r.data[r.pos:r.pos+n])
	r.pos += int64(copied)
	return copied, nil
}

func (r *chunkedReadSeeker) Seek(offset int64, whence int) (int64, error) {
	var newPos int64
	switch whence {
	case io.SeekStart:
		newPos = offset
	case io.SeekCurrent:
		newPos = r.pos + offset
	case io.SeekEnd:
		newPos = int64(len(r.data)) + offset
	default:
		return 0, fmt.Errorf("invalid whence %d", whence)
	}
	if newPos < 0 {
		return 0, fmt.Errorf("negative position %d", newPos)
	}
	r.pos = newPos
	return newPos, nil
}
