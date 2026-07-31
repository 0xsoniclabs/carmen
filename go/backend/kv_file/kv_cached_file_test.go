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
	"errors"
	"fmt"
	"iter"
	"maps"
	"sync"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
)

const (
	cacheSize            = 4
	flushBufferThreshold = 3

	key1   = 1
	value1 = "value1"
)

type K = int
type V = string

func TestKVCachedFile_Open_ReturnsErrorOnInvalidArguments(t *testing.T) {
	mock := NewMockKVFileWithMemoryFootprint[K, V](gomock.NewController(t))
	tests := map[string]struct {
		file                 KVFileWithMemoryFootprint[K, V]
		cacheSize            int
		flushBufferThreshold int
	}{
		"nil file":            {file: nil, cacheSize: cacheSize, flushBufferThreshold: flushBufferThreshold},
		"zero cache size":     {file: mock, cacheSize: 0, flushBufferThreshold: flushBufferThreshold},
		"negative cache size": {file: mock, cacheSize: -1, flushBufferThreshold: flushBufferThreshold},
		"zero threshold":      {file: mock, cacheSize: cacheSize, flushBufferThreshold: 0},
		"negative threshold":  {file: mock, cacheSize: cacheSize, flushBufferThreshold: -1},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			_, err := OpenKVCachedFile[K, V](test.file, test.cacheSize, test.flushBufferThreshold)
			require.Error(err)
		})
	}
}

func TestKVCachedFile_Open_SetsMaxPendingFlushesToThresholdPlusOne(t *testing.T) {
	tests := map[string]int{
		"threshold 1":   1,
		"threshold 3":   3,
		"threshold 100": 100,
	}
	for name, threshold := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			ctrl := gomock.NewController(t)
			mock := NewMockKVFileWithMemoryFootprint[K, V](ctrl)
			c, err := OpenKVCachedFile[K, V](mock, cacheSize, threshold)
			require.NoError(err)
			t.Cleanup(c.shutdownWriter)
			require.Equal(threshold+1, c.maxPendingFlushes)
		})
	}
}

func TestKVCachedFile_Get_ReturnsValueCorrectly(t *testing.T) {
	tests := map[string]struct {
		setup func(t *testing.T, c *KVCachedFile[K, V], mock *MockKVFileWithMemoryFootprint[K, V]) (want V)
	}{
		"returns from cache": {
			setup: func(t *testing.T, c *KVCachedFile[K, V], _ *MockKVFileWithMemoryFootprint[K, V]) V {
				want := "cache-value"
				require.NoError(t, c.Set(key1, want))
				return want
			},
		},
		"returns from flush buffer": {
			setup: func(t *testing.T, c *KVCachedFile[K, V], _ *MockKVFileWithMemoryFootprint[K, V]) V {
				want := "buffer-value"
				c.flushBuffer[key1] = want
				return want
			},
		},
		"returns from file": {
			setup: func(t *testing.T, _ *KVCachedFile[K, V], mock *MockKVFileWithMemoryFootprint[K, V]) V {
				want := "file-value"
				mock.EXPECT().Get(key1).Return(&want, nil)
				return want
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			c, mock := openTestKVCachedFile(t)
			want := test.setup(t, c, mock)

			got, err := c.Get(key1)
			require.NoError(err)
			require.NotNil(got)
			require.Equal(want, *got)
		})
	}
}

func TestKVCachedFile_Get_ReturnsNilOnUnknownKey(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	mock.EXPECT().Get(999).Return(nil, nil)

	got, err := c.Get(999)
	require.NoError(err)
	require.Nil(got)
	_, inCache := c.cache.Get(999)
	require.False(inCache)
}

func TestKVCachedFile_Get_ReturnsErrorOnFileReadError(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	injected := errors.New("read failed")
	mock.EXPECT().Get(key1).Return(nil, injected)

	got, err := c.Get(key1)
	require.ErrorIs(err, injected)
	require.Nil(got)
}

func TestKVCachedFile_Get_PromotesValuesInFlushBufferIntoCacheAsDirty(t *testing.T) {
	require := require.New(t)
	// No file interactions expected: the buffer never reaches the flush
	// threshold before the value is promoted back into the cache.
	c, _ := openTestKVCachedFile(t)

	for i := 0; i < cacheSize+1; i++ {
		require.NoError(c.Set(i, fmt.Sprintf("value%d", i)))
	}
	_, inBuffer := c.flushBuffer[0]
	require.True(inBuffer)

	got, err := c.Get(0)
	require.NoError(err)
	require.NotNil(got)
	require.Equal("value0", *got)

	_, inBuffer = c.flushBuffer[0]
	require.False(inBuffer)
	cached, found := c.cache.Get(0)
	require.True(found)
	require.Equal("value0", cached)
	// The value was removed from the buffer without being written, so it must
	// stay dirty: a clean promotion would drop it on eviction and lose the write.
	_, isDirty := c.dirty[0]
	require.True(isDirty, "value promoted from the flush buffer must be cached as dirty")
}

func TestKVCachedFile_Get_PromotesInFlightValueIntoCacheAsClean(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	// Block the background writer inside SetBatch so the sealed buffer stays
	// in flight while we read one of its keys.
	writing, unblockWriter := blockWriter(t, mock, nil)

	// Fill the cache and seal a buffer; key 0 is evicted into the in-flight buffer.
	for i := 0; i < cacheSize+flushBufferThreshold; i++ {
		require.NoError(c.Set(i, fmt.Sprintf("value%d", i)))
	}
	<-writing // the writer is blocked with key 0 still in a pending buffer

	// Reading key 0 serves it from the pending buffer and promotes it into the
	// cache as a clean entry.
	got, err := c.Get(0)
	require.NoError(err)
	require.NotNil(got)
	require.Equal("value0", *got)

	cached, found := c.cache.Get(0)
	require.True(found)
	require.Equal("value0", cached)
	_, isDirty := c.dirty[0]
	require.False(isDirty, "promoted in-flight value must be cached as clean")

	unblockWriter()
	require.NoError(c.Flush())
}

func TestKVCachedFile_Get_ReturnsNewestValueAcrossMultiplePendingBuffers(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	// Block the writer inside SetBatch so a sealed buffer stays in c.pending; the
	// writer keeps the buffer it is writing in the queue until the write returns,
	// so both sealed buffers coexist there while we read the key.
	writing, unblockWriter := blockWriter(t, mock, nil)

	// First buffer goes in flight; the second is queued behind it, so key1 now
	// lives in two pending buffers: {key1:"old"} then {key1:"new"}.
	sealBuffer(c, key1, "old")
	<-writing
	sealBuffer(c, key1, "new")

	got, err := c.Get(key1)
	require.NoError(err)
	require.NotNil(got)
	require.Equal("new", *got, "Get must return the newest pending value")

	unblockWriter()
	require.NoError(c.Flush())
}

func TestKVCachedFile_Has_ChecksCacheBufferAndFile(t *testing.T) {
	tests := map[string]struct {
		setup func(t *testing.T, c *KVCachedFile[K, V], mock *MockKVFileWithMemoryFootprint[K, V]) (want bool)
	}{
		"found in cache": {
			setup: func(t *testing.T, c *KVCachedFile[K, V], _ *MockKVFileWithMemoryFootprint[K, V]) bool {
				require.NoError(t, c.Set(key1, value1))
				return true
			},
		},
		"found in flush buffer": {
			setup: func(t *testing.T, c *KVCachedFile[K, V], _ *MockKVFileWithMemoryFootprint[K, V]) bool {
				c.flushBuffer[key1] = value1
				return true
			},
		},
		"found in file": {
			setup: func(t *testing.T, _ *KVCachedFile[K, V], mock *MockKVFileWithMemoryFootprint[K, V]) bool {
				mock.EXPECT().Has(key1).Return(true, nil)
				return true
			},
		},
		"not found": {
			setup: func(t *testing.T, _ *KVCachedFile[K, V], mock *MockKVFileWithMemoryFootprint[K, V]) bool {
				mock.EXPECT().Has(key1).Return(false, nil)
				return false
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			c, mock := openTestKVCachedFile(t)
			want := test.setup(t, c, mock)

			got, err := c.Has(key1)
			require.NoError(err)
			require.Equal(want, got)
		})
	}
}

func TestKVCachedFile_Has_FindsKeyInPendingBuffers(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	// Block the writer inside SetBatch so the sealed buffer holding key1 stays
	// in the pending queue while it is looked up.
	writing, unblockWriter := blockWriter(t, mock, nil)
	sealBuffer(c, key1, value1)
	<-writing

	has, err := c.Has(key1)
	require.NoError(err)
	require.True(has, "Has must find keys awaiting a background write")

	unblockWriter()
	require.NoError(c.Flush())
}

func TestKVCachedFile_Has_DoesNotLoadValueIntoCache(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	mock.EXPECT().Has(key1).Return(true, nil)

	has, err := c.Has(key1)
	require.NoError(err)
	require.True(has)

	_, inCache := c.cache.Get(key1)
	require.False(inCache, "Has must not populate the cache")
}

func TestKVCachedFile_Has_PropagatesFileError(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	injected := errors.New("has failed")
	mock.EXPECT().Has(key1).Return(false, injected)

	_, err := c.Has(key1)
	require.ErrorIs(err, injected)
}

func TestKVCachedFile_Get_PromotesValuesInFlushBufferIntoCache(t *testing.T) {
	require := require.New(t)
	// No file interactions expected: the buffer never reaches the flush
	// threshold before the value is promoted back into the cache.
	c, _ := openTestKVCachedFile(t)

	for i := 0; i < cacheSize+1; i++ {
		require.NoError(c.Set(i, fmt.Sprintf("value%d", i)))
	}
	_, inBuffer := c.flushBuffer[0]
	require.True(inBuffer)

	got, err := c.Get(0)
	require.NoError(err)
	require.NotNil(got)
	require.Equal("value0", *got)

	_, inBuffer = c.flushBuffer[0]
	require.False(inBuffer)
	cached, found := c.cache.Get(0)
	require.True(found)
	require.Equal("value0", cached)
}

func TestKVCachedFile_Set_UpdatesCache(t *testing.T) {
	require := require.New(t)
	c, _ := openTestKVCachedFile(t)

	require.NoError(c.Set(key1, value1))

	require.Equal(1, getCacheSize(c.cache))
	got, found := c.cache.Get(key1)
	require.True(found)
	require.Equal(value1, got)
}

func TestKVCachedFile_Set_ReDirtiesKeyAfterFlush(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	require.NoError(c.Set(1, "value-1"))

	mock.EXPECT().SetBatch(gomock.Any()).Return(nil).Times(1)
	mock.EXPECT().Flush().Return(nil).Times(1)
	require.NoError(c.Flush())

	// Key is now clean; a fresh Set marks it dirty again.
	require.NoError(c.Set(1, "value-1-new"))
	_, isDirty := c.dirty[1]
	require.True(isDirty)

	mock.EXPECT().SetBatch(gomock.Any()).Return(nil).Times(1)
	mock.EXPECT().Flush().Return(nil).Times(1)
	require.NoError(c.Flush())

	_, isDirty = c.dirty[1]
	require.False(isDirty)
}

func TestKVCachedFile_Set_ReSetWhileInFlightIsNotLost(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	// Seal a buffer for the background writer and block it inside SetBatch so
	// the buffer stays in flight while we re-Set one of its keys.
	written := map[K]V{}
	firstWrite, unblockWriter := blockWriter(t, mock, func(entries map[K]V) {
		maps.Copy(written, entries)
	})

	// Fill the cache and evict (threshold) entries to seal the first buffer.
	for i := 0; i < cacheSize+flushBufferThreshold; i++ {
		require.NoError(c.Set(i, fmt.Sprintf("value%d", i)))
	}
	<-firstWrite // the writer is now blocked inside SetBatch with key 0 in flight

	// Re-Set key 0 with a new value while its old value is being written.
	require.NoError(c.Set(0, "value0-new"))

	// Let the in-flight write finish, then flush everything.
	unblockWriter()
	require.NoError(c.Flush())

	// The re-Set value must have survived and reached disk.
	require.Equal("value0-new", written[0])

	got, err := c.Get(0)
	require.NoError(err)
	require.NotNil(got)
	require.Equal("value0-new", *got)
}

func TestKVCachedFile_Set_ReSetWhileInFlushBufferIsNotLost(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	written := map[K]V{}
	mock.EXPECT().SetBatch(gomock.Any()).DoAndReturn(func(entries map[K]V) error {
		maps.Copy(written, entries)
		return nil
	}).AnyTimes()
	mock.EXPECT().Flush().Return(nil).AnyTimes()

	// Fill the cache and evict key 0 into the (un-sealed) flush buffer.
	require.NoError(c.Set(0, "value0"))
	for i := 1; i <= cacheSize; i++ {
		require.NoError(c.Set(i, fmt.Sprintf("value%d", i)))
	}

	// Re-Set key 0 while its old value still sits in the flush buffer, then
	// trigger one more eviction so the buffer reaches the threshold and is
	// sealed for the background writer.
	require.NoError(c.Set(0, "value0-new"))
	require.NoError(c.Set(cacheSize+1, "trigger"))

	require.NoError(c.Flush())

	// The re-Set value must be the one persisted, not the stale buffered one.
	require.Equal("value0-new", written[0])
}

func TestKVCachedFile_Set_SurfacesAsyncWriteErrorOnNextOperation(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	// Fill enough entries so that the next Set pushes the flush buffer to the
	// threshold and triggers a background file write.
	for i := 0; i < cacheSize+flushBufferThreshold-1; i++ {
		require.NoError(c.Set(i, fmt.Sprintf("value%d", i)))
	}

	injected := errors.New("write failed")
	mock.EXPECT().SetBatch(gomock.Any()).Return(injected)

	// The write is asynchronous, so the triggering Set itself does not fail;
	// the error surfaces on the next operation (here a Flush barrier).
	require.NoError(c.Set(cacheSize+flushBufferThreshold-1, "trigger"))
	require.ErrorIs(c.Flush(), injected)

	// The error is sticky: subsequent operations keep reporting it.
	_, err := c.Get(0)
	require.ErrorIs(err, injected)
}

func TestKVCachedFile_SetBatch_StoresAllEntries(t *testing.T) {
	require := require.New(t)
	// No file interactions expected: the entries fit in the cache.
	c, _ := openTestKVCachedFile(t)

	entries := map[K]V{1: "value-1", 2: "value-2", 3: "value-3"}
	require.NoError(c.SetBatch(entries))

	for key, want := range entries {
		got, err := c.Get(key)
		require.NoError(err)
		require.NotNil(got)
		require.Equal(want, *got)
	}
}

func TestKVCachedFile_Flush_WritesCacheAndBufferToFile(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	require.NoError(c.Set(1, "cache-1"))
	require.NoError(c.Set(2, "cache-2"))
	c.flushBuffer[3] = "buffer-3"
	c.flushBuffer[4] = "buffer-4"

	written := map[K]V{}
	mock.EXPECT().SetBatch(gomock.Any()).DoAndReturn(func(entries map[K]V) error {
		maps.Copy(written, entries)
		return nil
	}).Times(1)
	mock.EXPECT().Flush().Return(nil).Times(1)

	require.NoError(c.Flush())

	require.Equal(map[K]V{1: "cache-1", 2: "cache-2", 3: "buffer-3", 4: "buffer-4"}, written)
}

func TestKVCachedFile_Flush_ClearsFlushBuffer(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	require.NoError(c.Set(key1, value1))
	c.flushBuffer[2] = "buffer-2"

	mock.EXPECT().SetBatch(gomock.Any()).Return(nil)
	mock.EXPECT().Flush().Return(nil)

	require.NoError(c.Flush())
	require.Equal(0, len(c.flushBuffer))
}

func TestKVCachedFile_Flush_ClearsDirtyForFlushedKeys(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	require.NoError(c.Set(1, "value-1"))
	require.NoError(c.Set(2, "value-2"))
	require.Len(c.dirty, 2)

	mock.EXPECT().SetBatch(gomock.Any()).Return(nil)
	mock.EXPECT().Flush().Return(nil)

	require.NoError(c.Flush())
	require.Empty(c.dirty)
}

func TestKVCachedFile_Flush_ReturnsErrorOnFileError(t *testing.T) {
	injected := errors.New("injected")
	tests := map[string]func(mock *MockKVFileWithMemoryFootprint[K, V]){
		"batch write fails": func(mock *MockKVFileWithMemoryFootprint[K, V]) {
			mock.EXPECT().SetBatch(gomock.Any()).Return(injected)
		},
		"file flush fails": func(mock *MockKVFileWithMemoryFootprint[K, V]) {
			mock.EXPECT().SetBatch(gomock.Any()).Return(nil)
			mock.EXPECT().Flush().Return(injected)
		},
	}

	for name, setup := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			c, mock := openTestKVCachedFile(t)
			setup(mock)

			c.flushBuffer[1] = "value"
			require.ErrorIs(c.Flush(), injected)
		})
	}
}

func TestKVCachedFile_Flush_IsNoopWhenNothingPending(t *testing.T) {
	require := require.New(t)
	// No file interactions expected when nothing is pending.
	c, _ := openTestKVCachedFile(t)

	require.NoError(c.Flush())
}

func TestKVCachedFile_Flush_DoesNotRewriteCleanEntries(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	// First flush writes both entries; second flush without any intervening
	// Set must be a no-op because the cache entries are now clean.
	require.NoError(c.Set(1, "value-1"))
	require.NoError(c.Set(2, "value-2"))

	mock.EXPECT().SetBatch(gomock.Any()).DoAndReturn(func(entries map[K]V) error {
		require.Len(entries, 2)
		return nil
	}).Times(1)
	mock.EXPECT().Flush().Return(nil).Times(1)

	require.NoError(c.Flush())
	require.NoError(c.Flush())
}

func TestKVCachedFile_Flush_OnlyWritesDirtyEntriesAfterPartialUpdate(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	require.NoError(c.Set(1, "value-1"))
	require.NoError(c.Set(2, "value-2"))

	// First flush: both entries written.
	mock.EXPECT().SetBatch(gomock.Any()).DoAndReturn(func(entries map[K]V) error {
		require.Len(entries, 2)
		return nil
	}).Times(1)
	mock.EXPECT().Flush().Return(nil).Times(1)
	require.NoError(c.Flush())

	// Only key 1 is re-set — key 2 remains clean.
	require.NoError(c.Set(1, "value-1-updated"))

	written := map[K]V{}
	mock.EXPECT().SetBatch(gomock.Any()).DoAndReturn(func(entries map[K]V) error {
		maps.Copy(written, entries)
		return nil
	}).Times(1)
	mock.EXPECT().Flush().Return(nil).Times(1)

	require.NoError(c.Flush())
	require.Equal(map[K]V{1: "value-1-updated"}, written)
}

func TestKVCachedFile_FileSize_ReturnsCorrectSize(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	mock.EXPECT().FileSize().Return(uint64(1234), nil)

	size, err := c.FileSize()
	require.NoError(err)
	require.Equal(uint64(1234), size)
}

func TestKVCachedFile_FileSize_ReturnsErrorOnFileReadError(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	injected := errors.New("file size read failed")
	mock.EXPECT().FileSize().Return(uint64(0), injected)

	_, err := c.FileSize()
	require.ErrorIs(err, injected)
}

func TestKVCachedFile_Iterate_YieldsFileContents(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	// No pending writes: draining is a no-op, Iterate delegates directly
	// to the underlying file.
	fileContents := map[K]V{0: "value0", 1: "value1", 2: "value2"}
	mock.EXPECT().Iterate().Return(mockIterateSeq(fileContents), nil)

	seq, err := c.Iterate()
	require.NoError(err)

	got := map[K]V{}
	for k, v := range seq {
		got[k] = v
	}
	require.Equal(fileContents, got)
}

func TestKVCachedFile_Iterate_FlushesDirtyEntriesBeforeIterating(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	require.NoError(c.Set(1, "value-1"))
	require.NoError(c.Set(2, "value-2"))
	c.flushBuffer[3] = "value-3"

	written := map[K]V{}
	gomock.InOrder(
		mock.EXPECT().SetBatch(gomock.Any()).DoAndReturn(func(entries map[K]V) error {
			maps.Copy(written, entries)
			return nil
		}),
		mock.EXPECT().Flush().Return(nil),
		mock.EXPECT().Iterate().DoAndReturn(func() (iter.Seq2[K, V], error) {
			return mockIterateSeq(written), nil
		}),
	)

	seq, err := c.Iterate()
	require.NoError(err)

	got := map[K]V{}
	for k, v := range seq {
		got[k] = v
	}
	require.Equal(map[K]V{1: "value-1", 2: "value-2", 3: "value-3"}, got)
}

func TestKVCachedFile_Iterate_ReturnsErrorOnFlushFailure(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	c.flushBuffer[1] = "value-1"

	injected := errors.New("flush failed")
	mock.EXPECT().SetBatch(gomock.Any()).Return(injected)

	seq, err := c.Iterate()
	require.ErrorIs(err, injected)
	require.Nil(seq)
}

func TestKVCachedFile_Iterate_ReturnsErrorFromFileIterate(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	injected := errors.New("iterate failed")
	mock.EXPECT().Iterate().Return(nil, injected)

	seq, err := c.Iterate()
	require.ErrorIs(err, injected)
	require.Nil(seq)
}

func TestKVCachedFile_Close_FlushesAndClosesUnderlyingFile(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	require.NoError(c.Set(key1, value1))
	c.flushBuffer[2] = "buffer-2"

	written := map[K]V{}
	gomock.InOrder(
		mock.EXPECT().SetBatch(gomock.Any()).DoAndReturn(func(entries map[K]V) error {
			maps.Copy(written, entries)
			return nil
		}),
		mock.EXPECT().Flush().Return(nil),
		mock.EXPECT().Close().Return(nil),
	)

	require.NoError(c.Close())
	require.Equal(value1, written[key1])
	require.Equal("buffer-2", written[2])
}

func TestKVCachedFile_Close_ReturnsErrorOnFlushError(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	c.flushBuffer[1] = "value"
	injected := errors.New("flush failed")
	mock.EXPECT().SetBatch(gomock.Any()).Return(injected)
	// Close on the underlying file must not be called when flush fails.

	err := c.Close()
	require.ErrorIs(err, injected)
}

func TestKVCachedFile_Close_ReturnsErrorOnFileCloseError(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	injected := errors.New("close failed")
	// No cache / buffer entries: draining is a no-op and we go straight
	// to file.Close().
	mock.EXPECT().Close().Return(injected)

	err := c.Close()
	require.ErrorIs(err, injected)
}

func TestKVCachedFile_Close_StopsBackgroundWriter(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)
	mock.EXPECT().Close().Return(nil)

	require.NoError(c.Close())

	// Close waits for the writer to exit before returning, so writerDone is
	// already closed by now.
	select {
	case <-c.writerDone:
	default:
		require.Fail("background writer still running after Close")
	}
}

func TestKVCachedFile_GetMemoryFootprint_IsNonZero(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	require.NoError(c.Set(key1, value1))
	c.flushBuffer[2] = "buffer-2"

	mock.EXPECT().GetMemoryFootprint().Return(common.NewMemoryFootprint(0))

	mf := c.GetMemoryFootprint()
	require.NotNil(mf)
	require.Greater(mf.Total(), uintptr(0))
}

func TestKVCachedFile_Operations_ReturnStickyErrorAfterBackgroundWriteFailure(t *testing.T) {
	injected := errors.New("background write failed")

	// induce drives the cache into its terminal error state by making the
	// background writer fail a flush.
	induce := func(t *testing.T, c *KVCachedFile[K, V], mock *MockKVFileWithMemoryFootprint[K, V]) {
		mock.EXPECT().SetBatch(gomock.Any()).Return(injected).AnyTimes()
		mock.EXPECT().Flush().Return(nil).AnyTimes()
		c.flushBuffer[999] = "boom"
		require.ErrorIs(t, c.Flush(), injected)
	}

	tests := map[string]func(c *KVCachedFile[K, V]) error{
		"Get":      func(c *KVCachedFile[K, V]) error { _, err := c.Get(key1); return err },
		"Has":      func(c *KVCachedFile[K, V]) error { _, err := c.Has(key1); return err },
		"Set":      func(c *KVCachedFile[K, V]) error { return c.Set(key1, value1) },
		"SetBatch": func(c *KVCachedFile[K, V]) error { return c.SetBatch(map[K]V{key1: value1}) },
		"Flush":    func(c *KVCachedFile[K, V]) error { return c.Flush() },
		"FileSize": func(c *KVCachedFile[K, V]) error { _, err := c.FileSize(); return err },
		"Iterate":  func(c *KVCachedFile[K, V]) error { _, err := c.Iterate(); return err },
	}

	for name, op := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			c, mock := openTestKVCachedFile(t)
			induce(t, c, mock)
			require.ErrorIs(op(c), injected)
		})
	}
}

// Concurrent callers and the background writer must not race. The fake file is
// deliberately NOT synchronized (see fakeKVFile), so this doubles as a guard on
// KVCachedFile's own serialization of file access: if that serialization (fileMu)
// were dropped, the writer and a foreground reader would touch the fake's map at
// once, which -race reports and Go's runtime turns into a fatal "concurrent map"
// panic. Run with -race for the hard signal. Get/Has target the same keys as
// Set, so reads exercise every layer -- cache, flush buffer, pending queue, and
// the underlying file once entries have been evicted and written.
func TestKVCachedFile_SetGetHasAndFlush_AreRaceFree(t *testing.T) {
	require := require.New(t)
	c, err := OpenKVCachedFile[K, V](newFakeKVFile(), cacheSize, flushBufferThreshold)
	require.NoError(err)
	t.Cleanup(c.shutdownWriter)

	const (
		workers      = 8
		opsPerWorker = 100000
	)
	var wg sync.WaitGroup
	for w := range workers {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := range opsPerWorker {
				key := w*opsPerWorker + i
				switch i % 4 {
				case 0:
					require.NoError(c.Set(key, fmt.Sprintf("w%d-i%d", w, i)))
				case 1:
					_, err := c.Get(key)
					require.NoError(err)
				case 2:
					_, err := c.Has(key)
					require.NoError(err)
				case 3:
					require.NoError(c.Flush())
				}
			}
		}(w)
	}
	wg.Wait()

	require.NoError(c.Flush())
	require.NoError(c.Close())
}

func TestKVCachedFile_enqueueCurrentBufferLocked_IsNoopWhenBufferIsEmpty(t *testing.T) {
	require := require.New(t)
	// No file interactions expected: an empty buffer is never handed to the writer.
	c, _ := openTestKVCachedFile(t)

	c.mu.Lock()
	defer c.mu.Unlock()
	c.enqueueCurrentBufferLocked()

	require.Empty(c.pending)
}

func TestKVCachedFile_enqueueCurrentBufferLocked_AppendsBufferAndClearsDirtyEntries(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	// Block the writer so the sealed buffer stays visible in c.pending for
	// inspection instead of being consumed immediately.
	_, unblockWriter := blockWriter(t, mock, nil)

	c.mu.Lock()
	c.flushBuffer[1] = "a"
	c.flushBuffer[2] = "b"
	c.dirty[1] = true
	c.dirty[2] = true
	// Dirty entry that is not in the flush buffer must survive the seal.
	c.dirty[3] = true
	sealed := maps.Clone(c.flushBuffer)

	c.enqueueCurrentBufferLocked()

	require.Empty(c.flushBuffer, "flushBuffer must be reset after enqueue")
	require.Len(c.pending, 1, "sealed buffer must be appended to pending")
	require.Equal(sealed, c.pending[0], "pending buffer must equal the sealed flushBuffer")

	require.NotContains(c.dirty, K(1), "dirty entries in the sealed buffer must be cleared")
	require.NotContains(c.dirty, K(2), "dirty entries in the sealed buffer must be cleared")
	require.Contains(c.dirty, K(3), "dirty entries not in the sealed buffer must be retained")
	c.mu.Unlock()

	unblockWriter()
	require.NoError(c.Flush())
}

func TestKVCachedFile_enqueueCurrentBufferLocked_RetainsBufferAfterWriteErrorOrClose(t *testing.T) {
	tests := map[string]func(c *KVCachedFile[K, V]){
		"after write error": func(c *KVCachedFile[K, V]) { c.writeErr = errors.New("terminal") },
		"after close":       func(c *KVCachedFile[K, V]) { c.closed = true },
	}

	for name, induce := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			// No file interactions expected: a stopped writer receives no buffers.
			c, _ := openTestKVCachedFile(t)

			c.mu.Lock()
			defer c.mu.Unlock()
			induce(c)
			c.flushBuffer[key1] = value1

			c.enqueueCurrentBufferLocked()

			require.Empty(c.pending, "no buffer must be handed to a stopped writer")
			require.Len(c.flushBuffer, 1, "unwritten entries must be retained")
		})
	}
}

// The background writer must persist queued buffers oldest-first so the newest
// value of a repeated key lands on disk last and wins.
func TestKVCachedFile_flushWorker_PersistsBuffersInFIFOOrder(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	var writeOrder []V
	writing, unblockWriter := blockWriter(t, mock, func(entries map[K]V) {
		writeOrder = append(writeOrder, entries[key1])
	})

	sealBuffer(c, key1, "old")
	<-writing                  // writer holds the first buffer in flight
	sealBuffer(c, key1, "new") // queued behind it

	unblockWriter()
	require.NoError(c.Flush())

	require.Equal([]V{"old", "new"}, writeOrder,
		"buffers must be written oldest-first so the newest value wins on disk")
}

func TestKVCachedFile_writeBuffer_WritesBatchThenFlushesFile(t *testing.T) {
	injected := errors.New("injected")
	tests := map[string]struct {
		setBatchErr error
		flushErr    error
	}{
		"success":           {},
		"batch write fails": {setBatchErr: injected},
		"file flush fails":  {flushErr: injected},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			c, mock := openTestKVCachedFile(t)

			entries := map[K]V{key1: value1}
			mock.EXPECT().SetBatch(entries).Return(test.setBatchErr)
			if test.setBatchErr == nil {
				// The file is only flushed when the batch write succeeded.
				mock.EXPECT().Flush().Return(test.flushErr)
			}

			err := c.writeBuffer(entries)
			if test.setBatchErr != nil || test.flushErr != nil {
				require.ErrorIs(err, injected)
			} else {
				require.NoError(err)
			}
		})
	}
}

func TestKVCachedFile_shutdownWriter_IsIdempotent(t *testing.T) {
	require := require.New(t)
	c, _ := openTestKVCachedFile(t)

	// Repeated calls must neither panic nor block, and the writer must exit.
	c.shutdownWriter()
	c.shutdownWriter()

	select {
	case <-c.writerDone:
	default:
		require.Fail("background writer still running after shutdown")
	}
}

func TestKVCachedFile_handleCacheSet_UpdatesCache(t *testing.T) {
	require := require.New(t)
	c, _ := openTestKVCachedFile(t)

	key, value := key1, value1
	require.NoError(c.handleCacheSet(&key, &value, true))

	require.Equal(1, getCacheSize(c.cache))
	got, found := c.cache.Get(key)
	require.True(found)
	require.Equal(value, got)
}

func TestKVCachedFile_handleCacheSet_IsNoOpWhenKeyOrValueIsNil(t *testing.T) {
	require := require.New(t)
	c, _ := openTestKVCachedFile(t)

	key := key1
	value := value1

	require.NoError(c.handleCacheSet(nil, nil, true))
	require.NoError(c.handleCacheSet(&key, nil, true))
	require.NoError(c.handleCacheSet(nil, &value, true))

	require.Equal(0, getCacheSize(c.cache))
}

func TestKVCachedFile_handleCacheSet_MovesEvictedEntryToFlushBuffer(t *testing.T) {
	require := require.New(t)
	// No file interactions are expected: a single eviction only fills the
	// flush buffer, and its size (1) stays below the flush threshold (3).
	c, _ := openTestKVCachedFile(t)

	for i := 0; i < cacheSize+1; i++ {
		v := fmt.Sprintf("value%d", i)
		require.NoError(c.handleCacheSet(&i, &v, true))
	}

	require.Equal(1, len(c.flushBuffer))
	require.Equal("value0", c.flushBuffer[0])
}

func TestKVCachedFile_handleCacheSet_FlushesBufferToFileWhenThresholdReached(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	written := map[K]V{}
	mock.EXPECT().SetBatch(gomock.Any()).DoAndReturn(func(entries map[K]V) error {
		maps.Copy(written, entries)
		return nil
	}).Times(1)
	mock.EXPECT().Flush().Return(nil).Times(1)

	// Insert enough entries so that the cache is full and (threshold - 1)
	// entries have been evicted into the buffer. No flush is triggered yet, so
	// the background writer stays parked and these unlocked calls are safe.
	total := cacheSize + flushBufferThreshold - 1
	for i := 0; i < total; i++ {
		v := fmt.Sprintf("value%d", i)
		require.NoError(c.handleCacheSet(&i, &v, true))
	}
	require.Equal(flushBufferThreshold-1, len(c.flushBuffer))

	// The next insert pushes the buffer to the threshold and seals it for the
	// background writer. This call wakes the writer, so it must hold the lock.
	key := total
	value := fmt.Sprintf("value%d", key)
	c.mu.Lock()
	err := c.handleCacheSet(&key, &value, true)
	c.mu.Unlock()
	require.NoError(err)

	// The buffer is sealed synchronously, so it is empty immediately; the
	// actual write completes asynchronously, so wait for it before inspecting
	// what was written.
	require.Equal(0, len(c.flushBuffer))
	require.NoError(c.waitForPendingFlushes())

	// The newly-inserted entry lives in the cache.
	cached, found := c.cache.Get(key)
	require.True(found)
	require.Equal(value, cached)

	require.Equal(flushBufferThreshold, len(written))
	for i := 0; i < flushBufferThreshold; i++ {
		require.Equal(fmt.Sprintf("value%d", i), written[i])
	}
}

func TestKVCachedFile_handleCacheSet_DoesNotMarkEntryDirtyWhenDirtyFlagIsFalse(t *testing.T) {
	require := require.New(t)
	c, _ := openTestKVCachedFile(t)

	key, value := key1, value1
	require.NoError(c.handleCacheSet(&key, &value, false))

	_, exists := c.dirty[key]
	require.False(exists)
}

func TestKVCachedFile_handleCacheSet_DoesNotInsertIntoFlushBufferWhenEvictedEntryIsClean(t *testing.T) {
	require := require.New(t)
	c, _ := openTestKVCachedFile(t)

	for i := 0; i < cacheSize; i++ {
		v := fmt.Sprintf("value%d", i)
		require.NoError(c.handleCacheSet(&i, &v, false))
	}

	// Trigger eviction
	v := fmt.Sprintf("value%d", cacheSize)
	key := cacheSize
	require.NoError(c.handleCacheSet(&key, &v, false))

	require.Equal(0, len(c.flushBuffer))
}

func TestKVCachedFile_handleCacheSet_UpdatedEntryIsRetrievedInAllLocations(t *testing.T) {
	testCases := map[string]struct {
		setup func(t *testing.T, c *KVCachedFile[K, V])
	}{
		"entry in cache": {
			setup: func(t *testing.T, c *KVCachedFile[K, V]) {
				require.NoError(t, c.Set(key1, value1))
			},
		},
		"entry in flush buffer": {
			setup: func(t *testing.T, c *KVCachedFile[K, V]) {
				c.flushBuffer[key1] = value1
			},
		},
		"entry on file only": {
			// handleCacheSet does not consult the file; this case verifies
			// that a rewrite of a key that only exists on disk still lands
			// in the cache.
			setup: func(t *testing.T, c *KVCachedFile[K, V]) {},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			c, _ := openTestKVCachedFile(t)
			tc.setup(t, c)

			key := key1
			newValue := "new-value"
			require.NoError(c.handleCacheSet(&key, &newValue, true))

			cached, found := c.cache.Get(key)
			require.True(found)
			require.Equal(newValue, cached)
		})
	}
}

// fakeKVFile is a minimal in-memory KVFile for concurrency tests. It is
// deliberately NOT synchronized: KVCachedFile must serialize all access to the
// wrapped file itself (KVFile is not required to be safe for concurrent use), so
// a lock here would hide a regression in that serialization instead of exposing
// it. Do not add a mutex.
type fakeKVFile struct {
	data map[K]V
}

func newFakeKVFile() *fakeKVFile {
	return &fakeKVFile{data: make(map[K]V)}
}

func (f *fakeKVFile) Get(key K) (*V, error) {
	if v, ok := f.data[key]; ok {
		return &v, nil
	}
	return nil, nil
}

func (f *fakeKVFile) Has(key K) (bool, error) {
	_, ok := f.data[key]
	return ok, nil
}

func (f *fakeKVFile) Set(key K, value V) error {
	f.data[key] = value
	return nil
}

func (f *fakeKVFile) SetBatch(entries map[K]V) error {
	maps.Copy(f.data, entries)
	return nil
}

func (f *fakeKVFile) Flush() error { return nil }

func (f *fakeKVFile) FileSize() (uint64, error) { return 0, nil }

func (f *fakeKVFile) Iterate() (iter.Seq2[K, V], error) {
	return mockIterateSeq(maps.Clone(f.data)), nil
}

func (f *fakeKVFile) Close() error { return nil }

func (f *fakeKVFile) GetMemoryFootprint() *common.MemoryFootprint {
	return common.NewMemoryFootprint(0)
}

func openTestKVCachedFile(t *testing.T) (*KVCachedFile[K, V], *MockKVFileWithMemoryFootprint[K, V]) {
	t.Helper()
	ctrl := gomock.NewController(t)
	mock := NewMockKVFileWithMemoryFootprint[K, V](ctrl)
	file, err := OpenKVCachedFile[K, V](mock, cacheSize, flushBufferThreshold)
	require.NoError(t, err)
	// Stop the background writer when the test ends so its goroutine never
	// outlives the test. This performs no file I/O and is idempotent, so it
	// is harmless for tests that already call Close. It runs before the
	// gomock controller's own cleanup verifies expectations.
	t.Cleanup(file.shutdownWriter)
	return file, mock
}

// waitForPendingFlushes blocks until the background writer has persisted every
// queued buffer, returning the sticky write error if any. It is a test-only
// synchronization aid; production callers use Flush, which also seals the
// current buffer.
func (c *KVCachedFile[K, V]) waitForPendingFlushes() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	for len(c.pending) > 0 && c.writeErr == nil {
		c.cond.Wait()
	}
	return c.writeErr
}

// blockWriter makes the mocked file park the background writer inside SetBatch
// until the returned unblock function is called. The returned channel is closed
// when the writer first enters SetBatch, and onWrite (if not nil) observes every
// batch once the writer is released; results it records are safe to read after a
// Flush. Unblock is also invoked during test cleanup so a failed assertion
// cannot leave the writer parked (see newWriterRelease).
func blockWriter(t *testing.T, mock *MockKVFileWithMemoryFootprint[K, V], onWrite func(entries map[K]V)) (writing <-chan struct{}, unblock func()) {
	t.Helper()
	release, unblock := newWriterRelease(t)
	writingCh := make(chan struct{})
	var once sync.Once
	mock.EXPECT().SetBatch(gomock.Any()).DoAndReturn(func(entries map[K]V) error {
		once.Do(func() { close(writingCh) })
		<-release
		if onWrite != nil {
			onWrite(entries)
		}
		return nil
	}).AnyTimes()
	mock.EXPECT().Flush().Return(nil).AnyTimes()
	return writingCh, unblock
}

// newWriterRelease returns a channel used to unblock a mocked background writer
// that parks inside SetBatch, together with a function that closes it. The
// channel is also closed during test cleanup if the test has not closed it
// itself: without this, a failed assertion would skip the test's own close and
// leave the writer parked, so the shutdownWriter cleanup registered by
// openTestKVCachedFile would deadlock waiting for the writer to exit. Because it
// is registered after that cleanup, LIFO ordering runs it first.
func newWriterRelease(t *testing.T) (release chan struct{}, unblock func()) {
	t.Helper()
	release = make(chan struct{})
	unblock = sync.OnceFunc(func() { close(release) })
	t.Cleanup(unblock)
	return release, unblock
}

// sealBuffer places a single entry into the flush buffer and hands the buffer
// over to the background writer.
func sealBuffer(c *KVCachedFile[K, V], key K, value V) {
	c.mu.Lock()
	c.flushBuffer[key] = value
	c.enqueueCurrentBufferLocked()
	c.mu.Unlock()
}

// mockIterateSeq returns an iter.Seq2 mock return value that yields the
// key/value pairs from the given map.
func mockIterateSeq(entries map[K]V) iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		for k, v := range entries {
			if !yield(k, v) {
				return
			}
		}
	}
}

func getCacheSize[K comparable, V any](cache *common.LruCache[K, V]) int {
	size := 0
	cache.Iterate(func(K, V) bool {
		size++
		return true
	})
	return size
}
