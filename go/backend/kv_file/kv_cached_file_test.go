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

func TestKVCachedFile_Open_ReturnsErrorOnNilFile(t *testing.T) {
	require := require.New(t)

	_, err := OpenKVCachedFile[K, V](nil, cacheSize, flushBufferThreshold)
	require.Error(err)
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

func TestKVCachedFile_Get_ReturnsNilOnUnknownKey(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	mock.EXPECT().Get(999).Return(nil, nil)

	got, err := c.Get(999)
	require.NoError(err)
	require.Nil(got)
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

func TestKVCachedFile_Set_ReturnsErrorWhenBackingFileWriteFails(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	// Fill enough entries so that the next Set pushes the flush buffer to the
	// threshold and triggers a file write.
	for i := 0; i < cacheSize+flushBufferThreshold-1; i++ {
		require.NoError(c.Set(i, fmt.Sprintf("value%d", i)))
	}

	injected := errors.New("write failed")
	mock.EXPECT().SetBatch(gomock.Any()).Return(injected)

	err := c.Set(cacheSize+flushBufferThreshold-1, "trigger")
	require.ErrorIs(err, injected)
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

func TestKVCachedFile_Flush_WritesCacheAndBufferToFile(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	require.NoError(c.Set(1, "cache-1"))
	require.NoError(c.Set(2, "cache-2"))
	c.flushBuffer[3] = "buffer-3"
	c.flushBuffer[4] = "buffer-4"

	written := map[K]V{}
	mock.EXPECT().SetBatch(gomock.Any()).DoAndReturn(func(entries map[K]V) error {
		for k, v := range entries {
			written[k] = v
		}
		return nil
	}).Times(1)
	mock.EXPECT().Flush().Return(nil).Times(1)

	require.NoError(c.Flush())

	require.Equal("cache-1", written[1])
	require.Equal("cache-2", written[2])
	require.Equal("buffer-3", written[3])
	require.Equal("buffer-4", written[4])
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

func TestKVCachedFile_Flush_ReturnsErrorOnFileWriteError(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	c.flushBuffer[1] = "value"
	injected := errors.New("write failed")
	mock.EXPECT().SetBatch(gomock.Any()).Return(injected)

	err := c.Flush()
	require.ErrorIs(err, injected)
}

func TestKVCachedFile_Flush_ReturnsErrorOnFileFlushError(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	c.flushBuffer[1] = "value"
	injected := errors.New("flush failed")
	mock.EXPECT().SetBatch(gomock.Any()).Return(nil)
	mock.EXPECT().Flush().Return(injected)

	err := c.Flush()
	require.ErrorIs(err, injected)
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
		for k, v := range entries {
			written[k] = v
		}
		return nil
	}).Times(1)
	mock.EXPECT().Flush().Return(nil).Times(1)

	require.NoError(c.Flush())
	require.Equal(map[K]V{1: "value-1-updated"}, written)
}

func TestKVCachedFile_Size_ReturnsCorrectSize(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	// Backing "storage" behind the mock. Values 1, 2, 3 are on the file.
	file := map[K]V{
		1: "file-1",
		2: "file-2",
		3: "file-3",
	}
	mock.EXPECT().Has(gomock.Any()).DoAndReturn(func(k K) (bool, error) {
		_, ok := file[k]
		return ok, nil
	}).AnyTimes()
	mock.EXPECT().Size().DoAndReturn(func() (uint64, error) { return uint64(len(file)), nil }).Times(1)

	c.Set(1, "cache-1") // overwrites file 1, not gonna be counted
	c.Set(4, "cache-4") // new key, will be counted

	size, err := c.Size()
	require.NoError(err)
	require.Equal(uint64(4), size, "unique keys: 1,2,3,4")
}

func TestKVCachedFile_Size_ReturnsErrorOnFileReadError(t *testing.T) {
	require := require.New(t)
	c, mock := openTestKVCachedFile(t)

	_, _, _ = c.cache.Set(1, "cache-1")
	c.dirty[1] = true

	injected := errors.New("size read failed")
	mock.EXPECT().Size().Return(uint64(0), nil)
	mock.EXPECT().Has(1).Return(false, injected)

	_, err := c.Size()
	require.ErrorIs(err, injected)
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

	// No pending writes: flushLocked is a no-op, Iterate delegates directly
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
			for k, v := range entries {
				written[k] = v
			}
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
			for k, v := range entries {
				written[k] = v
			}
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
	// No cache / buffer entries: flushLocked is a no-op and we go straight
	// to file.Close().
	mock.EXPECT().Close().Return(injected)

	err := c.Close()
	require.ErrorIs(err, injected)
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

// ---------------------------------------------------------------------------
// Private helper tests
// ---------------------------------------------------------------------------

func TestKVCachedFile_flushPending_ClearsDirtyForFlushedKeys(t *testing.T) {
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
		for k, v := range entries {
			written[k] = v
		}
		return nil
	}).Times(1)
	mock.EXPECT().Flush().Return(nil).Times(1)

	// Insert enough entries so that the cache is full and (threshold - 1)
	// entries have been evicted into the buffer.
	total := cacheSize + flushBufferThreshold - 1
	for i := 0; i < total; i++ {
		v := fmt.Sprintf("value%d", i)
		require.NoError(c.handleCacheSet(&i, &v, true))
	}
	require.Equal(flushBufferThreshold-1, len(c.flushBuffer))

	// The next insert pushes the buffer to the threshold and triggers a flush.
	key := total
	value := fmt.Sprintf("value%d", key)
	require.NoError(c.handleCacheSet(&key, &value, true))

	// The newly-inserted entry lives in the cache.
	cached, found := c.cache.Get(key)
	require.True(found)
	require.Equal(value, cached)

	// The buffer has been drained after being flushed to the file.
	require.Equal(0, len(c.flushBuffer))
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

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

func openTestKVCachedFile(t *testing.T) (*KVCachedFile[K, V], *MockKVFileWithMemoryFootprint[K, V]) {
	t.Helper()
	ctrl := gomock.NewController(t)
	mock := NewMockKVFileWithMemoryFootprint[K, V](ctrl)
	file, err := OpenKVCachedFile[K, V](mock, cacheSize, flushBufferThreshold)
	require.NoError(t, err)
	return file, mock
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
