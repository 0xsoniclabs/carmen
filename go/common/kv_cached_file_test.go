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

const (
	fileName             = "file.dat"
	cacheSize            = 10
	flushBufferThreshold = 5

	value1 = "value1"
	key1   = 1
)

type K = int
type V = string

func TestCachedFile_Add_CacheIsUpdated(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	c := openTestCachedFile(t, dir)

	require.NoError(c.Set(key1, value1))

	require.Equal(1, getCacheSize(c.cache))
	cachedValue, found := c.cache.Get(key1)
	require.True(found)
	require.Equal(value1, cachedValue)
}

func TestCachedFile_handleCacheSet_UpdatesCache(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	c := openTestCachedFile(t, dir)

	key := key1
	value := value1
	c.handleCacheSet(&key, &value)

	require.Equal(1, getCacheSize(c.cache))
	cachedValue, found := c.cache.Get(key)
	require.True(found)
	require.Equal(value, cachedValue)
}

func TestCachedFile_handleCacheSet_WritesToBufferOnEviction(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	c := openTestCachedFile(t, dir)

	// Fill the cache with values until it reaches the eviction threshold.
	for i := 0; i < cacheSize+1; i++ {
		value := fmt.Sprintf("value%d", i)
		c.handleCacheSet(&i, &value)
	}

	require.Equal(1, len(c.flushBuffer))
	value, found := c.flushBuffer[0]
	require.True(found)
	require.Equal(value, "value0")
}

func TestCachedFile_handleCacheSet_WritesToDiskWhenBufferIsFull(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	c := openTestCachedFile(t, dir)

	// Fill the cache until it reaches the eviction threshold.
	for i := range cacheSize + flushBufferThreshold - 1 {
		value := fmt.Sprintf("value%d", i)
		c.handleCacheSet(&i, &value)
	}

	// Check the first flushBufferThreshold-1 entries are in the flush buffer.
	for i := range flushBufferThreshold - 1 {
		value, found := c.flushBuffer[i]
		require.True(found)
		require.Equal(value, fmt.Sprintf("value%d", i))
	}

	// Check that the other ones are in the cache
	for i := flushBufferThreshold - 1; i < cacheSize+flushBufferThreshold-1; i++ {
		value, found := c.cache.Get(i)
		require.True(found)
		require.Equal(value, fmt.Sprintf("value%d", i))
	}

	// Add one more value to trigger flush to disk.
	lastValue := fmt.Sprintf("value%d", cacheSize+flushBufferThreshold-1)
	key := cacheSize + flushBufferThreshold - 1
	c.handleCacheSet(&key, &lastValue)

	// Check that the last value is in the cache.
	value, found := c.cache.Get(key)
	require.True(found)
	require.Equal(lastValue, value)

	// FLush buffer is now empty
	require.Equal(0, len(c.flushBuffer))
	// Flush buffer values are on disk now
	require.Equal(flushBufferThreshold, len(c.offsets))

	fileOffsets, _, err := readFileOffsets[K, V](c.filePath, 0, c.readValueFn)
	require.NoError(err)
	require.Equal(flushBufferThreshold, len(fileOffsets))
	for i := range flushBufferThreshold {
		key, value, err := readFromDiskAtOffset[K, V](c.filePath, fileOffsets[i], c.readValueFn)
		require.NoError(err)
		require.Equal(*key, i)
		require.Equal(*value, fmt.Sprintf("value%d", i))
	}
}

func TestCodes_handleCacheSet_EvictedEntryDiscardedWhenOnDisk(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	c := openTestCachedFile(t, dir)

	// Fill cache and mark {10} as already on disk.
	for i := range cacheSize {
		value := fmt.Sprintf("value%d", i)
		c.handleCacheSet(&i, &value)
	}
	c.offsets[0] = 0 // Simulate that the key is on disk

	// Trigger eviction
	key := 100
	value := "new-value"
	c.handleCacheSet(&key, &value)

	require.Empty(c.flushBuffer)
}

func TestCachedFile_Get_ReturnsValueCorrectly(t *testing.T) {
	key := 0
	tests := map[string]struct {
		setup func(t *testing.T, c *KVCachedFile[K, V]) (want V)
	}{
		"returns from cache": {
			setup: func(t *testing.T, c *KVCachedFile[K, V]) (want V) {
				want = "cache-value"
				require.NoError(t, c.Set(key, want))
				return want
			},
		},
		"returns from pending when evicted from cache": {
			setup: func(t *testing.T, c *KVCachedFile[K, V]) (want V) {
				// Use a tiny cache so we can force eviction.
				c.cache = NewLruCache[K, V](2)
				want = "pending-value"
				require.NoError(t, c.Set(key, want))
				// Fill the cache to evict the target into pending.
				c.Set(1, "filler-1")
				c.Set(2, "filler-2")
				return want
			},
		},
		"returns from disk": {
			setup: func(t *testing.T, c *KVCachedFile[K, V]) (want V) {
				want = "disk-value"
				require.NoError(t, c.Set(key, want))
				require.NoError(t, c.Flush())
				c.cache.Clear()
				return want
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			dir := t.TempDir()
			c := openTestCachedFile(t, dir)
			want := test.setup(t, c)
			got, err := c.Get(key)
			require.NoError(err)
			require.Equal(want, *got)
		})
	}
}

func TestCachedFile_Get_ReturnsNilOnUnknownKey(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	c := openTestCachedFile(t, dir)
	key := 999 // some key that doesn't exist
	got, err := c.Get(key)
	require.NoError(err)
	require.Nil(got)
}

func TestCodes_getCodeForHash_ReturnsNilOnDiskReadError(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	c := openTestCachedFile(t, dir)

	c.offsets[key1] = 0 // Simulate that the key is on disk
	// Remove the backing file to make readCodeFromDisk fail.
	require.NoError(os.Remove(c.filePath))

	got, err := c.Get(key1)
	require.Nil(got)
	require.Error(err)
}

func TestCodes_Get_ValuesInFlushBufferArePromotedIntoCache(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	c := openTestCachedFile(t, dir)

	for i := range cacheSize + 1 {
		value := fmt.Sprintf("value%d", i)
		c.Set(i, value)
	}

	// Verify that the first key is in pending.
	_, inPending := c.flushBuffer[0]
	require.True(inPending)

	// Request the first key, which should promote it back into the cache.
	got, err := c.Get(0)
	require.NoError(err)
	require.Equal("value0", *got)

	// Verify that the first key is no longer in pending and is now in cache.
	_, inPending = c.flushBuffer[0]
	require.False(inPending)
	cachedValue, found := c.cache.Get(0)
	require.True(found)
	require.Equal("value0", cachedValue)
}

func TestCodes_readCodeFromDisk_ReadsValueCorrectly(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	c := openTestCachedFile(t, dir)

	valuesToWrite := map[K]V{
		1: "value1",
		2: "value2",
	}

	_, _, err := appendToFile(valuesToWrite, c.filePath, c.writeValueFn)
	require.NoError(err)

	for key, offset := range c.offsets {
		gotKey, gotValue, err := c.readFromDiskAtOffset(offset)
		require.NoError(err)
		require.Equal(key, *gotKey)
		require.Equal(valuesToWrite[key], *gotValue)
	}
}

func TestKVCachedFile_appendToFile_AppendsValuesCorrectly(t *testing.T) {
	tests := map[string]struct {
		// existing codes already on disk before calling appendCodes
		existing map[K]V
		// new codes to append
		toAppend map[K]V
	}{
		"empty map appends nothing": {
			toAppend: map[K]V{},
		},
		"single code on empty file": {
			toAppend: map[K]V{
				1: "value1",
			},
		},
		"multiple codes on empty file": {
			toAppend: map[K]V{
				2: "value2",
				3: "value3",
				4: "value4",
			},
		},
		"append to existing file preserves previous data": {
			existing: map[K]V{
				5: "value5",
			},
			toAppend: map[K]V{
				6: "value6",
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			dir := t.TempDir()
			filepath := getTestFilePath(dir)

			// Write existing codes to establish a non-empty file.
			if len(test.existing) > 0 {
				_, _, err := appendToFile(test.existing, filepath, writeIntString)
				require.NoError(err)
			}

			offsets, size, err := appendToFile(test.toAppend, filepath, writeIntString)
			require.NoError(err)

			// Verify offsets map is populated for all appended codes.
			require.Equal(len(test.toAppend), len(offsets))
			for h := range test.toAppend {
				_, exists := offsets[h]
				require.True(exists, "offset missing for value %v", h)
			}

			// Verify fileSize matches actual file size.
			info, err := os.Stat(filepath)
			require.NoError(err)
			require.Equal(uint64(info.Size()), size)

			// Verify all appended values can be read back correctly from disk.
			file, err := os.Open(filepath)
			require.NoError(err)
			defer file.Close()
			for key, value := range test.toAppend {
				offset := offsets[key]
				_, err := file.Seek(int64(offset), io.SeekStart)
				require.NoError(err)
				readKey, readValue, err := readIntString(file)
				require.NoError(err)
				require.Equal(key, readKey)
				require.Equal(value, readValue)
			}
		})
	}
}

func TestCodes_GetAll_ReturnsAllValues(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	c := openTestCachedFile(t, dir)

	// Write somethings to disk
	valuesToWrite := map[K]V{
		0: "value0",
		1: "value1",
	}
	for key, value := range valuesToWrite {
		require.NoError(c.Set(key, value))
	}
	require.NoError(c.Flush())
	require.Equal(2, len(c.offsets))

	// Add some more values to the cache
	_, _, _ = c.cache.Set(2, "value2")
	_, _, _ = c.cache.Set(3, "value3")

	// Add some more values to the flush buffer
	c.flushBuffer[4] = "value4"
	c.flushBuffer[5] = "value5"

	allValues, err := c.GetAll()
	require.NoError(err)
	for key := range 6 {
		value, exists := allValues[key]
		require.True(exists)
		require.Equal(value, fmt.Sprintf("value%d", key))
	}
}

func TestCodese_readFromDiskAtOffset_ReadsValueCorrectly(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	c := openTestCachedFile(t, dir)

	valuesToWrite := map[K]V{
		1: "value1",
		2: "value2",
	}

	offsets, _, err := appendToFile(valuesToWrite, c.filePath, c.writeValueFn)
	require.NoError(err)

	for key, offset := range offsets {
		gotKey, gotValue, err := readFromDiskAtOffset[K, V](c.filePath, offset, c.readValueFn)
		require.NoError(err)
		require.Equal(key, *gotKey)
		require.Equal(valuesToWrite[key], *gotValue)
	}
}

func TestKVCachedFile_readFromDiskAtOffset_ErrorCases(t *testing.T) {
	tests := map[string]struct {
		prepare func(t *testing.T) (path string, offset uint64)
	}{
		"file does not exist": {
			prepare: func(t *testing.T) (string, uint64) {
				return filepath.Join(t.TempDir(), "missing.dat"), 0
			},
		},
		"truncated key field": {
			// readIntString reads 4 bytes for the key first; supplying fewer
			// than that triggers an unexpected EOF.
			prepare: func(t *testing.T) (string, uint64) {
				dir := t.TempDir()
				file := filepath.Join(dir, fileName)
				require.NoError(t, os.WriteFile(file, []byte{0x1, 0x2}, 0600))
				return file, 0
			},
		},
		"declared value length larger than available bytes": {
			// Well-formed 4-byte key and a 4-byte length that overstates the
			// number of value bytes actually present in the file.
			prepare: func(t *testing.T) (string, uint64) {
				dir := t.TempDir()
				file := filepath.Join(dir, fileName)
				content := []byte{
					0, 0, 0, 1, // key
					0, 0, 0, 100, // declared value length
					0x1, 0x2, // only 2 bytes of value present
				}
				require.NoError(t, os.WriteFile(file, content, 0600))
				return file, 0
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			path, offset := test.prepare(t)
			_, _, err := readFromDiskAtOffset[K, V](path, offset, readIntString)
			require.Error(err)
		})
	}
}

func TestKVCachedFile_appendToFile_ErrorCases(t *testing.T) {
	injectedErr := errors.New("injected write error")
	tests := map[string]struct {
		prepare   func(t *testing.T) (path string, writeFn writeValueFn[K, V])
		expectErr error // if non-nil, the returned error must wrap this
	}{
		"cannot open directory as file": {
			prepare: func(t *testing.T) (string, writeValueFn[K, V]) {
				return t.TempDir(), writeIntString
			},
		},
		"path in non-existent directory": {
			prepare: func(t *testing.T) (string, writeValueFn[K, V]) {
				return filepath.Join(t.TempDir(), "no", "such", "dir", fileName), writeIntString
			},
		},
		"read-only file causes flush error": {
			prepare: func(t *testing.T) (string, writeValueFn[K, V]) {
				dir := t.TempDir()
				file := filepath.Join(dir, fileName)
				require.NoError(t, os.WriteFile(file, nil, 0600))
				require.NoError(t, os.Chmod(file, 0444))
				return file, writeIntString
			},
		},
		"writeValueFn returns error": {
			prepare: func(t *testing.T) (string, writeValueFn[K, V]) {
				return filepath.Join(t.TempDir(), fileName), func(io.Writer, K, V) error {
					return injectedErr
				}
			},
			expectErr: injectedErr,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			path, writeFn := test.prepare(t)
			_, _, err := appendToFile(map[K]V{1: "value"}, path, writeFn)
			require.Error(err)
			if test.expectErr != nil {
				require.ErrorIs(err, test.expectErr)
			}
		})
	}
}

func TestKVCachedFile_Size(t *testing.T) {
	require := require.New(t)
	dir := t.TempDir()
	c := openTestCachedFile(t, dir)

	// Write some values to disk
	valuesToWrite := map[K]V{
		1: "value1",
		2: "value2",
	}
	for key, value := range valuesToWrite {
		require.NoError(c.Set(key, value))
	}
	require.NoError(c.Flush())

	// Add values to cache
	c.cache.Set(3, "value3")
	c.cache.Set(4, "value4")
	// Add values to flush buffer
	c.flushBuffer[5] = "value5"
	c.flushBuffer[6] = "value6"

	size := c.Size()
	require.Equal(6, size)
}

func TestKVCachedFile_readFileOffsets_ReadsOffsetsCorrectly(t *testing.T) {
	testCases := map[string]struct {
		header []byte
		data   map[K]V
	}{
		"empty file": {
			header: []byte{},
			data:   map[K]V{},
		},
		"no header": {
			header: []byte{},
			data:   map[K]V{1: "value"},
		},
		"with header": {
			header: []byte{0xDE, 0xAD, 0xBE, 0xEF},
			data:   map[K]V{1: "value"},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			dir := t.TempDir()
			filePath := getTestFilePath(dir)

			// Write header and data to file
			file, err := os.Create(filePath)
			require.NoError(err)
			defer file.Close()

			if len(tc.header) > 0 {
				_, err = file.Write(tc.header)
				require.NoError(err)
			}

			offsets, _, err := appendToFile(tc.data, filePath, writeIntString)
			require.NoError(err)

			readOffsets, _, err := readFileOffsets[K, V](filePath, uint64(len(tc.header)), readIntString)
			require.NoError(err)
			require.Equal(offsets, readOffsets)
		})
	}
}

func getTestFilePath(dir string) string {
	return dir + "/" + fileName
}

func writeIntString(writer io.Writer, key K, value V) error {
	// Format: K-len(V)-V Big endian
	var keyBytes [4]byte
	binary.BigEndian.PutUint32(keyBytes[:], uint32(key))
	if _, err := writer.Write(keyBytes[:]); err != nil {
		return err
	}
	var lenBytes [4]byte
	binary.BigEndian.PutUint32(lenBytes[:], uint32(len(value)))
	if _, err := writer.Write(lenBytes[:]); err != nil {
		return err
	}
	if _, err := writer.Write([]byte(value)); err != nil {
		return err
	}
	return nil
}

func readIntString(reader io.ReadSeeker) (K, V, error) {
	var keyBytes [4]byte
	var length [4]byte
	if _, err := io.ReadFull(reader, keyBytes[:]); err != nil {
		return 0, "", err
	}
	if _, err := io.ReadFull(reader, length[:]); err != nil {
		return 0, "", err
	}
	size := binary.BigEndian.Uint32(length[:])
	valueByte := make([]byte, size)
	if _, err := io.ReadFull(reader, valueByte); err != nil {
		return 0, "", err
	}
	key := K(binary.BigEndian.Uint32(keyBytes[:]))
	value := V(valueByte)
	return key, value, nil
}

func openTestCachedFile(t *testing.T, dir string) *KVCachedFile[K, V] {
	t.Helper()
	c, err := OpenKVCachedFile(
		getTestFilePath(dir),
		0,
		cacheSize,
		flushBufferThreshold,
		readIntString,
		writeIntString,
	)
	require.NoError(t, err)
	return c
}

func getCacheSize[K comparable, V any](cache *LruCache[K, V]) int {
	size := 0
	cache.Iterate(func(h K, b V) bool {
		size += 1
		return true
	})
	return size
}
