package kv_file

import (
	"encoding/binary"
	"fmt"
	"io"
	"iter"
	"maps"
	"sync"
	"testing"
	"testing/synctest"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
)

type K = uint64
type V = string

// fakeKVFile is a minimal in-memory KVFile for concurrency tests. It is
// deliberately NOT synchronized: KVCachedFile must serialize all access to the
// wrapped file itself (KVFile is not required to be safe for concurrent use), so
// a lock here would hide a regression in that serialization instead of exposing
// it. Do not add a mutex.
type fakeKVFile struct {
	data map[K]V
}

const (
	// testEntryValueCap is the longest value the test codec can store.
	testEntryValueCap = 20
	// testEntrySize is the fixed on-disk size of one test codec record:
	// [key(8), length(4), value padded to testEntryValueCap]. Records have a
	// fixed size so the codec is usable with OrderedFile as well.
	testEntrySize = 12 + testEntryValueCap
)

func readTestEntry(reader io.Reader) (K, V, error) {
	var record [testEntrySize]byte
	if _, err := io.ReadFull(reader, record[:]); err != nil {
		return 0, "", err
	}
	length := binary.BigEndian.Uint32(record[8:12])
	if length > testEntryValueCap {
		return 0, "", fmt.Errorf("invalid record: value length %d exceeds capacity %d", length, testEntryValueCap)
	}
	return K(binary.BigEndian.Uint64(record[0:8])), string(record[12 : 12+length]), nil
}

func writeTestEntry(writer io.Writer, key K, value V) error {
	if len(value) > testEntryValueCap {
		return fmt.Errorf("value %q exceeds the test codec capacity of %d bytes", value, testEntryValueCap)
	}
	var record [testEntrySize]byte
	binary.BigEndian.PutUint64(record[0:8], uint64(key))
	binary.BigEndian.PutUint32(record[8:12], uint32(len(value)))
	copy(record[12:], value)
	_, err := writer.Write(record[:])
	return err
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

func TestBlockWriter_ParksTheWriterAndReportsBatchesAfterRelease(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		require := require.New(t)
		c, mock := openTestKVCachedFile(t)

		var written []map[K]V
		writing, unblock := blockWriter(t, mock, func(entries map[K]V) {
			written = append(written, entries)
		})

		sealBuffer(c, key1, value1)
		<-writing // the writer has entered SetBatch
		synctest.Wait()
		require.Empty(written, "the writer must stay parked until released")

		unblock()
		require.NoError(c.waitForPendingFlushes())
		require.Equal([]map[K]V{{key1: value1}}, written)
	})
}

func TestNewWriterRelease_UnblockClosesTheChannelOnce(t *testing.T) {
	require := require.New(t)
	release, unblock := newWriterRelease(t)

	select {
	case <-release:
		t.Fatal("the channel must stay open until unblock is called")
	default:
	}

	unblock()
	unblock() // a second call must be a no-op instead of a double-close panic

	_, open := <-release
	require.False(open)
}

func TestNewWriterRelease_CleanupClosesTheChannel(t *testing.T) {
	require := require.New(t)
	var release chan struct{}
	t.Run("register without unblocking", func(t *testing.T) {
		release, _ = newWriterRelease(t)
	})

	// The subtest has ended, so its cleanup must have closed the channel; a
	// parked writer would now be released even though unblock was never called.
	_, open := <-release
	require.False(open)
}
