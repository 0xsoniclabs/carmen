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
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

// The tests in this file check the pre- and post-conditions of the KVFile
// contract against every implementation (see forEachKVFile). Keys are written
// densely: an OrderedFile allocates every slot up to the highest written key,
// so sparse writes would make its Has/Get semantics diverge from the others.
// Tests exercising an implementation concurrently skip the fake, which is
// deliberately not synchronized (see fakeKVFile).

// forEachKVFile runs the given test against every KVFile implementation --
// each backend alone and wrapped in a KVCachedFile -- identified by name.
// Repeated open calls within one test case reopen the same store, so tests
// can verify persistence across close and reopen. The fake is memory-backed
// and deliberately not synchronized (see fakeKVFile): concurrent tests must
// skip it, while its cached variant guards KVCachedFile's serialization of
// file access when tested with -race.
func forEachKVFile(t *testing.T, fn func(t *testing.T, name string, open func() KVFileWithMemoryFootprint[K, V])) {
	backends := map[string]func(t *testing.T) func() KVFileWithMemoryFootprint[K, V]{
		"offset": func(t *testing.T) func() KVFileWithMemoryFootprint[K, V] {
			path := filepath.Join(t.TempDir(), "kv.dat")
			return func() KVFileWithMemoryFootprint[K, V] {
				file, err := OpenOffsetFile[K, V](path, readTestEntry, writeTestEntry)
				require.NoError(t, err)
				return file
			}
		},
		"ordered": func(t *testing.T) func() KVFileWithMemoryFootprint[K, V] {
			path := filepath.Join(t.TempDir(), "kv.dat")
			return func() KVFileWithMemoryFootprint[K, V] {
				file, err := OpenOrderedFile[V](path, testEntrySize, readTestEntry, writeTestEntry)
				require.NoError(t, err)
				return file
			}
		},
		"fake": func(t *testing.T) func() KVFileWithMemoryFootprint[K, V] {
			fake := newFakeKVFile()
			return func() KVFileWithMemoryFootprint[K, V] { return fake }
		},
	}
	testCases := map[string]func(t *testing.T) func() KVFileWithMemoryFootprint[K, V]{}
	for name, makeOpen := range backends {
		testCases[name] = makeOpen
		testCases["cached-"+name] = func(t *testing.T) func() KVFileWithMemoryFootprint[K, V] {
			open := makeOpen(t)
			return func() KVFileWithMemoryFootprint[K, V] {
				cached, err := OpenKVCachedFile[K, V](open(), cacheSize, flushBufferThreshold)
				require.NoError(t, err)
				return cached
			}
		}
	}
	for name, makeOpen := range testCases {
		t.Run(name, func(t *testing.T) {
			fn(t, name, makeOpen(t))
		})
	}
}

func TestKVFile_Get_ReturnsNilForUnknownKey(t *testing.T) {
	forEachKVFile(t, func(t *testing.T, _ string, open func() KVFileWithMemoryFootprint[K, V]) {
		require := require.New(t)
		file := open()
		defer require.NoError(file.Close())

		got, err := file.Get(0)
		require.NoError(err)
		require.Nil(got)

		require.NoError(file.Set(0, "value-0"))
		got, err = file.Get(1) // beyond every written key
		require.NoError(err)
		require.Nil(got)
	})
}

func TestKVFile_SetAndGet_RoundTripsValues(t *testing.T) {
	forEachKVFile(t, func(t *testing.T, _ string, open func() KVFileWithMemoryFootprint[K, V]) {
		require := require.New(t)
		file := open()
		defer require.NoError(file.Close())

		const numKeys = 5
		for i := range K(numKeys) {
			require.NoError(file.Set(i, fmt.Sprintf("value-%d", i)))
		}
		for i := range K(numKeys) {
			got, err := file.Get(i)
			require.NoError(err)
			require.NotNil(got)
			require.Equal(fmt.Sprintf("value-%d", i), *got)
		}
	})
}

func TestKVFile_Set_OverwritesExistingValue(t *testing.T) {
	forEachKVFile(t, func(t *testing.T, _ string, open func() KVFileWithMemoryFootprint[K, V]) {
		require := require.New(t)
		file := open()
		defer require.NoError(file.Close())

		require.NoError(file.Set(0, "old"))
		require.NoError(file.Set(0, "new"))

		got, err := file.Get(0)
		require.NoError(err)
		require.NotNil(got)
		require.Equal("new", *got)
	})
}

func TestKVFile_Has_ReportsWrittenKeys(t *testing.T) {
	forEachKVFile(t, func(t *testing.T, _ string, open func() KVFileWithMemoryFootprint[K, V]) {
		require := require.New(t)
		file := open()
		defer require.NoError(file.Close())

		has, err := file.Has(0)
		require.NoError(err)
		require.False(has)

		const numKeys = 3
		for i := range K(numKeys) {
			require.NoError(file.Set(i, fmt.Sprintf("value-%d", i)))
		}
		for i := range K(numKeys) {
			has, err = file.Has(i)
			require.NoError(err)
			require.True(has, "key %d", i)
		}
		has, err = file.Has(numKeys)
		require.NoError(err)
		require.False(has)
	})
}

func TestKVFile_SetBatchAndIterate_YieldsAllWrittenEntries(t *testing.T) {
	forEachKVFile(t, func(t *testing.T, _ string, open func() KVFileWithMemoryFootprint[K, V]) {
		require := require.New(t)
		file := open()
		defer require.NoError(file.Close())

		entries := map[K]V{0: "value-0", 1: "value-1", 2: "value-2"}
		require.NoError(file.SetBatch(entries))

		seq, err := file.Iterate()
		require.NoError(err)
		require.Equal(entries, maps.Collect(seq))
	})
}

func TestKVFile_FlushAndClose_PersistValuesAcrossReopen(t *testing.T) {
	forEachKVFile(t, func(t *testing.T, _ string, open func() KVFileWithMemoryFootprint[K, V]) {
		require := require.New(t)
		file := open()

		const numKeys = 3
		for i := range K(numKeys) {
			require.NoError(file.Set(i, fmt.Sprintf("value-%d", i)))
		}
		require.NoError(file.Flush())
		require.NoError(file.Close())

		file = open()
		defer require.NoError(file.Close())
		for i := range K(numKeys) {
			got, err := file.Get(i)
			require.NoError(err)
			require.NotNil(got)
			require.Equal(fmt.Sprintf("value-%d", i), *got)
		}
	})
}

// Concurrent callers must not race, and no write may be lost: Get/Has target
// the same keys as Set, and after a close and reopen every key must hold its
// last written value. For the cached variants this stresses every layer --
// cache, flush buffer, pending queue, and the underlying file.
func TestKVFile_SetGetHasAndFlush_ConcurrentWritesToSharedKeysArePersisted(t *testing.T) {
	forEachKVFile(t, func(t *testing.T, name string, open func() KVFileWithMemoryFootprint[K, V]) {
		if name == "fake" {
			t.Skip("the fake is deliberately not synchronized")
		}
		require := require.New(t)
		file := open()

		const (
			workers      = uint64(8)
			numKeys      = uint64(16)
			opsPerWorker = uint64(2000)
		)
		value := func(key, version uint64) V { return fmt.Sprintf("k%d-v%d", key, version) }

		var keyLocks [numKeys]sync.Mutex
		versions := [numKeys]int{}

		var wg sync.WaitGroup
		for w := range workers {
			wg.Go(func() {
				for i := range opsPerWorker {
					key := (w + i) % numKeys
					switch i % 4 {
					case 0, 1:
						keyLocks[key].Lock()
						versions[key]++
						err := file.Set(key, value(key, uint64(versions[key])))
						keyLocks[key].Unlock()
						require.NoError(err)
					case 2:
						got, err := file.Get(key)
						require.NoError(err)
						// A zero-filled OrderedFile slot decodes as an empty value.
						if got != nil && *got != "" {
							require.True(strings.HasPrefix(*got, fmt.Sprintf("k%d-", key)),
								"value %q does not belong to key %d", *got, key)
						}
					case 3:
						if i%400 == 3 {
							require.NoError(file.Flush())
						} else {
							_, err := file.Has(key)
							require.NoError(err)
						}
					}
				}
			})
		}
		wg.Wait()
		require.NoError(file.Close())

		// Reopen the store: every key must hold its last written value.
		file = open()
		defer require.NoError(file.Close())
		for key := range numKeys {
			got, err := file.Get(key)
			require.NoError(err)
			require.NotNil(got)
			require.Equal(value(key, uint64(versions[key])), *got)
		}
	})
}

func TestKVFile_ConcurrentReadsAndWritesAreSafe(t *testing.T) {
	forEachKVFile(t, func(t *testing.T, name string, open func() KVFileWithMemoryFootprint[K, V]) {
		if name == "fake" {
			t.Skip("the fake is deliberately not synchronized")
		}
		require := require.New(t)
		file := open()
		defer require.NoError(file.Close())

		const numKeys = 8
		const rounds = 1000
		// The uniform tail detects torn reads of in-place overwrites, and the
		// key prefix detects values served for the wrong key.
		value := func(key K, version int) V {
			return fmt.Sprintf("k%d-", key) + strings.Repeat(string(rune('a'+version%26)), 8)
		}
		checkValue := func(key K, got V) {
			require.True(strings.HasPrefix(got, fmt.Sprintf("k%d-", key)),
				"value %q does not belong to key %d", got, key)
			tail := got[len(got)-8:]
			for i := range len(tail) {
				require.Equal(tail[0], tail[i], "torn read: value %q has mixed version bytes", got)
			}
		}

		for key := range K(numKeys) {
			require.NoError(file.Set(key, value(key, 0)))
		}

		var wg sync.WaitGroup
		wg.Go(func() { // overwriting writer
			for i := range rounds {
				key := K(i % numKeys)
				require.NoError(file.Set(key, value(key, i)))
			}
		})
		wg.Go(func() { // appending writer
			for i := range rounds {
				key := K(numKeys + i)
				require.NoError(file.Set(key, value(key, 0)))
			}
		})
		for r := range K(2) {
			wg.Go(func() { // point readers
				for i := range K(rounds) {
					key := (r + i) % numKeys
					got, err := file.Get(key)
					require.NoError(err)
					require.NotNil(got)
					checkValue(key, *got)
				}
			})
		}
		wg.Go(func() { // iterating reader
			for range rounds / 100 {
				seq, err := file.Iterate()
				require.NoError(err)
				for key, got := range seq {
					checkValue(key, got)
				}
			}
		})
		wg.Wait()
	})
}

func TestKVFile_Close_IsSafeWithConcurrentReadersAndCloses(t *testing.T) {
	forEachKVFile(t, func(t *testing.T, name string, open func() KVFileWithMemoryFootprint[K, V]) {
		if strings.Contains(name, "fake") {
			t.Skip("reads from the memory-backed fake never fail, so a close would never surface")
		}
		require := require.New(t)
		file := open()

		const numKeys = 16
		for key := range K(numKeys) {
			require.NoError(file.Set(key, fmt.Sprintf("value-%d", key)))
		}

		var wg sync.WaitGroup
		for r := range K(4) {
			wg.Go(func() { // readers run until the close surfaces
				for i := r; ; i++ {
					key := i % numKeys
					got, err := file.Get(key)
					if err != nil {
						require.ErrorIs(err, os.ErrClosed)
						return
					}
					require.NotNil(got)
					require.Equal(fmt.Sprintf("value-%d", key), *got)
				}
			})
		}
		for range 2 {
			wg.Go(func() {
				require.NoError(file.Close())
			})
		}
		wg.Wait()
	})
}

func TestKVFile_Iterate_YieldsCompleteValuesDuringConcurrentWrites(t *testing.T) {
	forEachKVFile(t, func(t *testing.T, name string, open func() KVFileWithMemoryFootprint[K, V]) {
		if name == "fake" {
			t.Skip("the fake is deliberately not synchronized")
		}
		require := require.New(t)
		file := open()

		const preloaded = 10000
		// The writers stop when the iteration is done, but are also capped so
		// the test always terminates even if the iterator is starved of the
		// file lock by the constant stream of writes.
		const maxWrites = 4 * preloaded
		for i := range K(preloaded) {
			require.NoError(file.Set(i, fmt.Sprintf("k%d-", i)))
		}

		stop := make(chan struct{})
		var wg sync.WaitGroup
		wg.Go(func() { // appends new keys, extending the file during iteration
			for i := range K(maxWrites) {
				select {
				case <-stop:
					return
				default:
				}
				key := preloaded + i
				if err := file.Set(key, fmt.Sprintf("k%d-", key)); err != nil {
					return
				}
			}
		})
		wg.Go(func() { // overwrites preloaded keys with versioned values
			for n := range K(maxWrites) {
				select {
				case <-stop:
					return
				default:
				}
				key := n % preloaded
				if err := file.Set(key, fmt.Sprintf("k%d-v%d", key, n)); err != nil {
					return
				}
			}
		})

		seq, err := file.Iterate()
		require.NoError(err)
		seen := map[K]bool{}
		for key, value := range seq {
			// Overwritten keys may yield any version, but never a value
			// belonging to another key or a torn record.
			require.True(strings.HasPrefix(value, fmt.Sprintf("k%d-", key)),
				"value %q does not belong to key %d", value, key)
			seen[key] = true
		}
		close(stop)
		wg.Wait()

		for i := range K(preloaded) {
			require.True(seen[i], "key %d written before Iterate is missing", i)
		}
		require.NoError(file.Close())
	})
}
