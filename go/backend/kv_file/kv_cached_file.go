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
	"slices"
	"sync"
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/common"
)

// KVCachedFile wraps a KVFile with an in-memory write-back cache. Buffered
// writes are persisted asynchronously by a background writer; Flush, Iterate
// and Close wait for those writes to complete, while a threshold-triggered
// write happens in the background. A background write failure is terminal and
// is reported by the next operation.
type KVCachedFile[K comparable, V any] struct {
	cache       *common.LruCache[K, V]
	flushBuffer map[K]V
	dirty       map[K]bool
	file        KVFileWithMemoryFootprint[K, V]

	// fileMu serializes every access to file, so the background writer and a
	// foreground operation never touch the wrapped KVFile at the same time
	// (KVFile is not required to be safe for concurrent use). It is independent
	// of mu: the writer performs its disk I/O holding only fileMu, so foreground
	// operations that stay in memory (cache/buffer hits, Set) proceed under mu
	// meanwhile, and only those that must reach the file wait on fileMu.
	//
	// Lock order is mu then fileMu: a foreground caller may take fileMu while
	// holding mu, but the writer takes fileMu on its own and only re-acquires mu
	// after releasing it, so the two never nest the other way and cannot deadlock.
	fileMu sync.Mutex

	flushBufferThreshold int
	// maxPendingFlushes bounds the number of sealed buffers queued for the
	// background writer, providing back-pressure that keeps memory bounded.
	maxPendingFlushes int

	// mu guards all mutable state below and is the locker of cond.
	mu           sync.Mutex
	cond         *sync.Cond    // signalled whenever pending, writeErr or closed change
	pending      []map[K]V     // sealed buffers awaiting a durable write (FIFO)
	writeErr     error         // sticky error from the background writer
	closed       bool          // set to stop the background writer
	writerDone   chan struct{} // closed when the writer goroutine has exited
	shutdownOnce sync.Once     // makes writer shutdown idempotent
}

// OpenKVCachedFile wraps the given KVFile with a cache of the specified size and a flush buffer threshold.
func OpenKVCachedFile[K comparable, V any](file KVFileWithMemoryFootprint[K, V], cacheSize int, flushBufferThreshold int) (*KVCachedFile[K, V], error) {
	if file == nil {
		return nil, fmt.Errorf("file cannot be nil")
	}

	if cacheSize <= 0 || flushBufferThreshold <= 0 {
		return nil, fmt.Errorf("cacheSize and flushBufferThreshold must be greater than 0, got %d and %d", cacheSize, flushBufferThreshold)
	}

	c := &KVCachedFile[K, V]{
		cache:                common.NewLruCache[K, V](cacheSize),
		flushBuffer:          make(map[K]V),
		dirty:                make(map[K]bool),
		file:                 file,
		flushBufferThreshold: flushBufferThreshold,
		maxPendingFlushes:    flushBufferThreshold + 1,
		writerDone:           make(chan struct{}),
	}
	c.cond = sync.NewCond(&c.mu)
	go c.flushWorker()

	return c, nil
}

// Get retrieves the value stored for a key, or nil if it does not exist.
func (c *KVCachedFile[K, V]) Get(key K) (*V, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.writeErr != nil {
		return nil, c.writeErr
	}
	if val, inCache := c.cache.Get(key); inCache {
		return &val, nil
	}
	if val, inFlushBuffer := c.flushBuffer[key]; inFlushBuffer {
		delete(c.flushBuffer, key)
		if err := c.handleCacheSet(&key, &val, true); err != nil {
			return nil, err
		}
		return &val, nil
	}
	// Consult buffers handed to the background writer but not yet on disk,
	// newest first. These buffers are owned by the writer and must not be
	// mutated, but their keys were marked clean when sealed, so the value can
	// be promoted into the cache as a clean entry: if it is later evicted it
	// will not be re-written, and the pending buffer stays its source of truth
	// until the writer persists it.
	for _, v := range slices.Backward(c.pending) {
		if val, ok := v[key]; ok {
			if err := c.handleCacheSet(&key, &val, false); err != nil {
				return nil, err
			}
			return &val, nil
		}
	}
	c.fileMu.Lock()
	value, err := c.file.Get(key)
	c.fileMu.Unlock()
	if err != nil {
		return nil, err
	}
	if value == nil {
		return nil, nil
	}
	err = c.handleCacheSet(&key, value, false)
	if err != nil {
		return nil, err
	}

	return value, nil
}

// Has reports whether a value is stored for the given key. Unlike Get, it does
// not load the value into the cache.
func (c *KVCachedFile[K, V]) Has(key K) (bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.writeErr != nil {
		return false, c.writeErr
	}
	if _, inCache := c.cache.Get(key); inCache {
		return true, nil
	}
	if _, inFlushBuffer := c.flushBuffer[key]; inFlushBuffer {
		return true, nil
	}
	for _, v := range slices.Backward(c.pending) {
		if _, ok := v[key]; ok {
			return true, nil
		}
	}
	c.fileMu.Lock()
	has, err := c.file.Has(key)
	c.fileMu.Unlock()
	return has, err
}

// Set stores a value for the given key.
func (c *KVCachedFile[K, V]) Set(key K, value V) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.writeErr != nil {
		return c.writeErr
	}
	return c.handleCacheSet(&key, &value, true)
}

// SetBatch stores multiple key-value pairs.
func (c *KVCachedFile[K, V]) SetBatch(entries map[K]V) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.writeErr != nil {
		return c.writeErr
	}

	for key, value := range entries {
		if err := c.handleCacheSet(&key, &value, true); err != nil {
			return err
		}
	}

	return nil
}

// Flush persists all buffered writes to disk before returning.
func (c *KVCachedFile[K, V]) Flush() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.drainLocked()
}

func (c *KVCachedFile[K, V]) FileSize() (uint64, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.writeErr != nil {
		return 0, c.writeErr
	}
	c.fileMu.Lock()
	size, err := c.file.FileSize()
	c.fileMu.Unlock()
	return size, err
}

// Iterate returns an iterator over all stored key-value pairs.
func (c *KVCachedFile[K, V]) Iterate() (iter.Seq2[K, V], error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Flush any pending writes (and wait for in-flight ones) so the underlying
	// file is up-to-date before iterating.
	if err := c.drainLocked(); err != nil {
		return nil, err
	}

	// After draining, every key/value pair lives on disk, so we can delegate
	// to the underlying file's iterator.
	c.fileMu.Lock()
	seq, err := c.file.Iterate()
	c.fileMu.Unlock()
	return seq, err
}

func (c *KVCachedFile[K, V]) Close() error {
	c.mu.Lock()
	err := c.drainLocked()
	c.mu.Unlock()

	// Stop the background writer regardless of whether the drain succeeded, so
	// its goroutine never outlives the cached file.
	c.shutdownWriter()

	// The writer has stopped, so fileMu is  uncontended here;
	// it is still taken to keep "every file access happens
	// under fileMu" a true invariant.
	c.fileMu.Lock()
	defer c.fileMu.Unlock()
	return errors.Join(err, c.file.Close())
}

func (c *KVCachedFile[K, V]) GetMemoryFootprint() *common.MemoryFootprint {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.fileMu.Lock()
	fileFootprint := c.file.GetMemoryFootprint()
	c.fileMu.Unlock()
	valueSize := func(v V) uintptr {
		switch vv := any(v).(type) {
		case []byte:
			return uintptr(cap(vv))
		case string:
			return uintptr(len(vv))
		default:
			return uintptr(unsafe.Sizeof(v))
		}
	}
	cacheFootprint := c.cache.GetDynamicMemoryFootprint(valueSize)
	var flushBufferSize uintptr
	for k, v := range c.flushBuffer {
		flushBufferSize += unsafe.Sizeof(k) + valueSize(v)
	}
	var pendingSize uintptr
	for _, buf := range c.pending {
		for k, v := range buf {
			pendingSize += unsafe.Sizeof(k) + valueSize(v)
		}
	}
	var dirtySize uintptr
	for k := range c.dirty {
		dirtySize += unsafe.Sizeof(k) + unsafe.Sizeof(true)
	}
	var pending uintptr
	for _, buf := range c.pending {
		pending += unsafe.Sizeof(buf)
	}
	footprint := common.NewMemoryFootprint(unsafe.Sizeof(*c))
	footprint.AddChild("file", fileFootprint)
	footprint.AddChild("cache", cacheFootprint)
	footprint.AddChild("flushBuffer", common.NewMemoryFootprint(flushBufferSize))
	footprint.AddChild("pending", common.NewMemoryFootprint(pendingSize+pending))
	footprint.AddChild("dirty", common.NewMemoryFootprint(dirtySize))
	return footprint
}

// drainLocked persists all buffered writes and waits for them to complete,
// returning the sticky write error if any. The caller must hold c.mu.
func (c *KVCachedFile[K, V]) drainLocked() error {
	if c.writeErr != nil {
		return c.writeErr
	}
	c.cache.Iterate(func(key K, value V) bool {
		if _, isDirty := c.dirty[key]; isDirty {
			c.flushBuffer[key] = value
		}
		return true
	})
	c.enqueueCurrentBufferLocked()
	for len(c.pending) > 0 && c.writeErr == nil {
		c.cond.Wait()
	}
	return c.writeErr
}

// enqueueCurrentBufferLocked hands the current flush buffer to the background
// writer and starts a fresh one. The caller must hold c.mu.
func (c *KVCachedFile[K, V]) enqueueCurrentBufferLocked() {
	if len(c.flushBuffer) == 0 {
		return
	}
	// Block while the queue is full to bound memory (back-pressure).
	for len(c.pending) >= c.maxPendingFlushes && c.writeErr == nil && !c.closed {
		c.cond.Wait()
	}
	if c.writeErr != nil || c.closed {
		return
	}
	// The buffered values are now committed to the background writer and will
	// be persisted, so mark them clean immediately. A subsequent Set of any of
	// these keys re-dirties it and is tracked in a later buffer; because
	// buffers are written in FIFO order, the newer value wins on disk and no
	// update is lost. (Clearing after the write completes instead would let a
	// re-Set-while-in-flight be wrongly marked clean and dropped.)
	for k := range c.flushBuffer {
		delete(c.dirty, k)
	}
	c.pending = append(c.pending, c.flushBuffer)
	c.flushBuffer = make(map[K]V)
	c.cond.Broadcast()
}

// flushWorker is the background goroutine that persists queued buffers to disk
// in FIFO order. On a write error it records the error as terminal and parks
// until the file is closed, retaining the unwritten buffers.
func (c *KVCachedFile[K, V]) flushWorker() {
	defer close(c.writerDone)
	c.mu.Lock()
	defer c.mu.Unlock()
	for {
		for len(c.pending) == 0 && !c.closed {
			c.cond.Wait()
		}
		if c.closed {
			return
		}
		if c.writeErr != nil {
			// Terminal error: keep the pending buffers readable and park
			// until Close wakes us.
			c.cond.Wait()
			continue
		}

		// Peek the oldest buffer but leave it in the queue so concurrent
		// readers can still find its entries while it is being written.
		buf := c.pending[0]
		c.mu.Unlock()
		err := c.writeBuffer(buf)
		c.mu.Lock()

		if err != nil {
			c.writeErr = err
			c.cond.Broadcast()
			continue
		}
		// The dirty flags for these keys were already cleared when the buffer
		// was sealed, so the writer only needs to drop the buffer from the
		// queue.
		c.pending = c.pending[1:]
		c.cond.Broadcast()
	}
}

// writeBuffer persists a single buffer to the underlying file. It is called
// without holding c.mu so in-memory foreground operations are not blocked on
// disk I/O; it holds fileMu instead, serializing against foreground operations
// that reach the file (see the fileMu field).
func (c *KVCachedFile[K, V]) writeBuffer(buf map[K]V) error {
	c.fileMu.Lock()
	defer c.fileMu.Unlock()
	if err := c.file.SetBatch(buf); err != nil {
		return err
	}
	return c.file.Flush()
}

// shutdownWriter stops the background writer and waits for it to exit. It is
// idempotent and performs no file I/O.
func (c *KVCachedFile[K, V]) shutdownWriter() {
	c.shutdownOnce.Do(func() {
		c.mu.Lock()
		c.closed = true
		c.cond.Broadcast()
		c.mu.Unlock()
		<-c.writerDone
	})
}

// handleCacheSet inserts a key-value pair into the cache, optionally marking it
// dirty. The caller must hold c.mu.
func (c *KVCachedFile[K, V]) handleCacheSet(key *K, value *V, dirty bool) error {
	if key == nil || value == nil {
		return nil // No-op
	}
	if dirty {
		c.dirty[*key] = true
		// Drop any stale copy of the key from the un-sealed flush buffer: the
		// cache now holds the newest value, tracked by its dirty flag. Keeping
		// the stale entry would let a later seal clear that flag and silently
		// drop the newer value.
		delete(c.flushBuffer, *key)
	}
	// An evicted dirty entry moves into the flush buffer; reaching the
	// threshold seals the buffer for the background writer.
	evictedKey, evictedValue, evicted := c.cache.Set(*key, *value)
	if evicted {
		if _, isDirty := c.dirty[evictedKey]; isDirty {
			c.flushBuffer[evictedKey] = evictedValue
			if len(c.flushBuffer) >= c.flushBufferThreshold {
				c.enqueueCurrentBufferLocked()
			}
		}
	}
	return nil
}
