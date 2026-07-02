// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package mpt

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"io"
	"os"
	"path/filepath"
	"sync"
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/backend/utils"
	"github.com/0xsoniclabs/carmen/go/backend/utils/checkpoint"
	"github.com/0xsoniclabs/carmen/go/common"
	"golang.org/x/crypto/sha3"
)

// codes is a simple data structure to store and manage the codes of accounts.
// All codes are retained in memory, incrementally backed up to disk during
// checkpoint and flush operations.
type codes struct {
	cache    *common.LruCache[common.Hash, []byte] // < a cache for the most recently used codes
	codes    map[common.Hash]uint64                // < all managed code offsets on disk
	pending  map[common.Hash][]byte                // < codes not yet written to disk
	file     string                                // < the file to store the codes
	fileSize uint64                                // < the current file size
	mutex    sync.Mutex
	hasher   hash.Hash

	directory  string                // < a directory for placing checkpoint data
	checkpoint checkpoint.Checkpoint // < the last checkpoint
}

var emptyCodeHash = common.GetHash(sha3.NewLegacyKeccak256(), []byte{})

const (
	fileNameCodes                    = "codes.dat"
	fileNameCodesCheckpointDirectory = "codes"
	fileNameCodesCommittedCheckpoint = "committed.json"
	fileNameCodesPrepareCheckpoint   = "prepare.json"
	pendingFlushThreshold            = 10_000
)

func openCodes(stateDirectory string) (*codes, error) {
	file, directory := getCodePaths(stateDirectory)
	if err := os.MkdirAll(directory, 0700); err != nil {
		return nil, err
	}

	// Create the code file if it does not exist.
	if _, err := os.Stat(file); os.IsNotExist(err) {
		if err := os.WriteFile(file, []byte{}, 0600); err != nil {
			return nil, err
		}
	}

	data, size, err := readCodeOffsetsAndSize(file)
	if err != nil {
		return nil, err
	}

	committed := filepath.Join(directory, fileNameCodesCommittedCheckpoint)
	meta, err := readCodeCheckpointMetaData(committed)
	if err != nil {
		return nil, err
	}

	return &codes{
		cache:      common.NewLruCache[common.Hash, []byte](100_000), // TODO: make this configurable
		codes:      data,
		pending:    make(map[common.Hash][]byte),
		file:       file,
		fileSize:   size,
		directory:  directory,
		hasher:     sha3.NewLegacyKeccak256(),
		checkpoint: meta.Checkpoint,
	}, nil
}

func (c *codes) add(code []byte) common.Hash {
	hash := common.GetHash(c.hasher, code)
	c.mutex.Lock()
	if _, onDisk := c.codes[hash]; !onDisk {
		if _, inPending := c.pending[hash]; !inPending {
			c.handleCacheSet(hash, code)
		}
	}
	c.mutex.Unlock()
	return hash
}

func (c *codes) getCodeForHash(hash common.Hash) []byte {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	// Check cache first
	if code, found := c.cache.Get(hash); found {
		return code
	}
	// Check pending (may have been evicted from cache but not yet flushed)
	if code, found := c.pending[hash]; found {
		delete(c.pending, hash)
		c.handleCacheSet(hash, code)
		return code
	}
	// Fall back to disk
	offset, onDisk := c.codes[hash]
	if !onDisk {
		return nil
	}
	code, err := c.readCodeFromDisk(offset)
	if err != nil {
		return nil
	}
	c.handleCacheSet(hash, code)
	return code
}

// handleCacheSet inserts a key/value into the cache and moves evicted entries
// to pending if they are not already persisted on disk. If pending grows above
// the threshold it is flushed.
// Must be called with c.mutex held.
func (c *codes) handleCacheSet(key common.Hash, value []byte) {
	evictedKey, evictedValue, evicted := c.cache.Set(key, value)
	if evicted {
		if _, onDisk := c.codes[evictedKey]; !onDisk {
			c.pending[evictedKey] = evictedValue
		}
	}
	if len(c.pending) >= pendingFlushThreshold {
		c.flushPending()
	}
}

// readCodeFromDisk reads a code at the given offset from the codes file.
func (c *codes) readCodeFromDisk(offset uint64) ([]byte, error) {
	f, err := os.Open(c.file)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	// Skip the hash (32 bytes) and read the length (4 bytes) and code.
	if _, err := f.Seek(int64(offset+32), io.SeekStart); err != nil {
		return nil, err
	}
	var length [4]byte
	if _, err := io.ReadFull(f, length[:]); err != nil {
		return nil, err
	}
	size := binary.BigEndian.Uint32(length[:])
	code := make([]byte, size)
	if _, err := io.ReadFull(f, code); err != nil {
		return nil, err
	}
	return code, nil
}

func (c *codes) Flush() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	// Move cache entries not yet on disk into pending before flushing.
	c.cache.Iterate(func(key common.Hash, value []byte) bool {
		if _, onDisk := c.codes[key]; !onDisk {
			c.pending[key] = value
		}
		return true
	})
	return c.flushPending()
}

// flushPending writes all pending codes to disk. Must be called with c.mutex held.
func (c *codes) flushPending() error {
	if len(c.pending) == 0 {
		return nil
	}
	size, err := appendCodes(c.pending, c.file, c.codes)
	if err != nil {
		return err
	}
	c.fileSize = size
	c.pending = make(map[common.Hash][]byte)
	return nil
}

func (c *codes) GetMemoryFootprint() *common.MemoryFootprint {
	var sizeCodes uint
	c.mutex.Lock()
	for k, v := range c.codes {
		sizeCodes += uint(uint(len(k)) + uint(unsafe.Sizeof(v)))
	}
	mf := c.cache.GetDynamicMemoryFootprint(func(v []byte) uintptr {
		return uintptr(len(v))
	})
	// Pending
	for k, v := range c.pending {
		sizeCodes += uint(len(k) + len(v))
	}
	c.mutex.Unlock()
	return common.NewMemoryFootprint(unsafe.Sizeof(*c) + uintptr(sizeCodes) + mf.Total())
}

func (c *codes) GuaranteeCheckpoint(checkpoint checkpoint.Checkpoint) error {
	if c.checkpoint == checkpoint {
		return nil
	}

	if c.checkpoint+1 == checkpoint {
		preparedFile := filepath.Join(c.directory, fileNameCodesPrepareCheckpoint)
		meta, err := readCodeCheckpointMetaData(preparedFile)
		if err != nil {
			return err
		}
		if meta.Checkpoint == checkpoint {
			return c.Commit(checkpoint)
		}
	}

	return fmt.Errorf("cannot guarantee checkpoint %d, current checkpoint is %d", checkpoint, c.checkpoint)
}

func (c *codes) Prepare(checkpoint checkpoint.Checkpoint) error {
	if c.checkpoint+1 != checkpoint {
		return fmt.Errorf("cannot prepare checkpoint %d, current checkpoint is %d", checkpoint, c.checkpoint)
	}
	if err := c.Flush(); err != nil {
		return err
	}
	preparedFile := filepath.Join(c.directory, fileNameCodesPrepareCheckpoint)
	return writeCodeCheckpointMetaData(preparedFile, codeCheckpointMetaData{
		Checkpoint: checkpoint,
		FileSize:   c.fileSize,
	})
}

func (c *codes) Commit(checkpoint checkpoint.Checkpoint) error {
	committedFile := filepath.Join(c.directory, fileNameCodesCommittedCheckpoint)
	preparedFile := filepath.Join(c.directory, fileNameCodesPrepareCheckpoint)
	meta, err := readCodeCheckpointMetaData(preparedFile)
	if err != nil {
		return err
	}
	if meta.Checkpoint != checkpoint {
		return fmt.Errorf("cannot commit checkpoint %d, prepared checkpoint is %d", checkpoint, meta.Checkpoint)
	}
	if err := os.Rename(preparedFile, committedFile); err != nil {
		return err
	}
	c.checkpoint = checkpoint
	return nil
}

func (c *codes) Abort(checkpoint checkpoint.Checkpoint) error {
	return os.Remove(filepath.Join(c.directory, fileNameCodesPrepareCheckpoint))
}

func getCodePaths(directory string) (codeFile, codeDir string) {
	return filepath.Join(directory, fileNameCodes),
		filepath.Join(directory, fileNameCodesCheckpointDirectory)
}

type codeRestorer struct {
	file      string
	directory string
}

func getCodeRestorer(stateDirectory string) codeRestorer {
	file, directory := getCodePaths(stateDirectory)
	return codeRestorer{
		file:      file,
		directory: directory,
	}
}

func (r codeRestorer) Restore(checkpoint checkpoint.Checkpoint) error {
	committedFile := filepath.Join(r.directory, fileNameCodesCommittedCheckpoint)
	meta, err := readCodeCheckpointMetaData(committedFile)
	if err != nil {
		return err
	}

	// If the given checkpoint is one step in the future, check whether there is a pending checkpoint.
	if meta.Checkpoint+1 == checkpoint {
		pending, err := readCodeCheckpointMetaData(filepath.Join(r.directory, fileNameCodesPrepareCheckpoint))
		if err == nil && pending.Checkpoint == checkpoint {
			meta = pending
		}
	}

	if meta.Checkpoint != checkpoint {
		return fmt.Errorf("cannot restore checkpoint %d, committed checkpoint is %d", checkpoint, meta.Checkpoint)
	}
	return os.Truncate(r.file, int64(meta.FileSize))
}

// readCodes parses the content of the given file if it exists or returns
// a an empty code collection if there is no such file.
func readCodes(path string) (_ map[common.Hash][]byte, retErr error) {
	file, err := os.Open(path)
	if os.IsNotExist(err) {
		return map[common.Hash][]byte{}, nil
	}
	if err != nil {
		return nil, err
	}
	defer func() { retErr = errors.Join(retErr, file.Close()) }()

	return readCodesFromReader(file)
}

func readCodesFromReader(reader io.Reader) (_ map[common.Hash][]byte, retErr error) {
	codes := map[common.Hash][]byte{}
	for {
		hash, code, err := readCode(reader)
		if err != nil {
			if err == io.EOF {
				return codes, nil
			}
			return nil, err
		}
		codes[hash] = code
	}
}

// readCodeOffsetsAndSize parses the content of the given file and returns the
// contained collection of code offsets and the size of the file.
func readCodeOffsetsAndSize(path string) (_ map[common.Hash]uint64, _ uint64, retErr error) {
	// If there is no file, initialize and return an empty code collection.
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return map[common.Hash]uint64{}, 0, nil
	}
	if err != nil {
		return nil, 0, err
	}

	file, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer func() { retErr = errors.Join(retErr, file.Close()) }()
	data, err := parseCodes(file)
	return data, uint64(info.Size()), err
}

func parseCodes(reader io.ReadSeeker) (map[common.Hash]uint64, error) {
	res := map[common.Hash]uint64{}
	for {
		offset, err := reader.Seek(0, io.SeekCurrent)
		if err != nil {
			return nil, err
		}
		hash, _, err := readCode(reader)
		if err != nil {
			if err == io.EOF {
				return res, nil
			}
			return nil, err
		}
		res[hash] = uint64(offset)
	}
}

// readCode reads a single code entry from the reader, returning the hash and
// the code bytes. It advances the reader past the entry.
// Returns io.EOF when there are no more entries.
func readCode(reader io.Reader) (common.Hash, []byte, error) {
	var hash common.Hash
	if _, err := io.ReadFull(reader, hash[:]); err != nil {
		return hash, nil, err
	}
	var length [4]byte
	if _, err := io.ReadFull(reader, length[:]); err != nil {
		return hash, nil, err
	}
	size := binary.BigEndian.Uint32(length[:])
	code := make([]byte, size)
	if _, err := io.ReadFull(reader, code); err != nil {
		return hash, nil, err
	}
	return hash, code, nil
}

// appendCodes appends the given map of codes to the given file.
// It updates the offsets map with the new offsets of the codes in the file.
func appendCodes(codes map[common.Hash][]byte, filename string, offsets map[common.Hash]uint64) (fileSize uint64, err error) {
	file, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return 0, err
	}
	// Get the current end-of-file position as the starting offset.
	offset, err := file.Seek(0, io.SeekEnd)
	if err != nil {
		return 0, errors.Join(err, file.Close())
	}
	fileSize = uint64(offset)
	buffer := bufio.NewWriter(file)
	for hash, code := range codes {
		err1 := writeCode(hash, code, buffer)
		if err1 != nil {
			return 0, errors.Join(err1, file.Close())
		}
		offsets[hash] = fileSize
		fileSize += uint64(len(code)) + 4 + 32 // 4 bytes for length, 32 bytes for hash
	}
	err2 := buffer.Flush()
	return fileSize, errors.Join(err2, file.Close())
}

// writeCode writes a single code entry to the given writer in the format:
// [<hash>, <length>, <code>]
func writeCode(hash common.Hash, code []byte, out io.Writer) (err error) {
	// The format is simple: [<key>, <length>, <code>]*
	if _, err := out.Write(hash[:]); err != nil {
		return err
	}
	var length [4]byte
	binary.BigEndian.PutUint32(length[:], uint32(len(code)))
	if _, err := out.Write(length[:]); err != nil {
		return err
	}
	if _, err := out.Write(code); err != nil {
		return err
	}
	return nil
}

// writeCodes write the given map of codes to the given file.
func writeCodes(codes map[common.Hash][]byte, filename string) (err error) {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	buffer := bufio.NewWriter(file)
	for hash, code := range codes {
		if err := writeCode(hash, code, buffer); err != nil {
			return errors.Join(err, file.Close())
		}
	}
	return errors.Join(
		buffer.Flush(),
		file.Close(),
	)
}

// getCodes returns a map of all codes.
func (c *codes) getCodes() (map[common.Hash][]byte, error) {
	c.mutex.Lock()
	res := map[common.Hash][]byte{}
	// Get values from the cache
	c.cache.Iterate(func(h common.Hash, b []byte) bool {
		res[h] = b
		return true
	})
	// Get values from pending buffer
	for h, b := range c.pending {
		res[h] = b
	}
	// Get values from disk
	diskCodes, err := readCodes(c.file)
	if err != nil {
		c.mutex.Unlock()
		return map[common.Hash][]byte{}, err
	}
	for h, b := range diskCodes {
		res[h] = b
	}

	c.mutex.Unlock()
	return res, nil
}

type codeCheckpointMetaData struct {
	Checkpoint checkpoint.Checkpoint
	FileSize   uint64
}

func readCodeCheckpointMetaData(path string) (codeCheckpointMetaData, error) {
	_, err := os.Stat(path)
	if os.IsNotExist(err) {
		return codeCheckpointMetaData{}, nil
	}
	return utils.ReadJsonFile[codeCheckpointMetaData](path)
}

func writeCodeCheckpointMetaData(path string, meta codeCheckpointMetaData) error {
	return utils.WriteJsonFile(path, meta)
}
