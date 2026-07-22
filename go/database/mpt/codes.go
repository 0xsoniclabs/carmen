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
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"io"
	"os"
	"path/filepath"
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/backend/kv_file"
	"github.com/0xsoniclabs/carmen/go/backend/utils"
	"github.com/0xsoniclabs/carmen/go/backend/utils/checkpoint"
	"github.com/0xsoniclabs/carmen/go/common"
	"golang.org/x/crypto/sha3"
)

// codes is a data structure to store and manage the codes of accounts.
// Codes are persisted on disk via an OffsetFile and served through a
// KVCachedFile in-memory cache; only recently accessed codes are kept in
// memory, while the remainder are fetched from disk on demand. Checkpoint
// and flush operations incrementally persist buffered writes to disk.
type codes struct {
	codes  kv_file.KVFileWithMemoryFootprint[common.Hash, []byte]
	hasher hash.Hash

	directory  string                // < a directory for placing checkpoint data
	checkpoint checkpoint.Checkpoint // < the last checkpoint
}

var emptyCodeHash = common.GetHash(sha3.NewLegacyKeccak256(), []byte{})

const (
	fileNameCodes                    = "codes.dat"
	fileNameCodesCheckpointDirectory = "codes"
	fileNameCodesCommittedCheckpoint = "committed.json"
	fileNameCodesPrepareCheckpoint   = "prepare.json"
	cacheSize                        = 20000
	flushBufferThreshold             = 2000
)

func openCodes(stateDirectory string) (*codes, error) {
	file, directory := getCodePaths(stateDirectory)
	if err := os.MkdirAll(directory, 0700); err != nil {
		return nil, err
	}

	codeFile, err := kv_file.OpenOffsetFile(file, readCode, writeCode)
	if err != nil {
		return nil, err
	}
	storedCodes, err := kv_file.OpenKVCachedFile[common.Hash, []byte](codeFile, cacheSize, flushBufferThreshold)
	if err != nil {
		return nil, errors.Join(err, codeFile.Close())
	}

	committed := filepath.Join(directory, fileNameCodesCommittedCheckpoint)
	meta, err := readCodeCheckpointMetaData(committed)
	if err != nil {
		return nil, errors.Join(err, storedCodes.Close())
	}

	return &codes{
		codes:      storedCodes,
		directory:  directory,
		hasher:     sha3.NewLegacyKeccak256(),
		checkpoint: meta.Checkpoint,
	}, nil
}

func (c *codes) add(code []byte) (common.Hash, error) {
	hash := common.GetHash(c.hasher, code)
	// Codes are content-addressed and immutable, so a known code must not be
	// re-written: re-setting it would append a duplicate record to the
	// underlying append-only file on the next flush.
	has, err := c.codes.Has(hash)
	if err != nil {
		return common.Hash{}, err
	}
	if has {
		return hash, nil
	}
	if err := c.codes.Set(hash, code); err != nil {
		return common.Hash{}, err
	}
	return hash, nil
}

func (c *codes) getCodeForHash(hash common.Hash) ([]byte, error) {
	code, err := c.codes.Get(hash)
	if err != nil {
		return []byte{}, err
	}
	if code == nil {
		return nil, nil
	}
	return *code, nil
}

func (c *codes) getCodes() map[common.Hash][]byte {
	// Preserve the historical non-nil contract so that callers can safely
	// write into the returned map even after a read failure.
	codes := map[common.Hash][]byte{}
	seq, err := c.codes.Iterate()
	if err != nil {
		return codes
	}
	for key, code := range seq {
		codes[key] = code
	}
	return codes
}

func (c *codes) Flush() error {
	return c.codes.Flush()
}

// Close releases the resources backing the code store.
func (c *codes) Close() error {
	return c.codes.Close()
}

func (c *codes) GetMemoryFootprint() *common.MemoryFootprint {
	codes := c.codes.GetMemoryFootprint()
	return common.NewMemoryFootprint(unsafe.Sizeof(*c) + codes.Total())
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
	fileSize, err := c.codes.FileSize()
	if err != nil {
		return err
	}
	preparedFile := filepath.Join(c.directory, fileNameCodesPrepareCheckpoint)
	return writeCodeCheckpointMetaData(preparedFile, codeCheckpointMetaData{
		Checkpoint: checkpoint,
		FileSize:   fileSize,
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
func readCodes(path string) (codes map[common.Hash][]byte, err error) {
	codeFile, err := kv_file.OpenOffsetFile(path, readCode, writeCode)
	if err != nil {
		return nil, err
	}
	defer func() {
		err = errors.Join(err, codeFile.Close())
	}()
	seq, err := codeFile.Iterate()
	if err != nil {
		return nil, err
	}
	codes = map[common.Hash][]byte{}
	for key, code := range seq {
		codes[key] = code
	}
	return codes, nil
}

// writeCodes writes the given map of codes to the given file.
func writeCodes(codes map[common.Hash][]byte, path string) (err error) {
	err = os.Remove(path)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	codeFile, err := kv_file.OpenOffsetFile(path, readCode, writeCode)
	if err != nil {
		return err
	}
	storedCodes, err := kv_file.OpenKVCachedFile[common.Hash, []byte](codeFile, cacheSize, flushBufferThreshold)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(err, storedCodes.Close())
	}()
	for hash, code := range codes {
		err := storedCodes.Set(hash, code)
		if err != nil {
			return err
		}
	}
	err = storedCodes.Flush()
	return err
}

func writeCode(out io.Writer, hash common.Hash, code []byte) (err error) {
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

func readCode(reader io.Reader) (common.Hash, []byte, error) {
	var hash common.Hash
	var length [4]byte
	if _, err := io.ReadFull(reader, hash[:]); err != nil {
		return common.Hash{}, nil, err
	}
	if _, err := io.ReadFull(reader, length[:]); err != nil {
		return common.Hash{}, nil, err
	}
	size := binary.BigEndian.Uint32(length[:])
	code := make([]byte, size)
	if _, err := io.ReadFull(reader, code[:]); err != nil {
		return common.Hash{}, nil, err
	}
	return hash, code, nil
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
