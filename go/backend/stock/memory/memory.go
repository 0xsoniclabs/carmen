// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package memory

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/backend/stock"
	"github.com/0xsoniclabs/carmen/go/backend/utils"
	"github.com/0xsoniclabs/carmen/go/backend/utils/checkpoint"
	"github.com/0xsoniclabs/carmen/go/common"
)

// inMemoryStock provides an in-memory implementation of the stock.Stock interface.
type inMemoryStock[I stock.Index, V any] struct {
	values                          []V
	freeList                        []I
	directory                       string
	encoder                         stock.ValueEncoder[V]
	checkpoint                      checkpoint.Checkpoint
	checkpointValueListLength       I
	checkpointFreeListLength        int
	backupCheckpointValueListLength I
	backupCheckpointFreeListLength  int
}

const (
	fileNameMetadata            = "meta.json"
	fileNameValues              = "values.dat"
	fileNameFreeList            = "freelist.dat"
	fileNameCommittedCheckpoint = "committed.json"
	fileNamePreparedCheckpoint  = "prepare.json"
)

func OpenStock[I stock.Index, V any](encoder stock.ValueEncoder[V], directory string) (stock.Stock[I, V], error) {
	res := &inMemoryStock[I, V]{
		values:    make([]V, 0, 10),
		freeList:  make([]I, 0, 10),
		directory: directory,
		encoder:   encoder,
	}

	// Create the directory if needed.
	if err := os.MkdirAll(directory, 0700); err != nil {
		return nil, err
	}

	// Test whether a meta file exists in this directory.
	metafile := filepath.Join(directory, fileNameMetadata)
	if _, err := os.Stat(metafile); err != nil {
		return res, nil
	}

	// If there are files in the directory, load the data.
	data, err := os.ReadFile(metafile)
	if err != nil {
		return nil, err
	}

	var meta metadata
	if err := json.Unmarshal(data, &meta); err != nil {
		return nil, err
	}

	// Check meta-data format information.
	if meta.Version != dataFormatVersion {
		return nil, fmt.Errorf("invalid file format version, got %d, wanted %d", meta.Version, dataFormatVersion)
	}
	indexSize := int(unsafe.Sizeof(I(0)))
	if meta.IndexTypeSize != indexSize {
		return nil, fmt.Errorf("invalid index type encoding, expected %d byte, found %d", indexSize, meta.IndexTypeSize)
	}
	valueSize := encoder.GetEncodedSize()
	if meta.ValueTypeSize != valueSize {
		return nil, fmt.Errorf("invalid value type encoding, expected %d byte, found %d", valueSize, meta.ValueTypeSize)
	}

	// Load list of values.
	{
		valuefile := filepath.Join(directory, fileNameValues)
		stats, err := os.Stat(valuefile)
		if err != nil {
			return nil, err
		}
		if got, want := stats.Size(), int64(meta.ValueListLength*valueSize); got != want {
			return nil, fmt.Errorf("invalid value file size, got %d, wanted %d", got, want)
		}
		res.values = make([]V, meta.ValueListLength)
		buffer := make([]byte, valueSize)
		file, err := os.Open(valuefile)
		if err != nil {
			return nil, err
		}
		defer file.Close()
		for i := 0; i < meta.ValueListLength; i++ {
			_, err := io.ReadFull(file, buffer)
			if err != nil {
				return nil, err
			}
			if err = encoder.Load(buffer, &res.values[i]); err != nil {
				return nil, err
			}
		}
	}

	// Load freelist.
	{
		freelistfile := filepath.Join(directory, fileNameFreeList)
		stats, err := os.Stat(freelistfile)
		if err != nil {
			return nil, err
		}
		if got, want := stats.Size(), int64(meta.FreeListLength*indexSize); got != want {
			return nil, fmt.Errorf("invalid free-list file size, got %d, wanted %d", got, want)
		}
		res.freeList = make([]I, meta.FreeListLength)
		buffer := make([]byte, indexSize)
		file, err := os.Open(freelistfile)
		if err != nil {
			return nil, err
		}
		defer file.Close()
		for i := 0; i < meta.FreeListLength; i++ {
			_, err := io.ReadFull(file, buffer)
			if err != nil {
				return nil, err
			}
			res.freeList[i] = stock.DecodeIndex[I](buffer)
		}
	}

	// Load last checkpoint.
	res.checkpoint = meta.LastCheckpoint
	res.checkpointValueListLength = I(meta.LastCheckpointValueListLength)
	res.checkpointFreeListLength = meta.LastCheckpointFreeListLength

	return res, nil
}

func (s *inMemoryStock[I, V]) New() (I, error) {
	lenValues := len(s.values)
	index := I(lenValues)

	// Reuse free index positions or grow list of values.
	lenFreeList := len(s.freeList)
	if lenFreeList > s.checkpointFreeListLength {
		index = s.freeList[lenFreeList-1]
		s.freeList = s.freeList[0 : lenFreeList-1]
	} else {
		var value V
		s.values = append(s.values, value)
	}

	return I(index), nil
}

func (s *inMemoryStock[I, V]) Get(index I) (V, error) {
	var res V
	if index >= I(len(s.values)) || index < 0 {
		return res, nil
	}
	return s.values[index], nil
}

func (s *inMemoryStock[I, V]) Set(index I, value V) error {
	if index >= I(len(s.values)) || index < 0 {
		return fmt.Errorf("index out of range, got %d, range [0,%d)", index, I(len(s.values)))
	}
	if index < s.checkpointValueListLength {
		return fmt.Errorf("index %d is read-only", index)
	}
	s.values[index] = value
	return nil
}

func (s *inMemoryStock[I, V]) Delete(index I) error {
	if index >= I(len(s.values)) || index < 0 {
		return nil
	}
	if index < s.checkpointValueListLength {
		return fmt.Errorf("index %d is read-only", index)
	}
	s.freeList = append(s.freeList, index)
	return nil
}

func (s *inMemoryStock[I, V]) GetIds() (stock.IndexSet[I], error) {
	res := stock.MakeComplementSet[I](0, I(len(s.values)))
	for _, i := range s.freeList {
		res.Remove(i)
	}
	return res, nil
}

func (s *inMemoryStock[I, V]) GetMemoryFootprint() *common.MemoryFootprint {
	indexSize := unsafe.Sizeof(I(0))
	valueSize := unsafe.Sizeof(s.values[0])
	res := common.NewMemoryFootprint(unsafe.Sizeof(*s))
	res.AddChild("values", common.NewMemoryFootprint(valueSize*uintptr(cap(s.values))))
	res.AddChild("freelist", common.NewMemoryFootprint(indexSize*uintptr(cap(s.freeList))))
	return res
}

func (s *inMemoryStock[I, V]) Flush() error {
	return s.writeTo(s.directory)
}

func (s *inMemoryStock[I, V]) writeTo(dir string) error {
	// Create the directory if needed.
	if err := os.MkdirAll(dir, 0700); err != nil {
		return err
	}

	// Write metadata.
	var index I
	indexSize := int(unsafe.Sizeof(index))
	metadata, err := json.Marshal(metadata{
		Version:                       dataFormatVersion,
		IndexTypeSize:                 indexSize,
		ValueTypeSize:                 s.encoder.GetEncodedSize(),
		ValueListLength:               len(s.values),
		FreeListLength:                len(s.freeList),
		LastCheckpoint:                s.checkpoint,
		LastCheckpointValueListLength: int(s.checkpointValueListLength),
		LastCheckpointFreeListLength:  int(s.checkpointFreeListLength),
	})
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, fileNameMetadata), metadata, 0600); err != nil {
		return err
	}

	// Write list of values.
	if f, err := os.Create(filepath.Join(dir, fileNameValues)); err != nil {
		return err
	} else {
		defer f.Close()

		buffer := make([]byte, s.encoder.GetEncodedSize())
		for _, v := range s.values {
			s.encoder.Store(buffer, &v)
			_, err := f.Write(buffer)
			if err != nil {
				return err
			}
		}

		if err := f.Close(); err != nil {
			return err
		}
	}

	// Write free list.
	if f, err := os.Create(filepath.Join(dir, fileNameFreeList)); err != nil {
		return err
	} else {
		defer f.Close()

		buffer := make([]byte, indexSize)
		for _, i := range s.freeList {
			stock.EncodeIndex(i, buffer)
			if _, err := f.Write(buffer); err != nil {
				return err
			}
		}

		if err := f.Close(); err != nil {
			return err
		}
	}

	return nil
}

func (s *inMemoryStock[I, V]) Close() error {
	return s.Flush()
}

func (s *inMemoryStock[I, V]) GuaranteeCheckpoint(checkpoint checkpoint.Checkpoint) error {
	// If requested checkpoint is the stock's current checkpoint, everything is fine.
	if s.checkpoint == checkpoint {
		return nil
	}

	// If the requested checkpoint is pending, it needs to be completed.
	if s.checkpoint+1 == checkpoint {
		return s.Commit(checkpoint)
	}

	// Otherwise, the stock is in an inconsistent state.
	return fmt.Errorf("unable to guarantee availability of checkpoint %d", checkpoint)
}

func (s *inMemoryStock[I, V]) Prepare(commit checkpoint.Checkpoint) error {
	if s.checkpoint+1 != commit {
		return fmt.Errorf("unable to prepare checkpoint %d, expected %d", commit, s.checkpoint+1)
	}
	if err := s.Flush(); err != nil {
		return err
	}
	return utils.WriteJsonFile(filepath.Join(s.directory, fileNamePreparedCheckpoint), checkpointData{
		Checkpoint:     commit,
		NumValues:      len(s.values),
		FreeListLength: len(s.freeList),
	})
}

func (s *inMemoryStock[I, V]) Commit(checkpoint checkpoint.Checkpoint) error {
	prepareFile := filepath.Join(s.directory, fileNamePreparedCheckpoint)
	commitFile := filepath.Join(s.directory, fileNameCommittedCheckpoint)
	meta, err := utils.ReadJsonFile[checkpointData](prepareFile)
	if err != nil {
		return err
	}
	if meta.Checkpoint != checkpoint {
		return fmt.Errorf("unable to commit checkpoint %d, expected %d", checkpoint, meta.Checkpoint)
	}
	s.checkpoint = checkpoint
	s.checkpointValueListLength = I(meta.NumValues)
	s.checkpointFreeListLength = meta.FreeListLength
	return os.Rename(prepareFile, commitFile)
}

func (s *inMemoryStock[I, V]) Abort(commit checkpoint.Checkpoint) error {
	if s.checkpoint+1 != commit {
		return fmt.Errorf("unable to abort checkpoint %d, expected %d", commit, s.checkpoint+1)
	}
	s.checkpointValueListLength = s.backupCheckpointValueListLength
	s.checkpointFreeListLength = s.backupCheckpointFreeListLength
	prepareFile := filepath.Join(s.directory, fileNamePreparedCheckpoint)
	return os.Remove(prepareFile)
}

const dataFormatVersion = 0

// metadata is the helper type to read and write metadata from/to the disk.
type metadata struct {
	Version                       int
	IndexTypeSize                 int
	ValueTypeSize                 int
	ValueListLength               int
	FreeListLength                int
	LastCheckpoint                checkpoint.Checkpoint
	LastCheckpointValueListLength int
	LastCheckpointFreeListLength  int
}

type checkpointData struct {
	Checkpoint     checkpoint.Checkpoint
	NumValues      int
	FreeListLength int
}
