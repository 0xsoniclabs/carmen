// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package io

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"iter"
	"os"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/cockroachdb/pebble"
)

// writeCodes serialises all codes yielded by the iterator to out, ordered by
// hash.
func writeCodes(codes iter.Seq2[common.Hash, []byte], out io.Writer, scratchDir string) (retErr error) {
	store, err := newCodeSortStore(scratchDir)
	if err != nil {
		return err
	}
	defer func() {
		retErr = errors.Join(retErr, store.close())
	}()

	for hash, code := range codes {
		if err := store.add(hash, code); err != nil {
			return err
		}
	}
	return store.writeTo(out)
}

func readCode(in io.Reader) ([]byte, error) {
	length := []byte{0, 0}
	if _, err := io.ReadFull(in, length[:]); err != nil {
		return nil, err
	}
	code := make([]byte, binary.BigEndian.Uint16(length))
	if _, err := io.ReadFull(in, code); err != nil {
		return nil, err
	}
	return code, nil
}

// codeSortStore is a disk-backed staging area collecting codes to be written
// out ordered by hash, keeping the full code set out of memory. Duplicated
// additions of the same hash are stored only once.
type codeSortStore struct {
	db  *pebble.DB
	dir string
}

// newCodeSortStore creates a store in a uniquely named directory under
// scratchDir; an empty scratchDir falls back to the system temp directory.
// The directory is removed again by close.
func newCodeSortStore(scratchDir string) (*codeSortStore, error) {
	dir, err := os.MkdirTemp(scratchDir, "carmen-export-codes-*")
	if err != nil {
		return nil, fmt.Errorf("failed to create staging dir: %w", err)
	}
	// The store is discarded after the export, so crash safety through the
	// write-ahead log is not needed.
	db, err := pebble.Open(dir, &pebble.Options{DisableWAL: true})
	if err != nil {
		return nil, errors.Join(
			fmt.Errorf("failed to open staging store: %w", err),
			os.RemoveAll(dir),
		)
	}
	return &codeSortStore{db: db, dir: dir}, nil
}

func (s *codeSortStore) add(hash common.Hash, code []byte) error {
	if err := s.db.Set(hash[:], code, pebble.NoSync); err != nil {
		return fmt.Errorf("failed to stage code: %w", err)
	}
	return nil
}

// writeTo serialises all staged codes to out, ordered by hash.
func (s *codeSortStore) writeTo(out io.Writer) (retErr error) {
	it, err := s.db.NewIter(nil)
	if err != nil {
		return fmt.Errorf("failed to open staging store iterator: %w", err)
	}
	defer func() {
		retErr = errors.Join(retErr, it.Close())
	}()

	for it.First(); it.Valid(); it.Next() {
		code := it.Value()
		b := []byte{byte('C'), 0, 0}
		binary.BigEndian.PutUint16(b[1:], uint16(len(code)))
		if _, err := out.Write(b); err != nil {
			return fmt.Errorf("output error: %v", err)
		}
		if _, err := out.Write(code); err != nil {
			return fmt.Errorf("output error: %v", err)
		}
	}
	if err := it.Error(); err != nil {
		return fmt.Errorf("failed to read staged codes: %w", err)
	}
	return nil
}

// close releases the store and removes its staging directory.
func (s *codeSortStore) close() error {
	return errors.Join(
		s.db.Close(),
		os.RemoveAll(s.dir),
	)
}
