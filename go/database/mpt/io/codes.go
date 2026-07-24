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
func writeCodes(codes iter.Seq2[common.Hash, []byte], out io.Writer) (retErr error) {
	dir, err := os.MkdirTemp("", "carmen-export-codes-*")
	if err != nil {
		return fmt.Errorf("failed to create temp dir: %w", err)
	}
	defer func() {
		if err := os.RemoveAll(dir); err != nil {
			retErr = errors.Join(retErr, fmt.Errorf("failed to remove temp dir: %w", err))
		}
	}()

	db, err := pebble.Open(dir, &pebble.Options{})
	if err != nil {
		return fmt.Errorf("failed to open temp pebble store: %w", err)
	}
	defer func() {
		if err := db.Close(); err != nil {
			retErr = errors.Join(retErr, fmt.Errorf("failed to close temp pebble store: %w", err))
		}
	}()

	for hash, code := range codes {
		if err := db.Set(hash[:], code, pebble.NoSync); err != nil {
			return fmt.Errorf("failed to persist code: %w", err)
		}
	}

	it, err := db.NewIter(nil)
	if err != nil {
		return fmt.Errorf("failed to open temp pebble iterator: %w", err)
	}
	defer func() {
		if err := it.Close(); err != nil {
			retErr = errors.Join(retErr, fmt.Errorf("failed to close temp pebble iterator: %w", err))
		}
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
	return nil
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
