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
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"maps"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/stretchr/testify/require"
)

func TestWriteCodes_WritesTagLengthAndBytesForEachCode(t *testing.T) {
	require := require.New(t)

	codes := map[common.Hash][]byte{
		{1}: {1, 2, 3},
		{2}: {4, 5},
		{3}: {6, 7, 8},
	}

	var buf bytes.Buffer
	require.NoError(writeCodes(maps.All(codes), &buf))

	got := make(map[string][]byte, len(codes))
	in := bytes.NewReader(buf.Bytes())
	for range codes {
		tag, err := in.ReadByte()
		require.NoError(err)
		require.Equal(byte('C'), tag)
		code, err := readCode(in)
		require.NoError(err)
		got[string(code)] = code
	}
	_, err := in.ReadByte()
	require.ErrorIs(err, io.EOF)

	want := make(map[string][]byte, len(codes))
	for _, code := range codes {
		want[string(code)] = code
	}
	require.Equal(want, got)
}

func TestWriteCodes_WritesNothingForEmptyIterator(t *testing.T) {
	require := require.New(t)

	var buf bytes.Buffer
	require.NoError(writeCodes(maps.All(map[common.Hash][]byte{}), &buf))
	require.Zero(buf.Len())
}

func TestWriteCodes_PropagatesWriteError(t *testing.T) {
	injected := errors.New("write failed")
	tests := map[string]struct {
		failAfter int
	}{
		"on header write": {failAfter: 0},
		"on body write":   {failAfter: 1},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)

			w := &failingWriter{failAfter: test.failAfter, err: injected}
			codes := map[common.Hash][]byte{{1}: {1, 2, 3}}

			err := writeCodes(maps.All(codes), w)
			require.ErrorContains(err, injected.Error())
		})
	}
}

func TestReadCode_ReturnsCodeWrittenByWriteCodes(t *testing.T) {
	require := require.New(t)

	want := []byte{9, 8, 7, 6}
	codes := map[common.Hash][]byte{{1}: want}

	var buf bytes.Buffer
	require.NoError(writeCodes(maps.All(codes), &buf))

	tag, err := buf.ReadByte()
	require.NoError(err)
	require.Equal(byte('C'), tag)

	got, err := readCode(&buf)
	require.NoError(err)
	require.Equal(want, got)
}

func TestReadCode_ReturnsEmptyCodeForZeroLength(t *testing.T) {
	require := require.New(t)

	code, err := readCode(bytes.NewReader([]byte{0, 0}))
	require.NoError(err)
	require.Empty(code)
}

func TestReadCode_ReturnsErrorOnTruncatedInput(t *testing.T) {
	// Header claims 4 bytes of body but only 2 are provided.
	truncatedBody := []byte{0, 0, 1, 2}
	binary.BigEndian.PutUint16(truncatedBody, 4)

	tests := map[string]struct {
		input []byte
	}{
		"short length header": {input: []byte{0}},
		"short code body":     {input: truncatedBody},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)

			_, err := readCode(bytes.NewReader(test.input))
			require.ErrorIs(err, io.ErrUnexpectedEOF)
		})
	}
}

// failingWriter is an io.Writer that succeeds for the first failAfter calls
// and then returns err on every subsequent call.
type failingWriter struct {
	failAfter int
	calls     int
	err       error
}

func (w *failingWriter) Write(p []byte) (int, error) {
	if w.calls >= w.failAfter {
		return 0, w.err
	}
	w.calls++
	return len(p), nil
}
