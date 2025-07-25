// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package geth

import (
	"testing"

	"github.com/ethereum/go-ethereum/common"
	"github.com/stretchr/testify/require"
)

func TestMemorySource_SetAndGetNode(t *testing.T) {
	src := newMemorySource()
	path := []byte{1, 2, 3}
	value := []byte{4, 5, 6}
	owner := common.Hash{}
	hash := common.Hash{}

	// Initially, Node should return nil
	got, err := src.Node(owner, path, hash)
	require.NoError(t, err, "unexpected error")
	require.Nil(t, got, "expected nil")

	// Set value and retrieve
	require.NoError(t, src.set(path, value), "set failed")
	got, err = src.Node(owner, path, hash)
	require.NoError(t, err, "unexpected error")
	require.Equal(t, value, got, "unexpected value")
}

func TestMemorySource_FlushAndClose(t *testing.T) {
	src := newMemorySource()
	require.NoError(t, src.Flush(), "Flush should not error")
	require.NoError(t, src.Close(), "Close should not error")
}
