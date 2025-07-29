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
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/ethereum/go-ethereum/triedb/database"

	ethcommon "github.com/ethereum/go-ethereum/common"
)

// nodeSource is an interface for a source of verkle nodes.
// It provides methods to get and set nodes at specific paths.
// It supports the adaptation for Geth's Verkle trie implementation.
// adopted from Geth implementation.
type nodeSource interface {
	common.FlushAndCloser
	database.NodeReader

	// Set sets the node at the given path.
	// The input is navigation path in tree and the serialised node.
	set(path []byte, value []byte) error
}

// singleNodeReader is a wrapper around a single NodeReader.
// When the method NodeReader is called, it returns always the same NodeReader.
type singleNodeReader struct {
	source nodeSource
}

func (r singleNodeReader) NodeReader(stateRoot ethcommon.Hash) (database.NodeReader, error) {
	return r.source, nil
}

// get is a convenience method to retrieve the underlying NodeSource.
func (r singleNodeReader) getSource() nodeSource {
	return r.source
}
