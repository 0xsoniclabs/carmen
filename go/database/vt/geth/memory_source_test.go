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
	"github.com/ethereum/go-ethereum/common"
	"testing"
)

func TestMemorySource_SetAndGetNode(t *testing.T) {
	src := newMemorySource()
	path := []byte{1, 2, 3}
	value := []byte{4, 5, 6}
	owner := common.Hash{}
	hash := common.Hash{}

	// Initially, Node should return nil
	got, err := src.Node(owner, path, hash)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != nil {
		t.Errorf("expected nil, got %v", got)
	}

	// Set value and retrieve
	if err := src.set(path, value); err != nil {
		t.Fatalf("set failed: %v", err)
	}
	got, err = src.Node(owner, path, hash)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(got) != string(value) {
		t.Errorf("expected %v, got %v", value, got)
	}
}

func TestMemorySource_FlushAndClose(t *testing.T) {
	src := newMemorySource()
	if err := src.Flush(); err != nil {
		t.Errorf("Flush should not error, got %v", err)
	}
	if err := src.Close(); err != nil {
		t.Errorf("Close should not error, got %v", err)
	}
}
