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
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/stretchr/testify/require"
)

func TestAccountInfo_EncodingAndDecoding(t *testing.T) {
	infos := []AccountInfo{
		{},
		{common.Nonce{1, 2, 3}, amount.New(456), common.Hash{7, 8, 9}},
	}

	encoder := AccountInfoEncoder{}
	buffer := make([]byte, encoder.GetEncodedSize())
	for _, info := range infos {
		encoder.Store(buffer[:], &info)
		restored := AccountInfo{}
		if encoder.Load(buffer[:], &restored); restored != info {
			t.Fatalf("failed to decode info %v: got %v", info, restored)
		}
	}
}

func TestAccountInfo_IsEmpty(t *testing.T) {
	testCases := map[string]struct {
		info     AccountInfo
		expected bool
	}{
		"empty": {
			info:     AccountInfo{},
			expected: true,
		},
		"non-empty": {
			info:     AccountInfo{common.Nonce{1, 2, 3}, amount.New(456), common.Hash{7, 8, 9}},
			expected: false,
		},
		"empty code hash": {
			info:     AccountInfo{CodeHash: emptyCodeHash},
			expected: true,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			require.Equal(t, tc.expected, tc.info.IsEmpty())
		})
	}
}
