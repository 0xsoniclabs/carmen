// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package geth2

import (
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/stretchr/testify/require"
)

var _ state.State = (*verkleState)(nil)

func TestState_ContentIsStoredPersistent(t *testing.T) {
	modes := map[string]state.Parameters{
		"path-based": {Archive: state.NoArchive},
		"archive":    {Archive: state.LevelDbArchive},
	}

	for name, params := range modes {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			dir := t.TempDir()

			params.Directory = dir
			s1, err := NewState(params)
			require.NoError(err)

			_, err = s1.Apply(0, common.Update{
				Nonces: []common.NonceUpdate{
					{Account: common.Address{1}, Nonce: common.ToNonce(12)},
				},
			})
			require.NoError(err)

			nonce, err := s1.GetNonce(common.Address{1})
			require.NoError(err)
			require.Equal(common.ToNonce(12), nonce)

			require.NoError(s1.Close())

			// Reopen
			s2, err := NewState(params)
			require.NoError(err)

			nonce, err = s2.GetNonce(common.Address{1})
			require.NoError(err)
			require.Equal(common.ToNonce(12), nonce)

			require.NoError(s2.Close())
		})
	}
}

func TestState_CanStoreAndRecoverCodes(t *testing.T) {
	require := require.New(t)

	params := state.Parameters{
		Directory: t.TempDir(),
		Archive:   state.NoArchive,
	}
	s1, err := NewState(params)
	require.NoError(err)

	addr := common.Address{1}
	code := make([]byte, 1<<16) // 64KB code
	for i := range code {
		code[i] = byte(i % 256)
	}

	_, err = s1.Apply(1, common.Update{
		Codes: []common.CodeUpdate{
			{Account: addr, Code: code},
		},
	})
	require.NoError(err)

	codeLength, err := s1.GetCodeSize(addr)
	require.NoError(err)
	require.Equal(len(code), codeLength)

	codeHash, err := s1.GetCodeHash(addr)
	require.NoError(err)
	require.Equal(common.Keccak256(code), codeHash)

	restored, err := s1.GetCode(addr)
	require.NoError(err)
	require.Equal(code, restored)

	require.NoError(s1.Close())

	// Reopen
	s2, err := NewState(params)
	require.NoError(err)

	codeLength, err = s2.GetCodeSize(addr)
	require.NoError(err)
	require.Equal(len(code), codeLength)

	codeHash, err = s2.GetCodeHash(addr)
	require.NoError(err)
	require.Equal(common.Keccak256(code), codeHash)

	restored, err = s2.GetCode(addr)
	require.NoError(err)
	require.Equal(code, restored)

	require.NoError(s2.Close())
}
