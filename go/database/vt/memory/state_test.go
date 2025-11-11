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
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/stretchr/testify/require"
)

func TestState_CanStoreAndRestoreNonces(t *testing.T) {
	require := require.New(t)

	state, err := NewState(state.Parameters{})
	require.NoError(err)

	address := common.Address{1}

	// Initially, the nonce should be zero
	nonce, err := state.GetNonce(address)
	require.NoError(err)
	require.Equal(common.ToNonce(0), nonce)

	// Set a nonce
	require.NoError(state.Apply(0, common.Update{
		Nonces: []common.NonceUpdate{{
			Account: address,
			Nonce:   common.ToNonce(42),
		}},
	}))

	// Retrieve the nonce again
	nonce, err = state.GetNonce(address)
	require.NoError(err)
	require.Equal(common.ToNonce(42), nonce)

	// Set another nonce
	require.NoError(state.Apply(0, common.Update{
		Nonces: []common.NonceUpdate{{
			Account: address,
			Nonce:   common.ToNonce(123),
		}},
	}))

	// Retrieve the updated nonce
	nonce, err = state.GetNonce(address)
	require.NoError(err)
	require.Equal(common.ToNonce(123), nonce)
}
