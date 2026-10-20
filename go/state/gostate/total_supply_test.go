// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package gostate

import (
	"errors"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/database/mpt"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
)

func TestGoState_TotalSupply_S5SumsBalances(t *testing.T) {
	mptState, err := mpt.OpenGoMemoryState(t.TempDir(), mpt.S5LiveConfig, mpt.NodeCacheConfig{Capacity: 1024})
	require.NoError(t, err)
	require.NoError(t, mptState.SetBalance(common.Address{1}, amount.New(5)))
	require.NoError(t, mptState.SetBalance(common.Address{2}, amount.New(7)))

	s := newGoState(&goSchema5{MptState: mptState}, nil, nil)
	defer s.Close()

	total, err := s.TotalSupply()
	require.NoError(t, err)
	require.Equal(t, amount.New(12), total)
	require.NoError(t, s.Check())
}

func TestGoState_TotalSupply_LiveDBWithoutSupportIsNotSupported(t *testing.T) {
	ctrl := gomock.NewController(t)
	live := state.NewMockLiveDB(ctrl)
	live.EXPECT().Flush()
	live.EXPECT().Close()

	s := newGoState(live, nil, nil)
	defer s.Close()

	_, err := s.TotalSupply()
	require.ErrorIs(t, err, state.TotalSupplyNotSupported)
}

func TestGoState_TotalSupply_PreviousStateErrorIsReported(t *testing.T) {
	ctrl := gomock.NewController(t)
	live := state.NewMockLiveDB(ctrl)

	issue := errors.New("test issue")
	goState := &GoState{live: live}
	goState.addStateError(issue)

	_, err := goState.TotalSupply()
	require.ErrorIs(t, err, issue)
}

func TestArchiveState_TotalSupply_IsNotSupported(t *testing.T) {
	_, err := (&ArchiveState{}).TotalSupply()
	require.ErrorIs(t, err, state.TotalSupplyNotSupported)
}
