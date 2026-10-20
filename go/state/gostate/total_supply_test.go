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

	"github.com/Fantom-foundation/Carmen/go/common"
	"github.com/Fantom-foundation/Carmen/go/common/amount"
	"github.com/Fantom-foundation/Carmen/go/database/mpt"
	"github.com/Fantom-foundation/Carmen/go/state"
	"go.uber.org/mock/gomock"
)

func TestGoState_TotalSupply_S5SumsBalances(t *testing.T) {
	mptState, err := mpt.OpenGoMemoryState(t.TempDir(), mpt.S5LiveConfig, mpt.NodeCacheConfig{Capacity: 1024})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := mptState.SetBalance(common.Address{1}, amount.New(5)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := mptState.SetBalance(common.Address{2}, amount.New(7)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	s := newGoState(&goSchema5{MptState: mptState}, nil, nil)
	defer s.Close()

	total, err := s.TotalSupply()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if amount.New(12) != total {
		t.Fatalf("want %v, got %v", amount.New(12), total)
	}
	if err := s.Check(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestGoState_TotalSupply_LiveDBWithoutSupportIsNotSupported(t *testing.T) {
	ctrl := gomock.NewController(t)
	live := state.NewMockLiveDB(ctrl)
	live.EXPECT().Flush()
	live.EXPECT().Close()

	s := newGoState(live, nil, nil)
	defer s.Close()

	_, err := s.TotalSupply()
	if !errors.Is(err, state.TotalSupplyNotSupported) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestGoState_TotalSupply_PreviousStateErrorIsReported(t *testing.T) {
	ctrl := gomock.NewController(t)
	live := state.NewMockLiveDB(ctrl)

	issue := errors.New("test issue")
	goState := &GoState{live: live}
	goState.stateError = issue

	_, err := goState.TotalSupply()
	if !errors.Is(err, issue) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestArchiveState_TotalSupply_IsNotSupported(t *testing.T) {
	_, err := (&ArchiveState{}).TotalSupply()
	if !errors.Is(err, state.TotalSupplyNotSupported) {
		t.Fatalf("unexpected error: %v", err)
	}
}
