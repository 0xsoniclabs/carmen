// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package state

import (
	"errors"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
)

func TestSyncedState_Apply_WrapsTheStagedBlock(t *testing.T) {
	ctrl := gomock.NewController(t)
	inner := NewMockState(ctrl)
	staged := NewMockStagedBlock(ctrl)
	inner.EXPECT().Apply(uint64(1), common.Update{}).Return(staged, nil)

	block, err := WrapIntoSyncedState(inner).Apply(1, common.Update{})
	require.NoError(t, err)
	wrapped, ok := block.(*syncedStagedBlock)
	require.True(t, ok, "the staged block must be wrapped")
	require.Same(t, staged, wrapped.block)
}

func TestSyncedState_Apply_ForwardsAFailure(t *testing.T) {
	ctrl := gomock.NewController(t)
	inner := NewMockState(ctrl)
	injected := errors.New("injected")
	inner.EXPECT().Apply(uint64(1), common.Update{}).Return(nil, injected)

	block, err := WrapIntoSyncedState(inner).Apply(1, common.Update{})
	require.ErrorIs(t, err, injected)
	require.Nil(t, block)
}

func TestSyncedStagedBlock_ForwardsTheDecisions(t *testing.T) {
	ctrl := gomock.NewController(t)
	staged := NewMockStagedBlock(ctrl)
	hash := common.Hash{1}
	done := NewWaitHandle(nil)
	injected := errors.New("injected")
	staged.EXPECT().StateHash().Return(hash)
	staged.EXPECT().Commit().Return(done, nil)
	staged.EXPECT().Rollback().Return(injected)

	block := &syncedStagedBlock{block: staged, mu: &WrapIntoSyncedState(NewMockState(ctrl)).(*syncedState).mu}
	require.Equal(t, hash, block.StateHash())
	got, err := block.Commit()
	require.NoError(t, err)
	require.Same(t, done, got)
	require.ErrorIs(t, block.Rollback(), injected)
}

func TestSyncedStagedBlock_DecidesUnderTheStateLock(t *testing.T) {
	for _, tc := range []struct {
		name   string
		expect func(*MockStagedBlock, func())
		decide func(StagedBlock)
	}{
		{"StateHash",
			func(m *MockStagedBlock, check func()) {
				m.EXPECT().StateHash().DoAndReturn(func() common.Hash { check(); return common.Hash{} })
			},
			func(b StagedBlock) { b.StateHash() }},
		{"Commit",
			func(m *MockStagedBlock, check func()) {
				m.EXPECT().Commit().DoAndReturn(func() (*WaitHandle, error) { check(); return nil, nil })
			},
			func(b StagedBlock) { _, _ = b.Commit() }},
		{"Rollback",
			func(m *MockStagedBlock, check func()) {
				m.EXPECT().Rollback().DoAndReturn(func() error { check(); return nil })
			},
			func(b StagedBlock) { _ = b.Rollback() }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctrl := gomock.NewController(t)
			inner := NewMockState(ctrl)
			staged := NewMockStagedBlock(ctrl)
			inner.EXPECT().Apply(uint64(1), common.Update{}).Return(staged, nil)

			synced := WrapIntoSyncedState(inner).(*syncedState)
			tc.expect(staged, func() {
				if synced.mu.TryLock() {
					synced.mu.Unlock()
					t.Error("the decision must run under the state lock")
				}
			})

			block, err := synced.Apply(1, common.Update{})
			require.NoError(t, err)
			tc.decide(block)
		})
	}
}
