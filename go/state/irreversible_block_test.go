// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package state_test

import (
	"errors"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/stretchr/testify/require"
)

func TestIrreversibleBlock_StateHash_ReportsTheHashOfTheState(t *testing.T) {
	require := require.New(t)
	want := common.Hash{0x42}

	block := state.NewIrreversibleBlock(1, func() common.Hash { return want }, nil)

	require.Equal(want, block.StateHash())
}

func TestIrreversibleBlock_Commit_HasNothingLeftToDo(t *testing.T) {
	require := require.New(t)

	block := state.NewIrreversibleBlock(1, func() common.Hash { return common.Hash{} }, nil)

	require.NoError(block.Commit())
}

func TestIrreversibleBlock_Wait_ReportsTheOutcomeOfTheAsynchronousWork(t *testing.T) {
	injected := errors.New("injected error")
	tests := map[string]struct {
		done func() chan error
		want error
	}{
		"no asynchronous work": {
			done: func() chan error { return nil },
		},
		"work succeeded": {
			done: func() chan error {
				done := make(chan error, 1)
				close(done)
				return done
			},
		},
		"work failed": {
			done: func() chan error {
				done := make(chan error, 1)
				done <- injected
				return done
			},
			want: injected,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			done := test.done()

			var block state.StagedBlock
			if done == nil {
				// A nil channel must be passed as a nil receive-only channel, not as
				// a non-nil interface wrapping one.
				block = state.NewIrreversibleBlock(1, func() common.Hash { return common.Hash{} }, nil)
			} else {
				block = state.NewIrreversibleBlock(1, func() common.Hash { return common.Hash{} }, done)
			}

			require.NoError(block.Commit())
			if test.want == nil {
				require.NoError(block.Wait())
			} else {
				require.ErrorIs(block.Wait(), test.want)
			}
		})
	}
}

func TestIrreversibleBlock_Rollback_IsRejected(t *testing.T) {
	require := require.New(t)

	block := state.NewIrreversibleBlock(7, func() common.Hash { return common.Hash{} }, nil)

	err := block.Rollback()
	require.ErrorIs(err, state.ErrStagedBlockMisuse)
	require.ErrorContains(err, "block 7")
}
