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
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
)

func TestIsEmptyAccount_BehavesCorrectly(t *testing.T) {
	testCases := map[string]struct {
		set_expectations func(s *MockState)
		expected         bool
	}{
		"empty account": {
			set_expectations: func(s *MockState) {
				s.EXPECT().GetBalance(gomock.Any()).Return(amount.New(0), nil)
				s.EXPECT().GetNonce(gomock.Any()).Return(common.Nonce{}, nil)
				s.EXPECT().GetCodeSize(gomock.Any()).Return(0, nil)
			},
			expected: true,
		},
		"non-empty balance": {
			set_expectations: func(s *MockState) {
				s.EXPECT().GetBalance(gomock.Any()).Return(amount.New(100), nil)
				s.EXPECT().GetNonce(gomock.Any()).Return(common.Nonce{}, nil)
				s.EXPECT().GetCodeSize(gomock.Any()).Return(0, nil)
			},
			expected: false,
		},
		"non-empty nonce": {
			set_expectations: func(s *MockState) {
				s.EXPECT().GetBalance(gomock.Any()).Return(amount.New(0), nil)
				s.EXPECT().GetNonce(gomock.Any()).Return(common.Nonce{1}, nil)
				s.EXPECT().GetCodeSize(gomock.Any()).Return(0, nil)
			},
			expected: false,
		},
		"non-empty code": {
			set_expectations: func(s *MockState) {
				s.EXPECT().GetBalance(gomock.Any()).Return(amount.New(0), nil)
				s.EXPECT().GetNonce(gomock.Any()).Return(common.Nonce{}, nil)
				s.EXPECT().GetCodeSize(gomock.Any()).Return(2, nil) // PUSH1 0x00
			},
			expected: false,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			ctrl := gomock.NewController(t)
			defer ctrl.Finish()

			mockState := NewMockState(ctrl)
			tc.set_expectations(mockState)

			result, err := IsEmptyAccount(mockState, common.Address{})
			require.NoError(err)
			require.Equal(tc.expected, result)
		})
	}
}

func TestIsEmptyAccount_PropagatesErrors(t *testing.T) {
	errTest := errors.New("test error")
	testCases := map[string]struct {
		set_expectations func(s *MockState)
	}{
		"GetBalance error": {
			set_expectations: func(s *MockState) {
				s.EXPECT().GetBalance(gomock.Any()).Return(amount.New(0), errTest)
			},
		},
		"GetNonce error": {
			set_expectations: func(s *MockState) {
				s.EXPECT().GetBalance(gomock.Any()).Return(amount.New(0), nil)
				s.EXPECT().GetNonce(gomock.Any()).Return(common.Nonce{}, errTest)
			},
		},
		"GetCodeSize error": {
			set_expectations: func(s *MockState) {
				s.EXPECT().GetBalance(gomock.Any()).Return(amount.New(0), nil)
				s.EXPECT().GetNonce(gomock.Any()).Return(common.Nonce{}, nil)
				s.EXPECT().GetCodeSize(gomock.Any()).Return(0, errTest)
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			ctrl := gomock.NewController(t)
			defer ctrl.Finish()

			mockState := NewMockState(ctrl)
			tc.set_expectations(mockState)

			result, err := IsEmptyAccount(mockState, common.Address{})
			require.ErrorIs(err, errTest)
			require.False(result)
		})
	}
}
