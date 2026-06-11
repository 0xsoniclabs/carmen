package state

import (
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
)

func TestIsEmptyAccount(t *testing.T) {
	testCases := map[string]struct {
		set_expectations func(s *MockState)
		expected         bool
	}{
		"empty account": {
			set_expectations: func(s *MockState) {
				s.EXPECT().GetBalance(gomock.Any()).Return(amount.New(0), nil)
				s.EXPECT().GetNonce(gomock.Any()).Return(common.Nonce{}, nil)
				s.EXPECT().GetCode(gomock.Any()).Return([]byte{}, nil)
			},
			expected: true,
		},
		"non-empty balance": {
			set_expectations: func(s *MockState) {
				s.EXPECT().GetBalance(gomock.Any()).Return(amount.New(100), nil)
				s.EXPECT().GetNonce(gomock.Any()).Return(common.Nonce{}, nil)
				s.EXPECT().GetCode(gomock.Any()).Return([]byte{}, nil)
			},
			expected: false,
		},
		"non-empty nonce": {
			set_expectations: func(s *MockState) {
				s.EXPECT().GetBalance(gomock.Any()).Return(amount.New(0), nil)
				s.EXPECT().GetNonce(gomock.Any()).Return(common.Nonce{1}, nil)
				s.EXPECT().GetCode(gomock.Any()).Return([]byte{}, nil)
			},
			expected: false,
		},
		"non-empty code": {
			set_expectations: func(s *MockState) {
				s.EXPECT().GetBalance(gomock.Any()).Return(amount.New(0), nil)
				s.EXPECT().GetNonce(gomock.Any()).Return(common.Nonce{}, nil)
				s.EXPECT().GetCode(gomock.Any()).Return([]byte{0x60, 0x00}, nil) // PUSH1 0x00
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
