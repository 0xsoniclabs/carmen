// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

// Code generated by MockGen. DO NOT EDIT.
// Source: ticker.go
//
// Generated by this command:
//
//	mockgen -source ticker.go -destination ticker_mocks.go -package ticker
//
// Package ticker is a generated GoMock package.
package ticker

import (
	reflect "reflect"
	time "time"

	gomock "go.uber.org/mock/gomock"
)

// MockTicker is a mock of Ticker interface.
type MockTicker struct {
	ctrl     *gomock.Controller
	recorder *MockTickerMockRecorder
}

// MockTickerMockRecorder is the mock recorder for MockTicker.
type MockTickerMockRecorder struct {
	mock *MockTicker
}

// NewMockTicker creates a new mock instance.
func NewMockTicker(ctrl *gomock.Controller) *MockTicker {
	mock := &MockTicker{ctrl: ctrl}
	mock.recorder = &MockTickerMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockTicker) EXPECT() *MockTickerMockRecorder {
	return m.recorder
}

// C mocks base method.
func (m *MockTicker) C() <-chan time.Time {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "C")
	ret0, _ := ret[0].(<-chan time.Time)
	return ret0
}

// C indicates an expected call of C.
func (mr *MockTickerMockRecorder) C() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "C", reflect.TypeOf((*MockTicker)(nil).C))
}

// Stop mocks base method.
func (m *MockTicker) Stop() {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "Stop")
}

// Stop indicates an expected call of Stop.
func (mr *MockTickerMockRecorder) Stop() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Stop", reflect.TypeOf((*MockTicker)(nil).Stop))
}
