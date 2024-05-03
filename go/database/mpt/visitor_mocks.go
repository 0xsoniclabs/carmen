//
// Copyright (c) 2024 Fantom Foundation
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at fantom.foundation/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use
// of this software will be governed by the GNU Lesser General Public License v3.
//

// Code generated by MockGen. DO NOT EDIT.
// Source: visitor.go
//
// Generated by this command:
//
//	mockgen -source visitor.go -destination visitor_mocks.go -package mpt
//

// Package mpt is a generated GoMock package.
package mpt

import (
	reflect "reflect"

	gomock "go.uber.org/mock/gomock"
)

// MockNodeVisitor is a mock of NodeVisitor interface.
type MockNodeVisitor struct {
	ctrl     *gomock.Controller
	recorder *MockNodeVisitorMockRecorder
}

// MockNodeVisitorMockRecorder is the mock recorder for MockNodeVisitor.
type MockNodeVisitorMockRecorder struct {
	mock *MockNodeVisitor
}

// NewMockNodeVisitor creates a new mock instance.
func NewMockNodeVisitor(ctrl *gomock.Controller) *MockNodeVisitor {
	mock := &MockNodeVisitor{ctrl: ctrl}
	mock.recorder = &MockNodeVisitorMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockNodeVisitor) EXPECT() *MockNodeVisitorMockRecorder {
	return m.recorder
}

// Visit mocks base method.
func (m *MockNodeVisitor) Visit(arg0 Node, arg1 NodeInfo) VisitResponse {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Visit", arg0, arg1)
	ret0, _ := ret[0].(VisitResponse)
	return ret0
}

// Visit indicates an expected call of Visit.
func (mr *MockNodeVisitorMockRecorder) Visit(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Visit", reflect.TypeOf((*MockNodeVisitor)(nil).Visit), arg0, arg1)
}
