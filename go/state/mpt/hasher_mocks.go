// Code generated by MockGen. DO NOT EDIT.
// Source: hasher.go

// Package mpt is a generated GoMock package.
package mpt

import (
	reflect "reflect"

	common "github.com/Fantom-foundation/Carmen/go/common"
	gomock "github.com/golang/mock/gomock"
)

// MockHasher is a mock of Hasher interface.
type MockHasher struct {
	ctrl     *gomock.Controller
	recorder *MockHasherMockRecorder
}

// MockHasherMockRecorder is the mock recorder for MockHasher.
type MockHasherMockRecorder struct {
	mock *MockHasher
}

// NewMockHasher creates a new mock instance.
func NewMockHasher(ctrl *gomock.Controller) *MockHasher {
	mock := &MockHasher{ctrl: ctrl}
	mock.recorder = &MockHasherMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockHasher) EXPECT() *MockHasherMockRecorder {
	return m.recorder
}

// GetHash mocks base method.
func (m *MockHasher) GetHash(arg0 Node, arg1 NodeSource, arg2 HashSource) (common.Hash, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetHash", arg0, arg1, arg2)
	ret0, _ := ret[0].(common.Hash)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetHash indicates an expected call of GetHash.
func (mr *MockHasherMockRecorder) GetHash(arg0, arg1, arg2 interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetHash", reflect.TypeOf((*MockHasher)(nil).GetHash), arg0, arg1, arg2)
}

// MockHashSource is a mock of HashSource interface.
type MockHashSource struct {
	ctrl     *gomock.Controller
	recorder *MockHashSourceMockRecorder
}

// MockHashSourceMockRecorder is the mock recorder for MockHashSource.
type MockHashSourceMockRecorder struct {
	mock *MockHashSource
}

// NewMockHashSource creates a new mock instance.
func NewMockHashSource(ctrl *gomock.Controller) *MockHashSource {
	mock := &MockHashSource{ctrl: ctrl}
	mock.recorder = &MockHashSourceMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockHashSource) EXPECT() *MockHashSourceMockRecorder {
	return m.recorder
}

// getHashFor mocks base method.
func (m *MockHashSource) getHashFor(arg0 NodeId) (common.Hash, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "getHashFor", arg0)
	ret0, _ := ret[0].(common.Hash)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// getHashFor indicates an expected call of getHashFor.
func (mr *MockHashSourceMockRecorder) getHashFor(arg0 interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "getHashFor", reflect.TypeOf((*MockHashSource)(nil).getHashFor), arg0)
}