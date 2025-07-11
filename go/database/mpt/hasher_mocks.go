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
// Source: hasher.go
//
// Generated by this command:
//
//	mockgen -source hasher.go -destination hasher_mocks.go -package mpt
//

// Package mpt is a generated GoMock package.
package mpt

import (
	reflect "reflect"

	common "github.com/0xsoniclabs/carmen/go/common"
	gomock "go.uber.org/mock/gomock"
)

// Mockhasher is a mock of hasher interface.
type Mockhasher struct {
	ctrl     *gomock.Controller
	recorder *MockhasherMockRecorder
	isgomock struct{}
}

// MockhasherMockRecorder is the mock recorder for Mockhasher.
type MockhasherMockRecorder struct {
	mock *Mockhasher
}

// NewMockhasher creates a new mock instance.
func NewMockhasher(ctrl *gomock.Controller) *Mockhasher {
	mock := &Mockhasher{ctrl: ctrl}
	mock.recorder = &MockhasherMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *Mockhasher) EXPECT() *MockhasherMockRecorder {
	return m.recorder
}

// getHash mocks base method.
func (m *Mockhasher) getHash(arg0 *NodeReference, arg1 NodeSource) (common.Hash, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "getHash", arg0, arg1)
	ret0, _ := ret[0].(common.Hash)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// getHash indicates an expected call of getHash.
func (mr *MockhasherMockRecorder) getHash(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "getHash", reflect.TypeOf((*Mockhasher)(nil).getHash), arg0, arg1)
}

// isEmbedded mocks base method.
func (m *Mockhasher) isEmbedded(arg0 Node, arg1 NodeSource) (bool, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "isEmbedded", arg0, arg1)
	ret0, _ := ret[0].(bool)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// isEmbedded indicates an expected call of isEmbedded.
func (mr *MockhasherMockRecorder) isEmbedded(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "isEmbedded", reflect.TypeOf((*Mockhasher)(nil).isEmbedded), arg0, arg1)
}

// updateHashes mocks base method.
func (m *Mockhasher) updateHashes(root *NodeReference, nodes NodeManager) (common.Hash, *NodeHashes, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "updateHashes", root, nodes)
	ret0, _ := ret[0].(common.Hash)
	ret1, _ := ret[1].(*NodeHashes)
	ret2, _ := ret[2].(error)
	return ret0, ret1, ret2
}

// updateHashes indicates an expected call of updateHashes.
func (mr *MockhasherMockRecorder) updateHashes(root, nodes any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "updateHashes", reflect.TypeOf((*Mockhasher)(nil).updateHashes), root, nodes)
}
