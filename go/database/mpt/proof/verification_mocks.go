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
// Source: verification.go
//
// Generated by this command:
//
//	mockgen -source verification.go -destination verification_mocks.go -package proof
//

// Package proof is a generated GoMock package.
package proof

import (
	reflect "reflect"

	common "github.com/0xsoniclabs/carmen/go/common"
	witness "github.com/0xsoniclabs/carmen/go/common/witness"
	mpt "github.com/0xsoniclabs/carmen/go/database/mpt"
	gomock "go.uber.org/mock/gomock"
)

// MockverifiableTrie is a mock of verifiableTrie interface.
type MockverifiableTrie struct {
	ctrl     *gomock.Controller
	recorder *MockverifiableTrieMockRecorder
	isgomock struct{}
}

// MockverifiableTrieMockRecorder is the mock recorder for MockverifiableTrie.
type MockverifiableTrieMockRecorder struct {
	mock *MockverifiableTrie
}

// NewMockverifiableTrie creates a new mock instance.
func NewMockverifiableTrie(ctrl *gomock.Controller) *MockverifiableTrie {
	mock := &MockverifiableTrie{ctrl: ctrl}
	mock.recorder = &MockverifiableTrieMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockverifiableTrie) EXPECT() *MockverifiableTrieMockRecorder {
	return m.recorder
}

// CreateWitnessProof mocks base method.
func (m *MockverifiableTrie) CreateWitnessProof(arg0 common.Address, arg1 ...common.Key) (witness.Proof, error) {
	m.ctrl.T.Helper()
	varargs := []any{arg0}
	for _, a := range arg1 {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "CreateWitnessProof", varargs...)
	ret0, _ := ret[0].(witness.Proof)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// CreateWitnessProof indicates an expected call of CreateWitnessProof.
func (mr *MockverifiableTrieMockRecorder) CreateWitnessProof(arg0 any, arg1 ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{arg0}, arg1...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "CreateWitnessProof", reflect.TypeOf((*MockverifiableTrie)(nil).CreateWitnessProof), varargs...)
}

// GetAccountInfo mocks base method.
func (m *MockverifiableTrie) GetAccountInfo(addr common.Address) (mpt.AccountInfo, bool, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetAccountInfo", addr)
	ret0, _ := ret[0].(mpt.AccountInfo)
	ret1, _ := ret[1].(bool)
	ret2, _ := ret[2].(error)
	return ret0, ret1, ret2
}

// GetAccountInfo indicates an expected call of GetAccountInfo.
func (mr *MockverifiableTrieMockRecorder) GetAccountInfo(addr any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetAccountInfo", reflect.TypeOf((*MockverifiableTrie)(nil).GetAccountInfo), addr)
}

// GetValue mocks base method.
func (m *MockverifiableTrie) GetValue(addr common.Address, key common.Key) (common.Value, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetValue", addr, key)
	ret0, _ := ret[0].(common.Value)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetValue indicates an expected call of GetValue.
func (mr *MockverifiableTrieMockRecorder) GetValue(addr, key any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetValue", reflect.TypeOf((*MockverifiableTrie)(nil).GetValue), addr, key)
}

// UpdateHashes mocks base method.
func (m *MockverifiableTrie) UpdateHashes() (common.Hash, *mpt.NodeHashes, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "UpdateHashes")
	ret0, _ := ret[0].(common.Hash)
	ret1, _ := ret[1].(*mpt.NodeHashes)
	ret2, _ := ret[2].(error)
	return ret0, ret1, ret2
}

// UpdateHashes indicates an expected call of UpdateHashes.
func (mr *MockverifiableTrieMockRecorder) UpdateHashes() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "UpdateHashes", reflect.TypeOf((*MockverifiableTrie)(nil).UpdateHashes))
}

// VisitAccountStorage mocks base method.
func (m *MockverifiableTrie) VisitAccountStorage(address common.Address, mode mpt.AccessMode, visitor mpt.NodeVisitor) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "VisitAccountStorage", address, mode, visitor)
	ret0, _ := ret[0].(error)
	return ret0
}

// VisitAccountStorage indicates an expected call of VisitAccountStorage.
func (mr *MockverifiableTrieMockRecorder) VisitAccountStorage(address, mode, visitor any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "VisitAccountStorage", reflect.TypeOf((*MockverifiableTrie)(nil).VisitAccountStorage), address, mode, visitor)
}

// VisitTrie mocks base method.
func (m *MockverifiableTrie) VisitTrie(mode mpt.AccessMode, visitor mpt.NodeVisitor) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "VisitTrie", mode, visitor)
	ret0, _ := ret[0].(error)
	return ret0
}

// VisitTrie indicates an expected call of VisitTrie.
func (mr *MockverifiableTrieMockRecorder) VisitTrie(mode, visitor any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "VisitTrie", reflect.TypeOf((*MockverifiableTrie)(nil).VisitTrie), mode, visitor)
}

// MockverifiableArchiveTrie is a mock of verifiableArchiveTrie interface.
type MockverifiableArchiveTrie struct {
	ctrl     *gomock.Controller
	recorder *MockverifiableArchiveTrieMockRecorder
	isgomock struct{}
}

// MockverifiableArchiveTrieMockRecorder is the mock recorder for MockverifiableArchiveTrie.
type MockverifiableArchiveTrieMockRecorder struct {
	mock *MockverifiableArchiveTrie
}

// NewMockverifiableArchiveTrie creates a new mock instance.
func NewMockverifiableArchiveTrie(ctrl *gomock.Controller) *MockverifiableArchiveTrie {
	mock := &MockverifiableArchiveTrie{ctrl: ctrl}
	mock.recorder = &MockverifiableArchiveTrieMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockverifiableArchiveTrie) EXPECT() *MockverifiableArchiveTrieMockRecorder {
	return m.recorder
}

// CreateWitnessProof mocks base method.
func (m *MockverifiableArchiveTrie) CreateWitnessProof(block uint64, address common.Address, keys ...common.Key) (witness.Proof, error) {
	m.ctrl.T.Helper()
	varargs := []any{block, address}
	for _, a := range keys {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "CreateWitnessProof", varargs...)
	ret0, _ := ret[0].(witness.Proof)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// CreateWitnessProof indicates an expected call of CreateWitnessProof.
func (mr *MockverifiableArchiveTrieMockRecorder) CreateWitnessProof(block, address any, keys ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{block, address}, keys...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "CreateWitnessProof", reflect.TypeOf((*MockverifiableArchiveTrie)(nil).CreateWitnessProof), varargs...)
}

// GetAccountInfo mocks base method.
func (m *MockverifiableArchiveTrie) GetAccountInfo(block uint64, addr common.Address) (mpt.AccountInfo, bool, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetAccountInfo", block, addr)
	ret0, _ := ret[0].(mpt.AccountInfo)
	ret1, _ := ret[1].(bool)
	ret2, _ := ret[2].(error)
	return ret0, ret1, ret2
}

// GetAccountInfo indicates an expected call of GetAccountInfo.
func (mr *MockverifiableArchiveTrieMockRecorder) GetAccountInfo(block, addr any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetAccountInfo", reflect.TypeOf((*MockverifiableArchiveTrie)(nil).GetAccountInfo), block, addr)
}

// GetBlockHeight mocks base method.
func (m *MockverifiableArchiveTrie) GetBlockHeight() (uint64, bool, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetBlockHeight")
	ret0, _ := ret[0].(uint64)
	ret1, _ := ret[1].(bool)
	ret2, _ := ret[2].(error)
	return ret0, ret1, ret2
}

// GetBlockHeight indicates an expected call of GetBlockHeight.
func (mr *MockverifiableArchiveTrieMockRecorder) GetBlockHeight() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetBlockHeight", reflect.TypeOf((*MockverifiableArchiveTrie)(nil).GetBlockHeight))
}

// GetHash mocks base method.
func (m *MockverifiableArchiveTrie) GetHash(block uint64) (common.Hash, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetHash", block)
	ret0, _ := ret[0].(common.Hash)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetHash indicates an expected call of GetHash.
func (mr *MockverifiableArchiveTrieMockRecorder) GetHash(block any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetHash", reflect.TypeOf((*MockverifiableArchiveTrie)(nil).GetHash), block)
}

// GetStorage mocks base method.
func (m *MockverifiableArchiveTrie) GetStorage(block uint64, addr common.Address, key common.Key) (common.Value, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetStorage", block, addr, key)
	ret0, _ := ret[0].(common.Value)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetStorage indicates an expected call of GetStorage.
func (mr *MockverifiableArchiveTrieMockRecorder) GetStorage(block, addr, key any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetStorage", reflect.TypeOf((*MockverifiableArchiveTrie)(nil).GetStorage), block, addr, key)
}

// VisitAccountStorage mocks base method.
func (m *MockverifiableArchiveTrie) VisitAccountStorage(block uint64, address common.Address, mode mpt.AccessMode, visitor mpt.NodeVisitor) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "VisitAccountStorage", block, address, mode, visitor)
	ret0, _ := ret[0].(error)
	return ret0
}

// VisitAccountStorage indicates an expected call of VisitAccountStorage.
func (mr *MockverifiableArchiveTrieMockRecorder) VisitAccountStorage(block, address, mode, visitor any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "VisitAccountStorage", reflect.TypeOf((*MockverifiableArchiveTrie)(nil).VisitAccountStorage), block, address, mode, visitor)
}

// VisitTrie mocks base method.
func (m *MockverifiableArchiveTrie) VisitTrie(block uint64, mode mpt.AccessMode, visitor mpt.NodeVisitor) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "VisitTrie", block, mode, visitor)
	ret0, _ := ret[0].(error)
	return ret0
}

// VisitTrie indicates an expected call of VisitTrie.
func (mr *MockverifiableArchiveTrieMockRecorder) VisitTrie(block, mode, visitor any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "VisitTrie", reflect.TypeOf((*MockverifiableArchiveTrie)(nil).VisitTrie), block, mode, visitor)
}
