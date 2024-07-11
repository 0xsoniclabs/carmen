// Copyright (c) 2024 Fantom Foundation
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at fantom.foundation/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

// Code generated by MockGen. DO NOT EDIT.
// Source: archive.go
//
// Generated by this command:
//
//	mockgen -source archive.go -destination archive_mock.go -package archive
//

// Package archive is a generated GoMock package.
package archive

import (
	reflect "reflect"

	common "github.com/Fantom-foundation/Carmen/go/common"
	witness "github.com/Fantom-foundation/Carmen/go/common/witness"
	gomock "go.uber.org/mock/gomock"
)

// MockArchive is a mock of Archive interface.
type MockArchive struct {
	ctrl     *gomock.Controller
	recorder *MockArchiveMockRecorder
}

// MockArchiveMockRecorder is the mock recorder for MockArchive.
type MockArchiveMockRecorder struct {
	mock *MockArchive
}

// NewMockArchive creates a new mock instance.
func NewMockArchive(ctrl *gomock.Controller) *MockArchive {
	mock := &MockArchive{ctrl: ctrl}
	mock.recorder = &MockArchiveMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockArchive) EXPECT() *MockArchiveMockRecorder {
	return m.recorder
}

// Add mocks base method.
func (m *MockArchive) Add(block uint64, update common.Update, hints any) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Add", block, update, hints)
	ret0, _ := ret[0].(error)
	return ret0
}

// Add indicates an expected call of Add.
func (mr *MockArchiveMockRecorder) Add(block, update, hints any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Add", reflect.TypeOf((*MockArchive)(nil).Add), block, update, hints)
}

// Close mocks base method.
func (m *MockArchive) Close() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Close")
	ret0, _ := ret[0].(error)
	return ret0
}

// Close indicates an expected call of Close.
func (mr *MockArchiveMockRecorder) Close() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Close", reflect.TypeOf((*MockArchive)(nil).Close))
}

// CreateWitnessProof mocks base method.
func (m *MockArchive) CreateWitnessProof(block uint64, address common.Address, keys ...common.Key) (witness.Proof, error) {
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
func (mr *MockArchiveMockRecorder) CreateWitnessProof(block, address any, keys ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{block, address}, keys...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "CreateWitnessProof", reflect.TypeOf((*MockArchive)(nil).CreateWitnessProof), varargs...)
}

// Exists mocks base method.
func (m *MockArchive) Exists(block uint64, account common.Address) (bool, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Exists", block, account)
	ret0, _ := ret[0].(bool)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// Exists indicates an expected call of Exists.
func (mr *MockArchiveMockRecorder) Exists(block, account any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Exists", reflect.TypeOf((*MockArchive)(nil).Exists), block, account)
}

// Flush mocks base method.
func (m *MockArchive) Flush() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Flush")
	ret0, _ := ret[0].(error)
	return ret0
}

// Flush indicates an expected call of Flush.
func (mr *MockArchiveMockRecorder) Flush() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Flush", reflect.TypeOf((*MockArchive)(nil).Flush))
}

// GetAccountHash mocks base method.
func (m *MockArchive) GetAccountHash(block uint64, account common.Address) (common.Hash, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetAccountHash", block, account)
	ret0, _ := ret[0].(common.Hash)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetAccountHash indicates an expected call of GetAccountHash.
func (mr *MockArchiveMockRecorder) GetAccountHash(block, account any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetAccountHash", reflect.TypeOf((*MockArchive)(nil).GetAccountHash), block, account)
}

// GetBalance mocks base method.
func (m *MockArchive) GetBalance(block uint64, account common.Address) (common.Balance, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetBalance", block, account)
	ret0, _ := ret[0].(common.Balance)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetBalance indicates an expected call of GetBalance.
func (mr *MockArchiveMockRecorder) GetBalance(block, account any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetBalance", reflect.TypeOf((*MockArchive)(nil).GetBalance), block, account)
}

// GetBlockHeight mocks base method.
func (m *MockArchive) GetBlockHeight() (uint64, bool, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetBlockHeight")
	ret0, _ := ret[0].(uint64)
	ret1, _ := ret[1].(bool)
	ret2, _ := ret[2].(error)
	return ret0, ret1, ret2
}

// GetBlockHeight indicates an expected call of GetBlockHeight.
func (mr *MockArchiveMockRecorder) GetBlockHeight() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetBlockHeight", reflect.TypeOf((*MockArchive)(nil).GetBlockHeight))
}

// GetCode mocks base method.
func (m *MockArchive) GetCode(block uint64, account common.Address) ([]byte, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetCode", block, account)
	ret0, _ := ret[0].([]byte)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetCode indicates an expected call of GetCode.
func (mr *MockArchiveMockRecorder) GetCode(block, account any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetCode", reflect.TypeOf((*MockArchive)(nil).GetCode), block, account)
}

// GetHash mocks base method.
func (m *MockArchive) GetHash(block uint64) (common.Hash, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetHash", block)
	ret0, _ := ret[0].(common.Hash)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetHash indicates an expected call of GetHash.
func (mr *MockArchiveMockRecorder) GetHash(block any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetHash", reflect.TypeOf((*MockArchive)(nil).GetHash), block)
}

// GetMemoryFootprint mocks base method.
func (m *MockArchive) GetMemoryFootprint() *common.MemoryFootprint {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetMemoryFootprint")
	ret0, _ := ret[0].(*common.MemoryFootprint)
	return ret0
}

// GetMemoryFootprint indicates an expected call of GetMemoryFootprint.
func (mr *MockArchiveMockRecorder) GetMemoryFootprint() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetMemoryFootprint", reflect.TypeOf((*MockArchive)(nil).GetMemoryFootprint))
}

// GetNonce mocks base method.
func (m *MockArchive) GetNonce(block uint64, account common.Address) (common.Nonce, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetNonce", block, account)
	ret0, _ := ret[0].(common.Nonce)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetNonce indicates an expected call of GetNonce.
func (mr *MockArchiveMockRecorder) GetNonce(block, account any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetNonce", reflect.TypeOf((*MockArchive)(nil).GetNonce), block, account)
}

// GetStorage mocks base method.
func (m *MockArchive) GetStorage(block uint64, account common.Address, slot common.Key) (common.Value, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetStorage", block, account, slot)
	ret0, _ := ret[0].(common.Value)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetStorage indicates an expected call of GetStorage.
func (mr *MockArchiveMockRecorder) GetStorage(block, account, slot any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetStorage", reflect.TypeOf((*MockArchive)(nil).GetStorage), block, account, slot)
}
