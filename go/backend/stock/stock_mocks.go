//
// Copyright (c) 2024 Fantom Foundation
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at fantom.foundation/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use
// of this software will be governed by the GNU Lesser General Public Licence v3.
//

// Code generated by MockGen. DO NOT EDIT.
// Source: stock.go
//
// Generated by this command:
//
//	mockgen -source stock.go -destination stock_mocks.go -package stock -exclude_interfaces Index
//

// Package stock is a generated GoMock package.
package stock

import (
	reflect "reflect"

	common "github.com/Fantom-foundation/Carmen/go/common"
	gomock "go.uber.org/mock/gomock"
)

// MockStock is a mock of Stock interface.
type MockStock[I Index, V any] struct {
	ctrl     *gomock.Controller
	recorder *MockStockMockRecorder[I, V]
}

// MockStockMockRecorder is the mock recorder for MockStock.
type MockStockMockRecorder[I Index, V any] struct {
	mock *MockStock[I, V]
}

// NewMockStock creates a new mock instance.
func NewMockStock[I Index, V any](ctrl *gomock.Controller) *MockStock[I, V] {
	mock := &MockStock[I, V]{ctrl: ctrl}
	mock.recorder = &MockStockMockRecorder[I, V]{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockStock[I, V]) EXPECT() *MockStockMockRecorder[I, V] {
	return m.recorder
}

// Close mocks base method.
func (m *MockStock[I, V]) Close() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Close")
	ret0, _ := ret[0].(error)
	return ret0
}

// Close indicates an expected call of Close.
func (mr *MockStockMockRecorder[I, V]) Close() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Close", reflect.TypeOf((*MockStock[I, V])(nil).Close))
}

// Delete mocks base method.
func (m *MockStock[I, V]) Delete(arg0 I) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Delete", arg0)
	ret0, _ := ret[0].(error)
	return ret0
}

// Delete indicates an expected call of Delete.
func (mr *MockStockMockRecorder[I, V]) Delete(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Delete", reflect.TypeOf((*MockStock[I, V])(nil).Delete), arg0)
}

// Flush mocks base method.
func (m *MockStock[I, V]) Flush() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Flush")
	ret0, _ := ret[0].(error)
	return ret0
}

// Flush indicates an expected call of Flush.
func (mr *MockStockMockRecorder[I, V]) Flush() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Flush", reflect.TypeOf((*MockStock[I, V])(nil).Flush))
}

// Get mocks base method.
func (m *MockStock[I, V]) Get(arg0 I) (V, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Get", arg0)
	ret0, _ := ret[0].(V)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// Get indicates an expected call of Get.
func (mr *MockStockMockRecorder[I, V]) Get(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Get", reflect.TypeOf((*MockStock[I, V])(nil).Get), arg0)
}

// GetIds mocks base method.
func (m *MockStock[I, V]) GetIds() (IndexSet[I], error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetIds")
	ret0, _ := ret[0].(IndexSet[I])
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetIds indicates an expected call of GetIds.
func (mr *MockStockMockRecorder[I, V]) GetIds() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetIds", reflect.TypeOf((*MockStock[I, V])(nil).GetIds))
}

// GetMemoryFootprint mocks base method.
func (m *MockStock[I, V]) GetMemoryFootprint() *common.MemoryFootprint {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetMemoryFootprint")
	ret0, _ := ret[0].(*common.MemoryFootprint)
	return ret0
}

// GetMemoryFootprint indicates an expected call of GetMemoryFootprint.
func (mr *MockStockMockRecorder[I, V]) GetMemoryFootprint() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetMemoryFootprint", reflect.TypeOf((*MockStock[I, V])(nil).GetMemoryFootprint))
}

// New mocks base method.
func (m *MockStock[I, V]) New() (I, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "New")
	ret0, _ := ret[0].(I)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// New indicates an expected call of New.
func (mr *MockStockMockRecorder[I, V]) New() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "New", reflect.TypeOf((*MockStock[I, V])(nil).New))
}

// Set mocks base method.
func (m *MockStock[I, V]) Set(arg0 I, arg1 V) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Set", arg0, arg1)
	ret0, _ := ret[0].(error)
	return ret0
}

// Set indicates an expected call of Set.
func (mr *MockStockMockRecorder[I, V]) Set(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Set", reflect.TypeOf((*MockStock[I, V])(nil).Set), arg0, arg1)
}

// MockIndexSet is a mock of IndexSet interface.
type MockIndexSet[I Index] struct {
	ctrl     *gomock.Controller
	recorder *MockIndexSetMockRecorder[I]
}

// MockIndexSetMockRecorder is the mock recorder for MockIndexSet.
type MockIndexSetMockRecorder[I Index] struct {
	mock *MockIndexSet[I]
}

// NewMockIndexSet creates a new mock instance.
func NewMockIndexSet[I Index](ctrl *gomock.Controller) *MockIndexSet[I] {
	mock := &MockIndexSet[I]{ctrl: ctrl}
	mock.recorder = &MockIndexSetMockRecorder[I]{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockIndexSet[I]) EXPECT() *MockIndexSetMockRecorder[I] {
	return m.recorder
}

// Contains mocks base method.
func (m *MockIndexSet[I]) Contains(arg0 I) bool {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Contains", arg0)
	ret0, _ := ret[0].(bool)
	return ret0
}

// Contains indicates an expected call of Contains.
func (mr *MockIndexSetMockRecorder[I]) Contains(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Contains", reflect.TypeOf((*MockIndexSet[I])(nil).Contains), arg0)
}

// GetLowerBound mocks base method.
func (m *MockIndexSet[I]) GetLowerBound() I {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetLowerBound")
	ret0, _ := ret[0].(I)
	return ret0
}

// GetLowerBound indicates an expected call of GetLowerBound.
func (mr *MockIndexSetMockRecorder[I]) GetLowerBound() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetLowerBound", reflect.TypeOf((*MockIndexSet[I])(nil).GetLowerBound))
}

// GetUpperBound mocks base method.
func (m *MockIndexSet[I]) GetUpperBound() I {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetUpperBound")
	ret0, _ := ret[0].(I)
	return ret0
}

// GetUpperBound indicates an expected call of GetUpperBound.
func (mr *MockIndexSetMockRecorder[I]) GetUpperBound() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetUpperBound", reflect.TypeOf((*MockIndexSet[I])(nil).GetUpperBound))
}

// MockValueEncoder is a mock of ValueEncoder interface.
type MockValueEncoder[V any] struct {
	ctrl     *gomock.Controller
	recorder *MockValueEncoderMockRecorder[V]
}

// MockValueEncoderMockRecorder is the mock recorder for MockValueEncoder.
type MockValueEncoderMockRecorder[V any] struct {
	mock *MockValueEncoder[V]
}

// NewMockValueEncoder creates a new mock instance.
func NewMockValueEncoder[V any](ctrl *gomock.Controller) *MockValueEncoder[V] {
	mock := &MockValueEncoder[V]{ctrl: ctrl}
	mock.recorder = &MockValueEncoderMockRecorder[V]{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockValueEncoder[V]) EXPECT() *MockValueEncoderMockRecorder[V] {
	return m.recorder
}

// GetEncodedSize mocks base method.
func (m *MockValueEncoder[V]) GetEncodedSize() int {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetEncodedSize")
	ret0, _ := ret[0].(int)
	return ret0
}

// GetEncodedSize indicates an expected call of GetEncodedSize.
func (mr *MockValueEncoderMockRecorder[V]) GetEncodedSize() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetEncodedSize", reflect.TypeOf((*MockValueEncoder[V])(nil).GetEncodedSize))
}

// Load mocks base method.
func (m *MockValueEncoder[V]) Load(arg0 []byte, arg1 *V) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Load", arg0, arg1)
	ret0, _ := ret[0].(error)
	return ret0
}

// Load indicates an expected call of Load.
func (mr *MockValueEncoderMockRecorder[V]) Load(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Load", reflect.TypeOf((*MockValueEncoder[V])(nil).Load), arg0, arg1)
}

// Store mocks base method.
func (m *MockValueEncoder[V]) Store(arg0 []byte, arg1 *V) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Store", arg0, arg1)
	ret0, _ := ret[0].(error)
	return ret0
}

// Store indicates an expected call of Store.
func (mr *MockValueEncoderMockRecorder[V]) Store(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Store", reflect.TypeOf((*MockValueEncoder[V])(nil).Store), arg0, arg1)
}
