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
// Source: carmen.go
//
// Generated by this command:
//
//	mockgen -source carmen.go -destination carmen_mock.go -package carmen
//

// Package carmen is a generated GoMock package.
package carmen

import (
	reflect "reflect"

	gomock "go.uber.org/mock/gomock"
)

// MockDatabase is a mock of Database interface.
type MockDatabase struct {
	ctrl     *gomock.Controller
	recorder *MockDatabaseMockRecorder
}

// MockDatabaseMockRecorder is the mock recorder for MockDatabase.
type MockDatabaseMockRecorder struct {
	mock *MockDatabase
}

// NewMockDatabase creates a new mock instance.
func NewMockDatabase(ctrl *gomock.Controller) *MockDatabase {
	mock := &MockDatabase{ctrl: ctrl}
	mock.recorder = &MockDatabaseMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockDatabase) EXPECT() *MockDatabaseMockRecorder {
	return m.recorder
}

// AddBlock mocks base method.
func (m *MockDatabase) AddBlock(block uint64, run func(HeadBlockContext) error) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "AddBlock", block, run)
	ret0, _ := ret[0].(error)
	return ret0
}

// AddBlock indicates an expected call of AddBlock.
func (mr *MockDatabaseMockRecorder) AddBlock(block, run any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "AddBlock", reflect.TypeOf((*MockDatabase)(nil).AddBlock), block, run)
}

// BeginBlock mocks base method.
func (m *MockDatabase) BeginBlock(block uint64) (HeadBlockContext, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "BeginBlock", block)
	ret0, _ := ret[0].(HeadBlockContext)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// BeginBlock indicates an expected call of BeginBlock.
func (mr *MockDatabaseMockRecorder) BeginBlock(block any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "BeginBlock", reflect.TypeOf((*MockDatabase)(nil).BeginBlock), block)
}

// Close mocks base method.
func (m *MockDatabase) Close() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Close")
	ret0, _ := ret[0].(error)
	return ret0
}

// Close indicates an expected call of Close.
func (mr *MockDatabaseMockRecorder) Close() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Close", reflect.TypeOf((*MockDatabase)(nil).Close))
}

// Flush mocks base method.
func (m *MockDatabase) Flush() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Flush")
	ret0, _ := ret[0].(error)
	return ret0
}

// Flush indicates an expected call of Flush.
func (mr *MockDatabaseMockRecorder) Flush() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Flush", reflect.TypeOf((*MockDatabase)(nil).Flush))
}

// GetArchiveBlockHeight mocks base method.
func (m *MockDatabase) GetArchiveBlockHeight() (int64, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetArchiveBlockHeight")
	ret0, _ := ret[0].(int64)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetArchiveBlockHeight indicates an expected call of GetArchiveBlockHeight.
func (mr *MockDatabaseMockRecorder) GetArchiveBlockHeight() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetArchiveBlockHeight", reflect.TypeOf((*MockDatabase)(nil).GetArchiveBlockHeight))
}

// GetHistoricContext mocks base method.
func (m *MockDatabase) GetHistoricContext(block uint64) (HistoricBlockContext, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetHistoricContext", block)
	ret0, _ := ret[0].(HistoricBlockContext)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetHistoricContext indicates an expected call of GetHistoricContext.
func (mr *MockDatabaseMockRecorder) GetHistoricContext(block any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetHistoricContext", reflect.TypeOf((*MockDatabase)(nil).GetHistoricContext), block)
}

// GetHistoricStateHash mocks base method.
func (m *MockDatabase) GetHistoricStateHash(block uint64) (Hash, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetHistoricStateHash", block)
	ret0, _ := ret[0].(Hash)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetHistoricStateHash indicates an expected call of GetHistoricStateHash.
func (mr *MockDatabaseMockRecorder) GetHistoricStateHash(block any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetHistoricStateHash", reflect.TypeOf((*MockDatabase)(nil).GetHistoricStateHash), block)
}

// QueryBlock mocks base method.
func (m *MockDatabase) QueryBlock(block uint64, run func(HistoricBlockContext) error) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "QueryBlock", block, run)
	ret0, _ := ret[0].(error)
	return ret0
}

// QueryBlock indicates an expected call of QueryBlock.
func (mr *MockDatabaseMockRecorder) QueryBlock(block, run any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "QueryBlock", reflect.TypeOf((*MockDatabase)(nil).QueryBlock), block, run)
}

// QueryHeadState mocks base method.
func (m *MockDatabase) QueryHeadState(query func(QueryContext)) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "QueryHeadState", query)
	ret0, _ := ret[0].(error)
	return ret0
}

// QueryHeadState indicates an expected call of QueryHeadState.
func (mr *MockDatabaseMockRecorder) QueryHeadState(query any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "QueryHeadState", reflect.TypeOf((*MockDatabase)(nil).QueryHeadState), query)
}

// QueryHistoricState mocks base method.
func (m *MockDatabase) QueryHistoricState(block uint64, query func(QueryContext)) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "QueryHistoricState", block, query)
	ret0, _ := ret[0].(error)
	return ret0
}

// QueryHistoricState indicates an expected call of QueryHistoricState.
func (mr *MockDatabaseMockRecorder) QueryHistoricState(block, query any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "QueryHistoricState", reflect.TypeOf((*MockDatabase)(nil).QueryHistoricState), block, query)
}

// StartBulkLoad mocks base method.
func (m *MockDatabase) StartBulkLoad(block uint64) (BulkLoad, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "StartBulkLoad", block)
	ret0, _ := ret[0].(BulkLoad)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// StartBulkLoad indicates an expected call of StartBulkLoad.
func (mr *MockDatabaseMockRecorder) StartBulkLoad(block any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "StartBulkLoad", reflect.TypeOf((*MockDatabase)(nil).StartBulkLoad), block)
}

// MockblockContext is a mock of blockContext interface.
type MockblockContext struct {
	ctrl     *gomock.Controller
	recorder *MockblockContextMockRecorder
}

// MockblockContextMockRecorder is the mock recorder for MockblockContext.
type MockblockContextMockRecorder struct {
	mock *MockblockContext
}

// NewMockblockContext creates a new mock instance.
func NewMockblockContext(ctrl *gomock.Controller) *MockblockContext {
	mock := &MockblockContext{ctrl: ctrl}
	mock.recorder = &MockblockContextMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockblockContext) EXPECT() *MockblockContextMockRecorder {
	return m.recorder
}

// BeginTransaction mocks base method.
func (m *MockblockContext) BeginTransaction() (TransactionContext, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "BeginTransaction")
	ret0, _ := ret[0].(TransactionContext)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// BeginTransaction indicates an expected call of BeginTransaction.
func (mr *MockblockContextMockRecorder) BeginTransaction() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "BeginTransaction", reflect.TypeOf((*MockblockContext)(nil).BeginTransaction))
}

// RunTransaction mocks base method.
func (m *MockblockContext) RunTransaction(run func(TransactionContext) error) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "RunTransaction", run)
	ret0, _ := ret[0].(error)
	return ret0
}

// RunTransaction indicates an expected call of RunTransaction.
func (mr *MockblockContextMockRecorder) RunTransaction(run any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "RunTransaction", reflect.TypeOf((*MockblockContext)(nil).RunTransaction), run)
}

// MockHeadBlockContext is a mock of HeadBlockContext interface.
type MockHeadBlockContext struct {
	ctrl     *gomock.Controller
	recorder *MockHeadBlockContextMockRecorder
}

// MockHeadBlockContextMockRecorder is the mock recorder for MockHeadBlockContext.
type MockHeadBlockContextMockRecorder struct {
	mock *MockHeadBlockContext
}

// NewMockHeadBlockContext creates a new mock instance.
func NewMockHeadBlockContext(ctrl *gomock.Controller) *MockHeadBlockContext {
	mock := &MockHeadBlockContext{ctrl: ctrl}
	mock.recorder = &MockHeadBlockContextMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockHeadBlockContext) EXPECT() *MockHeadBlockContextMockRecorder {
	return m.recorder
}

// Abort mocks base method.
func (m *MockHeadBlockContext) Abort() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Abort")
	ret0, _ := ret[0].(error)
	return ret0
}

// Abort indicates an expected call of Abort.
func (mr *MockHeadBlockContextMockRecorder) Abort() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Abort", reflect.TypeOf((*MockHeadBlockContext)(nil).Abort))
}

// BeginTransaction mocks base method.
func (m *MockHeadBlockContext) BeginTransaction() (TransactionContext, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "BeginTransaction")
	ret0, _ := ret[0].(TransactionContext)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// BeginTransaction indicates an expected call of BeginTransaction.
func (mr *MockHeadBlockContextMockRecorder) BeginTransaction() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "BeginTransaction", reflect.TypeOf((*MockHeadBlockContext)(nil).BeginTransaction))
}

// Commit mocks base method.
func (m *MockHeadBlockContext) Commit() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Commit")
	ret0, _ := ret[0].(error)
	return ret0
}

// Commit indicates an expected call of Commit.
func (mr *MockHeadBlockContextMockRecorder) Commit() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Commit", reflect.TypeOf((*MockHeadBlockContext)(nil).Commit))
}

// RunTransaction mocks base method.
func (m *MockHeadBlockContext) RunTransaction(run func(TransactionContext) error) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "RunTransaction", run)
	ret0, _ := ret[0].(error)
	return ret0
}

// RunTransaction indicates an expected call of RunTransaction.
func (mr *MockHeadBlockContextMockRecorder) RunTransaction(run any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "RunTransaction", reflect.TypeOf((*MockHeadBlockContext)(nil).RunTransaction), run)
}

// MockHistoricBlockContext is a mock of HistoricBlockContext interface.
type MockHistoricBlockContext struct {
	ctrl     *gomock.Controller
	recorder *MockHistoricBlockContextMockRecorder
}

// MockHistoricBlockContextMockRecorder is the mock recorder for MockHistoricBlockContext.
type MockHistoricBlockContextMockRecorder struct {
	mock *MockHistoricBlockContext
}

// NewMockHistoricBlockContext creates a new mock instance.
func NewMockHistoricBlockContext(ctrl *gomock.Controller) *MockHistoricBlockContext {
	mock := &MockHistoricBlockContext{ctrl: ctrl}
	mock.recorder = &MockHistoricBlockContextMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockHistoricBlockContext) EXPECT() *MockHistoricBlockContextMockRecorder {
	return m.recorder
}

// BeginTransaction mocks base method.
func (m *MockHistoricBlockContext) BeginTransaction() (TransactionContext, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "BeginTransaction")
	ret0, _ := ret[0].(TransactionContext)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// BeginTransaction indicates an expected call of BeginTransaction.
func (mr *MockHistoricBlockContextMockRecorder) BeginTransaction() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "BeginTransaction", reflect.TypeOf((*MockHistoricBlockContext)(nil).BeginTransaction))
}

// Close mocks base method.
func (m *MockHistoricBlockContext) Close() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Close")
	ret0, _ := ret[0].(error)
	return ret0
}

// Close indicates an expected call of Close.
func (mr *MockHistoricBlockContextMockRecorder) Close() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Close", reflect.TypeOf((*MockHistoricBlockContext)(nil).Close))
}

// RunTransaction mocks base method.
func (m *MockHistoricBlockContext) RunTransaction(run func(TransactionContext) error) error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "RunTransaction", run)
	ret0, _ := ret[0].(error)
	return ret0
}

// RunTransaction indicates an expected call of RunTransaction.
func (mr *MockHistoricBlockContextMockRecorder) RunTransaction(run any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "RunTransaction", reflect.TypeOf((*MockHistoricBlockContext)(nil).RunTransaction), run)
}

// MockTransactionContext is a mock of TransactionContext interface.
type MockTransactionContext struct {
	ctrl     *gomock.Controller
	recorder *MockTransactionContextMockRecorder
}

// MockTransactionContextMockRecorder is the mock recorder for MockTransactionContext.
type MockTransactionContextMockRecorder struct {
	mock *MockTransactionContext
}

// NewMockTransactionContext creates a new mock instance.
func NewMockTransactionContext(ctrl *gomock.Controller) *MockTransactionContext {
	mock := &MockTransactionContext{ctrl: ctrl}
	mock.recorder = &MockTransactionContextMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockTransactionContext) EXPECT() *MockTransactionContextMockRecorder {
	return m.recorder
}

// Abort mocks base method.
func (m *MockTransactionContext) Abort() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Abort")
	ret0, _ := ret[0].(error)
	return ret0
}

// Abort indicates an expected call of Abort.
func (mr *MockTransactionContextMockRecorder) Abort() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Abort", reflect.TypeOf((*MockTransactionContext)(nil).Abort))
}

// AddAddressToAccessList mocks base method.
func (m *MockTransactionContext) AddAddressToAccessList(arg0 Address) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "AddAddressToAccessList", arg0)
}

// AddAddressToAccessList indicates an expected call of AddAddressToAccessList.
func (mr *MockTransactionContextMockRecorder) AddAddressToAccessList(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "AddAddressToAccessList", reflect.TypeOf((*MockTransactionContext)(nil).AddAddressToAccessList), arg0)
}

// AddBalance mocks base method.
func (m *MockTransactionContext) AddBalance(arg0 Address, arg1 Amount) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "AddBalance", arg0, arg1)
}

// AddBalance indicates an expected call of AddBalance.
func (mr *MockTransactionContextMockRecorder) AddBalance(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "AddBalance", reflect.TypeOf((*MockTransactionContext)(nil).AddBalance), arg0, arg1)
}

// AddLog mocks base method.
func (m *MockTransactionContext) AddLog(arg0 *Log) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "AddLog", arg0)
}

// AddLog indicates an expected call of AddLog.
func (mr *MockTransactionContextMockRecorder) AddLog(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "AddLog", reflect.TypeOf((*MockTransactionContext)(nil).AddLog), arg0)
}

// AddRefund mocks base method.
func (m *MockTransactionContext) AddRefund(arg0 uint64) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "AddRefund", arg0)
}

// AddRefund indicates an expected call of AddRefund.
func (mr *MockTransactionContextMockRecorder) AddRefund(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "AddRefund", reflect.TypeOf((*MockTransactionContext)(nil).AddRefund), arg0)
}

// AddSlotToAccessList mocks base method.
func (m *MockTransactionContext) AddSlotToAccessList(arg0 Address, arg1 Key) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "AddSlotToAccessList", arg0, arg1)
}

// AddSlotToAccessList indicates an expected call of AddSlotToAccessList.
func (mr *MockTransactionContextMockRecorder) AddSlotToAccessList(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "AddSlotToAccessList", reflect.TypeOf((*MockTransactionContext)(nil).AddSlotToAccessList), arg0, arg1)
}

// ClearAccessList mocks base method.
func (m *MockTransactionContext) ClearAccessList() {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "ClearAccessList")
}

// ClearAccessList indicates an expected call of ClearAccessList.
func (mr *MockTransactionContextMockRecorder) ClearAccessList() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ClearAccessList", reflect.TypeOf((*MockTransactionContext)(nil).ClearAccessList))
}

// Commit mocks base method.
func (m *MockTransactionContext) Commit() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Commit")
	ret0, _ := ret[0].(error)
	return ret0
}

// Commit indicates an expected call of Commit.
func (mr *MockTransactionContextMockRecorder) Commit() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Commit", reflect.TypeOf((*MockTransactionContext)(nil).Commit))
}

// CreateAccount mocks base method.
func (m *MockTransactionContext) CreateAccount(arg0 Address) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "CreateAccount", arg0)
}

// CreateAccount indicates an expected call of CreateAccount.
func (mr *MockTransactionContextMockRecorder) CreateAccount(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "CreateAccount", reflect.TypeOf((*MockTransactionContext)(nil).CreateAccount), arg0)
}

// CreateContract mocks base method.
func (m *MockTransactionContext) CreateContract(arg0 Address) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "CreateContract", arg0)
}

// CreateContract indicates an expected call of CreateContract.
func (mr *MockTransactionContextMockRecorder) CreateContract(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "CreateContract", reflect.TypeOf((*MockTransactionContext)(nil).CreateContract), arg0)
}

// Empty mocks base method.
func (m *MockTransactionContext) Empty(arg0 Address) bool {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Empty", arg0)
	ret0, _ := ret[0].(bool)
	return ret0
}

// Empty indicates an expected call of Empty.
func (mr *MockTransactionContextMockRecorder) Empty(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Empty", reflect.TypeOf((*MockTransactionContext)(nil).Empty), arg0)
}

// Exist mocks base method.
func (m *MockTransactionContext) Exist(arg0 Address) bool {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Exist", arg0)
	ret0, _ := ret[0].(bool)
	return ret0
}

// Exist indicates an expected call of Exist.
func (mr *MockTransactionContextMockRecorder) Exist(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Exist", reflect.TypeOf((*MockTransactionContext)(nil).Exist), arg0)
}

// GetBalance mocks base method.
func (m *MockTransactionContext) GetBalance(arg0 Address) Amount {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetBalance", arg0)
	ret0, _ := ret[0].(Amount)
	return ret0
}

// GetBalance indicates an expected call of GetBalance.
func (mr *MockTransactionContextMockRecorder) GetBalance(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetBalance", reflect.TypeOf((*MockTransactionContext)(nil).GetBalance), arg0)
}

// GetCode mocks base method.
func (m *MockTransactionContext) GetCode(arg0 Address) []byte {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetCode", arg0)
	ret0, _ := ret[0].([]byte)
	return ret0
}

// GetCode indicates an expected call of GetCode.
func (mr *MockTransactionContextMockRecorder) GetCode(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetCode", reflect.TypeOf((*MockTransactionContext)(nil).GetCode), arg0)
}

// GetCodeHash mocks base method.
func (m *MockTransactionContext) GetCodeHash(arg0 Address) Hash {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetCodeHash", arg0)
	ret0, _ := ret[0].(Hash)
	return ret0
}

// GetCodeHash indicates an expected call of GetCodeHash.
func (mr *MockTransactionContextMockRecorder) GetCodeHash(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetCodeHash", reflect.TypeOf((*MockTransactionContext)(nil).GetCodeHash), arg0)
}

// GetCodeSize mocks base method.
func (m *MockTransactionContext) GetCodeSize(arg0 Address) int {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetCodeSize", arg0)
	ret0, _ := ret[0].(int)
	return ret0
}

// GetCodeSize indicates an expected call of GetCodeSize.
func (mr *MockTransactionContextMockRecorder) GetCodeSize(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetCodeSize", reflect.TypeOf((*MockTransactionContext)(nil).GetCodeSize), arg0)
}

// GetCommittedState mocks base method.
func (m *MockTransactionContext) GetCommittedState(arg0 Address, arg1 Key) Value {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetCommittedState", arg0, arg1)
	ret0, _ := ret[0].(Value)
	return ret0
}

// GetCommittedState indicates an expected call of GetCommittedState.
func (mr *MockTransactionContextMockRecorder) GetCommittedState(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetCommittedState", reflect.TypeOf((*MockTransactionContext)(nil).GetCommittedState), arg0, arg1)
}

// GetLogs mocks base method.
func (m *MockTransactionContext) GetLogs() []*Log {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetLogs")
	ret0, _ := ret[0].([]*Log)
	return ret0
}

// GetLogs indicates an expected call of GetLogs.
func (mr *MockTransactionContextMockRecorder) GetLogs() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetLogs", reflect.TypeOf((*MockTransactionContext)(nil).GetLogs))
}

// GetNonce mocks base method.
func (m *MockTransactionContext) GetNonce(arg0 Address) uint64 {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetNonce", arg0)
	ret0, _ := ret[0].(uint64)
	return ret0
}

// GetNonce indicates an expected call of GetNonce.
func (mr *MockTransactionContextMockRecorder) GetNonce(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetNonce", reflect.TypeOf((*MockTransactionContext)(nil).GetNonce), arg0)
}

// GetRefund mocks base method.
func (m *MockTransactionContext) GetRefund() uint64 {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetRefund")
	ret0, _ := ret[0].(uint64)
	return ret0
}

// GetRefund indicates an expected call of GetRefund.
func (mr *MockTransactionContextMockRecorder) GetRefund() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetRefund", reflect.TypeOf((*MockTransactionContext)(nil).GetRefund))
}

// GetState mocks base method.
func (m *MockTransactionContext) GetState(arg0 Address, arg1 Key) Value {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetState", arg0, arg1)
	ret0, _ := ret[0].(Value)
	return ret0
}

// GetState indicates an expected call of GetState.
func (mr *MockTransactionContextMockRecorder) GetState(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetState", reflect.TypeOf((*MockTransactionContext)(nil).GetState), arg0, arg1)
}

// GetTransientState mocks base method.
func (m *MockTransactionContext) GetTransientState(arg0 Address, arg1 Key) Value {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetTransientState", arg0, arg1)
	ret0, _ := ret[0].(Value)
	return ret0
}

// GetTransientState indicates an expected call of GetTransientState.
func (mr *MockTransactionContextMockRecorder) GetTransientState(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetTransientState", reflect.TypeOf((*MockTransactionContext)(nil).GetTransientState), arg0, arg1)
}

// HasSelfDestructed mocks base method.
func (m *MockTransactionContext) HasSelfDestructed(arg0 Address) bool {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "HasSelfDestructed", arg0)
	ret0, _ := ret[0].(bool)
	return ret0
}

// HasSelfDestructed indicates an expected call of HasSelfDestructed.
func (mr *MockTransactionContextMockRecorder) HasSelfDestructed(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "HasSelfDestructed", reflect.TypeOf((*MockTransactionContext)(nil).HasSelfDestructed), arg0)
}

// IsAddressInAccessList mocks base method.
func (m *MockTransactionContext) IsAddressInAccessList(arg0 Address) bool {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "IsAddressInAccessList", arg0)
	ret0, _ := ret[0].(bool)
	return ret0
}

// IsAddressInAccessList indicates an expected call of IsAddressInAccessList.
func (mr *MockTransactionContextMockRecorder) IsAddressInAccessList(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "IsAddressInAccessList", reflect.TypeOf((*MockTransactionContext)(nil).IsAddressInAccessList), arg0)
}

// IsSlotInAccessList mocks base method.
func (m *MockTransactionContext) IsSlotInAccessList(arg0 Address, arg1 Key) (bool, bool) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "IsSlotInAccessList", arg0, arg1)
	ret0, _ := ret[0].(bool)
	ret1, _ := ret[1].(bool)
	return ret0, ret1
}

// IsSlotInAccessList indicates an expected call of IsSlotInAccessList.
func (mr *MockTransactionContextMockRecorder) IsSlotInAccessList(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "IsSlotInAccessList", reflect.TypeOf((*MockTransactionContext)(nil).IsSlotInAccessList), arg0, arg1)
}

// RevertToSnapshot mocks base method.
func (m *MockTransactionContext) RevertToSnapshot(arg0 int) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "RevertToSnapshot", arg0)
}

// RevertToSnapshot indicates an expected call of RevertToSnapshot.
func (mr *MockTransactionContextMockRecorder) RevertToSnapshot(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "RevertToSnapshot", reflect.TypeOf((*MockTransactionContext)(nil).RevertToSnapshot), arg0)
}

// SelfDestruct mocks base method.
func (m *MockTransactionContext) SelfDestruct(arg0 Address) bool {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "SelfDestruct", arg0)
	ret0, _ := ret[0].(bool)
	return ret0
}

// SelfDestruct indicates an expected call of SelfDestruct.
func (mr *MockTransactionContextMockRecorder) SelfDestruct(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "SelfDestruct", reflect.TypeOf((*MockTransactionContext)(nil).SelfDestruct), arg0)
}

// SelfDestruct6780 mocks base method.
func (m *MockTransactionContext) SelfDestruct6780(arg0 Address) bool {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "SelfDestruct6780", arg0)
	ret0, _ := ret[0].(bool)
	return ret0
}

// SelfDestruct6780 indicates an expected call of SelfDestruct6780.
func (mr *MockTransactionContextMockRecorder) SelfDestruct6780(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "SelfDestruct6780", reflect.TypeOf((*MockTransactionContext)(nil).SelfDestruct6780), arg0)
}

// SetCode mocks base method.
func (m *MockTransactionContext) SetCode(arg0 Address, arg1 []byte) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "SetCode", arg0, arg1)
}

// SetCode indicates an expected call of SetCode.
func (mr *MockTransactionContextMockRecorder) SetCode(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "SetCode", reflect.TypeOf((*MockTransactionContext)(nil).SetCode), arg0, arg1)
}

// SetNonce mocks base method.
func (m *MockTransactionContext) SetNonce(arg0 Address, arg1 uint64) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "SetNonce", arg0, arg1)
}

// SetNonce indicates an expected call of SetNonce.
func (mr *MockTransactionContextMockRecorder) SetNonce(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "SetNonce", reflect.TypeOf((*MockTransactionContext)(nil).SetNonce), arg0, arg1)
}

// SetState mocks base method.
func (m *MockTransactionContext) SetState(arg0 Address, arg1 Key, arg2 Value) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "SetState", arg0, arg1, arg2)
}

// SetState indicates an expected call of SetState.
func (mr *MockTransactionContextMockRecorder) SetState(arg0, arg1, arg2 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "SetState", reflect.TypeOf((*MockTransactionContext)(nil).SetState), arg0, arg1, arg2)
}

// SetTransientState mocks base method.
func (m *MockTransactionContext) SetTransientState(arg0 Address, arg1 Key, arg2 Value) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "SetTransientState", arg0, arg1, arg2)
}

// SetTransientState indicates an expected call of SetTransientState.
func (mr *MockTransactionContextMockRecorder) SetTransientState(arg0, arg1, arg2 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "SetTransientState", reflect.TypeOf((*MockTransactionContext)(nil).SetTransientState), arg0, arg1, arg2)
}

// Snapshot mocks base method.
func (m *MockTransactionContext) Snapshot() int {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Snapshot")
	ret0, _ := ret[0].(int)
	return ret0
}

// Snapshot indicates an expected call of Snapshot.
func (mr *MockTransactionContextMockRecorder) Snapshot() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Snapshot", reflect.TypeOf((*MockTransactionContext)(nil).Snapshot))
}

// SubBalance mocks base method.
func (m *MockTransactionContext) SubBalance(arg0 Address, arg1 Amount) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "SubBalance", arg0, arg1)
}

// SubBalance indicates an expected call of SubBalance.
func (mr *MockTransactionContextMockRecorder) SubBalance(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "SubBalance", reflect.TypeOf((*MockTransactionContext)(nil).SubBalance), arg0, arg1)
}

// SubRefund mocks base method.
func (m *MockTransactionContext) SubRefund(arg0 uint64) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "SubRefund", arg0)
}

// SubRefund indicates an expected call of SubRefund.
func (mr *MockTransactionContextMockRecorder) SubRefund(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "SubRefund", reflect.TypeOf((*MockTransactionContext)(nil).SubRefund), arg0)
}

// MockQueryContext is a mock of QueryContext interface.
type MockQueryContext struct {
	ctrl     *gomock.Controller
	recorder *MockQueryContextMockRecorder
}

// MockQueryContextMockRecorder is the mock recorder for MockQueryContext.
type MockQueryContextMockRecorder struct {
	mock *MockQueryContext
}

// NewMockQueryContext creates a new mock instance.
func NewMockQueryContext(ctrl *gomock.Controller) *MockQueryContext {
	mock := &MockQueryContext{ctrl: ctrl}
	mock.recorder = &MockQueryContextMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockQueryContext) EXPECT() *MockQueryContextMockRecorder {
	return m.recorder
}

// GetBalance mocks base method.
func (m *MockQueryContext) GetBalance(arg0 Address) Amount {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetBalance", arg0)
	ret0, _ := ret[0].(Amount)
	return ret0
}

// GetBalance indicates an expected call of GetBalance.
func (mr *MockQueryContextMockRecorder) GetBalance(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetBalance", reflect.TypeOf((*MockQueryContext)(nil).GetBalance), arg0)
}

// GetCode mocks base method.
func (m *MockQueryContext) GetCode(arg0 Address) []byte {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetCode", arg0)
	ret0, _ := ret[0].([]byte)
	return ret0
}

// GetCode indicates an expected call of GetCode.
func (mr *MockQueryContextMockRecorder) GetCode(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetCode", reflect.TypeOf((*MockQueryContext)(nil).GetCode), arg0)
}

// GetCodeHash mocks base method.
func (m *MockQueryContext) GetCodeHash(arg0 Address) Hash {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetCodeHash", arg0)
	ret0, _ := ret[0].(Hash)
	return ret0
}

// GetCodeHash indicates an expected call of GetCodeHash.
func (mr *MockQueryContextMockRecorder) GetCodeHash(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetCodeHash", reflect.TypeOf((*MockQueryContext)(nil).GetCodeHash), arg0)
}

// GetCodeSize mocks base method.
func (m *MockQueryContext) GetCodeSize(arg0 Address) int {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetCodeSize", arg0)
	ret0, _ := ret[0].(int)
	return ret0
}

// GetCodeSize indicates an expected call of GetCodeSize.
func (mr *MockQueryContextMockRecorder) GetCodeSize(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetCodeSize", reflect.TypeOf((*MockQueryContext)(nil).GetCodeSize), arg0)
}

// GetNonce mocks base method.
func (m *MockQueryContext) GetNonce(arg0 Address) uint64 {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetNonce", arg0)
	ret0, _ := ret[0].(uint64)
	return ret0
}

// GetNonce indicates an expected call of GetNonce.
func (mr *MockQueryContextMockRecorder) GetNonce(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetNonce", reflect.TypeOf((*MockQueryContext)(nil).GetNonce), arg0)
}

// GetState mocks base method.
func (m *MockQueryContext) GetState(arg0 Address, arg1 Key) Value {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetState", arg0, arg1)
	ret0, _ := ret[0].(Value)
	return ret0
}

// GetState indicates an expected call of GetState.
func (mr *MockQueryContextMockRecorder) GetState(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetState", reflect.TypeOf((*MockQueryContext)(nil).GetState), arg0, arg1)
}

// GetStateHash mocks base method.
func (m *MockQueryContext) GetStateHash() Hash {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetStateHash")
	ret0, _ := ret[0].(Hash)
	return ret0
}

// GetStateHash indicates an expected call of GetStateHash.
func (mr *MockQueryContextMockRecorder) GetStateHash() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetStateHash", reflect.TypeOf((*MockQueryContext)(nil).GetStateHash))
}

// MockBulkLoad is a mock of BulkLoad interface.
type MockBulkLoad struct {
	ctrl     *gomock.Controller
	recorder *MockBulkLoadMockRecorder
}

// MockBulkLoadMockRecorder is the mock recorder for MockBulkLoad.
type MockBulkLoadMockRecorder struct {
	mock *MockBulkLoad
}

// NewMockBulkLoad creates a new mock instance.
func NewMockBulkLoad(ctrl *gomock.Controller) *MockBulkLoad {
	mock := &MockBulkLoad{ctrl: ctrl}
	mock.recorder = &MockBulkLoadMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockBulkLoad) EXPECT() *MockBulkLoadMockRecorder {
	return m.recorder
}

// CreateAccount mocks base method.
func (m *MockBulkLoad) CreateAccount(arg0 Address) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "CreateAccount", arg0)
}

// CreateAccount indicates an expected call of CreateAccount.
func (mr *MockBulkLoadMockRecorder) CreateAccount(arg0 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "CreateAccount", reflect.TypeOf((*MockBulkLoad)(nil).CreateAccount), arg0)
}

// Finalize mocks base method.
func (m *MockBulkLoad) Finalize() error {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Finalize")
	ret0, _ := ret[0].(error)
	return ret0
}

// Finalize indicates an expected call of Finalize.
func (mr *MockBulkLoadMockRecorder) Finalize() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Finalize", reflect.TypeOf((*MockBulkLoad)(nil).Finalize))
}

// SetBalance mocks base method.
func (m *MockBulkLoad) SetBalance(arg0 Address, arg1 Amount) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "SetBalance", arg0, arg1)
}

// SetBalance indicates an expected call of SetBalance.
func (mr *MockBulkLoadMockRecorder) SetBalance(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "SetBalance", reflect.TypeOf((*MockBulkLoad)(nil).SetBalance), arg0, arg1)
}

// SetCode mocks base method.
func (m *MockBulkLoad) SetCode(arg0 Address, arg1 []byte) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "SetCode", arg0, arg1)
}

// SetCode indicates an expected call of SetCode.
func (mr *MockBulkLoadMockRecorder) SetCode(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "SetCode", reflect.TypeOf((*MockBulkLoad)(nil).SetCode), arg0, arg1)
}

// SetNonce mocks base method.
func (m *MockBulkLoad) SetNonce(arg0 Address, arg1 uint64) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "SetNonce", arg0, arg1)
}

// SetNonce indicates an expected call of SetNonce.
func (mr *MockBulkLoadMockRecorder) SetNonce(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "SetNonce", reflect.TypeOf((*MockBulkLoad)(nil).SetNonce), arg0, arg1)
}

// SetState mocks base method.
func (m *MockBulkLoad) SetState(arg0 Address, arg1 Key, arg2 Value) {
	m.ctrl.T.Helper()
	m.ctrl.Call(m, "SetState", arg0, arg1, arg2)
}

// SetState indicates an expected call of SetState.
func (mr *MockBulkLoadMockRecorder) SetState(arg0, arg1, arg2 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "SetState", reflect.TypeOf((*MockBulkLoad)(nil).SetState), arg0, arg1, arg2)
}
