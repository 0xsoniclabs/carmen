package state

import (
	"fmt"
	"iter"
	"maps"
	"math"
	"math/rand/v2"
	reflect "reflect"
	"slices"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
)

func TestStateDB_CreateAccount_RevertsClearedAccountsOnInterTxSnapshotRevert(t *testing.T) {
	t.Parallel()
	ctrl := gomock.NewController(t)
	require := require.New(t)

	db := NewMockState(ctrl)
	state := createStateDBWith(db, 1, true)
	addr := common.Address{0x1}
	db.EXPECT().Check().Return(nil).AnyTimes()
	db.EXPECT().Flush().Return(nil).AnyTimes()
	db.EXPECT().Close().Return(nil).AnyTimes()
	db.EXPECT().GetBalance(gomock.Any()).Return(amount.New(0), nil).AnyTimes()
	db.EXPECT().GetNonce(gomock.Any()).Return(common.Nonce{0}, nil).AnyTimes()
	db.EXPECT().GetCode(gomock.Any()).Return(nil, nil).AnyTimes()
	db.EXPECT().GetCodeSize(gomock.Any()).Return(0, nil).AnyTimes()
	db.EXPECT().Exists(gomock.Any()).Return(false, nil).AnyTimes()

	val, exists := state.clearedAccounts[addr]
	require.False(exists)
	require.Equal(noClearing, val)

	snapshot := state.InterTxSnapshot()
	state.BeginTransaction()
	state.CreateAccount(addr)
	state.Suicide(addr)
	state.CreateAccount(addr)
	state.EndTransaction()

	if err := db.Check(); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	val, exists = state.clearedAccounts[addr]
	require.True(exists)
	require.Equal(cleared, val)

	err := state.RevertToInterTxSnapshot(snapshot)
	if err != nil {
		t.Fatalf("Error while reverting to inter-transaction snapshot: %v", err)
	}
	val, exists = state.clearedAccounts[addr]
	require.False(exists)
	require.Equal(noClearing, val)
}

func TestStateDB_Suicide_RevertsClearedAccountsOnSnapshotRevert(t *testing.T) {
	t.Parallel()
	ctrl := gomock.NewController(t)
	require := require.New(t)

	db := NewMockState(ctrl)
	addr := common.Address{0x1}
	db.EXPECT().Exists(addr).Return(true, nil).AnyTimes()
	db.EXPECT().Check().Return(nil).AnyTimes()
	state := createStateDBWith(db, 1, true)

	val, exists := state.clearedAccounts[addr]
	require.False(exists)
	require.Equal(noClearing, val)

	snapshot := state.InterTxSnapshot()
	state.BeginTransaction()
	state.Suicide(addr)
	state.CreateAccount(addr)
	state.clearedAccounts[addr] = noClearing
	state.Suicide(addr)
	state.EndTransaction()

	if err := db.Check(); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	val, exists = state.clearedAccounts[addr]
	require.True(exists)
	require.Equal(cleared, val)

	err := state.RevertToInterTxSnapshot(snapshot)
	if err != nil {
		t.Fatalf("Error while reverting to inter-transaction snapshot: %v", err)
	}
	val, exists = state.clearedAccounts[addr]
	require.False(exists)
	require.Equal(noClearing, val)
}

func TestStateDB_EndTransaction_RevertsStateOnInterTxSnapshotRevert(t *testing.T) {
	t.Parallel()
	ctrl := gomock.NewController(t)
	require := require.New(t)
	addr := common.Address{0x1}

	db := NewMockState(ctrl)
	db.EXPECT().Check().Return(nil).AnyTimes()
	db.EXPECT().Exists(addr).Return(true, nil).AnyTimes()
	db.EXPECT().GetBalance(gomock.Any()).Return(amount.New(0), nil).AnyTimes()
	db.EXPECT().GetNonce(gomock.Any()).Return(common.Nonce{0}, nil).AnyTimes()
	db.EXPECT().GetCode(gomock.Any()).Return(nil, nil).AnyTimes()
	db.EXPECT().GetCodeSize(gomock.Any()).Return(0, nil).AnyTimes()
	state := createStateDBWith(db, 1, true)

	// To Revert:
	// - Empty account: pending clearing (delete)
	// - Cleared account (existing)
	val, exists := state.clearedAccounts[addr]
	require.False(exists)
	require.Equal(noClearing, val)
	snapshot := state.InterTxSnapshot()
	state.BeginTransaction()
	state.emptyCandidates = append(state.emptyCandidates, addr)
	state.accountsToDelete = append(state.accountsToDelete, addr)
	state.EndTransaction()
	val, exists = state.clearedAccounts[addr]
	require.True(exists)
	require.Equal(cleared, val)

	err := state.RevertToInterTxSnapshot(snapshot)
	require.NoError(err)
	val, exists = state.clearedAccounts[addr]
	require.False(exists)
	require.Equal(noClearing, val)

	// To Revert:
	// - Cleared account (delete)
	clearedAccountValue, clearedAccountExists := state.clearedAccounts[addr]
	require.False(clearedAccountExists)
	require.Equal(noClearing, clearedAccountValue)
	snapshot = state.InterTxSnapshot()
	state.BeginTransaction()
	state.accountsToDelete = append(state.accountsToDelete, addr)
	state.EndTransaction()
	err = state.Check()

	require.NoError(err)
	clearedAccountValue, clearedAccountExists = state.clearedAccounts[addr]
	require.True(clearedAccountExists)
	require.Equal(cleared, clearedAccountValue)

	err = state.RevertToInterTxSnapshot(snapshot)
	require.NoError(err)
	clearedAccountValue, clearedAccountExists = state.clearedAccounts[addr]
	require.False(clearedAccountExists)
	require.Equal(noClearing, clearedAccountValue)

	// To Revert:
	// - Written slots
	// - Empty account: pending clearing (existing)
	// - Has suicided: pending clearing
	// - Data
	key := common.Key{0x2}
	clearedAccountVal, clearedAccountExists := state.clearedAccounts[addr]
	require.False(clearedAccountExists)
	require.Equal(noClearing, clearedAccountVal)

	// Set data to test data undo
	valueToWrite := common.Value{0x3}
	slotId := slotId{addr, key}
	_, slotExists := state.data.Get(slotId)
	require.False(slotExists)
	dataSnapshot := state.InterTxSnapshot()
	state.BeginTransaction()
	state.SetState(addr, key, valueToWrite)
	state.EndTransaction()
	slotValue, slotExists := state.data.Get(slotId)
	require.True(slotExists)
	require.Equal(valueToWrite, slotValue.committed)

	snapshot = state.InterTxSnapshot()
	state.BeginTransaction()
	state.emptyCandidates = append(state.emptyCandidates, addr)
	state.Suicide(addr)
	state.EndTransaction()
	require.NoError(state.Check())

	slotValue, slotExists = state.data.Get(slotId)
	require.True(slotExists)
	require.Equal(slotValue.committed, common.Value{}) // Value reset by suicide
	clearedAccountVal, clearedAccountExists = state.clearedAccounts[addr]
	require.True(clearedAccountExists)
	require.Equal(cleared, clearedAccountVal)

	err = state.RevertToInterTxSnapshot(snapshot)
	require.NoError(err)
	slotValue, slotExists = state.data.Get(slotId)
	require.True(slotExists)
	require.Equal(valueToWrite, slotValue.committed)

	// Back to initial state
	err = state.RevertToInterTxSnapshot(dataSnapshot)
	require.NoError(err)
	_, slotExists = state.data.Get(slotId)
	require.False(slotExists)
}

func TestStateDB_InterTxSnapshot_ThrowsErrorIfWithinTransaction(t *testing.T) {
	t.Parallel()
	require := require.New(t)
	ctrl := gomock.NewController(t)
	db := NewMockState(ctrl)
	state := createStateDBWith(db, 1, true)

	state.BeginTransaction()
	require.True(state.withinTransaction)
	_ = state.InterTxSnapshot()
	require.Equal(len(state.errors), 1)
	err := state.errors[0]
	require.EqualError(err, "cannot create inter-transaction snapshot in a transaction")
}

func TestStateDB_InterTxSnapshot_ReturnsSnapshotID(t *testing.T) {
	t.Parallel()
	require := require.New(t)
	ctrl := gomock.NewController(t)
	db := NewMockState(ctrl)
	state := createStateDBWith(db, 1, true)

	snapshotID := state.InterTxSnapshot()
	require.Equal(snapshotID, InterTxSnapshotID(0), "unexpected snapshot ID: want %d, got %d", 0, snapshotID)

	state.BeginTransaction()
	state.EndTransaction()
	snapshotID2 := state.InterTxSnapshot()
	require.Equal(snapshotID2, InterTxSnapshotID(1), "unexpected snapshot ID: want %d, got %d", 1, snapshotID2)
}

func TestStateDB_RevertToInterTxSnapshot_ThrowsErrorIfWithinTransaction(t *testing.T) {
	t.Parallel()
	require := require.New(t)
	ctrl := gomock.NewController(t)
	db := NewMockState(ctrl)
	state := createStateDBWith(db, 1, true)

	state.BeginTransaction()
	require.True(state.withinTransaction)
	err := state.RevertToInterTxSnapshot(0)
	require.EqualError(err, "cannot revert to inter-transaction snapshot in a transaction")
}

func TestStateDB_RevertToInterTxSnapshot_ThrowsErrorIfInvalidSnapshotID(t *testing.T) {
	t.Parallel()
	require := require.New(t)
	ctrl := gomock.NewController(t)
	db := NewMockState(ctrl)
	state := createStateDBWith(db, 1, true)

	err := state.RevertToInterTxSnapshot(1)
	require.EqualError(err, "cannot revert to inter-transaction snapshot 1, only 0 snapshots in the current block")

	state.BeginTransaction()
	state.EndTransaction()
	err = state.RevertToInterTxSnapshot(2)
	require.EqualError(err, "cannot revert to inter-transaction snapshot 2, only 1 snapshots in the current block")
}

func TestStateDB_RevertToInterTxSnapshot_RevertsStateCorrectly(t *testing.T) {
	type InterTxSnapshotWithStateCheck struct {
		stateBackup *stateDB
		snapshotID  InterTxSnapshotID
	}

	operationListWithAddress := map[string]func(ctx *StateDBContext, args OpArgs){
		"setNonce":      setNonceOp,
		"setCode":       setCodeOp,
		"addBalance":    addBalanceOp,
		"subBalance":    subBalanceOp,
		"createAccount": createAccountOp,
		"suicide":       suicideOp,
	}

	operationListWithAddressAndKey := map[string]func(ctx *StateDBContext, args OpArgs){
		"setState": setStateOp,
	}

	var opWithNameList []StateDBOperation
	for opName, op := range operationListWithAddress {
		for i, address := range addresses {
			opWithNameList = append(opWithNameList, StateDBOperation{
				name: fmt.Sprintf("%s addr %d", opName, i),
				op:   op,
				args: OpArgs{address: address},
			})
		}
	}
	for opName, op := range operationListWithAddressAndKey {
		for i, address := range addresses {
			for j, key := range keys {
				op := StateDBOperation{
					name: fmt.Sprintf("%s addr %d key %d", opName, i, j),
					op:   op,
					args: OpArgs{address: address, key: key},
				}
				opWithNameList = append(opWithNameList, op)
				// Simulate multiple writes on the same address and key
				opWithNameList = append(opWithNameList, op)
			}
		}
	}

	testNameFunc := func(t1, t2, t3 StateDBOperation) string {
		return fmt.Sprintf("%s %s  %s", t1.name, t2.name, t3.name)
	}

	for testCaseName, testCaseList := range CartesianTriple(opWithNameList, testNameFunc) {
		t.Run(testCaseName, func(t *testing.T) {
			t.Parallel()
			require := require.New(t)

			ctx := NewStateDBContext(t)

			var statesToCheck []InterTxSnapshotWithStateCheck
			for _, testCase := range testCaseList {
				snapshotID := ctx.state.InterTxSnapshot()
				require.Empty(ctx.state.accountsToDelete)
				require.Empty(ctx.state.writtenSlots)
				oldStateDB := backupStateDB(ctx.state)

				ctx.state.BeginTransaction()
				testCase.Execute(ctx)
				ctx.state.EndTransaction()

				statesToCheck = append(statesToCheck, InterTxSnapshotWithStateCheck{
					stateBackup: oldStateDB,
					snapshotID:  snapshotID,
				})
			}

			slices.Reverse(statesToCheck)
			for _, cur := range statesToCheck {
				require.NoError(
					ctx.state.RevertToInterTxSnapshot(cur.snapshotID),
				)
				checkStateDB(t, cur.stateBackup, ctx.state)
			}
		})
	}
}

// StateDBContext is a helper struct containing a state and values to be used in subsequent operations on it, to be used in the tests of StateDB transaction revert functionality. Such values are supposed to be mutated by the operations to properly check after reverts that the state is properly reverted to the previous state.
type StateDBContext struct {
	state *stateDB
	db    *MockState
	rng   *rand.Rand
}

// OpArgs is a struct containing an address and a key, to be used as arguments for operations that require them in the tests of StateDB transaction revert functionality.
type OpArgs struct {
	address common.Address
	key     common.Key
}

// StateDBOperation is an helper struct representing an operation to be performed on the StateDB, containing the operation function, its name for better test readability, and the address and key arguments to be used in the operation if needed.
type StateDBOperation struct {
	name string
	op   func(ctx *StateDBContext, args OpArgs)
	args OpArgs
}

// Execute executes the operation on the given StateDBContext with the stored arguments.
func (op *StateDBOperation) Execute(ctx *StateDBContext) {
	op.op(ctx, op.args)
}

var (
	mockCode       = []byte{0x1}
	mockBalance    = amount.New(0)
	mockNonce      = common.Nonce{0}
	mockStateValue = common.Value{0x1}
	mockCodeSize   = math.MaxInt
	addresses      = []common.Address{
		{0x1},
		{0x2},
		{0x3},
	}
	keys = []common.Key{
		{0x4},
		{0x5},
		{0x6},
	}
)

// NewStateDBContext creates a new StateDBContext with a mocked State. It sets up an address and key with initial values, and sets expectations on the mocked State for operations that might be performed on the address and key.
func NewStateDBContext(t *testing.T) *StateDBContext {
	t.Helper()

	ctrl := gomock.NewController(t)
	db := NewMockState(ctrl)
	db.EXPECT().Check().Return(nil).AnyTimes()
	db.EXPECT().Flush().Return(nil).AnyTimes()
	db.EXPECT().Close().Return(nil).AnyTimes()

	state := createStateDBWith(db, 1, true)

	// Set expectation in case values are not cached, i.e. they are untouched.
	for _, address := range addresses {
		db.EXPECT().Exists(address).Return(false, nil).MaxTimes(1)
		db.EXPECT().GetBalance(address).Return(mockBalance, nil).MaxTimes(1)
		db.EXPECT().GetNonce(address).Return(mockNonce, nil).MaxTimes(1)
		db.EXPECT().GetCode(address).Return(mockCode, nil).MaxTimes(1)
		db.EXPECT().GetCodeSize(address).Return(mockCodeSize, nil).MaxTimes(1)
		for _, key := range keys {
			db.EXPECT().GetStorage(address, key).Return(mockStateValue, nil).MaxTimes(1)
		}
	}

	return &StateDBContext{
		state: state,
		db:    db,
		rng:   rand.New(rand.NewPCG(42, 42)),
	}
}

func setStateOp(ctx *StateDBContext, args OpArgs) {
	randomValue := common.Value(randomByteArrayWithPrefix(ctx.rng, 32, []byte{0x2}))
	ctx.state.SetState(args.address, args.key, randomValue)
}

func setNonceOp(ctx *StateDBContext, args OpArgs) {
	ctx.state.SetNonce(args.address, ctx.rng.Uint64N(math.MaxUint64)+1)
}

func setCodeOp(ctx *StateDBContext, args OpArgs) {
	randomCode := randomByteArrayWithPrefix(ctx.rng, 8, []byte{0x2})
	ctx.state.SetCode(args.address, randomCode)
}

func addBalanceOp(ctx *StateDBContext, args OpArgs) {
	ctx.state.AddBalance(args.address, amount.New(10))
}

func subBalanceOp(ctx *StateDBContext, args OpArgs) {
	ctx.state.SubBalance(args.address, amount.New(10))
}

func createAccountOp(ctx *StateDBContext, args OpArgs) {
	ctx.state.CreateAccount(args.address)
}

func suicideOp(ctx *StateDBContext, args OpArgs) {
	ctx.state.Suicide(args.address)
}

func Test_partialCopyStateDB_copiesStateDBApartFromStoredDataCache(t *testing.T) {
	t.Parallel()

	ctrl := gomock.NewController(t)
	st := NewMockState(ctrl)

	st.EXPECT().Check().Return(nil).AnyTimes()
	st.EXPECT().Flush().Return(nil).AnyTimes()
	st.EXPECT().Close().Return(nil).AnyTimes()

	state := createStateDBWith(st, defaultStoredDataCacheSize, true)
	defer func() {
		require.NoError(t, state.Close())
	}()

	// Create a state with some data and set all fields that are copied by partialCopyStateDB
	addr := common.Address{0x1}
	state.BeginTransaction()

	state.accounts[addr] = &accountState{}
	state.balances[addr] = &balanceValue{original: func() *amount.Amount { a := amount.New(100); return &a }(), current: amount.New(10)}
	state.nonces[addr] = &nonceValue{original: func() *uint64 { n := uint64(100); return &n }(), current: 200}
	state.reincarnation[addr] = 42
	state.codes[addr] = &codeValue{code: []byte{0x1}}
	state.undo = [][]func(){{func() {}}}
	state.clearedAccounts[addr] = 1
	state.canApplyChanges = true

	copiedState := backupStateDB(state)

	require.NoError(t, checkStateDB(t, state, copiedState))
}

// backupStateDB creates a partial copy of the stateDB to be used to check transaction revert effects.
func backupStateDB(s *stateDB) *stateDB {
	ns := createStateDBWith(s.state, 1, true)
	ns.accounts = cloneMapWith(s.accounts, cloneValue)
	ns.balances = cloneMapWith(s.balances, cloneBalanceValue)
	ns.nonces = cloneMapWith(s.nonces, cloneNonceValue)
	copyFastMapWith(s.data, ns.data, cloneValue)
	s.transientStorage.CopyTo(ns.transientStorage)
	ns.reincarnation = maps.Clone(s.reincarnation)
	ns.codes = cloneMapWith(s.codes, cloneCodeValue)
	for undo := range s.undo {
		ns.undo = append(ns.undo, slices.Clone(s.undo[undo]))
	}
	ns.clearedAccounts = maps.Clone(s.clearedAccounts)
	ns.createdContracts = maps.Clone(s.createdContracts)
	ns.emptyCandidates = slices.Clone(s.emptyCandidates)
	ns.canApplyChanges = s.canApplyChanges

	return ns
}

// checkStateDB checks if the `actual` reverted stateDB fields match the backup `expected` stateDB ones, along with postconditions on transaction context fields.
func checkStateDB(t *testing.T, expected *stateDB, actual *stateDB) error {
	t.Helper()

	for addr, account := range actual.accounts {
		value, exists := expected.accounts[addr]
		if !((exists && reflect.DeepEqual(account, value)) || (!exists && account.current == account.original && account.current == accountNonExisting)) {
			return fmt.Errorf("accounts differ at address %v: got %v", addr, account)
		}
	}
	for addr, balance := range actual.balances {
		value, exists := expected.balances[addr]
		if !((exists && reflect.DeepEqual(balance, value)) || (!exists && balance.current == mockBalance && balance.original == &balance.current)) {
			return fmt.Errorf("balances differ at address %v: got %v", addr, balance)
		}
	}
	for addr, nonce := range actual.nonces {
		value, exists := expected.nonces[addr]
		if !((exists && reflect.DeepEqual(nonce, value)) || (!exists && nonce.current == mockNonce.ToUint64())) {
			return fmt.Errorf("nonces differ at address %v: got %v", addr, nonce)
		}
	}

	if !fastMapEqual(expected.data, actual.data) {
		return fmt.Errorf("data maps differ: expected %v, got %v", expected.data, actual.data)
	}
	if !reflect.DeepEqual(expected.reincarnation, actual.reincarnation) {
		return fmt.Errorf("reincarnation maps differ: expected %v, got %v", expected.reincarnation, actual.reincarnation)
	}

	for addr, code := range actual.codes {
		value, exists := expected.codes[addr]
		if !((exists && reflect.DeepEqual(code, value)) || (!exists && (code.code == nil || slices.Equal(code.code, mockCode)))) {
			return fmt.Errorf("codes differ at address %v: expected %v, got %v", addr, value, code)
		}
	}

	if !reflect.DeepEqual(expected.clearedAccounts, actual.clearedAccounts) {
		return fmt.Errorf("clearedAccounts maps differ: expected %v, got %v", expected.clearedAccounts, actual.clearedAccounts)
	}
	if !reflect.DeepEqual(expected.canApplyChanges, actual.canApplyChanges) {
		return fmt.Errorf("canApplyChanges differ: expected %v, got %v", expected.canApplyChanges, actual.canApplyChanges)
	}

	// For the undo, we check function pointers as they will stay in memory
	for i := range expected.undo {
		if len(expected.undo[i]) != len(actual.undo[i]) {
			return fmt.Errorf("undo functions length differ at transaction %d: expected %d, got %d", i, len(expected.undo[i]), len(actual.undo[i]))
		}
		for j := range expected.undo[i] {
			addr1 := reflect.ValueOf(expected.undo[i][j]).Pointer()
			addr2 := reflect.ValueOf(actual.undo[i][j]).Pointer()
			if addr1 != addr2 {
				return fmt.Errorf("undo functions differ at transaction %d, function %d: expected %v, got %v", i, j, addr1, addr2)
			}
		}
	}

	// Transaction context field should have been reset
	if actual.refund != 0 {
		return fmt.Errorf("refund mismatch: expected 0, got %d", actual.refund)
	}
	if actual.accessedSlots.Size() != 0 {
		return fmt.Errorf("accessedSlots size mismatch: expected 0, got %d", actual.accessedSlots.Size())
	}
	if len(actual.accessedAddresses) != 0 {
		return fmt.Errorf("accessedAddresses not empty: got %v", actual.accessedAddresses)
	}
	if len(actual.accountsToDelete) != 0 {
		return fmt.Errorf("accountsToDelete not empty: got %v", actual.accountsToDelete)
	}
	if len(actual.logs) != 0 {
		return fmt.Errorf("logs not empty: got %v", actual.logs)
	}
	if len(actual.emptyCandidates) != 0 {
		return fmt.Errorf("emptyCandidates not empty: got %v", actual.emptyCandidates)
	}
	if len(actual.createdContracts) != 0 {
		return fmt.Errorf("createdContracts not empty: got %v", actual.createdContracts)
	}
	if actual.transientStorage.Size() != 0 {
		return fmt.Errorf("transientStorage size mismatch: expected 0, got %d", actual.transientStorage.Size())
	}

	return nil
}

func Test_checkStateDB_checksStateDBRevertedFieldsAndPostconditions(t *testing.T) {
	t.Parallel()

	// Setup addresses, keys, and values
	addr := common.Address{0x1}
	key := common.Key{0x1}
	nonce := uint64(42)
	code := []byte{0x2}

	s1 := createStateDBWith(NewMockState(gomock.NewController(t)), 1, true)
	s2 := createStateDBWith(NewMockState(gomock.NewController(t)), 1, true)

	s1.accounts[addr] = &accountState{current: 1, original: 1}
	s2.accounts[addr] = &accountState{current: 1, original: 1}
	bv := &balanceValue{}
	s1.balances[addr] = bv
	s2.balances[addr] = bv
	nv := &nonceValue{current: nonce}
	s1.nonces[addr] = nv
	s2.nonces[addr] = nv
	slot := slotId{addr, key}
	sv := &slotValue{}
	s1.data.Put(slot, sv)
	s2.data.Put(slot, sv)
	s1.reincarnation[addr] = 99
	s2.reincarnation[addr] = 99
	cv := &codeValue{code: code}
	s1.codes[addr] = cv
	s2.codes[addr] = cv
	s1.clearedAccounts[addr] = 1
	s2.clearedAccounts[addr] = 1
	s1.canApplyChanges = true
	s2.canApplyChanges = true
	f := func() {}
	s1.undo = [][]func(){{f}}
	s2.undo = [][]func(){{f}}
	// Caches populated by read-only functions with mock values should not cause checkStateDB to fail
	newAddr := common.Address{0x2}
	s1.accounts[newAddr] = &accountState{current: accountNonExisting, original: accountNonExisting}
	s1.balances[newAddr] = &balanceValue{current: mockBalance, original: &mockBalance}
	s1.nonces[newAddr] = &nonceValue{current: mockNonce.ToUint64(), original: nil}
	s1.codes[newAddr] = &codeValue{code: nil}
	s1.codes[common.Address{0x3}] = &codeValue{code: mockCode}

	require.NoError(t, checkStateDB(t, s1, s2))
}

func Test_checkStateDB_failsOnDifferentStateDB(t *testing.T) {
	t.Parallel()

	require := require.New(t)
	addr := common.Address{0x1}
	key := common.Key{0x1}
	value := common.Value{0x1}
	nonce := uint64(42)
	code := []byte{0x2}
	slot := slotId{addr, key}
	bv := &balanceValue{current: amount.New(42)}
	nv := &nonceValue{current: nonce}
	sv := &slotValue{stored: value}
	cv := &codeValue{code: code}
	as := accountState{current: accountExists, original: accountExists}
	f := func() {}

	s1 := createStateDBWith(NewMockState(gomock.NewController(t)), 1, true)
	s2 := createStateDBWith(NewMockState(gomock.NewController(t)), 1, true)

	// Set all fields to be equal
	s1.accounts[addr] = &as
	s2.accounts[addr] = &as
	s1.balances[addr] = bv
	s2.balances[addr] = bv
	s1.nonces[addr] = nv
	s2.nonces[addr] = nv
	s1.data.Put(slot, sv)
	s2.data.Put(slot, sv)
	s1.reincarnation[addr] = 99
	s2.reincarnation[addr] = 99
	s1.codes[addr] = cv
	s2.codes[addr] = cv
	s1.clearedAccounts[addr] = 1
	s2.clearedAccounts[addr] = 1
	s1.canApplyChanges = true
	s2.canApplyChanges = true
	s1.undo = [][]func(){{f}}
	s2.undo = [][]func(){{f}}
	require.NoError(checkStateDB(t, s1, s2))

	s2.accounts[addr] = &accountState{current: accountNonExisting, original: accountExists}
	require.Error(checkStateDB(t, s1, s2))
	s2.accounts[addr] = &as
	require.NoError(checkStateDB(t, s1, s2))

	tmpBalance := balanceValue{current: amount.New(100), original: &bv.current}
	s2.balances[addr] = &tmpBalance
	require.Error(checkStateDB(t, s1, s2))
	s2.balances[addr] = bv
	require.NoError(checkStateDB(t, s1, s2))

	s2.nonces[addr] = &nonceValue{current: nonce + 1}
	require.Error(checkStateDB(t, s1, s2))
	s2.nonces[addr] = nv
	require.NoError(checkStateDB(t, s1, s2))

	s2.data.Put(slot, &slotValue{})
	require.Error(checkStateDB(t, s1, s2))
	s2.data.Put(slot, sv)
	require.NoError(checkStateDB(t, s1, s2))

	s2.reincarnation[addr] = 100
	require.Error(checkStateDB(t, s1, s2))
	s2.reincarnation[addr] = 99
	require.NoError(checkStateDB(t, s1, s2))

	s2.codes[addr] = &codeValue{code: []byte{0x3}}
	require.Error(checkStateDB(t, s1, s2))
	s2.codes[addr] = cv
	require.NoError(checkStateDB(t, s1, s2))

	s2.clearedAccounts[addr] = 2
	require.Error(checkStateDB(t, s1, s2))
	s2.clearedAccounts[addr] = 1
	require.NoError(checkStateDB(t, s1, s2))

	s2.canApplyChanges = false
	require.Error(checkStateDB(t, s1, s2))
	s2.canApplyChanges = true
	require.NoError(checkStateDB(t, s1, s2))

	s2.undo = [][]func(){{func() {}}}
	require.Error(checkStateDB(t, s1, s2))
	s2.undo = [][]func(){{f}}
	require.NoError(checkStateDB(t, s1, s2))

}

// fastMapEqual checks if two FastMaps are equal. If the values are pointers, the pointed values are compared for equality instead of the pointers themselves.
func fastMapEqual[K comparable, V comparable](m1, m2 *common.FastMap[K, V]) bool {
	equal := m1.Size() == m2.Size()
	if equal {
		m1.ForEach(func(key K, value V) {
			v2, ok := m2.Get(key)
			if !ok {
				equal = false
				return
			}
			if reflect.ValueOf(value).Kind() == reflect.Pointer {
				equal = (reflect.ValueOf(value).IsNil() && reflect.ValueOf(v2).IsNil()) || (reflect.ValueOf(value).Elem().Interface() == reflect.ValueOf(v2).Elem().Interface())
			} else {
				equal = reflect.DeepEqual(value, v2)
			}
		})
	}
	return equal
}

func copyFastMapWith[K comparable, V any](src *common.FastMap[K, V], dst *common.FastMap[K, V], cloneFunc func(V) V) {
	if src == nil || dst == nil {
		return
	}
	src.ForEach(func(key K, value V) {
		dst.Put(key, cloneFunc(value))
	})
}

func cloneMapWith[K comparable, V any](m map[K]V, cloneFunc func(V) V) map[K]V {
	if m == nil {
		return nil
	}
	cloned := make(map[K]V, len(m))
	for k, v := range m {
		cloned[k] = cloneFunc(v)
	}
	return cloned
}

func cloneValue[V any](v *V) *V {
	if v == nil {
		return nil
	}
	cloned := *v
	return &cloned
}

func cloneBalanceValue(bv *balanceValue) *balanceValue {
	if bv == nil {
		return nil
	}
	cloned := *bv
	if bv.original != nil {
		originalCopy := *bv.original
		cloned.original = &originalCopy
	}
	return &cloned
}

func cloneNonceValue(nv *nonceValue) *nonceValue {
	if nv == nil {
		return nil
	}
	cloned := *nv
	if nv.original != nil {
		originalCopy := *nv.original
		cloned.original = &originalCopy
	}
	return &cloned
}

func cloneCodeValue(cv *codeValue) *codeValue {
	if cv == nil {
		return nil
	}
	cloned := *cv
	if cv.hash != nil {
		hashCopy := *cv.hash
		cloned.hash = &hashCopy
	}
	cloned.code = slices.Clone(cv.code)
	return &cloned
}

// randomByteArrayWithPrefix generates a random byte array of the given size, where the first bytes are the given prefix.
func randomByteArrayWithPrefix(rng *rand.Rand, size int, prefix []byte) []byte {
	b := make([]byte, size)
	copy(b, prefix)
	for i := len(prefix); i < size; i++ {
		b[i] = byte(rng.Uint64N(256))
	}
	return b
}

// CartesianTriple generates the cartesian product of a slice with itself three times, yielding the result as a sequence of tuples containing the values and string representations of them.
func CartesianTriple[T any](slice []T, genName func(T, T, T) string) iter.Seq2[string, [3]T] {
	return func(yield func(string, [3]T) bool) {
		for _, a := range slice {
			for _, b := range slice {
				for _, c := range slice {
					if !yield(genName(a, b, c), [3]T{a, b, c}) {
						return
					}
				}
			}
		}
	}
}
