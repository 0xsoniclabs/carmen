// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package state

import (
	"fmt"
	"iter"
	"maps"
	"math"
	"math/rand/v2"
	reflect "reflect"
	"slices"
	"strings"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
)

func TestStateDB_InterTxSnapshot_ReturnsSnapshotID(t *testing.T) {
	t.Parallel()
	require := require.New(t)
	ctrl := gomock.NewController(t)
	db := NewMockState(ctrl)
	db.EXPECT().Exists(gomock.Any()).Return(false, nil).AnyTimes()
	db.EXPECT().Check().Return(nil).AnyTimes()
	state := createStateDBWith(db, 1, true)

	snapshotID := state.InterTxSnapshot()
	require.Equal(InterTxSnapshotID(0), snapshotID)

	// Empty transaction does not increment snapshot ID
	state.BeginTransaction()
	state.EndTransaction()
	snapshotID2 := state.InterTxSnapshot()
	require.Equal(InterTxSnapshotID(0), snapshotID2)

	state.BeginTransaction()
	state.SetCode(common.Address{}, []byte{0x1})
	state.EndTransaction()
	snapshotID3 := state.InterTxSnapshot()
	require.Equal(InterTxSnapshotID(3), snapshotID3)
}

func TestStateDB_RevertToInterTxSnapshot_RecordsErrorIfInvalidSnapshotID(t *testing.T) {
	t.Parallel()
	require := require.New(t)
	ctrl := gomock.NewController(t)
	db := NewMockState(ctrl)
	db.EXPECT().Exists(gomock.Any()).Return(false, nil).AnyTimes()
	db.EXPECT().Check().Return(nil).AnyTimes()
	state := createStateDBWith(db, 1, true)

	state.RevertToInterTxSnapshot(1)
	require.EqualError(state.Check(), "cannot revert to inter-transaction snapshot: the snapshot point is ahead of the last recorded one")
	state.errors = state.errors[:0]

	state.BeginTransaction()
	state.SetCode(common.Address{}, []byte{0x1})
	state.EndTransaction()
	state.RevertToInterTxSnapshot(10)
	require.EqualError(state.Check(), "cannot revert to inter-transaction snapshot: the snapshot point is ahead of the last recorded one")
}

func TestStateDB_EndTransaction_AddsUndoForRestoreWhen(t *testing.T) {
	runTests := func(t *testing.T, beginOp func(state *stateDB)) {
		testCases := map[string](func(t *testing.T, state *stateDB, beginOp func(state *stateDB))){
			"UpdatingWrittenSlots":                UpdatingWrittenSlots,
			"AddingEmptyAccountWithoutState":      AddingEmptyAccountWithState,
			"AddingEmptyAccountWithState":         AddingEmptyAccountWithState,
			"DeletingSuicidedAccountWithoutState": DeletingSuicidedAccountWithoutState,
			"DeletingSuicidedAccountWithState":    DeletingSuicidedAccountWithState,
			"DeletingAccountWithData":             DeletingAccountWithData,
			"DeletingAccountWithoutState":         DeletingAccountWithoutState,
			"DeletingAccountWithState":            DeletingAccountWithState,
		}

		for name, testFunc := range testCases {
			t.Run(name, func(t *testing.T) {
				ctrl := gomock.NewController(t)
				db := NewMockState(ctrl)
				db.EXPECT().Check().Return(nil).AnyTimes()
				db.EXPECT().Check().Return(nil).AnyTimes()
				db.EXPECT().Exists(gomock.Any()).Return(true, nil).AnyTimes()
				db.EXPECT().GetBalance(gomock.Any()).Return(amount.New(0), nil).AnyTimes()
				db.EXPECT().GetNonce(gomock.Any()).Return(common.Nonce{0}, nil).AnyTimes()
				db.EXPECT().GetCode(gomock.Any()).Return(nil, nil).AnyTimes()
				db.EXPECT().GetCodeSize(gomock.Any()).Return(0, nil).AnyTimes()
				state := createStateDBWith(db, 1, true)

				testFunc(t, state, beginOp)
			})
		}
	}

	t.Run("WithBeginTransaction", func(t *testing.T) {
		runTests(t, func(state *stateDB) {
			state.BeginTransaction()
		})
	})

	t.Run("WithoutBeginTransaction", func(t *testing.T) {
		runTests(t, func(state *stateDB) {
			// No-op
		})
	})
}

func UpdatingWrittenSlots(t *testing.T, state *stateDB, beginOp func(state *stateDB)) {
	t.Parallel()
	require := require.New(t)

	backup := state.InterTxSnapshot()

	beginOp(state)
	entry := slotValue{committed: common.Value{0x1}, current: common.Value{0x2}}
	state.writtenSlots[&entry] = true
	state.EndTransaction()

	require.Equal(entry.committed, common.Value{0x2})

	state.RevertToInterTxSnapshot(backup)
	require.Equal(entry.committed, common.Value{0x1})
	require.Equal(entry.current, common.Value{0x2})
}

func AddingEmptyAccountWithoutState(t *testing.T, state *stateDB, beginOp func(state *stateDB)) {
	t.Parallel()
	require := require.New(t)

	addr := common.Address{0x1}
	backup := state.InterTxSnapshot()
	beginOp(state)
	state.emptyCandidates = append(state.emptyCandidates, addr)
	state.EndTransaction()
	require.Equal(state.clearedAccounts[addr], cleared)

	state.RevertToInterTxSnapshot(backup)
	require.Empty(state.clearedAccounts)

}

func AddingEmptyAccountWithState(t *testing.T, state *stateDB, beginOp func(state *stateDB)) {
	t.Parallel()
	require := require.New(t)

	addr := common.Address{0x1}
	backup := state.InterTxSnapshot()

	beginOp(state)
	state.clearedAccounts[addr] = noClearing
	state.emptyCandidates = append(state.emptyCandidates, addr)
	state.EndTransaction()
	require.Equal(state.clearedAccounts[addr], cleared)

	state.RevertToInterTxSnapshot(backup)
	require.Equal(state.clearedAccounts[addr], noClearing)
}

func DeletingSuicidedAccountWithoutState(t *testing.T, state *stateDB, beginOp func(state *stateDB)) {
	t.Parallel()
	require := require.New(t)

	addr := common.Address{0x1}
	backup := state.InterTxSnapshot()

	beginOp(state)
	state.accounts[addr] = &accountState{current: accountSelfDestructed}
	state.accountsToDelete = append(state.accountsToDelete, addr)
	state.EndTransaction()
	require.Equal(state.clearedAccounts[addr], cleared)

	state.RevertToInterTxSnapshot(backup)
	require.Empty(state.clearedAccounts)
}

func DeletingSuicidedAccountWithState(t *testing.T, state *stateDB, beginOp func(state *stateDB)) {
	t.Parallel()
	require := require.New(t)

	addr := common.Address{0x1}
	backup := state.InterTxSnapshot()

	beginOp(state)
	state.clearedAccounts[addr] = noClearing
	state.accounts[addr] = &accountState{current: accountSelfDestructed}
	state.accountsToDelete = append(state.accountsToDelete, addr)
	state.EndTransaction()
	require.Equal(state.clearedAccounts[addr], cleared)

	state.RevertToInterTxSnapshot(backup)
	require.Equal(state.clearedAccounts[addr], noClearing)
}

func DeletingAccountWithData(t *testing.T, state *stateDB, beginOp func(state *stateDB)) {
	t.Parallel()
	require := require.New(t)
	addr := common.Address{0x1}
	key := common.Key{0x2}

	defaultSlotValue := slotValue{
		stored:         common.Value{0x1},
		storedKnown:    false,
		committed:      common.Value{0x1},
		committedKnown: false,
		current:        common.Value{0x1},
	}
	testSlotValue := defaultSlotValue

	state.data.Put(slotId{addr, key}, &testSlotValue)
	backup := state.InterTxSnapshot()

	beginOp(state)
	state.accountsToDelete = append(state.accountsToDelete, addr)
	state.EndTransaction()

	require.Equal(state.clearedAccounts[addr], cleared)
	require.Equal(testSlotValue.stored, common.Value{})
	require.Equal(testSlotValue.storedKnown, true)
	require.Equal(testSlotValue.committed, common.Value{})
	require.Equal(testSlotValue.committedKnown, true)
	require.Equal(testSlotValue.current, common.Value{})

	state.RevertToInterTxSnapshot(backup)
	require.Empty(state.clearedAccounts)
	require.Equal(testSlotValue, defaultSlotValue)
}

func DeletingAccountWithoutState(t *testing.T, state *stateDB, beginOp func(state *stateDB)) {
	t.Parallel()
	require := require.New(t)
	addr := common.Address{0x1}

	backup := state.InterTxSnapshot()

	beginOp(state)
	state.accountsToDelete = append(state.accountsToDelete, addr)
	state.EndTransaction()
	require.Equal(state.clearedAccounts[addr], cleared)

	state.RevertToInterTxSnapshot(backup)
	require.Empty(state.clearedAccounts)
}

func DeletingAccountWithState(t *testing.T, state *stateDB, beginOp func(state *stateDB)) {
	t.Parallel()
	require := require.New(t)
	addr := common.Address{0x1}

	backup := state.InterTxSnapshot()

	beginOp(state)
	state.clearedAccounts[addr] = noClearing
	state.accountsToDelete = append(state.accountsToDelete, addr)
	state.EndTransaction()
	require.Equal(state.clearedAccounts[addr], cleared)

	state.RevertToInterTxSnapshot(backup)
	require.Equal(state.clearedAccounts[addr], noClearing)
}

func TestStateDB_RevertToInterTxSnapshot_RevertsStateCorrectly(t *testing.T) {
	type InterTxSnapshotWithStateCheck struct {
		stateBackup *stateDB
		snapshotID  InterTxSnapshotID
	}

	addresses := []common.Address{
		{0x1},
		{0x2},
		{0x3},
	}
	keys := []common.Key{
		{0x4},
		{0x5},
		{0x6},
	}

	operationWithAddress := map[string]func(ctx *StateDBContext, args OpArgs){
		"setNonce":      setNonceOp,
		"setCode":       setCodeOp,
		"addBalance":    addBalanceOp,
		"subBalance":    subBalanceOp,
		"createAccount": createAccountOp,
		"suicide":       suicideOp,
	}

	operationWithAddressAndKey := map[string]func(ctx *StateDBContext, args OpArgs){
		"setState": setStateOp,
	}

	var opWithNameList []StateDBOperation
	for opName, op := range operationWithAddress {
		for _, address := range addresses {
			opWithNameList = append(opWithNameList, NewStateDBOperation(op, opName, OpArgs{address: &address}))
		}
	}
	for opName, op := range operationWithAddressAndKey {
		for _, address := range addresses {
			for _, key := range keys {
				op := NewStateDBOperation(
					op,
					opName,
					OpArgs{address: &address, key: &key},
				)
				opWithNameList = append(opWithNameList, op)
				// Multiple writes to to the same slot to trigger already existing case
				opWithNameList = append(opWithNameList, op)
			}
		}
	}

	testCaseNameFunc := func(s [][]StateDBOperation) string {
		var nameParts []string
		for _, opList := range s {
			name := "["
			for _, op := range opList {
				name += op.name + " "
			}
			name = strings.TrimSpace(name) + "]"
			nameParts = append(nameParts, name)
		}
		return strings.Join(nameParts, " ")
	}

	for operationTriple := range cartesianProductTriple(opWithNameList) {
		for testCase := range orderedPartitions(operationTriple) {
			t.Run(testCaseNameFunc(testCase), func(t *testing.T) {
				t.Parallel()
				require := require.New(t)

				ctx := NewStateDBContext(t)

				var statesToCheck []InterTxSnapshotWithStateCheck
				for _, opList := range testCase {
					snapshotID := ctx.state.InterTxSnapshot()
					require.Empty(ctx.state.accountsToDelete)
					require.Empty(ctx.state.writtenSlots)
					oldStateDB := backupStateDB(ctx.state)

					ctx.state.BeginTransaction()
					for _, op := range opList {
						op.Execute(ctx)
					}
					ctx.state.EndTransaction()

					statesToCheck = append(statesToCheck, InterTxSnapshotWithStateCheck{
						stateBackup: oldStateDB,
						snapshotID:  snapshotID,
					})
				}

				slices.Reverse(statesToCheck)
				for _, cur := range statesToCheck {
					ctx.state.RevertToInterTxSnapshot(cur.snapshotID)
					checkStateDB(t, cur.stateBackup, ctx.state, ctx.db, common.Address{0x0})
				}
			})
		}
	}
}

// StateDBContext is a helper struct wrapping a StateDB and its underlying mocked db.
// It also contains a random number generator to generate random values for the operations performed on the StateDB.
type StateDBContext struct {
	state *stateDB
	db    *MockState
	rng   *rand.Rand
}

// OpArgs is a struct containing an address and a key, to be used as arguments for operations performed on the StateDB.
type OpArgs struct {
	address *common.Address
	key     *common.Key
}

// StateDBOperation is an helper struct representing an operation to be performed on the StateDB, with its name and arguments.
type StateDBOperation struct {
	op   func(ctx *StateDBContext, args OpArgs)
	name string
	args OpArgs
}

// NewStateDBContext creates a new StateDBContext with a mocked State and a StateDB using that mocked State.
// It sets up a random number generator for generating random values within stateDB operations.
func NewStateDBContext(t *testing.T) *StateDBContext {
	t.Helper()

	defaultCode := []byte{0x1}
	defaultBalance := amount.New(0)
	defaultNonce := common.Nonce{0}
	defaultStateValue := common.Value{0x1}
	defaultCodeSize := math.MaxInt

	ctrl := gomock.NewController(t)
	db := NewMockState(ctrl)
	any := gomock.Any()
	db.EXPECT().Check().Return(nil).AnyTimes()
	db.EXPECT().Flush().Return(nil).AnyTimes()
	db.EXPECT().Close().Return(nil).AnyTimes()

	state := createStateDBWith(db, 1, true)

	// Set expectation in case values are not cached, i.e. they are untouched.
	db.EXPECT().Exists(any).Return(false, nil).AnyTimes()
	db.EXPECT().GetBalance(any).Return(defaultBalance, nil).AnyTimes()
	db.EXPECT().GetNonce(any).Return(defaultNonce, nil).AnyTimes()
	db.EXPECT().GetCode(any).Return(defaultCode, nil).AnyTimes()
	db.EXPECT().GetCodeSize(any).Return(defaultCodeSize, nil).AnyTimes()
	db.EXPECT().GetStorage(any, any).Return(defaultStateValue, nil).AnyTimes()

	return &StateDBContext{
		state: state,
		db:    db,
		rng:   rand.New(rand.NewPCG(42, 42)),
	}
}

// NewStateDBOperation creates a new StateDBOperation with the given operation function, name and arguments.
// The name is generated based on the provided name and the presence of address and key in the arguments.
func NewStateDBOperation(op func(ctx *StateDBContext, args OpArgs), opName string, args OpArgs) StateDBOperation {
	name := fmt.Sprintf("%s", opName)
	if args.address != nil {
		name += fmt.Sprintf(" address %v", *args.address)
	}
	if args.key != nil {
		name += fmt.Sprintf(" key %v", *args.key)
	}
	return StateDBOperation{
		op:   op,
		name: name,
		args: args,
	}
}

// Execute executes the operation on the given StateDBContext with the stored arguments.
func (op *StateDBOperation) Execute(ctx *StateDBContext) {
	op.op(ctx, op.args)
}

func setStateOp(ctx *StateDBContext, args OpArgs) {
	randomValue := common.Value(randomByteArrayWithPrefix(ctx.rng, 32, []byte{0x2}))
	ctx.state.SetState(*args.address, *args.key, randomValue)
}

func setNonceOp(ctx *StateDBContext, args OpArgs) {
	ctx.state.SetNonce(*args.address, ctx.rng.Uint64N(math.MaxUint64)+1)
}

func setCodeOp(ctx *StateDBContext, args OpArgs) {
	randomCode := randomByteArrayWithPrefix(ctx.rng, 8, []byte{0x2})
	ctx.state.SetCode(*args.address, randomCode)
}

func addBalanceOp(ctx *StateDBContext, args OpArgs) {
	ctx.state.AddBalance(*args.address, amount.New(10))
}

func subBalanceOp(ctx *StateDBContext, args OpArgs) {
	ctx.state.SubBalance(*args.address, amount.New(10))
}

func createAccountOp(ctx *StateDBContext, args OpArgs) {
	ctx.state.CreateAccount(*args.address)
}

func suicideOp(ctx *StateDBContext, args OpArgs) {
	ctx.state.Suicide(*args.address)
}

// backupStateDB creates a partial copy of the stateDB to be used to check transaction revert effects.
func backupStateDB(s *stateDB) *stateDB {
	ns := createStateDBWith(s.state, 1, true)
	ns.accounts = cloneMapWith(s.accounts, cloneValue)
	ns.balances = cloneMapWith(s.balances, cloneBalanceValue)
	ns.nonces = cloneMapWith(s.nonces, cloneNonceValue)
	s.data.CopyToWith(ns.data, cloneValue)
	ns.reincarnation = maps.Clone(s.reincarnation)
	ns.codes = cloneMapWith(s.codes, cloneCodeValue)
	ns.undo = slices.Clone(s.undo)
	ns.clearedAccounts = maps.Clone(s.clearedAccounts)
	ns.canApplyChanges = s.canApplyChanges

	return ns
}

// checkStateDB checks if the `actual` reverted stateDB fields match the `expected` stateDB one, along with postconditions on transaction context fields.
// Caches populated by read-only functions are checked against default values and are retrieved by querying `mock` with an untouched `defaultAccount`.
func checkStateDB(t *testing.T, expected *stateDB, actual *stateDB, mock *MockState, defaultAccount common.Address) error {
	t.Helper()

	for addr, account := range actual.accounts {
		value, exists := expected.accounts[addr]
		if !((exists && reflect.DeepEqual(account, value)) || (!exists && account.current == account.original && account.current == accountNonExisting)) {
			return fmt.Errorf("accounts differ at address %v: got %v", addr, account)
		}
	}
	for addr, balance := range actual.balances {
		value, exists := expected.balances[addr]
		if exists && reflect.DeepEqual(balance, value) {
			continue
		}
		defaultBalance, _ := mock.GetBalance(defaultAccount)
		if !(balance.current == defaultBalance && balance.original == &balance.current) {
			return fmt.Errorf("balances differ at address %v: got %v", addr, balance)
		}
	}
	for addr, nonce := range actual.nonces {
		value, exists := expected.nonces[addr]
		if exists && reflect.DeepEqual(nonce, value) {
			continue
		}
		defaultNonce, _ := mock.GetNonce(defaultAccount)
		if !(nonce.current == defaultNonce.ToUint64()) {
			return fmt.Errorf("nonces differ at address %v: got %v", addr, nonce)
		}
	}

	if !expected.data.DeepEqual(actual.data) {
		return fmt.Errorf("data maps differ: expected %v, got %v", expected.data, actual.data)
	}
	if !reflect.DeepEqual(expected.reincarnation, actual.reincarnation) {
		return fmt.Errorf("reincarnation maps differ: expected %v, got %v", expected.reincarnation, actual.reincarnation)
	}

	for addr, code := range actual.codes {
		value, exists := expected.codes[addr]
		if exists && reflect.DeepEqual(code, value) {
			continue
		}
		defaultCode, _ := mock.GetCode(defaultAccount)
		if !(code.code == nil || slices.Equal(code.code, defaultCode)) {
			return fmt.Errorf("codes differ at address %v: expected %v, got %v", addr, value, code)
		}
	}

	if !reflect.DeepEqual(expected.clearedAccounts, actual.clearedAccounts) {
		return fmt.Errorf("clearedAccounts maps differ: expected %v, got %v", expected.clearedAccounts, actual.clearedAccounts)
	}
	if !reflect.DeepEqual(expected.canApplyChanges, actual.canApplyChanges) {
		return fmt.Errorf("canApplyChanges differ: expected %v, got %v", expected.canApplyChanges, actual.canApplyChanges)
	}

	// For the undo, we check function pointers
	if len(expected.undo) != len(actual.undo) {
		return fmt.Errorf("undo functions length differ: expected %d, got %d", len(expected.undo), len(actual.undo))
	}
	for i := range expected.undo {
		addr1 := reflect.ValueOf(expected.undo[i]).Pointer()
		addr2 := reflect.ValueOf(actual.undo[i]).Pointer()
		if addr1 != addr2 {
			return fmt.Errorf("undo functions differ, function %d: expected %v, got %v", i, addr1, addr2)
		}
	}

	// Transaction context field should have been reset
	if actual.refund != 0 {
		return fmt.Errorf("refund mismatch: expected 0, got %d", actual.refund)
	}
	if actual.accessedSlots.Size() > 0 {
		return fmt.Errorf("accessedSlots size mismatch: expected 0, got %d", actual.accessedSlots.Size())
	}
	if len(actual.accessedAddresses) > 0 {
		return fmt.Errorf("accessedAddresses not empty: got %v", actual.accessedAddresses)
	}
	if len(actual.accountsToDelete) > 0 {
		return fmt.Errorf("accountsToDelete not empty: got %v", actual.accountsToDelete)
	}
	if len(actual.logs) > 0 {
		return fmt.Errorf("logs not empty: got %v", actual.logs)
	}
	if len(actual.emptyCandidates) > 0 {
		return fmt.Errorf("emptyCandidates not empty: got %v", actual.emptyCandidates)
	}
	if len(actual.createdContracts) > 0 {
		return fmt.Errorf("createdContracts not empty: got %v", actual.createdContracts)
	}
	if actual.transientStorage.Size() > 0 {
		return fmt.Errorf("transientStorage size mismatch: expected 0, got %d", actual.transientStorage.Size())
	}

	return nil
}

// cloneMapWith clones `m` using `cloneFunc` to clone the values. If `m` is nil, it returns nil.
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

// cartesianProductTriple generates the cartesian product of a slice with itself three times,
// yielding the result as a sequence of tuples containing the values.
func cartesianProductTriple[T any](slice []T) iter.Seq[[]T] {
	return func(yield func([]T) bool) {
		for _, a := range slice {
			for _, b := range slice {
				for _, c := range slice {
					if !yield([]T{a, b, c}) {
						return
					}
				}
			}
		}
	}
}

// orderedPartitions yields each ordered partition of the input slice.
func orderedPartitions[T any](input []T) iter.Seq[[][]T] {
	return func(yield func([][]T) bool) {
		n := len(input)
		if n == 0 {
			return
		}

		numCombinations := 1 << (n - 1)
		for i := range numCombinations {
			var result [][]T
			currentGroup := []T{input[0]}

			for j := 0; j < n-1; j++ {
				if (i>>j)&1 == 1 {
					result = append(result, currentGroup)
					currentGroup = []T{input[j+1]}
				} else {
					currentGroup = append(currentGroup, input[j+1])
				}
			}
			result = append(result, currentGroup)
			if !yield(result) {
				return
			}
		}
	}
}

func Test_backupStateDB_performPartialStateDBCopy(t *testing.T) {
	t.Parallel()

	ctrl := gomock.NewController(t)
	st := NewMockState(ctrl)

	st.EXPECT().Check().Return(nil).AnyTimes()
	st.EXPECT().Flush().Return(nil).AnyTimes()
	st.EXPECT().Close().Return(nil).AnyTimes()

	state := createStateDBWith(st, defaultStoredDataCacheSize, true)

	addr := common.Address{0x1}
	key := common.Key{0x2}

	state.accounts[addr] = &accountState{}
	state.balances[addr] = &balanceValue{original: func() *amount.Amount { a := amount.New(100); return &a }(), current: amount.New(10)}
	state.nonces[addr] = &nonceValue{original: func() *uint64 { n := uint64(100); return &n }(), current: 200}
	state.data.Put(slotId{addr: addr, key: key}, &slotValue{stored: common.Value{0x1}})
	state.reincarnation[addr] = 42
	state.codes[addr] = &codeValue{code: []byte{0x1}}
	state.undo = []func(){func() {}}
	state.clearedAccounts[addr] = 1
	state.canApplyChanges = true

	backup := backupStateDB(state)

	require.NoError(t, checkStateDB(t, state, backup, nil, common.Address{0x0}))
}

func Test_checkStateDB_checksStateDBRevertedFieldsAndPostconditions(t *testing.T) {
	t.Parallel()

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
	s1.undo = []func(){f}
	s2.undo = []func(){f}
	// Caches populated by read-only functions with mock values should not cause checkStateDB to fail
	newAddr := common.Address{0x2}
	mockBalance := amount.New(100)
	mockNonce := common.Nonce{100}
	mockCode := []byte{0x3}
	mockState := NewMockState(gomock.NewController(t))
	mockState.EXPECT().GetBalance(newAddr).Return(mockBalance, nil).AnyTimes()
	mockState.EXPECT().GetNonce(newAddr).Return(mockNonce, nil).AnyTimes()
	mockState.EXPECT().GetCode(newAddr).Return(mockCode, nil).AnyTimes()
	mockState.EXPECT().GetCodeSize(newAddr).Return(len(mockCode), nil).AnyTimes()
	s1.accounts[newAddr] = &accountState{current: accountNonExisting, original: accountNonExisting}
	s1.balances[newAddr] = &balanceValue{current: mockBalance, original: &mockBalance}
	s1.nonces[newAddr] = &nonceValue{current: mockNonce.ToUint64(), original: nil}
	s1.codes[newAddr] = &codeValue{code: nil}
	s1.codes[common.Address{0x3}] = &codeValue{code: mockCode}

	require.NoError(t, checkStateDB(t, s1, s2, mockState, newAddr))
}

func Test_checkStateDB_failsOnDifferentStateDB(t *testing.T) {
	t.Parallel()

	require := require.New(t)
	addr := common.Address{0x1}
	defaultAddr := common.Address{0x0}
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

	mockState := NewMockState(gomock.NewController(t))
	mockState.EXPECT().GetBalance(defaultAddr).Return(amount.New(math.MaxUint64), nil).AnyTimes()
	mockState.EXPECT().GetNonce(defaultAddr).Return(common.Nonce{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, nil).AnyTimes()
	mockState.EXPECT().GetCode(defaultAddr).Return([]byte{0x10}, nil).AnyTimes()
	mockState.EXPECT().GetCodeSize(defaultAddr).Return(100, nil).AnyTimes()

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
	s1.undo = []func(){f}
	s2.undo = []func(){f}
	require.NoError(checkStateDB(t, s1, s2, nil, defaultAddr))

	s2.accounts[addr] = &accountState{current: accountNonExisting, original: accountExists}
	require.Error(checkStateDB(t, s1, s2, mockState, defaultAddr))
	s2.accounts[addr] = &as
	require.NoError(checkStateDB(t, s1, s2, mockState, defaultAddr))

	tmpBalance := balanceValue{current: amount.New(100), original: &bv.current}
	s2.balances[addr] = &tmpBalance
	require.Error(checkStateDB(t, s1, s2, mockState, defaultAddr))
	s2.balances[addr] = bv
	require.NoError(checkStateDB(t, s1, s2, mockState, defaultAddr))

	s2.nonces[addr] = &nonceValue{current: nonce + 1}
	require.Error(checkStateDB(t, s1, s2, mockState, defaultAddr))
	s2.nonces[addr] = nv
	require.NoError(checkStateDB(t, s1, s2, mockState, defaultAddr))

	s2.data.Put(slot, &slotValue{})
	require.Error(checkStateDB(t, s1, s2, mockState, defaultAddr))
	s2.data.Put(slot, sv)
	require.NoError(checkStateDB(t, s1, s2, mockState, defaultAddr))

	s2.reincarnation[addr] = 100
	require.Error(checkStateDB(t, s1, s2, mockState, defaultAddr))
	s2.reincarnation[addr] = 99
	require.NoError(checkStateDB(t, s1, s2, mockState, defaultAddr))

	s2.codes[addr] = &codeValue{code: []byte{0x3}}
	require.Error(checkStateDB(t, s1, s2, mockState, defaultAddr))
	s2.codes[addr] = cv
	require.NoError(checkStateDB(t, s1, s2, mockState, defaultAddr))

	s2.clearedAccounts[addr] = 2
	require.Error(checkStateDB(t, s1, s2, mockState, defaultAddr))
	s2.clearedAccounts[addr] = 1
	require.NoError(checkStateDB(t, s1, s2, mockState, defaultAddr))

	s2.canApplyChanges = false
	require.Error(checkStateDB(t, s1, s2, mockState, defaultAddr))
	s2.canApplyChanges = true
	require.NoError(checkStateDB(t, s1, s2, mockState, defaultAddr))

	s2.undo = []func(){func() {}}
	require.Error(checkStateDB(t, s1, s2, mockState, defaultAddr))
	s2.undo = []func(){f}
	require.NoError(checkStateDB(t, s1, s2, mockState, defaultAddr))

}

func Test_CartesianTriple(t *testing.T) {
	t.Parallel()

	input := []string{"a", "b"}
	var triples [][]string
	for t := range cartesianProductTriple(input) {
		triples = append(triples, t)
	}

	expected := [][]string{
		{"a", "a", "a"},
		{"a", "a", "b"},
		{"a", "b", "a"},
		{"a", "b", "b"},
		{"b", "a", "a"},
		{"b", "a", "b"},
		{"b", "b", "a"},
		{"b", "b", "b"},
	}

	require.Equal(t, expected, triples)
}

func Test_OrderedPartitions(t *testing.T) {
	t.Parallel()

	input := []int{1, 2, 3}
	var partitions [][][]int
	for p := range orderedPartitions(input) {
		partitions = append(partitions, p)
	}

	expected := [][][]int{
		{{1, 2, 3}},
		{{1}, {2, 3}},
		{{1, 2}, {3}},
		{{1}, {2}, {3}},
	}

	require.Equal(t, expected, partitions)
}
