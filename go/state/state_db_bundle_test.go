package state

import (
	"encoding/binary"
	"iter"
	"maps"
	"math/rand/v2"
	reflect "reflect"
	"slices"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
)

func TestStateDB_RevertToCheckpoint(t *testing.T) {
	t.Parallel()

	type checkpointWithStateCheck struct {
		stateCheck   func()
		checkpointID int
	}

	require := require.New(t)
	ctx := NewStateDBOpContext(t)

	opList := map[string]func(){
		"setState":      makeSetStateFunc(ctx),
		"setNonce":      makeSetNonceFunc(ctx),
		"setCode":       makeSetCodeFunc(ctx),
		"addRefund":     makeAddRefundFunc(ctx),
		"subRefund":     makeSubRefundFunc(ctx),
		"addBalance":    makeAddBalanceFunc(ctx),
		"subBalance":    makeSubBalanceFunc(ctx),
		"addAddress":    makeAddAddressToAccessListFunc(ctx),
		"createAccount": makeCreateAccountFunc(ctx),
		"suicide":       makeSuicideFunc(ctx),
	}

	var revertCheckerList []checkpointWithStateCheck
	runTx := func(op_call func()) {
		checkpointID := ctx.state.Checkpoint()
		require.Empty(ctx.state.accountsToDelete)
		require.Empty(ctx.state.writtenSlots)
		oldStateDB := copyStateDB(ctx.state)

		ctx.state.BeginTransaction()
		op_call()
		ctx.state.EndTransaction()

		revertCheckerList = append(revertCheckerList, checkpointWithStateCheck{
			stateCheck: func() {
				checkStateDBEqual(t, oldStateDB, ctx.state)
			},
			checkpointID: checkpointID,
		})
	}

	for tx1Name, tx1Func := range opList {
		for tx2Name, tx2Func := range opList {
			for tx3Name, tx3Func := range opList {
				t.Run(tx1Name+"-"+tx2Name+"-"+tx3Name, func(t *testing.T) {
					*ctx = *NewStateDBOpContext(t)

					runTx(tx1Func)
					runTx(tx2Func)
					runTx(tx3Func)

					for curCheckpoint := range PopStackIterator(&revertCheckerList) {
						require.NoError(
							ctx.state.RevertToCheckpoint(curCheckpoint.checkpointID),
						)
						curCheckpoint.stateCheck()
					}
				})
			}
		}
	}
}

// StateDBOpContext is a helper struct containing a state and values to be used in subsequent operations on it, to be used in the tests of StateDB transaction revert functionality. Such values are supposed to be mutated by the operations to properly check after reverts that the state is properly reverted to the previous state.
type StateDBOpContext struct {
	state *stateDB
	db    MockState
	addr  common.Address
	key   common.Key
	value common.Value
	nonce uint64
	// Test stuff
	t   *testing.T
	pcg *rand.PCG
}

// NewStateDBOpContext creates a new StateDBOpContext with a mocked State. It sets up an address and key with initial values, and sets expectations on the mocked State for operations that might be performed on the address and key.
func NewStateDBOpContext(t *testing.T) *StateDBOpContext {
	t.Helper()

	ctrl := gomock.NewController(t)
	db := NewMockState(ctrl)
	db.EXPECT().Check().Return(nil).AnyTimes()
	db.EXPECT().Flush().Return(nil).AnyTimes()
	db.EXPECT().Close().Return(nil).AnyTimes()

	state := createStateDBWith(db, 1, true)

	addr := common.Address{0x1}
	// Set expectation in case values are not cached, i.e. they are untouched.
	setAddrDbExpectations := func(addr common.Address) {
		db.EXPECT().Exists(addr).Return(false, nil).AnyTimes()
		db.EXPECT().GetBalance(addr).Return(amount.New(0), nil).AnyTimes()
		db.EXPECT().GetNonce(addr).Return(common.Nonce([common.NonceSize]byte{}), nil).AnyTimes()
		db.EXPECT().GetCodeSize(addr).Return(0, nil).AnyTimes()
	}
	setAddrDbExpectations(addr)

	return &StateDBOpContext{
		state: state,
		db:    *db,
		addr:  addr,
		key:   common.Key{0x1},
		value: common.Value{0x1},
		nonce: 1,
		t:     t,
		pcg:   rand.NewPCG(42, 42),
	}
}

func makeSetStateFunc(ctx *StateDBOpContext) func() {
	return func() {
		ctx.db.EXPECT().GetStorage(ctx.addr, ctx.key).Return(common.Value{}, nil).MinTimes(0).MaxTimes(1)
		ctx.state.SetState(ctx.addr, ctx.key, ctx.value)
		incValue(&ctx.value, 1)
	}
}

func makeSetNonceFunc(ctx *StateDBOpContext) func() {
	return func() {
		ctx.state.SetNonce(ctx.addr, ctx.nonce)
		ctx.nonce++
	}
}

func makeSetCodeFunc(ctx *StateDBOpContext) func() {
	return func() {
		randomCode := make([]byte, 8)
		for i := range randomCode {
			randomCode[i] = byte(ctx.pcg.Uint64())
		}
		ctx.state.SetCode(ctx.addr, randomCode)
	}
}

func makeAddRefundFunc(ctx *StateDBOpContext) func() {
	return func() {
		ctx.state.AddRefund(10)
	}
}

func makeSubRefundFunc(ctx *StateDBOpContext) func() {
	return func() {
		ctx.state.SubRefund(10)
	}
}

func makeAddAddressToAccessListFunc(ctx *StateDBOpContext) func() {
	return func() {
		addr := common.Address{byte(ctx.pcg.Uint64())}
		ctx.state.AddAddressToAccessList(addr)
	}
}

func makeAddBalanceFunc(ctx *StateDBOpContext) func() {
	return func() {
		ctx.state.AddBalance(ctx.addr, amount.New(10))
	}
}

func makeSubBalanceFunc(ctx *StateDBOpContext) func() {
	return func() {
		ctx.state.SubBalance(ctx.addr, amount.New(10))
	}
}

func makeCreateAccountFunc(ctx *StateDBOpContext) func() {
	return func() {
		ctx.state.CreateAccount(ctx.addr)
	}
}

func makeSuicideFunc(ctx *StateDBOpContext) func() {
	return func() {
		ctx.state.Suicide(ctx.addr)
	}
}

func Test_StateDB_copyStateDB(t *testing.T) {
	// TODO: The proper implementation should make sure every field is mutated before copying
	ctrl := gomock.NewController(t)
	st := NewMockState(ctrl)

	st.EXPECT().Check().Return(nil).AnyTimes()
	st.EXPECT().Flush().Return(nil).AnyTimes()
	st.EXPECT().Close().Return(nil).AnyTimes()

	state := createStateDBWith(st, defaultStoredDataCacheSize, true)
	defer func() {
		require.NoError(t, state.Close())
	}()

	// Create a state with some data
	addr := common.Address{0x1}
	key := common.Key{0x1}
	value := common.Value{0x1}
	state.BeginTransaction()
	st.EXPECT().Exists(gomock.Any()).Return(true, nil).Times(1)
	state.CreateAccount(addr)
	st.EXPECT().GetBalance(addr).Return(amount.New(0), nil).Times(1)
	state.AddBalance(addr, amount.New(10))
	state.SetState(addr, key, value)
	state.EndTransaction()

	copiedState := copyStateDB(state)

	checkStateDBEqual(t, state, copiedState)
}

// copyStateDB creates a deep copy of the given stateDB, excluding the `storedDataCache` field.
func copyStateDB(s *stateDB) *stateDB {
	ns := createStateDBWith(s.state, 1, true)
	ns.accounts = maps.Clone(s.accounts)
	ns.balances = maps.Clone(s.balances)
	ns.nonces = maps.Clone(s.nonces)
	s.data.CopyTo(ns.data)
	s.transientStorage.CopyTo(ns.transientStorage)
	ns.reincarnation = maps.Clone(s.reincarnation)
	ns.codes = maps.Clone(s.codes)
	ns.refund = s.refund
	ns.accessedAddresses = maps.Clone(s.accessedAddresses)
	s.accessedSlots.CopyTo(ns.accessedSlots)
	ns.accountsToDelete = slices.Clone(s.accountsToDelete)
	for undo := range s.undo {
		ns.undo = append(ns.undo, slices.Clone(s.undo[undo]))
	}
	ns.clearedAccounts = maps.Clone(s.clearedAccounts)
	ns.createdContracts = maps.Clone(s.createdContracts)
	ns.emptyCandidates = slices.Clone(s.emptyCandidates)
	ns.canApplyChanges = s.canApplyChanges

	return ns
}

// checkStateDBEqual checks if two stateDB instances are equal, excluding the `storedDataCache` field.
func checkStateDBEqual(t *testing.T, expected *stateDB, actual *stateDB) {
	t.Helper()
	require := require.New(t)

	require.Equal(expected.accounts, actual.accounts)
	require.Equal(expected.balances, actual.balances)
	require.Equal(expected.nonces, actual.nonces)
	require.True(fastMapEqual(expected.data, actual.data))
	require.True(fastMapEqual(expected.transientStorage, actual.transientStorage))
	require.Equal(expected.reincarnation, actual.reincarnation)
	require.Equal(expected.codes, actual.codes)
	require.Equal(expected.refund, actual.refund)
	require.Equal(expected.accessedAddresses, actual.accessedAddresses)
	require.True(fastMapEqual(expected.accessedSlots, actual.accessedSlots))
	require.Equal(expected.writtenSlots, actual.writtenSlots)
	require.Equal(expected.accountsToDelete, actual.accountsToDelete)
	require.Equal(expected.clearedAccounts, actual.clearedAccounts)
	require.Equal(expected.createdContracts, actual.createdContracts)
	require.Equal(expected.emptyCandidates, actual.emptyCandidates)
	require.Equal(expected.canApplyChanges, actual.canApplyChanges)

	// For the undo, we check function pointers as they will stay in memory
	for i := range expected.undo {
		require.Len(expected.undo[i], len(actual.undo[i]))
		for j := range expected.undo[i] {
			addr1 := reflect.ValueOf(expected.undo[i][j]).Pointer()
			addr2 := reflect.ValueOf(actual.undo[i][j]).Pointer()
			require.Equal(addr1, addr2)
		}
	}
}

func incValue(value *common.Value, amount uint64) {
	newValue := binary.LittleEndian.Uint64(value[:8]) + amount
	binary.LittleEndian.PutUint64(value[:8], newValue)
}

func fastMapEqual[K comparable, V comparable](m1, m2 *common.FastMap[K, V]) bool {
	equal := true
	m1.ForEach(func(key K, value V) {
		v2, ok := m2.Get(key)
		if !ok || v2 != value {
			equal = false
		}
	})
	return equal
}

// PopStackIterator returns an iterator that pops elements from the given slice in a stack-like manner.
func PopStackIterator[T any](s *[]T) iter.Seq[*T] {
	return func(yield func(*T) bool) {
		for i := len(*s) - 1; i >= 0; i-- {
			v := &(*s)[i]
			*s = (*s)[:i]
			if !yield(v) {
				return
			}
		}
	}
}
