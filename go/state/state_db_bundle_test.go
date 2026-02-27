package state

import (
	"fmt"
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

func TestStateDB_RevertToInterTxSnapshot_RevertsStateCorrectly(t *testing.T) {
	type InterTxSnapshotWithStateCheck struct {
		stateBackup *stateDB
		snapshotID  interTxSnapshotID
	}

	operationList := map[string]func(ctx *StateDBOpContext){
		"setState":      setStateOp,
		"setNonce":      setNonceOp,
		"setCode":       setCodeOp,
		"addRefund":     addRefundOp,
		"subRefund":     subRefundOp,
		"addBalance":    addBalanceOp,
		"subBalance":    subBalanceOp,
		"addAddress":    addAddressToAccessListOp,
		"createAccount": createAccountOp,
		"suicide":       suicideOp,
	}

	for n1, op1 := range operationList {
		for n2, op2 := range operationList {
			for n3, op3 := range operationList {
				t.Run(fmt.Sprintf("%s %s %s", n1, n2, n3), func(t *testing.T) {
					t.Parallel()
					require := require.New(t)

					ctx := NewStateDBOpContext(t)

					var statesToCheck []InterTxSnapshotWithStateCheck
					for _, op := range []func(ctx *StateDBOpContext){
						op1,
						op2,
						op3,
					} {
						snapshotID := ctx.state.InterTxSnapshot()
						require.Empty(ctx.state.accountsToDelete)
						require.Empty(ctx.state.writtenSlots)
						oldStateDB := partialCopyStateDB(ctx.state)

						ctx.state.BeginTransaction()
						op(ctx)
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
						checkStateDBEqual(t, cur.stateBackup, ctx.state)
					}
				})
			}
		}
	}
}

// StateDBOpContext is a helper struct containing a state and values to be used in subsequent operations on it, to be used in the tests of StateDB transaction revert functionality. Such values are supposed to be mutated by the operations to properly check after reverts that the state is properly reverted to the previous state.
type StateDBOpContext struct {
	state   *stateDB
	db      *MockState
	address common.Address
	key     common.Key
	value   common.Value
	nonce   uint64
	pcg     *rand.PCG
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

	address := common.Address{0x1}
	// Set expectation in case values are not cached, i.e. they are untouched.
	db.EXPECT().Exists(address).Return(false, nil).AnyTimes()
	db.EXPECT().GetBalance(address).Return(amount.New(0), nil).AnyTimes()
	db.EXPECT().GetNonce(address).Return(common.Nonce([common.NonceSize]byte{}), nil).AnyTimes()
	db.EXPECT().GetCodeSize(address).Return(0, nil).AnyTimes()

	return &StateDBOpContext{
		state:   state,
		db:      db,
		address: address,
		key:     common.Key{0x1},
		value:   common.Value{0x1},
		nonce:   1,
		pcg:     rand.NewPCG(42, 42),
	}
}

func setStateOp(ctx *StateDBOpContext) {
	ctx.db.EXPECT().GetStorage(ctx.address, ctx.key).Return(common.Value{}, nil).MinTimes(0).MaxTimes(1)
	ctx.state.SetState(ctx.address, ctx.key, ctx.value)
	ctx.value[0] += byte(1)
}

func setNonceOp(ctx *StateDBOpContext) {
	ctx.state.SetNonce(ctx.address, ctx.nonce)
	ctx.nonce++
}

func setCodeOp(ctx *StateDBOpContext) {
	randomCode := make([]byte, 8)
	for i := range randomCode {
		randomCode[i] = byte(ctx.pcg.Uint64())
	}
	ctx.state.SetCode(ctx.address, randomCode)
}

func addRefundOp(ctx *StateDBOpContext) {
	ctx.state.AddRefund(10)
}

func subRefundOp(ctx *StateDBOpContext) {
	ctx.state.SubRefund(10)
}

func addAddressToAccessListOp(ctx *StateDBOpContext) {
	addr := common.Address{byte(ctx.pcg.Uint64())}
	ctx.state.AddAddressToAccessList(addr)
}

func addBalanceOp(ctx *StateDBOpContext) {
	ctx.state.AddBalance(ctx.address, amount.New(10))
}

func subBalanceOp(ctx *StateDBOpContext) {
	ctx.state.SubBalance(ctx.address, amount.New(10))
}

func createAccountOp(ctx *StateDBOpContext) {
	ctx.state.CreateAccount(ctx.address)
}

func suicideOp(ctx *StateDBOpContext) {
	ctx.state.Suicide(ctx.address)
}

func Test_partialCopyStateDB_copiesStateDBApartFromStoredDataCache(t *testing.T) {
	t.Parallel()

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

	copiedState := partialCopyStateDB(state)

	checkStateDBEqual(t, state, copiedState)
}

// partialCopyStateDB creates a deep copy of the given stateDB, excluding the `storedDataCache` field.
func partialCopyStateDB(s *stateDB) *stateDB {
	ns := createStateDBWith(s.state, 1, true)
	ns.accounts = cloneMapWith(s.accounts, cloneValue)
	ns.balances = cloneMapWith(s.balances, cloneBalanceValue)
	ns.nonces = cloneMapWith(s.nonces, cloneNonceValue)
	copyFastMapWith(s.data, ns.data, cloneValue)
	s.transientStorage.CopyTo(ns.transientStorage)
	ns.reincarnation = maps.Clone(s.reincarnation)
	ns.codes = cloneMapWith(s.codes, cloneCodeValue)
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

	for addr, account := range actual.accounts {
		value, exists := expected.accounts[addr]
		ok := (exists && reflect.DeepEqual(account, value)) || (!exists && account.current == account.original && account.current == accountNonExisting)
		require.True(ok, fmt.Sprintf("accounts differ at address %v: expected %v, got %v", addr, value, account))
	}
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
	// TODO: This panics if one of the two maps is nil
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
