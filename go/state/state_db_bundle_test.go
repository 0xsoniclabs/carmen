package state

import (
	"encoding/binary"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
)

func TestStateDB_revertTransactionsRevertToPreviousState(t *testing.T) {
	ctrl := gomock.NewController(t)
	st := NewMockState(ctrl)

	st.EXPECT().Exists(gomock.Any()).AnyTimes().Return(true, nil)
	st.EXPECT().GetBalance(gomock.Any()).Return(amount.New(0), nil).AnyTimes()

	statedb := CreateStateDBUsing(st)
	targetAddress := common.Address{}

	// Tx 1: Set balance to 10
	statedb.BeginTransaction()
	statedb.AddBalance(targetAddress, amount.New(10))
	statedb.EndTransaction()
	require.Equal(t, statedb.GetBalance(targetAddress), amount.New(10))

	// Tx 2: Add 10 balance
	statedb.BeginTransaction()
	statedb.AddBalance(targetAddress, amount.New(10))
	statedb.EndTransaction()
	require.Equal(t, statedb.GetBalance(targetAddress), amount.New(20))

	// Tx 3: Add 10 balance
	statedb.BeginTransaction()
	statedb.AddBalance(targetAddress, amount.New(10))
	statedb.EndTransaction()
	require.Equal(t, statedb.GetBalance(targetAddress), amount.New(30))

	// Revert last two txs
	statedb.RevertTransactions(2)
	require.Equal(t, statedb.GetBalance(targetAddress), amount.New(10))
}

func TestStateDB_RollbackOfTransactionsRestoresCommittedState(t *testing.T) {
	require := require.New(t)
	ctrl := gomock.NewController(t)
	st := NewMockState(ctrl)

	any := gomock.Any()
	st.EXPECT().Exists(any).Return(true, nil).AnyTimes()
	st.EXPECT().GetStorage(any, any).AnyTimes()
	st.EXPECT().Check().AnyTimes()
	st.EXPECT().Flush().AnyTimes()
	st.EXPECT().Close()

	state := CreateStateDBUsing(st)
	defer func() {
		require.NoError(state.Close())
	}()

	addr := common.Address{0x1}
	key := common.Key{0x1}
	val0 := common.Value{}
	val1 := common.Value{0x1}
	val2 := common.Value{0x2}

	// Tx 1: updates a storage location
	state.BeginTransaction()
	require.Equal(val0, state.GetCommittedState(addr, key))
	state.SetState(addr, key, val1)
	require.Equal(val0, state.GetCommittedState(addr, key))
	state.EndTransaction()

	// Tx 2: updates the same storage location
	state.BeginTransaction()
	require.Equal(val1, state.GetCommittedState(addr, key))
	state.SetState(addr, key, val2)
	require.Equal(val1, state.GetCommittedState(addr, key))
	state.EndTransaction()

	// Revert last transaction
	state.RevertTransactions(1)

	// Now, the committed state should be the state as before Tx 2.
	state.BeginTransaction()
	require.Equal(val1, state.GetCommittedState(addr, key))
	state.EndTransaction()
}

type StateDBOpContext struct {
	state           StateDB
	st              MockState
	existingAddr    common.Address
	nonExistingAddr common.Address
	key             common.Key
	value           common.Value
	nonce           uint64
	// Test stuff
	require *require.Assertions
}

func NewStateDBOpContext(t *testing.T, require *require.Assertions) *StateDBOpContext {
	ctrl := gomock.NewController(t)
	st := NewMockState(ctrl)
	any := gomock.Any()
	st.EXPECT().Check().Return(nil).AnyTimes()
	st.EXPECT().Flush().Return(nil).AnyTimes()
	st.EXPECT().Close().Return(nil).AnyTimes()

	state := CreateStateDBUsing(st)

	addr := common.Address{0x1}
	key := common.Key{0x1}

	// Create an account to be used in the tests
	// TODO: Not sure if we can avoid this
	state.BeginTransaction()
	st.EXPECT().Exists(any).Return(true, nil).Times(1)
	state.CreateAccount(addr)
	// Add some balance to avoid the account to be deleted at the ends
	st.EXPECT().GetBalance(addr).Return(amount.New(0), nil).Times(1)
	state.AddBalance(addr, amount.New(10))
	state.EndTransaction()
	require.True(state.Exist(addr))

	return &StateDBOpContext{
		state:        state,
		st:           *st,
		existingAddr: addr,
		key:          key,
		value:        common.Value{0x1},
		nonce:        1,
		require:      require,
	}
}

func (ctx *StateDBOpContext) Reset(t *testing.T) {
	_ = ctx.state.Close()
	*ctx = *NewStateDBOpContext(t, ctx.require)
}

func makeSetStateFunc(ctx *StateDBOpContext) func() func() {
	return func() func() {
		ctx.st.EXPECT().GetStorage(ctx.existingAddr, ctx.key).Return(common.Value{}, nil).AnyTimes()
		oldValue := ctx.state.GetCommittedState(ctx.existingAddr, ctx.key)
		ctx.state.SetState(ctx.existingAddr, ctx.key, ctx.value)
		defer incValue(&ctx.value, 1)
		return func() {
			ctx.require.Equal(oldValue, ctx.state.GetCommittedState(ctx.existingAddr, ctx.key))
		}
	}
}

func makeSetNonceFunc(ctx *StateDBOpContext) func() func() {
	return func() func() {
		oldValue := ctx.state.GetNonce(ctx.existingAddr)
		ctx.state.SetNonce(ctx.existingAddr, ctx.nonce)
		defer func() {
			ctx.nonce++
		}()
		return func() {
			ctx.require.Equal(oldValue, ctx.state.GetNonce(ctx.existingAddr))
		}
	}
}

func TestStateDB_RevertTransactions(t *testing.T) {
	require := require.New(t)
	ctx := NewStateDBOpContext(t, require)

	opList := []func() func(){
		makeSetStateFunc(ctx),
		makeSetNonceFunc(ctx),
	}
	var revertCheckerList []func()

	runTx := func(op func() func()) {
		ctx.state.BeginTransaction()
		revertCheckerList = append(revertCheckerList, op())
		ctx.state.EndTransaction()
	}

	// Cartesian product of operations
	for i := range opList {
		for j := range opList {
			for k := range opList {
				ctx.Reset(t)

				runTx(opList[i])
				runTx(opList[j])
				runTx(opList[k])

				for range 2 {
					ctx.state.RevertTransactions(1)
					revertCheckerList[len(revertCheckerList)-1]()
					revertCheckerList = revertCheckerList[:len(revertCheckerList)-1]
				}
			}
		}
	}
}

func incValue(value *common.Value, amount uint64) {
	newValue := binary.LittleEndian.Uint64(value[:8]) + amount
	binary.LittleEndian.PutUint64(value[:8], newValue)
}
