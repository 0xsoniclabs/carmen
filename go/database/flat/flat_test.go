// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package flat

import (
	"errors"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/common/future"
	"github.com/0xsoniclabs/carmen/go/common/result"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
)

var _ state.State = (*State)(nil)

func TestWrapFactory_ProducesAStateFactoryWrappingGivenFactory(t *testing.T) {
	ctrl := gomock.NewController(t)
	wrapped := state.NewMockState(ctrl)
	wrapped.EXPECT().Close()

	params := state.Parameters{
		Variant: "test",
	}

	counter := 0
	inner := func(p state.Parameters) (state.State, error) {
		counter++
		require.Equal(t, params, p)
		return wrapped, nil
	}

	factory := WrapFactory(inner)
	state, err := factory(params)
	require.NoError(t, err)
	require.Equal(t, 1, counter)
	require.IsType(t, &State{}, state)

	require.NoError(t, state.Close())
}

func TestWrapFactory_FailingNestedFactory_ForwardsError(t *testing.T) {
	issues := errors.New("factory failed")
	inner := func(state.Parameters) (state.State, error) {
		return nil, issues
	}

	factory := WrapFactory(inner)
	state, err := factory(state.Parameters{})
	require.ErrorIs(t, err, issues)
	require.Nil(t, state)
}

func TestState_CanBeOpenedAndClosed(t *testing.T) {
	ctrl := gomock.NewController(t)
	backend := state.NewMockState(ctrl)

	backend.EXPECT().Close().Return(nil)

	flatState := NewState(backend)
	require.NoError(t, flatState.Close())
}

func TestState_Close_IssueDuringCloseIsReported(t *testing.T) {
	ctrl := gomock.NewController(t)
	backend := state.NewMockState(ctrl)

	issue := errors.New("backend close failed")
	backend.EXPECT().Close().Return(issue)

	flatState := NewState(backend)
	err := flatState.Close()
	require.ErrorIs(t, err, issue)
}

func TestState_Exists_ReturnsFromLocalMap(t *testing.T) {
	require := require.New(t)
	addr := common.Address{0x01}
	state := &State{
		accounts: map[common.Address]account{
			addr: {},
		},
	}

	exists, err := state.Exists(addr)
	require.NoError(err)
	require.True(exists)
	exists, err = state.Exists(common.Address{0x03})
	require.NoError(err)
	require.False(exists)
}

func TestState_GetBalance_ReturnsFromLocalMap(t *testing.T) {
	require := require.New(t)
	addr := common.Address{0x01}
	expectedBalance := amount.New(12345)
	state := &State{
		accounts: map[common.Address]account{
			addr: {balance: expectedBalance},
		},
	}

	balance, err := state.GetBalance(addr)
	require.NoError(err)
	require.Equal(expectedBalance, balance)
}

func TestState_GetNonce_ReturnsFromLocalMap(t *testing.T) {
	require := require.New(t)
	addr := common.Address{0x01}
	expectedNonce := common.Nonce{42}
	state := &State{
		accounts: map[common.Address]account{
			addr: {nonce: expectedNonce},
		},
	}

	nonce, err := state.GetNonce(addr)
	require.NoError(err)
	require.Equal(expectedNonce, nonce)
}

func TestState_GetStorage_ReturnsFromLocalMap(t *testing.T) {
	require := require.New(t)
	addr := common.Address{0x01}
	key := common.Key{0x02}
	expectedValue := common.Value{0xAB, 0xCD}
	state := &State{
		storage: map[slotKey]common.Value{
			{addr, key}: expectedValue,
		},
	}

	value, err := state.GetStorage(addr, key)
	require.NoError(err)
	require.Equal(expectedValue, value)
}

func TestState_GetCode_ReturnsFromLocalMap(t *testing.T) {
	require := require.New(t)
	addr := common.Address{0x01}
	codeHash := common.Hash{0x0A, 0x0B}
	expectedCode := []byte{0xDE, 0xAD, 0xBE, 0xEF}
	state := &State{
		accounts: map[common.Address]account{
			addr: {codeHash: codeHash},
		},
		codes: map[common.Hash][]byte{
			codeHash: expectedCode,
		},
	}

	code, err := state.GetCode(addr)
	require.NoError(err)
	require.Equal(expectedCode, code)
}

func TestState_GetCodeSize_ReturnsFromLocalMap(t *testing.T) {
	require := require.New(t)
	addr := common.Address{0x01}
	expectedSize := 256
	state := &State{
		accounts: map[common.Address]account{
			addr: {codeSize: expectedSize},
		},
	}

	size, err := state.GetCodeSize(addr)
	require.NoError(err)
	require.Equal(expectedSize, size)
}

func TestState_GetCodeHash_ReturnsFromLocalMap(t *testing.T) {
	require := require.New(t)
	addr := common.Address{0x01}
	expectedHash := common.Hash{0x0A, 0x0B, 0x0C}
	state := &State{
		accounts: map[common.Address]account{
			addr: {codeHash: expectedHash},
		},
	}

	hash, err := state.GetCodeHash(addr)
	require.NoError(err)
	require.Equal(expectedHash, hash)
}

func TestState_HasEmptyStorage_AlwaysReturnsTrue(t *testing.T) {
	require := require.New(t)
	state := &State{}
	empty, err := state.HasEmptyStorage(common.Address{0x01})
	require.NoError(err)
	require.True(empty)
}

func TestState_Apply_SetsValuesInLocalMaps(t *testing.T) {
	require := require.New(t)

	update := common.Update{
		CreatedAccounts: []common.Address{{0x01}, {0x02}},
		Nonces: []common.NonceUpdate{
			{Account: common.Address{0x03}, Nonce: common.Nonce{42}},
			{Account: common.Address{0x04}, Nonce: common.Nonce{84}},
		},
		Balances: []common.BalanceUpdate{
			{Account: common.Address{0x05}, Balance: amount.New(100)},
			{Account: common.Address{0x06}, Balance: amount.New(200)},
		},
		Slots: []common.SlotUpdate{
			{Account: common.Address{0x07}, Key: common.Key{0x0A}, Value: common.Value{0xFF}},
			{Account: common.Address{0x08}, Key: common.Key{0x0B}, Value: common.Value{0xEE}},
		},
		Codes: []common.CodeUpdate{
			{Account: common.Address{0x09}, Code: []byte{0xAA, 0xBB}},
			{Account: common.Address{0x0A}, Code: []byte{0xCC, 0xDD}},
		},
	}

	commands := make(chan command, 1)
	state := &State{
		accounts: make(map[common.Address]account),
		storage:  make(map[slotKey]common.Value),
		codes:    make(map[common.Hash][]byte),
		commands: commands,
	}

	require.NoError(state.Apply(1, update))

	// The update is send to the commands channel.
	command := <-commands
	require.NotNil(command.update)
	require.Equal(uint64(1), command.update.block)
	require.Equal(update, command.update.data)

	// The local maps are updated immediately.
	for _, address := range update.CreatedAccounts {
		acc, exists := state.accounts[address]
		require.True(exists)
		require.Equal(common.Hash(types.EmptyCodeHash), acc.codeHash)
		require.Equal(0, acc.codeSize)
		require.Equal(amount.New(0), acc.balance)
		require.Equal(common.Nonce{}, acc.nonce)
	}

	for _, nonceUpdate := range update.Nonces {
		acc, exists := state.accounts[nonceUpdate.Account]
		require.True(exists)
		require.Equal(nonceUpdate.Nonce, acc.nonce)
	}

	for _, balanceUpdate := range update.Balances {
		acc, exists := state.accounts[balanceUpdate.Account]
		require.True(exists)
		require.Equal(balanceUpdate.Balance, acc.balance)
	}

	for _, slotUpdate := range update.Slots {
		value, exists := state.storage[slotKey{slotUpdate.Account, slotUpdate.Key}]
		require.True(exists)
		require.Equal(slotUpdate.Value, value)
	}

	for _, codeUpdate := range update.Codes {
		acc, exists := state.accounts[codeUpdate.Account]
		require.True(exists)
		require.Equal(len(codeUpdate.Code), acc.codeSize)
		expectedHash := common.Keccak256(codeUpdate.Code)
		require.Equal(expectedHash, acc.codeHash)

		code, exists := state.codes[expectedHash]
		require.True(exists)
		require.Equal(codeUpdate.Code, code)
	}
}

func TestState_Apply_IsForwardedToBackend(t *testing.T) {
	ctrl := gomock.NewController(t)
	backend := state.NewMockState(ctrl)

	block := uint64(21)
	update := common.Update{
		Nonces: []common.NonceUpdate{
			{Account: common.Address{0x01}, Nonce: common.Nonce{42}},
		},
	}

	gomock.InOrder(
		backend.EXPECT().Apply(block, update),
		backend.EXPECT().Close(),
	)

	flatState := NewState(backend)
	err := flatState.Apply(block, update)
	require.NoError(t, err)

	require.NoError(t, flatState.Close())
}

func TestState_GetHash_IsForwardedToBackendGetCommitment(t *testing.T) {
	ctrl := gomock.NewController(t)
	backend := state.NewMockState(ctrl)

	hash := common.Hash{1, 2, 3}

	future := future.Immediate(result.Ok(hash))
	gomock.InOrder(
		backend.EXPECT().GetCommitment().Return(future),
		backend.EXPECT().Close(),
	)

	flatState := NewState(backend)
	got, err := flatState.GetHash()
	require.NoError(t, err)
	require.Equal(t, hash, got)

	require.NoError(t, flatState.Close())
}

func TestState_GetCommitment_IsForwardedToBackend(t *testing.T) {
	ctrl := gomock.NewController(t)
	backend := state.NewMockState(ctrl)

	hash := common.Hash{1, 2, 3}
	future := future.Immediate(result.Ok(hash))

	gomock.InOrder(
		backend.EXPECT().GetCommitment().Return(future),
		backend.EXPECT().Close(),
	)

	flatState := NewState(backend)
	got, err := flatState.GetCommitment().Await().Get()
	require.NoError(t, err)
	require.Equal(t, hash, got)

	require.NoError(t, flatState.Close())
}

func TestState_Check_SyncsAndConsultsBackend(t *testing.T) {
	ctrl := gomock.NewController(t)
	backend := state.NewMockState(ctrl)

	issue := errors.New("test issue")

	// Expect a sync call to the background worker, returning no issue.
	syncs := make(chan error, 1)
	syncs <- nil

	// Expect the backend check to be called, returning an issue.
	backend.EXPECT().Check().Return(issue)

	state := &State{
		backend:  backend,
		commands: make(chan command, 1),
		syncs:    syncs,
	}

	require.ErrorIs(t, state.Check(), issue)
	select {
	case <-syncs:
		t.Fatal("syncs should have been consumed")
	default:
	}
}

func TestState_Check_IssueReportedBySyncIsForwarded(t *testing.T) {
	issue := errors.New("test issue")

	syncs := make(chan error, 1)
	syncs <- issue

	state := &State{
		commands: make(chan command, 1),
		syncs:    syncs,
	}

	require.ErrorIs(t, state.Check(), issue)
}

func TestState_Flush_SyncsAndConsultsBackend(t *testing.T) {
	ctrl := gomock.NewController(t)
	backend := state.NewMockState(ctrl)

	issue := errors.New("test issue")

	// Expect a sync call to the background worker, returning no issue.
	syncs := make(chan error, 1)
	syncs <- nil

	// Expect the backend flush to be called, returning an issue.
	backend.EXPECT().Flush().Return(issue)

	state := &State{
		backend:  backend,
		commands: make(chan command, 1),
		syncs:    syncs,
	}

	require.ErrorIs(t, state.Flush(), issue)
	select {
	case <-syncs:
		t.Fatal("syncs should have been consumed")
	default:
	}
}

func TestState_Flush_IssueReportedBySyncIsForwarded(t *testing.T) {
	issue := errors.New("test issue")

	syncs := make(chan error, 1)
	syncs <- issue

	state := &State{
		commands: make(chan command, 1),
		syncs:    syncs,
	}

	require.ErrorIs(t, state.Flush(), issue)
}

func TestState_Close_SyncsAndConsultsBackend(t *testing.T) {
	ctrl := gomock.NewController(t)
	backend := state.NewMockState(ctrl)

	issue := errors.New("test issue")

	// Expect a sync call to the background worker, returning no issue.
	syncs := make(chan error, 1)
	syncs <- nil

	// Signal the background worker to be done.
	done := make(chan struct{})
	close(done)

	// Expect the backend close to be called, returning an issue.
	backend.EXPECT().Close().Return(issue)

	state := &State{
		backend:  backend,
		commands: make(chan command, 1),
		syncs:    syncs,
		done:     done,
	}

	require.ErrorIs(t, state.Close(), issue)
	select {
	case <-syncs:
		t.Fatal("syncs should have been consumed")
	default:
	}
}

func TestState_Close_IssueReportedBySyncIsForwarded(t *testing.T) {
	issue := errors.New("test issue")

	syncs := make(chan error, 1)
	syncs <- issue

	state := &State{
		commands: make(chan command, 1),
		syncs:    syncs,
	}

	require.ErrorIs(t, state.Close(), issue)
}

func TestState_GetMemoryFootprint_ListsInternalMaps(t *testing.T) {
	require := require.New(t)
	ctrl := gomock.NewController(t)
	backend := state.NewMockState(ctrl)

	backendFootprint := common.NewMemoryFootprint(1234)
	backend.EXPECT().GetMemoryFootprint().Return(backendFootprint)

	state := &State{
		accounts: map[common.Address]account{
			{0x01}: {},
			{0x02}: {},
		},
		storage: map[slotKey]common.Value{
			{common.Address{0x03}, common.Key{0x0A}}: {0xFF},
			{common.Address{0x04}, common.Key{0x0B}}: {0xEE},
			{common.Address{0x05}, common.Key{0x0C}}: {0xDD},
		},
		codes: map[common.Hash][]byte{
			{0x0A}: {0xAA, 0xBB},
		},
		backend: backend,
	}

	footprint := state.GetMemoryFootprint()
	accounts := footprint.GetChild("accounts")
	require.EqualValues(memoryFootprintOfMap(state.accounts), accounts)
	storage := footprint.GetChild("storage")
	require.EqualValues(memoryFootprintOfMap(state.storage), storage)
	codes := footprint.GetChild("codes")
	require.EqualValues(memoryFootprintOfMap(state.codes), codes)

	require.Equal(footprint.GetChild("backend"), backendFootprint)
}

func TestState_Sync_SyncsWithBackgroundWorker(t *testing.T) {
	commands := make(chan command, 1)
	syncs := make(chan error, 1)

	state := &State{
		commands: commands,
		syncs:    syncs,
	}

	// Emulate background worker processing the sync command.
	issue := errors.New("sync issue")
	go func() {
		// Wait for the sync command (empty command).
		command := <-commands
		require.Empty(t, command)
		// Send the sync result.
		syncs <- issue
	}()

	require.ErrorIs(t, state.sync(), issue)
}

func TestState_GetArchiveState_ForwardsQueryToBackend(t *testing.T) {
	ctrl := gomock.NewController(t)
	backend := state.NewMockState(ctrl)
	archiveState := state.NewMockState(ctrl)

	block := uint64(42)
	backend.EXPECT().GetArchiveState(block).Return(archiveState, nil)

	flatState := &State{
		backend: backend,
	}
	state, err := flatState.GetArchiveState(block)
	require.NoError(t, err)
	require.Equal(t, archiveState, state)
}

func TestState_GetArchiveBlockHeight_ForwardsQueryToBackend(t *testing.T) {
	ctrl := gomock.NewController(t)
	backend := state.NewMockState(ctrl)

	height := uint64(100)
	empty := false
	backend.EXPECT().GetArchiveBlockHeight().Return(height, empty, nil)

	flatState := &State{
		backend: backend,
	}
	gotHeight, gotEmpty, err := flatState.GetArchiveBlockHeight()
	require.NoError(t, err)
	require.Equal(t, height, gotHeight)
	require.Equal(t, empty, gotEmpty)
}

func TestState_CreateWitnessProof_ForwardsToBackend(t *testing.T) {
	ctrl := gomock.NewController(t)
	backend := state.NewMockState(ctrl)

	address := common.Address{0x01}
	keys := []common.Key{{0x0A}, {0x0B}}

	backend.EXPECT().CreateWitnessProof(address, keys).Return(nil, nil)

	flatState := &State{
		backend: backend,
	}
	_, err := flatState.CreateWitnessProof(address, keys...)
	require.NoError(t, err)
}

func TestState_Export_SyncsAndForwardsToBackend(t *testing.T) {
	ctrl := gomock.NewController(t)
	backend := state.NewMockState(ctrl)

	// Expect a sync call to the background worker, returning no issue.
	syncs := make(chan error, 1)
	syncs <- nil

	// Expect the backend export to be called.
	backend.EXPECT().Export(gomock.Any(), gomock.Any()).Return(common.Hash{0xAA}, nil)

	state := &State{
		backend:  backend,
		commands: make(chan command, 1),
		syncs:    syncs,
	}

	hash, err := state.Export(t.Context(), nil)
	require.NoError(t, err)
	require.Equal(t, common.Hash{0xAA}, hash)

	select {
	case <-syncs:
		t.Fatal("syncs should have been consumed")
	default:
	}
}

func TestState_Export_IssueReportedBySyncIsForwarded(t *testing.T) {
	issue := errors.New("test issue")

	syncs := make(chan error, 1)
	syncs <- issue

	state := &State{
		commands: make(chan command, 1),
		syncs:    syncs,
	}

	_, err := state.Export(t.Context(), nil)
	require.ErrorIs(t, err, issue)
}

func TestState_GetProof_ForwardsToBackend(t *testing.T) {
	ctrl := gomock.NewController(t)
	backend := state.NewMockState(ctrl)

	backend.EXPECT().GetProof().Return(nil, nil)

	flatState := &State{
		backend: backend,
	}
	gotProof, err := flatState.GetProof()
	require.NoError(t, err)
	require.Nil(t, gotProof)
}

func TestState_CreateSnapshot_ForwardsToBackend(t *testing.T) {
	ctrl := gomock.NewController(t)
	backend := state.NewMockState(ctrl)
	backend.EXPECT().CreateSnapshot().Return(nil, nil)

	flatState := &State{
		backend: backend,
	}
	gotSnapshot, err := flatState.CreateSnapshot()
	require.NoError(t, err)
	require.Nil(t, gotSnapshot)
}

func TestState_Restore_ForwardsToBackend(t *testing.T) {
	ctrl := gomock.NewController(t)
	inner := state.NewMockState(ctrl)

	inner.EXPECT().Restore(nil).Return(nil)
	flatState := &State{
		backend: inner,
	}

	require.NoError(t, flatState.Restore(nil))
}

func TestState_GetSnapshotVerifier_ForwardsToBackend(t *testing.T) {
	ctrl := gomock.NewController(t)
	inner := state.NewMockState(ctrl)

	inner.EXPECT().GetSnapshotVerifier([]byte{0x01, 0x02}).Return(nil, nil)
	flatState := &State{
		backend: inner,
	}
	_, err := flatState.GetSnapshotVerifier([]byte{0x01, 0x02})
	require.NoError(t, err)
}

// --- issue collector tests ---

func TestIssueCollector_HandleIssue_CollectsIssues(t *testing.T) {
	require := require.New(t)
	collector := issueCollector{}

	issueA := errors.New("issue A")
	issueB := errors.New("issue B")
	issueC := errors.New("issue C")

	require.Empty(collector.issues)
	require.Zero(collector.extraIssues)

	collector.HandleIssue(issueA)

	require.Equal([]error{issueA}, collector.issues)
	require.Zero(collector.extraIssues)

	collector.HandleIssue(issueB)

	require.Equal([]error{issueA, issueB}, collector.issues)
	require.Zero(collector.extraIssues)

	collector.HandleIssue(issueC)

	require.Equal([]error{issueA, issueB, issueC}, collector.issues)
	require.Zero(collector.extraIssues)
}

func TestIssueCollector_HandleIssue_CapsIssuesAtTen(t *testing.T) {
	require := require.New(t)
	collector := issueCollector{}

	issue := errors.New("test issue")
	for i := range 20 {
		collector.HandleIssue(issue)
		require.Equal(min(i+1, 10), len(collector.issues))
		require.Equal(max(0, i-9), collector.extraIssues)
	}
}

func TestIssueCollector_HandleIssue_IgnoresNilErrors(t *testing.T) {
	require := require.New(t)
	collector := issueCollector{}

	require.Empty(collector.issues)
	require.Zero(collector.extraIssues)

	collector.HandleIssue(nil)

	require.Empty(collector.issues)
	require.Zero(collector.extraIssues)
}

func TestIssueCollector_Collect_ReturnsNilIfThereIsNoIssue(t *testing.T) {
	collector := issueCollector{}
	require.Nil(t, collector.Collect())
}

func TestIssueCollector_Collect_JoinsReportedIssues(t *testing.T) {
	require := require.New(t)
	collector := issueCollector{}

	issueA := errors.New("issue A")
	issueB := errors.New("issue B")

	collector.HandleIssue(issueA)
	collector.HandleIssue(issueB)
	err := collector.Collect()
	require.ErrorIs(err, issueA)
	require.ErrorIs(err, issueB)

	require.Equal(errors.Join(issueA, issueB), err)
}

func TestIssueCollector_Collect_AppendsInfoOnTruncatedIssues(t *testing.T) {
	require := require.New(t)
	collector := issueCollector{}

	issue := errors.New("test issue")
	for range 15 {
		collector.HandleIssue(issue)
	}

	err := collector.Collect()
	require.ErrorContains(err, "5 additional errors truncated")
}

func TestMemoryFootprintOfMap_GivesApproximationBasedOnKeyValueTypes(t *testing.T) {

	int32ToInt64Map := map[int32]int64{
		1: 10,
		2: 20,
	}
	footprint := memoryFootprintOfMap(int32ToInt64Map)
	require.EqualValues(t, footprint.Total(), (4+8)*2)

	int16ToByteMap := map[int16]byte{
		1: 0xAA,
		2: 0xBB,
		3: 0xCC,
	}
	footprint = memoryFootprintOfMap(int16ToByteMap)
	require.EqualValues(t, footprint.Total(), (2+1)*3)
}
