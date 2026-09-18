// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package state_test

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/0xsoniclabs/carmen/go/tests/nightly"
	"github.com/stretchr/testify/require"
)

// TestCarmen_StagedBlock_RollbackRestoresStateForEveryOperationCombination checks
// that rolling staged blocks back restores the live state exactly, for every
// combination of state-mutating operations and every way of splitting them into
// blocks. It is exhaustive and runs only nightly.
func TestCarmen_StagedBlock_RollbackRestoresStateForEveryOperationCombination(t *testing.T) {
	if !nightly.IsNightly() {
		t.Skip("exhaustive combination test runs only nightly")
	}

	// The enumeration mirrors TestStateDB_RevertToInterTxSnapshot_RevertsStateCorrectly:
	// operations are bound to addresses and keys, the cartesian product of triples is
	// taken, and each triple is split into ordered partitions. Here each partition
	// group becomes a staged block, and the check is on the state root rather than the
	// internal stateDB fields.

	addresses := []common.Address{address1, address2, address3}
	keys := []common.Key{key1, key2, key3}

	// Operations that can delete an account are left out: an update that leaves a
	// storage-owning account empty releases its storage, which the LiveDB cannot
	// restore on rollback (see MptState.RevertLastBlock). Real execution never
	// produces such an update, but these operations would:
	//   - state.CreateAccountOp on an existing contract resets nonce and code while
	//     keeping the balance, a CREATE collision the EVM rejects (EIP-684).
	//   - state.SuicideOp, combined with the above, empties and recreates accounts
	//     in orders no contract can execute.
	// The one deletion EIP-6780 permits, a contract created and destroyed within one
	// transaction, has a dedicated test below.
	operationWithAddress := map[string]func(db state.StateDB, rng *rand.Rand, args state.OpArgs){
		"setNonce":   state.SetNonceOp,
		"setCode":    state.SetCodeOp,
		"addBalance": state.AddBalanceOp,
		"subBalance": state.SubBalanceOp,
	}
	// Storage is written together with code. Only contracts hold storage: an
	// account with storage but no code is a state the EVM cannot construct. If such
	// an artificial account were later drained to empty, the block boundary would
	// prune it and drop its storage, and account deletion is not something a block
	// rollback can undo. Giving the account code keeps it non-empty, so it is never
	// pruned, matching real execution. (CreateAccount clears code and storage
	// together, so the "storage implies code" invariant is never broken.)
	setStorage := func(db state.StateDB, rng *rand.Rand, args state.OpArgs) {
		db.SetCode(args.Address, []byte{0x1})
		state.SetStateOp(db, rng, args)
	}
	operationWithAddressAndKey := map[string]func(db state.StateDB, rng *rand.Rand, args state.OpArgs){
		"setState": setStorage,
	}

	var opWithNameList []state.StateDBOperation
	for opName, op := range operationWithAddress {
		for i, address := range addresses {
			opWithNameList = append(opWithNameList, state.StateDBOperation{
				Op:   op,
				Name: fmt.Sprintf("%s addr %d", opName, i),
				Args: state.OpArgs{Address: address},
			})
		}
	}
	for opName, op := range operationWithAddressAndKey {
		for i, address := range addresses {
			for j, key := range keys {
				opWithNameList = append(opWithNameList, state.StateDBOperation{
					Op:   op,
					Name: fmt.Sprintf("%s addr %d key %d", opName, i, j),
					Args: state.OpArgs{Address: address, Key: key},
				})
			}
		}
	}

	tests := make(map[string][][]state.StateDBOperation)
	for operationTriple := range state.CartesianProductTriple(opWithNameList) {
		for testCase := range state.OrderedPartitions(operationTriple) {
			tests[state.OperationPartitionTestName(testCase)] = testCase
		}
	}

	forEachStagingState(t, func(t *testing.T, _ namedStateConfig, _ state.State, db state.StateDB) {
		// The state is reused across subtests: every case rolls all of its blocks
		// back, so it must leave the state exactly as it found it. Block numbers keep
		// increasing so no height is re-used, and the subtests must therefore not run
		// in parallel on the shared
		block := uint64(0)
		for name, testCase := range tests {
			t.Run(name, func(t *testing.T) {
				require := require.New(t)
				rng := rand.New(rand.NewPCG(42, 42))

				hashesBefore := make([]common.Hash, 0, len(testCase))
				staged := make([]state.StagedBlock, 0, len(testCase))

				for _, group := range testCase {
					hashesBefore = append(hashesBefore, db.GetHash())

					block++
					db.BeginBlock()
					db.BeginTransaction()
					for _, op := range group {
						op.Execute(db, rng)
					}
					db.EndTransaction()
					sb, err := db.EndBlock(block)
					require.NoError(err)
					staged = append(staged, sb)
				}

				// Roll every block back newest-first; each rollback must restore the
				// root its block found when it started.
				for i, s := range slices.Backward(staged) {
					require.NoError(s.Rollback())
					require.Equal(hashesBefore[i], db.GetHash())
				}
			})
		}
	})
}

// TestCarmen_StagedBlock_RollbackOfASameBlockCreateAndSuicideRestoresTheState covers
// the one deletion EIP-6780 permits: an account created and self-destructed within
// the same transaction. The combination test excludes the operations producing it.
func TestCarmen_StagedBlock_RollbackOfASameBlockCreateAndSuicideRestoresTheState(t *testing.T) {
	forEachStagingState(t, func(t *testing.T, _ namedStateConfig, _ state.State, db state.StateDB) {
		require := require.New(t)

		// Seed an unrelated account so the state is not empty to begin with.
		db.BeginBlock()
		db.BeginTransaction()
		db.AddBalance(address2, balance1)
		db.EndTransaction()
		endAndCommitBlock(t, db, 1)

		initialHash := db.GetHash()

		// Create a contract with storage and destroy it again, all in one
		// transaction, as the EVM does: CreateContract marks it as created in this
		// transaction, which is what makes its self-destruct a deletion under
		// EIP-6780. The block nets out to no change at all.
		db.BeginBlock()
		db.BeginTransaction()
		db.CreateAccount(address1)
		db.CreateContract(address1)
		require.True(db.IsNewContract(address1))
		db.SetNonce(address1, 1)
		db.AddBalance(address1, balance1)
		db.SetCode(address1, []byte{0x01})
		db.SetState(address1, key1, val1)
		db.AddBalance(address2, balance2)
		require.True(db.Suicide(address1))
		db.EndTransaction()
		staged, err := db.EndBlock(2)
		require.NoError(err)
		require.NotEqual(initialHash, staged.StateHash())

		require.NoError(staged.Rollback())
		require.Equal(initialHash, db.GetHash())

		db.BeginBlock()
		db.BeginTransaction()
		require.False(db.Exist(address1))
		require.Equal(balance1, db.GetBalance(address2))
		db.EndTransaction()
	})
}
