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
// combination of state-mutating operations spread across the blocks. It is
// exhaustive and runs only nightly.
func TestCarmen_StagedBlock_RollbackRestoresStateForEveryOperationCombination(t *testing.T) {
	if !nightly.IsNightly() {
		t.Skip("exhaustive combination test runs only nightly")
	}

	// The enumeration mirrors TestStateDB_RevertToInterTxSnapshot_RevertsStateCorrectly:
	// operations are bound to addresses and keys, the cartesian product of triples is
	// taken, and each triple is split into ordered partitions. Here each partition
	// group becomes a staged block, and the check is on the state root rather than the
	// internal stateDB fields.

	addresses := []common.Address{address1, address2}
	keys := []common.Key{key1, key2}

	// Operations that delete an account are intentionally left out, because a block
	// rollback cannot reverse a deletion, and every way these operations would delete
	// an account across separate blocks is a state the EVM itself cannot construct:
	//
	//   - Self-destruct (state.SuicideOp): EIP-6780, which Carmen follows, only lets a
	//     self-destruct delete an account created within the same block. Operations
	//     here are spread across separate blocks, so a self-destruct could never share
	//     a block with the creation it would have to undo, and thus could never
	//     trigger a valid deletion.
	//   - Account creation (state.CreateAccountOp): CREATE at an address that already
	//     has code or a nonce is a collision the EVM rejects (EIP-684); and a creation
	//     that leaves the account empty (no code, nonce, or balance) is pruned at the
	//     block boundary, deleting it. Either way the account ends up deleted through a
	//     path real execution does not take.
	//
	// Reversing account creation and same-block self-destruct belongs in dedicated
	// same-block tests. What remains below are the value-updating operations, whose
	// undo a block rollback must reverse exactly.
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
		db.SetCode(*args.Address, []byte{0x1})
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
				Args: state.OpArgs{Address: &address},
			})
		}
	}
	for opName, op := range operationWithAddressAndKey {
		for i, address := range addresses {
			for j, key := range keys {
				opWithNameList = append(opWithNameList, state.StateDBOperation{
					Op:   op,
					Name: fmt.Sprintf("%s addr %d key %d", opName, i, j),
					Args: state.OpArgs{Address: &address, Key: &key},
				})
			}
		}
	}

	tests := make(map[string][][]state.StateDBOperation)
	for operationTriple := range state.CartesianProductTriple(opWithNameList) {
		for testCase := range state.OrderedPartitions(operationTriple) {
			tests[state.OperationPartitionName(testCase)] = testCase
		}
	}

	forEachStagingState(t, func(t *testing.T, _ namedStateConfig, _ state.State, db state.StateDB) {
		// The state is reused across subtests: every case rolls all of its blocks
		// back, so it must leave the state exactly as it found it. Block numbers keep
		// increasing so no height is re-used, and the subtests must therefore not run
		// in parallel on the shared state.
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
					require.Equal(hashesBefore[i], db.GetHash(),
						"rolling back block %d must restore its predecessor's root", i)
				}
			})
		}
	})
}
