// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package reference

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/common/future"
	"github.com/0xsoniclabs/carmen/go/common/result"
	"github.com/0xsoniclabs/carmen/go/common/witness"
	"github.com/0xsoniclabs/carmen/go/database/vt/reference/trie"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/ethereum/go-ethereum/core/types"
)

// State is an in-memory implementation of a chain-state tracking account and
// storage data using a Verkle Trie. It implements the state.State interface.
type State struct {
	trie *trie.Trie
}

// NewState creates a new, empty in-memory state instance.
func NewState(_ state.Parameters) (state.State, error) {
	return &State{
		trie: &trie.Trie{},
	}, nil
}

// newState creates a new, empty in-memory state instance.
func newState() *State {
	return &State{
		trie: &trie.Trie{},
	}
}

func (s *State) Exists(address common.Address) (bool, error) {
	key := getBasicDataKey(address)
	value := s.trie.Get(key)
	var empty [24]byte // nonce and balance are layed out in bytes 8-32
	return !bytes.Equal(value[8:32], empty[:]), nil

}

func (s *State) GetBalance(address common.Address) (amount.Amount, error) {
	key := getBasicDataKey(address)
	value := s.trie.Get(key)
	return amount.NewFromBytes(value[16:32]...), nil
}

func (s *State) GetNonce(address common.Address) (common.Nonce, error) {
	key := getBasicDataKey(address)
	value := s.trie.Get(key)
	return common.Nonce(value[8:16]), nil
}

func (s *State) GetStorage(address common.Address, key common.Key) (common.Value, error) {
	return common.Value(s.trie.Get(getStorageKey(address, key))), nil
}

func (s *State) GetCode(address common.Address) ([]byte, error) {
	size, _ := s.GetCodeSize(address)
	chunks := make([]chunk, 0, size)
	for i := 0; i < size/31+1; i++ {
		key := getCodeChunkKey(address, i)
		value := s.trie.Get(key)
		chunks = append(chunks, chunk(value))
	}
	return merge(chunks, size), nil
}

func (s *State) GetCodeSize(address common.Address) (int, error) {
	key := getBasicDataKey(address)
	value := s.trie.Get(key)
	return int(binary.BigEndian.Uint32(value[4:8])), nil
}

func (s *State) GetCodeHash(address common.Address) (common.Hash, error) {
	key := getCodeHashKey(address)
	value := s.trie.Get(key)
	return common.Hash(value[:]), nil
}

func (s *State) HasEmptyStorage(addr common.Address) (bool, error) {
	return true, nil
}

func (s *State) Apply(block uint64, update common.Update) error {

	// init potentially empty accounts with empty code hash,
	for _, address := range update.CreatedAccounts {
		accountKey := getBasicDataKey(address)
		value := s.trie.Get(accountKey)
		var empty [28]byte
		// empty accnout has empty code size, nonce, and balance
		if bytes.Equal(value[4:32], empty[:]) {
			codeHashKey := getCodeHashKey(address)
			s.trie.Set(accountKey, value) // must be initialized to empty account
			s.trie.Set(codeHashKey, trie.Value(types.EmptyCodeHash))
		}
	}

	for _, update := range update.Nonces {
		key := getBasicDataKey(update.Account)
		value := s.trie.Get(key)
		copy(value[8:16], update.Nonce[:])
		s.trie.Set(key, value)
	}

	for _, update := range update.Balances {
		key := getBasicDataKey(update.Account)
		value := s.trie.Get(key)
		amount := update.Balance.Bytes32()
		copy(value[16:32], amount[16:])
		s.trie.Set(key, value)
	}

	for _, update := range update.Slots {
		key := getStorageKey(update.Account, update.Key)
		s.trie.Set(key, trie.Value(update.Value))
	}

	for _, update := range update.Codes {
		// Store the code length.
		key := getBasicDataKey(update.Account)
		value := s.trie.Get(key)
		size := len(update.Code)
		binary.BigEndian.PutUint32(value[4:8], uint32(size))
		s.trie.Set(key, value)

		// Store the code hash.
		key = getCodeHashKey(update.Account)
		hash := common.Keccak256(update.Code)
		s.trie.Set(key, trie.Value(hash))

		// Store the actual code.
		chunks := splitCode(update.Code)
		for i, chunk := range chunks {
			key := getCodeChunkKey(update.Account, i)
			s.trie.Set(key, trie.Value(chunk))
		}
	}

	return nil
}

func (s *State) GetHash() (common.Hash, error) {
	return s.GetCommitment().Await().Get()
}

func (s *State) GetCommitment() future.Future[result.Result[common.Hash]] {
	hash := common.Hash(s.trie.Commit().Compress())
	return future.Immediate(result.Ok(hash))
}

// --- Operational Features ---

func (s *State) Check() error {
	return nil
}

func (s *State) Flush() error {
	return nil
}
func (s *State) Close() error {
	return nil
}

func (s *State) GetMemoryFootprint() *common.MemoryFootprint {
	return common.NewMemoryFootprint(1)
}

func (s *State) GetArchiveState(block uint64) (state.State, error) {
	return nil, state.NoArchiveError
}

func (s *State) GetArchiveBlockHeight() (height uint64, empty bool, err error) {
	return 0, true, state.NoArchiveError
}

func (s *State) CreateWitnessProof(address common.Address, keys ...common.Key) (witness.Proof, error) {
	return nil, fmt.Errorf("witness proof not supported yet")
}

func (s *State) Export(ctx context.Context, out io.Writer) (common.Hash, error) {
	panic("not implemented")
}
