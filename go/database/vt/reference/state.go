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

	"github.com/0xsoniclabs/carmen/go/backend"
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/common/future"
	"github.com/0xsoniclabs/carmen/go/common/result"
	"github.com/0xsoniclabs/carmen/go/common/witness"
	"github.com/0xsoniclabs/carmen/go/database/vt/commit"
	"github.com/0xsoniclabs/carmen/go/database/vt/reference/trie"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/ethereum/go-ethereum/core/types"
)

// State is an in-memory implementation of a chain-state tracking account and
// storage data using a Verkle Trie. It implements the state.State interface.
type State struct {
	store     store
	embedding embedding
}

// store is the interface that wraps basic Verkle trie operations used by the
// reference state implementation.
type store interface {
	Get(key trie.Key) trie.Value
	Set(key trie.Key, value trie.Value)
	Commit() commit.Commitment
}

// embedding defines the mapping from high-level state concepts like accounts,
// storage slots, and code chunks to Verkle trie keys.
type embedding interface {
	GetBasicDataKey(address common.Address) trie.Key
	GetStorageKey(address common.Address, key common.Key) trie.Key
	GetCodeChunkKey(address common.Address, index int) trie.Key
	GetCodeHashKey(address common.Address) trie.Key
}

// NewState creates a new, empty in-memory state instance.
func NewState(params state.Parameters) (state.State, error) {
	return NewCustomState(params, &trie.Trie{}, Embedding{})
}

// NewCustomState creates a new Verkle state instance using the provided
// trie and embedding implementations.
func NewCustomState(
	_ state.Parameters,
	store store,
	embedding embedding,
) (state.State, error) {
	return &State{
		store:     store,
		embedding: embedding,
	}, nil
}

func (s *State) Exists(address common.Address) (bool, error) {
	key := s.embedding.GetBasicDataKey(address)
	value := s.store.Get(key)
	var empty [24]byte // nonce and balance are layed out in bytes 8-32
	return !bytes.Equal(value[8:32], empty[:]), nil

}

func (s *State) GetBalance(address common.Address) (amount.Amount, error) {
	key := s.embedding.GetBasicDataKey(address)
	value := s.store.Get(key)
	return amount.NewFromBytes(value[16:32]...), nil
}

func (s *State) GetNonce(address common.Address) (common.Nonce, error) {
	key := s.embedding.GetBasicDataKey(address)
	value := s.store.Get(key)
	return common.Nonce(value[8:16]), nil
}

func (s *State) GetStorage(address common.Address, key common.Key) (common.Value, error) {
	return common.Value(s.store.Get(s.embedding.GetStorageKey(address, key))), nil
}

func (s *State) GetCode(address common.Address) ([]byte, error) {
	size, _ := s.GetCodeSize(address)
	chunks := make([]chunk, 0, size)
	for i := 0; i < size/31+1; i++ {
		key := s.embedding.GetCodeChunkKey(address, i)
		value := s.store.Get(key)
		chunks = append(chunks, chunk(value))
	}
	return merge(chunks, size), nil
}

func (s *State) GetCodeSize(address common.Address) (int, error) {
	key := s.embedding.GetBasicDataKey(address)
	value := s.store.Get(key)
	return int(binary.BigEndian.Uint32(value[4:8])), nil
}

func (s *State) GetCodeHash(address common.Address) (common.Hash, error) {
	key := s.embedding.GetCodeHashKey(address)
	value := s.store.Get(key)
	return common.Hash(value[:]), nil
}

func (s *State) HasEmptyStorage(addr common.Address) (bool, error) {
	return true, nil
}

func (s *State) Apply(block uint64, update common.Update) error {

	// init potentially empty accounts with empty code hash,
	for _, address := range update.CreatedAccounts {
		accountKey := s.embedding.GetBasicDataKey(address)
		value := s.store.Get(accountKey)
		var empty [28]byte
		// empty accnout has empty code size, nonce, and balance
		if bytes.Equal(value[4:32], empty[:]) {
			codeHashKey := s.embedding.GetCodeHashKey(address)
			s.store.Set(accountKey, value) // must be initialized to empty account
			s.store.Set(codeHashKey, trie.Value(types.EmptyCodeHash))
		}
	}

	for _, update := range update.Nonces {
		key := s.embedding.GetBasicDataKey(update.Account)
		value := s.store.Get(key)
		copy(value[8:16], update.Nonce[:])
		s.store.Set(key, value)
	}

	for _, update := range update.Balances {
		key := s.embedding.GetBasicDataKey(update.Account)
		value := s.store.Get(key)
		amount := update.Balance.Bytes32()
		copy(value[16:32], amount[16:])
		s.store.Set(key, value)
	}

	for _, update := range update.Slots {
		key := s.embedding.GetStorageKey(update.Account, update.Key)
		s.store.Set(key, trie.Value(update.Value))
	}

	for _, update := range update.Codes {
		// Store the code length.
		key := s.embedding.GetBasicDataKey(update.Account)
		value := s.store.Get(key)
		size := len(update.Code)
		binary.BigEndian.PutUint32(value[4:8], uint32(size))
		s.store.Set(key, value)

		// Store the code hash.
		key = s.embedding.GetCodeHashKey(update.Account)
		hash := common.Keccak256(update.Code)
		s.store.Set(key, trie.Value(hash))

		// Store the actual code.
		chunks := splitCode(update.Code)
		for i, chunk := range chunks {
			key := s.embedding.GetCodeChunkKey(update.Account, i)
			s.store.Set(key, trie.Value(chunk))
		}
	}

	return nil
}

func (s *State) GetHash() (common.Hash, error) {
	return s.GetCommitment().Await().Get()
}

func (s *State) GetCommitment() future.Future[result.Result[common.Hash]] {
	hash := common.Hash(s.store.Commit().Compress())
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

// Snapshot & Recovery
func (s *State) GetProof() (backend.Proof, error) {
	return nil, backend.ErrSnapshotNotSupported
}

func (s *State) CreateSnapshot() (backend.Snapshot, error) {
	return nil, backend.ErrSnapshotNotSupported
}
func (s *State) Restore(backend.SnapshotData) error {
	return backend.ErrSnapshotNotSupported
}
func (s *State) GetSnapshotVerifier([]byte) (backend.SnapshotVerifier, error) {
	return nil, backend.ErrSnapshotNotSupported
}
