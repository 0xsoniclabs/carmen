package memory

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"

	"github.com/0xsoniclabs/carmen/go/backend"
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/common/witness"
	"github.com/0xsoniclabs/carmen/go/database/vt/memory/trie"
	"github.com/0xsoniclabs/carmen/go/state"
)

type State struct {
	trie *trie.Trie
}

func NewState() *State {
	return &State{
		trie: &trie.Trie{},
	}
}

func (s *State) Exists(address common.Address) (bool, error) {
	// TODO: figure out whether this is actually required
	panic("not implemented")
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
	return false, fmt.Errorf("this is not supported by Verkle Tries")
}

func (s *State) Apply(block uint64, update common.Update) error {

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

	for _, update := range update.Slots {
		key := getStorageKey(update.Account, update.Key)
		s.trie.Set(key, trie.Value(update.Value))
	}

	return nil
}

func (s *State) GetHash() (common.Hash, error) {
	return s.trie.Commit().Compress(), nil
}

// --- Operational Features ---

func (s *State) Flush() error {
	return nil
}
func (s *State) Close() error {
	return nil
}

func (s *State) GetMemoryFootprint() *common.MemoryFootprint {
	panic("not implemented")
}

func (s *State) GetArchiveState(block uint64) (state.State, error) {
	return nil, state.NoArchiveError
}

func (s *State) GetArchiveBlockHeight() (height uint64, empty bool, err error) {
	return 0, true, state.NoArchiveError
}

func (s *State) Check() error {
	return nil
}

func (s *State) CreateWitnessProof(address common.Address, keys ...common.Key) (witness.Proof, error) {
	return nil, fmt.Errorf("witness proof not supported")
}

func (s *State) Export(ctx context.Context, out io.Writer) (common.Hash, error) {
	return common.Hash{}, fmt.Errorf("export not supported")
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
