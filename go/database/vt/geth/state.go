// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package geth

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/0xsoniclabs/carmen/go/backend"
	"github.com/0xsoniclabs/carmen/go/backend/archive"
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/common/witness"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/trie"
	"github.com/ethereum/go-ethereum/trie/utils"

	ethcommon "github.com/ethereum/go-ethereum/common"
)

// NewState creates a new verkle state using an in-memory source.
// It uses the Verkle Trie from the Ethereum Geth implementation.
// This state is experimental, stores data in-memory only,
// and not intended for production use.
func NewState(params state.Parameters) (state.State, error) {
	return newState(params)
}

// NewMemoryState creates a new verkle state where the whole trie
// is always kept in memory.
// It uses the Verkle Trie from the Ethereum Geth implementation.
// This state is experimental, stores data in-memory only,
// and not intended for production use.
func NewMemoryState(params state.Parameters) (state.State, error) {
	vs, err := newState(params)
	if err != nil {
		return nil, err
	}

	source := singleNodeReader{source: newMemorySource()}
	return &persistentVerkleState{*vs, source}, nil
}

func newState(_ state.Parameters) (*verkleState, error) {
	pointCache := utils.NewPointCache(4096)
	vt, err := trie.NewVerkleTrie(ethcommon.Hash{}, nil, pointCache)
	if err != nil {
		return nil, errors.Join(nil, err)
	}
	return &verkleState{
		pointCache: pointCache,
		verkle:     vt,
		codes:      make(map[common.Address][]byte),
	}, nil
}

// verkleState implements the state.State interface for a verkle trie.
// It adapts to the VerkleTrie implementation from the Ethereum Geth library.
// This is a reference implementation to compare with the original Geth.
type verkleState struct {
	pointCache *utils.PointCache
	verkle     *trie.VerkleTrie
	codes      map[common.Address][]byte // current Verkle Trie does not support code retrieval, so we use a map to store codes
}

func (s *verkleState) DeleteAccount(address common.Address) error {
	return fmt.Errorf("not supported: verkle trie does not support deleting accounts")
}

func (s *verkleState) SetNonce(address common.Address, nonce common.Nonce) error {
	account, err := s.getAccount(address)
	if err != nil {
		return err
	}

	account.Nonce = nonce.ToUint64()

	size, err := s.GetCodeSize(address)
	if err != nil {
		return err
	}

	return s.verkle.UpdateAccount(ethcommon.Address(address), account, size)
}

func (s *verkleState) SetStorage(address common.Address, key common.Key, value common.Value) error {
	return s.verkle.UpdateStorage(ethcommon.Address(address), key[:], value[:])
}

func (s *verkleState) SetCode(address common.Address, code []byte) error {
	account, err := s.getAccount(address)
	if err != nil {
		return err
	}

	// update code len and code hash first
	codeHash := common.Keccak256(code)
	account.CodeHash = codeHash[:]
	if err := s.verkle.UpdateAccount(ethcommon.Address(address), account, len(code)); err != nil {
		return err
	}

	// insert code into the trie
	if err := s.verkle.UpdateContractCode(ethcommon.Address(address), ethcommon.Hash(codeHash), code); err != nil {
		return err
	}

	// put in the local map for retrieval
	s.codes[address] = bytes.Clone(code)
	return nil
}

func (s *verkleState) Exists(address common.Address) (bool, error) {
	account, err := s.getAccount(address)
	if err != nil {
		return false, err
	}

	return account.Nonce != 0 || account.Balance.Uint64() != 0, nil
}

func (s *verkleState) GetNonce(address common.Address) (common.Nonce, error) {
	account, err := s.getAccount(address)
	if err != nil {
		return common.Nonce{}, err
	}

	return common.ToNonce(account.Nonce), nil
}

func (s *verkleState) GetStorage(address common.Address, key common.Key) (common.Value, error) {
	value, err := s.verkle.GetStorage(ethcommon.Address(address), key[:])
	if err != nil {
		return common.Value{}, err
	}

	var commonValue common.Value
	copy(commonValue[32-len(value):], value)
	return commonValue, nil
}

func (s *verkleState) GetCode(address common.Address) ([]byte, error) {
	// current Verkle Trie does not support retrieval of codes, i.e. we pick them from the map
	return s.codes[address], nil
}

func (s *verkleState) GetCodeSize(address common.Address) (int, error) {
	code := s.codes[address]
	if code == nil {
		return 0, nil
	}

	return len(code), nil
}

func (s *verkleState) GetCodeHash(address common.Address) (common.Hash, error) {
	account, err := s.getAccount(address)
	if err != nil {
		return common.Hash{}, err
	}

	return common.Hash(account.CodeHash), nil
}

func (s *verkleState) HasEmptyStorage(addr common.Address) (bool, error) {
	return false, fmt.Errorf("not supported: verkle trie does not support has empty storage")
}

func (s *verkleState) GetHash() (common.Hash, error) {
	return common.Hash(s.verkle.Hash()), nil
}

func (s *verkleState) GetMemoryFootprint() *common.MemoryFootprint {
	return common.NewMemoryFootprint(uintptr(1))
}

func (s *verkleState) GetArchiveState(block uint64) (state.State, error) {
	return nil, state.NoArchiveError
}

func (s *verkleState) GetArchiveBlockHeight() (height uint64, empty bool, err error) {
	return 0, true, state.NoArchiveError
}

func (s *verkleState) CreateAccount(address common.Address) error {
	account, err := s.verkle.GetAccount(ethcommon.Address(address))
	if account != nil || err != nil {
		return err
	}

	account = types.NewEmptyStateAccount()
	return s.verkle.UpdateAccount(ethcommon.Address(address), account, 0)
}

func (s *verkleState) SetBalance(address common.Address, balance amount.Amount) error {
	account, err := s.getAccount(address)
	if err != nil {
		return err
	}

	val := balance.Uint256()
	account.Balance = &val

	size, err := s.GetCodeSize(address)
	if err != nil {
		return err
	}

	return s.verkle.UpdateAccount(ethcommon.Address(address), account, size)
}

func (s *verkleState) GetBalance(address common.Address) (amount.Amount, error) {
	account, err := s.getAccount(address)
	if err != nil {
		return amount.Amount{}, err
	}

	return amount.NewFromUint256(account.Balance), nil
}

func (s *verkleState) Apply(block uint64, update common.Update) error {
	if err := update.ApplyTo(s); err != nil {
		return err
	}

	return nil
}

//
//		Witness Proof features -- not supported at the moment
//

func (s *verkleState) CreateWitnessProof(address common.Address, keys ...common.Key) (witness.Proof, error) {
	return nil, archive.ErrWitnessProofNotSupported // not supported at the moment, will be implemented later
}

//
//		Snapshot features -- not supported in Verkle Trie
//

func (s *verkleState) GetProof() (backend.Proof, error) {
	return nil, backend.ErrSnapshotNotSupported // not supported at the moment, will be implemented later
}

func (s *verkleState) CreateSnapshot() (backend.Snapshot, error) {
	return nil, backend.ErrSnapshotNotSupported
}

func (s *verkleState) Restore(data backend.SnapshotData) error {
	return backend.ErrSnapshotNotSupported
}

func (s *verkleState) GetSnapshotVerifier(metadata []byte) (backend.SnapshotVerifier, error) {
	return nil, backend.ErrSnapshotNotSupported
}

//
//	Operation features -- not supported
//

func (s *verkleState) Export(ctx context.Context, out io.Writer) (common.Hash, error) {
	return common.Hash{}, fmt.Errorf("not supported: verkle trie does not support export")
}

func (s *verkleState) Check() error {
	return nil
}

//
//	I/O features
//

func (s *verkleState) Flush() error {
	return nil
}

func (s *verkleState) Close() error {
	return nil
}

func (s *verkleState) getAccount(address common.Address) (*types.StateAccount, error) {
	account, err := s.verkle.GetAccount(ethcommon.Address(address))
	if err != nil {
		return nil, err
	}
	if account == nil {
		account = types.NewEmptyStateAccount()
	}

	return account, nil
}

// persistentVerkleState is a verkleState that persists changes to a singleNodeReader source.
// It adapts to the VerkleTrie implementation from the Ethereum Geth library.
// This is a reference implementation to compare with the original Geth.
type persistentVerkleState struct {
	verkleState
	source singleNodeReader
}

func (s *persistentVerkleState) Apply(block uint64, update common.Update) error {
	if err := update.ApplyTo(s); err != nil {
		return err
	}

	rootHash, nodeSet := s.verkle.Commit(false)
	var errs []error
	for path, node := range nodeSet.Nodes {
		errs = append(errs, s.source.getSource().set([]byte(path), node.Blob))
	}

	if err := errors.Join(errs...); err != nil {
		return err
	}

	// recreate the verkle trie to flush the in-memory nodes
	vt, err := trie.NewVerkleTrie(rootHash, s.source, s.pointCache)
	if err != nil {
		return err
	}
	s.verkle = vt

	return nil
}

func (s *persistentVerkleState) Flush() error {
	return s.source.getSource().Flush()
}

func (s *persistentVerkleState) Close() error {
	return errors.Join(
		s.Flush(),
		s.source.getSource().Close(),
	)
}
