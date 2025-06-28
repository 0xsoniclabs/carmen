// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package io

import (
	"context"
	"errors"
	"fmt"
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/interrupt"
	"github.com/0xsoniclabs/carmen/go/common/witness"
	"github.com/0xsoniclabs/carmen/go/database/mpt"
	"golang.org/x/exp/maps"
	"math/rand"
)

//go:generate mockgen -source verification_proof.go -destination verification_proof_mocks.go -package io

// ErrInvalidProof is an error returned when a witness proof is not invalid.
const ErrInvalidProof = common.ConstError("invalid proof")

// VerifyArchiveTrieProof verifies the consistency of witness proofs for an archive trie.
// It reads the trie for each block within the input range.
// It gets account and storage slots and extracts a witness proof for each account and its storage.
// It is checked that values in the database and the proof match.
func VerifyArchiveTrieProof(ctx context.Context, dir string, config mpt.MptConfig, from, to int, observer mpt.VerificationObserver) error {
	trie, err := mpt.OpenArchiveTrie(dir, config, mpt.NodeCacheConfig{}, mpt.ArchiveConfig{})
	if err != nil {
		return err
	}

	observer.StartVerification()
	err = errors.Join(
		verifyArchiveTrieProof(ctx, trie, from, to, dir, config, observer),
		trie.Close(),
	)
	observer.EndVerification(err)
	return err
}

func verifyArchiveTrieProof(ctx context.Context, trie verifiableArchiveTrie, from, to int, dir string, config mpt.MptConfig, observer mpt.VerificationObserver) error {
	blockHeight, empty, err := trie.GetBlockHeight()
	if err != nil {
		return err
	}
	if empty {
		return nil
	}

	if to > int(blockHeight) {
		to = int(blockHeight)
		observer.Progress(fmt.Sprintf("setting a maximum block height: %d", blockHeight))
	}

	observer.Progress(fmt.Sprintf("Verifying total block range [%d;%d]", from, to))
	for i := from; i <= to; i++ {
		root, err := trie.GetBlockRoot(uint64(i))
		if err != nil {
			return err
		}
		trieView := &archiveTrieView{trie: trie, block: uint64(i)}
		observer.Progress(fmt.Sprintf("Verifying block: %d ", i))
		if err := verifyTrieProof(ctx, trieView, root, dir, config, observer); err != nil {
			return err
		}
	}

	return nil
}

// VerifyLiveTrieProof verifies the consistency of witness proofs for a live trie.
// It reads the trie for the head block and loads accounts and storage slots.
// It extracts witness proofs for these accounts and its storage,
// and checks that values in the proof and the database match.
func VerifyLiveTrieProof(ctx context.Context, dir string, config mpt.MptConfig, observer mpt.VerificationObserver) error {
	trie, err := mpt.OpenFileLiveTrie(dir, config, mpt.NodeCacheConfig{})
	if err != nil {
		return err
	}

	root := trie.RootNodeId()
	observer.StartVerification()
	err = errors.Join(
		verifyTrieProof(ctx, trie, root, dir, config, observer),
		trie.Close(),
	)
	observer.EndVerification(err)
	return err
}

func verifyTrieProof(ctx context.Context, trie verifiableTrie, root mpt.NodeId, dir string, config mpt.MptConfig, observer mpt.VerificationObserver) error {
	rootHash, hints, err := trie.UpdateHashes()
	if hints != nil {
		hints.Release()
	}
	if err != nil {
		return err
	}

	observer.Progress("Collecting and Verifying proofs ... ")
	visitor := accountVerifyingProofVisitor{
		ctx:       ctx,
		rootHash:  rootHash,
		trie:      trie,
		observer:  observer,
		directory: dir,
		config:    config,
		logWindow: 1000_000,
	}

	if err := visitAll(dir, config, root, &visitor, true); err != nil {
		return err
	}

	const numAddresses = 10
	return verifyEmptyAccountProof(trie, rootHash, numAddresses, observer)
}

// verifyEmptyAccountProof verifies the consistency of witness proofs for empty accounts that are not present in the trie.
func verifyEmptyAccountProof(trie verifiableTrie, rootHash common.Hash, numAddresses int, observer mpt.VerificationObserver) error {
	observer.Progress(fmt.Sprintf("Veryfing %d empty addresses...", numAddresses))
	addresses, err := generateUnusedAddresses(trie, numAddresses)
	if err != nil {
		return err
	}
	for _, addr := range addresses {
		proof, err := trie.CreateWitnessProof(addr)
		if err != nil {
			return err
		}
		if !proof.IsValid() {
			return ErrInvalidProof
		}
		// expect an empty account
		if err := verifyAccountProof(rootHash, proof, addr, mpt.AccountInfo{}); err != nil {
			return err
		}
	}

	return nil
}

// verifyUnusedStorageSlotsProof verifies the consistency of witness proofs for empty storage that are not present in the trie.
func verifyUnusedStorageSlotsProof(trie verifiableTrie, rootHash common.Hash, addr common.Address) error {
	const numKeys = 10
	keys, err := generateUnusedKeys(trie, numKeys, addr)
	if err != nil {
		return err
	}
	proof, err := trie.CreateWitnessProof(addr, keys...)
	if err != nil {
		return err
	}
	if !proof.IsValid() {
		return ErrInvalidProof
	}

	if err := verifyStorageProof(rootHash, proof, addr, keys, nil); err != nil {
		return err
	}

	return nil
}

// verifyAccountProof verifies the consistency between the input witness proofs and the account.
func verifyAccountProof(root common.Hash, proof witness.Proof, addr common.Address, info mpt.AccountInfo) error {
	balance, complete, err := proof.GetBalance(root, addr)
	if err != nil {
		return err
	}
	if !complete {
		return fmt.Errorf("proof incomplete for 0x%x", addr)
	}
	if got, want := balance, info.Balance; got != want {
		return fmt.Errorf("balance mismatch for 0x%x, got %v, want %v", addr, got, want)
	}
	nonce, complete, err := proof.GetNonce(root, addr)
	if err != nil {
		return err
	}
	if !complete {
		return fmt.Errorf("proof incomplete for 0x%x", addr)
	}
	if got, want := nonce, info.Nonce; got != want {
		return fmt.Errorf("nonce mismatch for 0x%x, got %v, want %v", addr, got, want)
	}
	code, complete, err := proof.GetCodeHash(root, addr)
	if err != nil {
		return err
	}
	if !complete {
		return fmt.Errorf("proof incomplete for 0x%x", addr)
	}
	if got, want := code, info.CodeHash; got != want {
		return fmt.Errorf("code mismatch for 0x%x, got %v, want %v", addr, got, want)
	}

	return nil
}

// verifyStorageProof verifies the consistency between the input witness proofs and the storage.
func verifyStorageProof(root common.Hash, proof witness.Proof, addr common.Address, keys []common.Key, storage map[common.Key]common.Value) error {
	for _, key := range keys {
		proofValue, complete, err := proof.GetState(root, addr, key)
		if err != nil {
			return err
		}
		if !complete {
			return fmt.Errorf("proof incomplete for address: 0x%x, key: 0x%x", addr, key)
		}
		if got, want := proofValue, storage[key]; got != want {
			return fmt.Errorf("storage mismatch for 0x%x, key 0x%x, got %v, want %v", addr, key, got, want)
		}
	}

	return nil
}

// generateUnusedAddresses generates a slice of addresses that do not appear in the input MPT.
func generateUnusedAddresses(trie verifiableTrie, number int) ([]common.Address, error) {
	res := make([]common.Address, 0, number)
	for len(res) < number {
		j := rand.Int()
		addr := common.Address{byte(j), byte(j >> 8), byte(j >> 16), byte(j >> 24), 1}

		// if an unlikely situation happens and the address is in the trie, skip it
		_, exists, err := trie.GetAccountInfo(addr)
		if err != nil {
			return nil, err
		}

		if exists {
			continue
		}

		res = append(res, addr)
	}

	return res, nil
}

// generateUnusedKeys generates a slice of keys  that do not appear in the input MPT under the given account.
func generateUnusedKeys(trie verifiableTrie, number int, address common.Address) ([]common.Key, error) {
	res := make([]common.Key, 0, number)
	for len(res) < number {
		j := rand.Int()
		key := common.Key{byte(j), byte(j >> 8), byte(j >> 16), byte(j >> 24), 1}

		// if an unlikely situation happens and the key is not in the trie, skip it
		val, err := trie.GetValue(address, key)
		if err != nil {
			return nil, err
		}

		if val != (common.Value{}) {
			continue
		}

		res = append(res, key)
	}

	return res, nil
}

// accountVerifyingProofVisitor is a visitor that verifies the consistency of witness proofs for a live trie.
// It collects account and storage slots and extracts witness proofs for each account and its storage.
// It checks that values in the database and the proof match.
// The process can be interrupted by the input context.
type accountVerifyingProofVisitor struct {
	ctx       context.Context
	rootHash  common.Hash
	trie      verifiableTrie
	observer  mpt.VerificationObserver
	directory string
	config    mpt.MptConfig

	logWindow    int
	counter      int
	numAddresses int
}

func (v *accountVerifyingProofVisitor) Visit(n mpt.Node, nodeInfo mpt.NodeInfo) error {
	if v.counter%100 == 0 && interrupt.IsCancelled(v.ctx) {
		return interrupt.ErrCanceled
	}
	v.counter++

	switch n := n.(type) {
	case *mpt.AccountNode:
		proof, err := v.trie.CreateWitnessProof(n.Address())
		if err != nil {
			return err
		}
		if !proof.IsValid() {
			return ErrInvalidProof
		}

		if err := verifyAccountProof(v.rootHash, proof, n.Address(), n.Info()); err != nil {
			return err
		}

		storageVisitor := storageVerifyingProofVisitor{
			ctx:            v.ctx,
			rootHash:       v.rootHash,
			trie:           v.trie,
			currentAddress: n.Address(),
			storage:        make(map[common.Key]common.Value)}

		// visit the storage of the account
		if err := visitAll(v.directory, v.config, nodeInfo.Id, &storageVisitor, false); err != nil {
			return err
		}

		// verify remaining storage if not done inside the visitor
		if len(storageVisitor.storage) > 0 {
			if err := storageVisitor.verifyStorage(); err != nil {
				return err
			}
		}

		// add empty storages check
		if err := verifyUnusedStorageSlotsProof(v.trie, v.rootHash, n.Address()); err != nil {
			return err
		}

		v.numAddresses++
		if (v.numAddresses)%v.logWindow == 0 {
			v.observer.Progress(fmt.Sprintf("  ... verified %d addresses", v.numAddresses))
		}

		return nil
	}

	return nil
}

// storageVerifyingProofVisitor is a visitor that verifies the consistency of witness proofs for storage slots.
// It collects storage slots and extracts witness proofs for each storage slot.
// It checks that values in the database and the proof match.
// The process can be interrupted by the input context.
// Storage keys are verified in batches of 10 to save on memory and allowing for responsive cancellation.
type storageVerifyingProofVisitor struct {
	ctx            context.Context
	rootHash       common.Hash
	trie           verifiableTrie
	counter        int
	currentAddress common.Address
	storage        map[common.Key]common.Value
}

func (v *storageVerifyingProofVisitor) Visit(n mpt.Node, _ mpt.NodeInfo) error {
	if v.counter%100 == 0 && interrupt.IsCancelled(v.ctx) {
		return interrupt.ErrCanceled
	}
	v.counter++

	switch n := n.(type) {
	case *mpt.ValueNode:
		v.storage[n.Key()] = n.Value()
	}

	// when ten keys accumulate, verify the storage
	if len(v.storage) >= 10 {
		if err := v.verifyStorage(); err != nil {
			return err
		}
	}

	return nil
}

// verifyStorageProof verifies the consistency of witness proofs for storage slots.
func (v *storageVerifyingProofVisitor) verifyStorage() error {
	keys := maps.Keys(v.storage)

	proof, err := v.trie.CreateWitnessProof(v.currentAddress, keys...)
	if err != nil {
		return err
	}
	if !proof.IsValid() {
		return ErrInvalidProof
	}

	if err := verifyStorageProof(v.rootHash, proof, v.currentAddress, keys, v.storage); err != nil {
		return err
	}

	v.storage = make(map[common.Key]common.Value)
	return nil
}

// verifiableTrie is an interface for a trie that can provide witness proofs
// and trie properties to validate the witness proofs against the trie.
type verifiableTrie interface {

	// GetAccountInfo returns the account info for the given address.
	GetAccountInfo(addr common.Address) (mpt.AccountInfo, bool, error)

	// GetValue returns the value for the given address and key.
	GetValue(addr common.Address, key common.Key) (common.Value, error)

	// UpdateHashes updates the hashes of the trie, and returns the resulting root hash.
	UpdateHashes() (common.Hash, *mpt.NodeHashes, error)

	// CreateWitnessProof creates a witness proof for the given address and keys.
	CreateWitnessProof(common.Address, ...common.Key) (witness.Proof, error)
}

// verifiableArchiveTrie is an interface for an archive trie that can provide witness proofs
// and trie properties to validate the witness proofs against the trie.
type verifiableArchiveTrie interface {

	// GetAccountInfo returns the account info for the given address at the given block.
	GetAccountInfo(block uint64, addr common.Address) (mpt.AccountInfo, bool, error)

	// GetStorage returns the value for the given address and key at the given block.
	GetStorage(block uint64, addr common.Address, key common.Key) (common.Value, error)

	// GetHash returns the root hash of the trie at the given block.
	GetHash(block uint64) (common.Hash, error)

	// CreateWitnessProof creates a witness proof for the given address and keys at the given block.
	CreateWitnessProof(block uint64, address common.Address, keys ...common.Key) (witness.Proof, error)

	// GetBlockHeight returns the block height of the trie.
	GetBlockHeight() (uint64, bool, error)

	// GetBlockRoot returns the root hash of the trie at the given block.
	GetBlockRoot(block uint64) (mpt.NodeId, error)
}

// archiveTrieView is a wrapper for an archive trie that implements the verifiableTrie interface.
// It bounds the archive trie to a specific block.
type archiveTrieView struct {
	trie  verifiableArchiveTrie
	block uint64
}

func (v *archiveTrieView) GetAccountInfo(addr common.Address) (mpt.AccountInfo, bool, error) {
	return v.trie.GetAccountInfo(v.block, addr)
}

func (v *archiveTrieView) GetValue(addr common.Address, key common.Key) (common.Value, error) {
	return v.trie.GetStorage(v.block, addr, key)
}

func (v *archiveTrieView) UpdateHashes() (common.Hash, *mpt.NodeHashes, error) {
	hash, err := v.trie.GetHash(v.block)
	return hash, nil, err
}

func (v *archiveTrieView) CreateWitnessProof(address common.Address, keys ...common.Key) (witness.Proof, error) {
	return v.trie.CreateWitnessProof(v.block, address, keys...)
}
