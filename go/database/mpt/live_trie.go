// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package mpt

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path"
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/witness"
)

// LiveTrie retains a single trie encoding state information with destructible
// updates. Thus, whenever updating some information, the previous state is
// lost.
//
// Its main role is to adapt the maintain a root node and to provide a single-
// trie view on a forest.
type LiveTrie struct {
	// The node structure of the trie.
	forest Database
	// The root node of the trie.
	root NodeReference
	// The file name for storing trie metadata.
	metaDataFile string
}

// OpenInMemoryLiveTrie loads trie information from the given directory and
// creates a LiveTrie instance retaining all information in memory. If the
// directory is empty, an empty trie is created.
func OpenInMemoryLiveTrie(directory string, config MptConfig, cacheConfig NodeCacheConfig) (*LiveTrie, error) {
	forestConfig := ForestConfig{Mode: Mutable, NodeCacheConfig: cacheConfig}
	forest, err := OpenInMemoryForest(directory, config, forestConfig)
	if err != nil {
		return nil, err
	}

	return openTrieOnForest(directory, forest)
}

// OpenFileLiveTrie loads trie information from the given directory and
// creates a LiveTrie instance using a fixed-size cache for retaining nodes in
// memory, backed by a file-based storage automatically kept in sync. If the
// directory is empty, an empty trie is created.
func OpenFileLiveTrie(directory string, config MptConfig, cacheConfig NodeCacheConfig) (*LiveTrie, error) {
	forestConfig := ForestConfig{Mode: Mutable, NodeCacheConfig: cacheConfig}
	forest, err := OpenFileForest(directory, config, forestConfig)
	if err != nil {
		return nil, err
	}

	return openTrieOnForest(directory, forest)
}

// openTrieOnForest creates a LiveTrie instance based on the given forest.
// The forest is closed if the trie creation fails.
func openTrieOnForest(directory string, forest *Forest) (*LiveTrie, error) {
	trie, err := makeTrie(directory, forest)
	if err != nil {
		return nil, errors.Join(err, forest.Close())
	}
	return trie, nil
}

// VerifyFileLiveTrie validates a file-based live trie stored in the given
// directory. If the test passes, the data stored in the respective directory
// can be considered to be a valid Live Trie of the given configuration.
func VerifyFileLiveTrie(ctx context.Context, directory string, config MptConfig, observer VerificationObserver) error {
	metadata, exists, err := readMetadata(getLiveTrieMetadataPath(directory))
	if err != nil {
		return err
	}
	if !exists {
		return nil
	}
	return verifyFileForest(ctx, directory, config, []Root{{
		NewNodeReference(metadata.RootNode),
		metadata.RootHash,
	}}, observer)
}

func makeTrie(
	directory string,
	forest *Forest,
) (*LiveTrie, error) {
	if err := forest.CheckErrors(); err != nil {
		return nil, fmt.Errorf("unable to open corrupted forest: %w", err)
	}
	// Parse metadata file.
	metaDataFile := getLiveTrieMetadataPath(directory)
	metadata, _, err := readMetadata(metaDataFile)
	if err != nil {
		return nil, err
	}
	return &LiveTrie{
		root:         NewNodeReference(metadata.RootNode),
		metaDataFile: metaDataFile,
		forest:       forest,
	}, nil
}

// getTrieView creates a live trie based on an existing Forest instance.
func getTrieView(root NodeReference, forest Database) *LiveTrie {
	return &LiveTrie{
		root:   root,
		forest: forest,
	}
}

// HasEmptyStorage returns true if account has empty storage.
func (s *LiveTrie) HasEmptyStorage(addr common.Address) (bool, error) {
	return s.forest.HasEmptyStorage(&s.root, addr)
}

func (s *LiveTrie) GetAccountInfo(addr common.Address) (AccountInfo, bool, error) {
	return s.forest.GetAccountInfo(&s.root, addr)
}

func (s *LiveTrie) SetAccountInfo(addr common.Address, info AccountInfo) error {
	newRoot, err := s.forest.SetAccountInfo(&s.root, addr, info)
	if err != nil {
		return err
	}
	s.root = newRoot
	return nil
}

func (s *LiveTrie) GetValue(addr common.Address, key common.Key) (common.Value, error) {
	return s.forest.GetValue(&s.root, addr, key)
}

func (s *LiveTrie) SetValue(addr common.Address, key common.Key, value common.Value) error {
	newRoot, err := s.forest.SetValue(&s.root, addr, key, value)
	if err != nil {
		return err
	}
	s.root = newRoot
	return nil
}

func (s *LiveTrie) ClearStorage(addr common.Address) error {
	newRoot, err := s.forest.ClearStorage(&s.root, addr)
	if err != nil {
		return err
	}
	s.root = newRoot
	return nil
}

func (s *LiveTrie) UpdateHashes() (common.Hash, *NodeHashes, error) {
	return s.forest.updateHashesFor(&s.root)
}

func (s *LiveTrie) setHashes(hashes *NodeHashes) error {
	return s.forest.setHashesFor(&s.root, hashes)
}

func (s *LiveTrie) VisitTrie(mode AccessMode, visitor NodeVisitor) error {
	return s.forest.VisitTrie(&s.root, mode, visitor)
}

// VisitAccountStorage visits the storage nodes of an account with the given address.
// The visited nodes do not contain the account node itself.
func (s *LiveTrie) VisitAccountStorage(
	address common.Address,
	mode AccessMode,
	visitor NodeVisitor,
) error {
	var innerError error
	// this visitor finds the account node with the given address
	accountVisitor := MakeVisitor(func(node Node, info NodeInfo) VisitResponse {
		switch n := node.(type) {
		case *AccountNode:
			if n.Address() == address {
				// and then it sends the storage node to the visitor
				_, innerError = n.visitStorage(s.forest, 0, mode, visitor)
			}
			return VisitResponseAbort
		}

		return VisitResponseContinue
	})

	_, err := VisitPathToAccount(s.forest, &s.root, address, mode, accountVisitor)
	return errors.Join(innerError, err)
}

func (s *LiveTrie) CreateWitnessProof(addr common.Address, keys ...common.Key) (witness.Proof, error) {
	return CreateWitnessProof(s.forest, &s.root, addr, keys...)
}

func (s *LiveTrie) Flush() error {
	// Update hashes to eliminate dirty hashes before flushing.
	hash, _, err := s.UpdateHashes()
	if err != nil {
		return err
	}

	// Update on-disk meta-data.
	metadata, err := json.Marshal(metadata{
		RootNode: s.root.Id(),
		RootHash: hash,
	})

	if err == nil {
		if err := os.WriteFile(s.metaDataFile, metadata, 0600); err != nil {
			return err
		}
	}

	return errors.Join(err, s.forest.Flush())
}

func (s *LiveTrie) Close() error {
	return errors.Join(
		s.Flush(),
		s.forest.Close(),
	)
}

// GetMemoryFootprint provides sizes of individual components of the state in the memory
func (s *LiveTrie) GetMemoryFootprint() *common.MemoryFootprint {
	mf := common.NewMemoryFootprint(unsafe.Sizeof(*s))
	mf.AddChild("forest", s.forest.GetMemoryFootprint())
	return mf
}

// Dump prints the content of the Trie to the console. Mainly intended for debugging.
func (s *LiveTrie) Dump() {
	s.forest.Dump(&s.root)
}

// GetConfig returns the configuration of the trie.
func (s *LiveTrie) GetConfig() MptConfig {
	return s.forest.getConfig()
}

// Check verifies internal invariants of the Trie instance. If the trie is
// self-consistent, nil is returned and the Trie is read to be accessed. If
// errors are detected, the Trie is to be considered in an invalid state and
// the behavior of all other operations is undefined.
func (s *LiveTrie) Check() error {
	return s.forest.Check(&s.root)
}

// -- LiveTrie metadata --

// metadata is the helper type to read and write metadata from/to the disk.
type metadata struct {
	RootNode NodeId
	RootHash common.Hash
}

// getLiveTrieMetadataPath returns the path to the LiveTrie's metadata file of
// the given directory.
func getLiveTrieMetadataPath(directory string) string {
	return path.Join(directory, "meta.json")
}

// readMetadata parses the content of the given file if it exists or returns
// a default-initialized metadata struct if there is no such file.
func readMetadata(filename string) (metadata, bool, error) {

	// If there is no file, initialize and return default metadata.
	if _, err := os.Stat(filename); err != nil {
		return metadata{}, false, nil
	}

	// If the file exists, parse it and return its content.
	data, err := os.ReadFile(filename)
	if err != nil {
		return metadata{}, false, err
	}

	var meta metadata
	if err := json.Unmarshal(data, &meta); err != nil {
		return meta, false, err
	}
	return meta, true, nil
}
