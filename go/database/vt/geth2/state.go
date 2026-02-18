package geth2

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/ethereum/go-verkle"
)

type verkleState struct {
	root verkle.VerkleNode

	directory string
	store     NodeStore

	pathBased bool
}

func newState(
	params state.Parameters,
) (_ *verkleState, err error) {
	dataDir := params.Directory
	pathBased := params.Archive == state.NoArchive

	// Open the node store (e.g., LevelDB) for the given directory.
	store, err := newLevelDbStore(filepath.Join(dataDir, "nodes"))
	if err != nil {
		return nil, fmt.Errorf("failed to open node store: %w", err)
	}
	success := false
	defer func() {
		if !success {
			err = errors.Join(err, store.Close())
		}
	}()

	// Load the key for the root node.
	rootNodeKey := []byte{} // empty for path-based storage
	if !pathBased {
		// If not path-based, the root file contains the root node key.
		rootFile := getRootFilePath(dataDir)
		if _, err := os.Stat(rootFile); err == nil {
			rootId, err := os.ReadFile(rootFile)
			if err != nil || len(rootId) != 32 {
				return nil, fmt.Errorf("failed to read root file: %w", err)
			}
			rootNodeKey = rootId
		}
	}

	// Load the root node.
	var root verkle.VerkleNode
	if rootData, err := store.Get(rootNodeKey); err == ErrNotFound {
		root = verkle.New()
	} else if err != nil {
		return nil, fmt.Errorf("failed to read root node from store: %w", err)
	} else {
		rootNode, err := verkle.ParseNode(rootData, 0)
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize root node: %w", err)
		}
		root = rootNode
	}

	success = true
	return &verkleState{
		root:      root,
		directory: dataDir,
		store:     store,
		pathBased: pathBased,
	}, nil
}

func (s *verkleState) SetNonce(
	address common.Address,
	nonce common.Nonce,
) error {
	key := common.Keccak256ForAddress(address)
	return s.root.Insert(key[:], nonce[:], s.readNode)
}

func (s *verkleState) GetNonce(
	address common.Address,
) (common.Nonce, error) {
	key := common.Keccak256ForAddress(address)
	fmt.Printf("Getting nonce for address %v with key %x\n", address, key)
	data, err := s.root.Get(key[:], s.readNode)
	if err != nil {
		return common.Nonce{}, err
	}
	var nonce common.Nonce
	copy(nonce[:], data)
	return nonce, nil
}

func (s *verkleState) Flush() error {
	var issues []error
	flush := func(key []byte, node verkle.VerkleNode) {
		if err := s.flushNode(key, node); err != nil {
			issues = append(issues, err)
		}
	}

	// Flush all nodes.
	switch node := s.root.(type) {
	case *verkle.InternalNode:
		node.Flush(flush)
		// TODO: use FlushAtDepth to manage memory usage and performance trade-offs
		// node.FlushAtDepth(3, flush)
	}

	// Write root key to file.
	rootId := s.root.Commitment().Bytes()
	rootFilePath := getRootFilePath(s.directory)
	if err := os.WriteFile(rootFilePath, rootId[:], 0644); err != nil {
		issues = append(issues, fmt.Errorf("failed to write root file: %w", err))
	}

	return errors.Join(issues...)
}

func (s *verkleState) Close() error {
	return errors.Join(s.Flush(), s.store.Close())
}

func (s *verkleState) readNode(path []byte) ([]byte, error) {
	fmt.Printf("Reading node %x\n", path)
	return s.store.Get(path)
}

func (s *verkleState) flushNode(path []byte, node verkle.VerkleNode) error {
	data, err := node.Serialize()
	if err != nil {
		return fmt.Errorf("node serialization failed: %w", err)
	}
	hash := node.Commitment().Bytes()

	// TODO: support live and archive modes by optionally only writing the node
	// to the path and not the hash.

	fmt.Printf("Flushing node %x / %x\n", path, hash)
	if s.pathBased {
		return s.store.Set(path, data)
	}
	return errors.Join(
		s.store.Set(path, data),
		s.store.Set(hash[:], data),
	)
}

func getRootFilePath(dir string) string {
	return filepath.Join(dir, "root")
}
