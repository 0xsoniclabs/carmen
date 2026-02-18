package geth2

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"path/filepath"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/0xsoniclabs/carmen/go/common/future"
	"github.com/0xsoniclabs/carmen/go/common/result"
	"github.com/0xsoniclabs/carmen/go/common/witness"
	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/trie/utils"
	"github.com/ethereum/go-verkle"
	"github.com/holiman/uint256"
)

type verkleState struct {
	root   verkle.VerkleNode
	store  Store
	source NodeSource
}

func newState(
	params state.Parameters,
) (_ *verkleState, err error) {
	dataDir := params.Directory

	var liveOnly bool
	switch params.Archive {
	case state.NoArchive:
		liveOnly = true
	case state.LevelDbArchive:
		liveOnly = false
	default:
		return nil, fmt.Errorf("unsupported archive mode: %v", params.Archive)
	}

	// Open the node store (e.g., LevelDB) for the given directory.
	var store Store
	if liveOnly {
		store, err = newLevelDbLiveStore(filepath.Join(dataDir, "nodes"))
	} else {
		store, err = newLevelDbArchiveStore(filepath.Join(dataDir, "nodes"))
	}
	if err != nil {
		return nil, fmt.Errorf("failed to open node store: %w", err)
	}
	success := false
	defer func() {
		if !success {
			err = errors.Join(err, store.Close())
		}
	}()

	res, err := _newStateWithSource(store.HeadState())
	if err != nil {
		return nil, fmt.Errorf("failed to initialize state: %w", err)
	}
	res.store = store
	success = true
	return res, nil
}

func _newStateWithSource(source NodeSource) (_ *verkleState, err error) {

	// Load the root node, if present.
	var root verkle.VerkleNode
	if rootData, err := source.GetNode(nil); err == ErrNotFound {
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

	return &verkleState{
		root:   root,
		source: source,
	}, nil
}

// --- State interface implementation ---

func (s *verkleState) Exists(address common.Address) (bool, error) {
	panic("not implemented")
}

func (s *verkleState) GetBalance(address common.Address) (amount.Amount, error) {
	panic("not implemented")
}

func (s *verkleState) GetNonce(address common.Address) (common.Nonce, error) {

	// TODO: replace with proper tree embedding
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

func (s *verkleState) GetStorage(address common.Address, key common.Key) (common.Value, error) {
	panic("not implemented")
}

func (s *verkleState) GetCode(address common.Address) ([]byte, error) {
	panic("not implemented")
}

func (s *verkleState) GetCodeSize(address common.Address) (int, error) {
	panic("not implemented")
}

func (s *verkleState) GetCodeHash(address common.Address) (common.Hash, error) {
	panic("not implemented")
}

func (s *verkleState) HasEmptyStorage(addr common.Address) (bool, error) {
	return true, nil
}

func (s *verkleState) Apply(block uint64, update common.Update) error {

	// ------------------------------------------------------------------
	// 				  Implement all state modifications
	// ------------------------------------------------------------------

	// TODO: do proper updates, including embedding.
	for _, update := range update.Nonces {
		key := common.Keccak256ForAddress(update.Account)
		s.root.Insert(key[:], update.Nonce[:], s.readNode)
	}

	// ------------------------------------------------------------------

	// Compute and commit all changes to the store.
	nodes, err := s.root.(*verkle.InternalNode).BatchSerialize()
	if err != nil {
		return fmt.Errorf("failed to serialize nodes: %w", err)
	}
	changes := make([]Entry, 0, len(nodes))
	for _, node := range nodes {
		changes = append(changes, Entry{
			Path: node.Path,
			Blob: node.SerializedBytes,
		})
	}

	// Limit memory usage by pruning in-memory tree structure at a certain depth.
	// The nodes will be reloaded from the store when needed.
	s.root.(*verkle.InternalNode).FlushAtDepth(3, func([]byte, verkle.VerkleNode) {
		// no-op callback, since all nodes are stored as deltas.
	})

	return s.store.AddBlock(block, changes)
}

func (s *verkleState) GetHash() (common.Hash, error) {
	return s.GetCommitment().Await().Get()
}

func (s *verkleState) GetCommitment() future.Future[result.Result[common.Hash]] {
	return future.Immediate(result.Ok(common.Hash(s.root.Commitment().Bytes())))
}

func (s *verkleState) Flush() error {
	if s.store == nil {
		return nil
	}
	return s.store.Flush()
}

func (s *verkleState) Close() error {
	if s.store == nil {
		return nil
	}
	err := s.store.Close()
	s.store = nil
	return err
}

func (s *verkleState) GetMemoryFootprint() *common.MemoryFootprint {
	panic("not implemented")
}

func (s *verkleState) GetArchiveState(block uint64) (state.State, error) {
	source, err := s.store.HistoricState(block)
	if err != nil {
		return nil, err
	}
	// For simplicity, we are re-using the verkleState implementation for the
	// historic state. The store field remains nil set in this state view.
	return _newStateWithSource(source)
}

func (s *verkleState) GetArchiveBlockHeight() (height uint64, empty bool, err error) {
	next := s.store.NextBlock()
	if next == 0 {
		return 0, true, nil
	}
	return next - 1, false, nil
}

func (s *verkleState) Check() error {
	panic("not implemented")
}

func (s *verkleState) CreateWitnessProof(address common.Address, keys ...common.Key) (witness.Proof, error) {
	panic("not implemented")
}

func (s *verkleState) Export(ctx context.Context, out io.Writer) (common.Hash, error) {
	panic("not implemented")
}

// --- utility functions ---

func (s *verkleState) readNode(path []byte) ([]byte, error) {
	fmt.Printf("Reading node %x\n", path)
	if s.source == nil {
		// In live mode, the source is nil, and we read directly from the
		// store's head state.
		return s.store.HeadState().GetNode(path)
	}
	// In archive mode, the source is initialized with a view on the node store
	// at the requested block height.
	return s.source.GetNode(path)
}

func (s *verkleState) getAccountOrEmptyState(address common.Address) (*types.StateAccount, error) {
	account, err := s.getAccount(address)
	if err != nil {
		return nil, err
	}
	if account == nil {
		account = types.NewEmptyStateAccount()
	}
	return account, nil
}

// --- Ported from go-Ethereum, as this code got removed from the main branch --

var (
	errInvalidRootType = errors.New("invalid node type for root")
)

func (s *verkleState) getAccount(address common.Address) (*types.StateAccount, error) {
	// TODO: reference source of this logic;

	var (
		acc    = &types.StateAccount{}
		values [][]byte
		err    error
	)
	switch n := s.root.(type) {
	case *verkle.InternalNode:
		values, err = n.GetValuesAtStem(s.cache.GetStem(address[:]), s.readNode)
		if err != nil {
			return nil, fmt.Errorf("GetAccount (%x) error: %v", address, err)
		}
	default:
		return nil, errInvalidRootType
	}
	if values == nil {
		return nil, nil
	}
	basicData := values[utils.BasicDataLeafKey]
	acc.Nonce = binary.BigEndian.Uint64(basicData[utils.BasicDataNonceOffset:])
	acc.Balance = new(uint256.Int).SetBytes(basicData[utils.BasicDataBalanceOffset : utils.BasicDataBalanceOffset+16])
	acc.CodeHash = values[utils.CodeHashLeafKey]

	// TODO account.Root is leave as empty. How should we handle the legacy account?
	return acc, nil
}

// UpdateAccount implements state.Trie, writing the provided account into the tree.
// If the tree is corrupted, an error will be returned.
func (s *verkleState) UpdateAccount(addr common.Address, acc *types.StateAccount, codeLen int) error {
	// TODO: reference source of this logic;

	var (
		err       error
		basicData [32]byte
		values    = make([][]byte, verkle.NodeWidth)
		stem      = s.cache.GetStem(addr[:])
	)

	// Code size is encoded in BasicData as a 3-byte big-endian integer. Spare bytes are present
	// before the code size to support bigger integers in the future. PutUint32(...) requires
	// 4 bytes, so we need to shift the offset 1 byte to the left.
	binary.BigEndian.PutUint32(basicData[utils.BasicDataCodeSizeOffset-1:], uint32(codeLen))
	binary.BigEndian.PutUint64(basicData[utils.BasicDataNonceOffset:], acc.Nonce)
	if acc.Balance.ByteLen() > 16 {
		panic("balance too large")
	}
	acc.Balance.WriteToSlice(basicData[utils.BasicDataBalanceOffset : utils.BasicDataBalanceOffset+16])
	values[utils.BasicDataLeafKey] = basicData[:]
	values[utils.CodeHashLeafKey] = acc.CodeHash[:]

	switch root := s.root.(type) {
	case *verkle.InternalNode:
		err = root.InsertValuesAtStem(stem, values, s.readNode)
	default:
		return errInvalidRootType
	}
	if err != nil {
		return fmt.Errorf("UpdateAccount (%x) error: %v", addr, err)
	}

	return nil
}
