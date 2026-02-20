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
	"github.com/0xsoniclabs/carmen/go/database/vt/geth2/utils"
	"github.com/0xsoniclabs/carmen/go/state"
	geth_common "github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-verkle"
	"github.com/holiman/uint256"
)

type verkleState struct {
	root   verkle.VerkleNode
	store  Store
	source NodeSource

	// pointCache for computing paths in the verkle trie.
	pointCache *utils.PointCache

	// Resolving codes is relatively expensive, so we keep a cache of codes in memory.
	codeCache map[common.Address][]byte
}

func NewState(params state.Parameters) (state.State, error) {
	return newState(params)
}

func newState(
	params state.Parameters,
) (_ *verkleState, err error) {
	dataDir := params.Directory

	// Determine the mode, live or archive.
	var liveOnly bool
	switch params.Archive {
	case state.NoArchive:
		liveOnly = true
	case state.LevelDbArchive, state.S5Archive:
		liveOnly = false
	default:
		return nil, fmt.Errorf("unsupported archive mode: %v", params.Archive)
	}

	// Open the node store (e.g., LevelDB) in the required mode.
	var store Store
	if liveOnly {
		store, err = newLevelDbLiveStore(filepath.Join(dataDir, "live"))
	} else {
		store, err = newLevelDbArchiveStore(filepath.Join(dataDir, "live"))
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
		root:       root,
		source:     source,
		pointCache: utils.NewPointCache(4096),
		codeCache:  make(map[common.Address][]byte),
	}, nil
}

// --- State interface implementation ---

func (s *verkleState) Exists(address common.Address) (bool, error) {
	account, err := s.getAccountData(address)
	if err != nil {
		return false, err
	}

	return account.Nonce != 0 || account.Balance.Uint64() != 0, nil
}

func (s *verkleState) GetBalance(address common.Address) (amount.Amount, error) {
	account, err := s.getAccountData(address)
	if err != nil {
		return amount.Amount{}, err
	}

	return amount.NewFromUint256(&account.Balance), nil
}

func (s *verkleState) GetNonce(address common.Address) (common.Nonce, error) {
	account, err := s.getAccountData(address)
	if err != nil {
		return common.Nonce{}, err
	}

	return common.ToNonce(account.Nonce), nil
}

func (s *verkleState) GetStorage(address common.Address, key common.Key) (common.Value, error) {
	value, err := s.getStorage(address, key[:])
	if err != nil {
		return common.Value{}, err
	}

	var commonValue common.Value
	copy(commonValue[32-len(value):], value)
	return commonValue, nil
}

func (s *verkleState) GetCode(address common.Address) ([]byte, error) {
	if code, ok := s.codeCache[address]; ok {
		return code, nil
	}
	size, err := s.GetCodeSize(address)
	if err != nil {
		return nil, err
	}
	if size == 0 {
		return nil, nil
	}
	code := make([]byte, size)

	// Fetch code chunks from trie and stitch them together.
	out := code
	addressPoint := s.pointCache.Get(address[:])
	for i := 0; len(out) > 0; i++ {
		key := utils.CodeChunkKeyWithEvaluatedAddress(addressPoint, uint256.NewInt(uint64(i)))
		value, err := s.root.Get(key, s.readNode)
		if err != nil {
			return nil, fmt.Errorf("failed to read code chunk %d for address %x: %w", i, address, err)
		}
		out = out[copy(out, value[1:]):]
	}

	s.codeCache[address] = code
	return code, nil
}

func (s *verkleState) GetCodeSize(address common.Address) (int, error) {
	account, err := s.getAccountData(address)
	if err != nil {
		return 0, err
	}
	return account.CodeLength, nil
}

func (s *verkleState) GetCodeHash(address common.Address) (common.Hash, error) {
	account, err := s.getAccountData(address)
	if err != nil {
		return common.Hash{}, err
	}

	return common.Hash(account.CodeHash), nil
}

func (s *verkleState) HasEmptyStorage(addr common.Address) (bool, error) {
	return true, nil
}

func (s *verkleState) Apply(block uint64, update common.Update) (<-chan error, error) {

	// Aggregate changes to the account data.
	modifiedAccounts := map[common.Address]*accountData{}
	getAccountData := func(addr common.Address) (*accountData, error) {
		if data, ok := modifiedAccounts[addr]; ok {
			return data, nil
		}
		res, err := s.getAccountData(addr)
		if err != nil {
			return nil, err
		}
		modifiedAccounts[addr] = &res
		return &res, nil
	}

	// Process deleted accounts.
	if len(update.DeletedAccounts) > 0 {
		return nil, fmt.Errorf("not supported: verkle trie does not support deleting accounts")
	}

	// Process created accounts.
	for _, newAccount := range update.CreatedAccounts {
		data, err := getAccountData(newAccount)
		if err != nil {
			return nil, err
		}
		data.Balance = *uint256.NewInt(0)
		data.Nonce = 0
		data.CodeHash = common.Hash(types.EmptyCodeHash[:])
		data.CodeLength = 0
	}

	// update balances
	for _, update := range update.Balances {
		account, err := getAccountData(update.Account)
		if err != nil {
			return nil, err
		}
		account.Balance = update.Balance.Uint256()
	}

	// update nonces
	for _, update := range update.Nonces {
		account, err := getAccountData(update.Account)
		if err != nil {
			return nil, err
		}
		account.Nonce = update.Nonce.ToUint64()
	}

	// Update codes.
	for _, update := range update.Codes {
		account, err := getAccountData(update.Account)
		if err != nil {
			return nil, err
		}

		// update code len and code hash
		codeHash := common.Keccak256(update.Code)
		account.CodeHash = codeHash
		account.CodeLength = len(update.Code)

		// insert code into the trie
		if err := s.updateContractCode(update.Account, codeHash, update.Code); err != nil {
			return nil, err
		}

		s.codeCache[update.Account] = update.Code
	}

	// Update storage slots.
	for _, update := range update.Slots {
		if err := s.updateStorage(update.Account, update.Key[:], update.Value[:]); err != nil {
			return nil, err
		}
	}

	// Write back all modified accounts.
	for addr, data := range modifiedAccounts {
		if err := s.setAccountData(addr, *data); err != nil {
			return nil, err
		}
	}

	// ------------------------------------------------------------------

	// Compute and commit all changes to the store.
	nodes, err := s.root.(*verkle.InternalNode).BatchSerialize()
	if err != nil {
		return nil, fmt.Errorf("failed to serialize nodes: %w", err)
	}
	changes := make([]Entry, 0, len(nodes))
	for _, node := range nodes {
		changes = append(changes, Entry{
			Path: node.Path,
			Blob: node.SerializedBytes,
		})
	}

	if err := s.store.AddBlock(block, changes); err != nil {
		return nil, fmt.Errorf("failed to add block changes to store: %w", err)
	}

	// Reload root node, pruning the in-memory tree to the root.
	// Note: flush is crashing, so we re-load the root only, discard the rest.
	rootData, err := s.source.GetNode(nil)
	if err != nil {
		return nil, err
	}
	rootNode, err := verkle.ParseNode(rootData, 0)
	if err != nil {
		return nil, err
	}
	s.root = rootNode

	/*
		// Limit memory usage by pruning in-memory tree structure at a certain depth.
		// The nodes will be reloaded from the store when needed.
		s.root.(*verkle.InternalNode).Flush(func([]byte, verkle.VerkleNode) {
			// no-op callback, since all nodes are stored as deltas.
		})
	*/
	/*
		s.root.(*verkle.InternalNode).FlushAtDepth(4, func(path []byte, node verkle.VerkleNode) {
			// no-op callback, since all nodes are stored as deltas.
		})
	*/

	return nil, nil
	//return nil, s.store.AddBlock(block, changes)
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
	if err := s.Flush(); err != nil {
		return err
	}
	if s.store == nil {
		return nil
	}
	err := s.store.Close()
	s.store = nil
	return err
}

func (s *verkleState) GetMemoryFootprint() *common.MemoryFootprint {
	return common.NewMemoryFootprint(uintptr(1))
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
	return nil
}

func (s *verkleState) CreateWitnessProof(address common.Address, keys ...common.Key) (witness.Proof, error) {
	panic("not implemented")
}

func (s *verkleState) Export(ctx context.Context, out io.Writer) (common.Hash, error) {
	panic("not implemented")
}

// --- utility functions ---

func (s *verkleState) readNode(path []byte) ([]byte, error) {
	return s.source.GetNode(path)
}

// accountData summarizes the metadata stored per account in the verkle trie.
type accountData struct {
	Nonce      uint64
	Balance    uint256.Int
	CodeHash   common.Hash
	CodeLength int
}

func (s *verkleState) getAccountData(address common.Address) (accountData, error) {
	account, codeLength, err := s.getAccount(address)
	if err != nil {
		return accountData{}, err
	}
	if account == nil {
		return accountData{
			CodeHash: common.Hash(types.EmptyCodeHash),
		}, nil
	}
	return accountData{
		Nonce:      account.Nonce,
		Balance:    *account.Balance,
		CodeHash:   common.Hash(account.CodeHash),
		CodeLength: codeLength,
	}, nil
}

func (s *verkleState) setAccountData(address common.Address, data accountData) error {
	account := &types.StateAccount{
		Nonce:    data.Nonce,
		Balance:  &data.Balance,
		CodeHash: data.CodeHash[:],
	}
	return s.updateAccount(address, account, data.CodeLength)
}

// --- Ported from go-Ethereum, as this code got removed from the main branch --

// The code below has been ported from the original Geth implementation of the
// Verkle trie, as it got discontinued and removed from go-ethereum.
// The code was taken from trie/verkle.go of the v1.16.8 version of go-ethereum
// and only slightly adapted to fit into the local verkleState implementation.
// The original code can be found here:
// https://github.com/ethereum/go-ethereum/blob/v1.16.9/trie/verkle.go

var (
	errInvalidRootType = errors.New("invalid node type for root")
)

// getAccount implements state.Trie, retrieving the account with the specified
// account address. If the specified account is not in the verkle tree, nil will
// be returned. If the tree is corrupted, an error will be returned.
func (s *verkleState) getAccount(addr common.Address) (*types.StateAccount, int, error) {
	var (
		acc    = &types.StateAccount{}
		values [][]byte
		err    error
	)
	switch n := s.root.(type) {
	case *verkle.InternalNode:
		values, err = n.GetValuesAtStem(s.pointCache.GetStem(addr[:]), s.readNode)
		if err != nil {
			return nil, 0, fmt.Errorf("GetAccount (%x) error: %v", addr, err)
		}
	default:
		return nil, 0, errInvalidRootType
	}
	if values == nil {
		return nil, 0, nil
	}
	basicData := values[utils.BasicDataLeafKey]
	acc.Nonce = binary.BigEndian.Uint64(basicData[utils.BasicDataNonceOffset:])
	acc.Balance = new(uint256.Int).SetBytes(basicData[utils.BasicDataBalanceOffset : utils.BasicDataBalanceOffset+16])
	acc.CodeHash = values[utils.CodeHashLeafKey]

	codeLength := binary.BigEndian.Uint32(basicData[utils.BasicDataCodeSizeOffset-1:])

	// TODO account.Root is leave as empty. How should we handle the legacy account?
	return acc, int(codeLength), nil
}

// updateAccount implements state.Trie, writing the provided account into the tree.
// If the tree is corrupted, an error will be returned.
func (s *verkleState) updateAccount(addr common.Address, acc *types.StateAccount, codeLen int) error {
	var (
		err       error
		basicData [32]byte
		values    = make([][]byte, verkle.NodeWidth)
		stem      = s.pointCache.GetStem(addr[:])
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

// updateStorage implements state.Trie, writing the provided storage slot into
// the tree. If the tree is corrupted, an error will be returned.
func (s *verkleState) updateStorage(address common.Address, key, value []byte) error {
	// Left padding the slot value to 32 bytes.
	var v [32]byte
	if len(value) >= 32 {
		copy(v[:], value[:32])
	} else {
		copy(v[32-len(value):], value[:])
	}
	k := utils.StorageSlotKeyWithEvaluatedAddress(s.pointCache.Get(address[:]), key)
	return s.root.Insert(k, v[:], s.readNode)
}

// getStorage implements state.Trie, retrieving the storage slot with the specified
// account address and storage key. If the specified slot is not in the verkle tree,
// nil will be returned. If the tree is corrupted, an error will be returned.
func (s *verkleState) getStorage(addr common.Address, key []byte) ([]byte, error) {
	k := utils.StorageSlotKeyWithEvaluatedAddress(s.pointCache.Get(addr[:]), key)
	val, err := s.root.Get(k, s.readNode)
	if err != nil {
		return nil, err
	}
	return geth_common.TrimLeftZeroes(val), nil
}

// ChunkedCode represents a sequence of 32-bytes chunks of code (31 bytes of which
// are actual code, and 1 byte is the pushdata offset).
type ChunkedCode []byte

// Copy the values here so as to avoid an import cycle
const (
	PUSH1  = byte(0x60)
	PUSH32 = byte(0x7f)
)

// ChunkifyCode generates the chunked version of an array representing EVM bytecode
func ChunkifyCode(code []byte) ChunkedCode {
	var (
		chunkOffset = 0 // offset in the chunk
		chunkCount  = len(code) / 31
		codeOffset  = 0 // offset in the code
	)
	if len(code)%31 != 0 {
		chunkCount++
	}
	chunks := make([]byte, chunkCount*32)
	for i := 0; i < chunkCount; i++ {
		// number of bytes to copy, 31 unless the end of the code has been reached.
		end := 31 * (i + 1)
		if len(code) < end {
			end = len(code)
		}
		copy(chunks[i*32+1:], code[31*i:end]) // copy the code itself

		// chunk offset = taken from the last chunk.
		if chunkOffset > 31 {
			// skip offset calculation if push data covers the whole chunk
			chunks[i*32] = 31
			chunkOffset = 1
			continue
		}
		chunks[32*i] = byte(chunkOffset)
		chunkOffset = 0

		// Check each instruction and update the offset it should be 0 unless
		// a PUSH-N overflows.
		for ; codeOffset < end; codeOffset++ {
			if code[codeOffset] >= PUSH1 && code[codeOffset] <= PUSH32 {
				codeOffset += int(code[codeOffset] - PUSH1 + 1)
				if codeOffset+1 >= 31*(i+1) {
					codeOffset++
					chunkOffset = codeOffset - 31*(i+1)
					break
				}
			}
		}
	}
	return chunks
}

// updateContractCode implements state.Trie, writing the provided contract code
// into the trie.
// Note that the code-size *must* be already saved by a previous UpdateAccount call.
func (s *verkleState) updateContractCode(addr common.Address, codeHash common.Hash, code []byte) error {
	var (
		chunks = ChunkifyCode(code)
		values [][]byte
		key    []byte
		err    error
	)
	for i, chunknr := 0, uint64(0); i < len(chunks); i, chunknr = i+32, chunknr+1 {
		groupOffset := (chunknr + 128) % 256
		if groupOffset == 0 /* start of new group */ || chunknr == 0 /* first chunk in header group */ {
			values = make([][]byte, verkle.NodeWidth)
			key = utils.CodeChunkKeyWithEvaluatedAddress(s.pointCache.Get(addr[:]), uint256.NewInt(chunknr))
		}
		values[groupOffset] = chunks[i : i+32]

		if groupOffset == 255 || len(chunks)-i <= 32 {
			switch root := s.root.(type) {
			case *verkle.InternalNode:
				err = root.InsertValuesAtStem(key[:31], values, s.readNode)
				if err != nil {
					return fmt.Errorf("UpdateContractCode (addr=%x) error: %w", addr[:], err)
				}
			default:
				return errInvalidRootType
			}
		}
	}
	return nil
}
