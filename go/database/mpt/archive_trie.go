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
	"errors"
	"fmt"
	"io"
	"iter"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"time"
	"unsafe"

	"github.com/0xsoniclabs/carmen/go/backend/archive"
	"github.com/0xsoniclabs/carmen/go/backend/kv_file"
	"github.com/0xsoniclabs/carmen/go/backend/stock/file"
	"github.com/0xsoniclabs/carmen/go/backend/utils"
	"github.com/0xsoniclabs/carmen/go/backend/utils/checkpoint"
	"github.com/0xsoniclabs/carmen/go/common/iter_utils"
	"github.com/0xsoniclabs/carmen/go/common/witness"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
)

// ArchiveTrie retains a per-block history of the state trie. Each state is
// a trie in a Forest of which the root node is retained. Updates can only
// be applied through the `Add` method, according to the `archive.Archive“
// interface, which this type is implementing.
//
// Its main task is to keep track of state roots and to freeze the head
// state after each block.
type ArchiveTrie struct {
	directory    string
	head         LiveState  // the current head-state
	forest       Database   // global forest with all versions of LiveState
	roots        *rootList  // the roots of individual blocks indexed by block height
	rootsMutex   sync.Mutex // protecting access to the roots list
	addMutex     sync.Mutex // a mutex to make sure that at any time only one thread is adding new blocks
	errorMutex   sync.RWMutex
	archiveError error // a non-nil error will be stored here should it occur during any archive operation

	// Check-point support for DB healing.
	checkpointCoordinator checkpoint.Coordinator
	checkpointInterval    int
	checkpointPeriod      time.Duration
	lastCheckpointTime    time.Time
}

// ArchiveConfig is the configuration for the archive trie.
type ArchiveConfig struct {
	// The number of blocks after which the latest a checkpoint is created.
	CheckpointInterval int
	// The system-time period after which the latest a checkpoint is created.
	CheckpointPeriod time.Duration
}

const (
	fileNameArchiveCheckpointDirectory      = "checkpoint"
	fileNameArchiveRoots                    = "roots.dat"
	fileNameArchiveRootsCheckpointDirectory = "roots"
	fileNameArchiveRootsCommittedCheckpoint = "committed.json"
	fileNameArchiveRootsPreparedCheckpoint  = "prepare.json"
)

func OpenArchiveTrie(
	directory string,
	config MptConfig,
	cacheConfig NodeCacheConfig,
	archiveConfig ArchiveConfig,
) (*ArchiveTrie, error) {
	lock, err := openStateDirectory(directory)
	if err != nil {
		return nil, err
	}
	roots, err := loadRoots(directory)
	if err != nil {
		return nil, err
	}

	forestConfig := ForestConfig{Mode: Immutable, NodeCacheConfig: cacheConfig}
	forest, err := OpenFileForest(directory, config, forestConfig)
	if err != nil {
		return nil, errors.Join(err, roots.Close())
	}

	head, err := makeTrie(directory, forest)
	if err != nil {
		return nil, errors.Join(err, roots.Close(), forest.Close())
	}

	root := NewNodeReference(EmptyId())
	if roots.length() > 0 {
		rootValue, err := roots.get(uint64(roots.length() - 1))
		if err != nil {
			return nil, errors.Join(err, roots.Close(), head.Close())
		}
		root = rootValue.NodeRef
	}
	head.root = root

	state, err := newMptState(directory, lock, head)
	if err != nil {
		return nil, errors.Join(err, roots.Close(), head.Close())
	}

	checkpointDir := filepath.Join(directory, fileNameArchiveCheckpointDirectory)
	coordinator, err := checkpoint.NewCoordinator(
		checkpointDir,
		forest.accounts,
		forest.branches,
		forest.extensions,
		forest.values,
		state.codes,
		roots,
	)
	if err != nil {
		return nil, errors.Join(err, roots.Close(), head.Close())
	}

	// Load the checkpointing configuration and set
	// default values.
	checkpointInterval := archiveConfig.CheckpointInterval
	if checkpointInterval <= 0 {
		checkpointInterval = 1_000_000
	}
	checkpointPeriod := archiveConfig.CheckpointPeriod
	if checkpointPeriod <= 0 {
		checkpointPeriod = 10 * time.Minute
	}

	// Pick a random time in the past to introduce an offset
	// between archive instances started at roughly the same time.
	lastCheckpointTime := time.Now()
	lastCheckpointTime = lastCheckpointTime.Add(time.Duration(-1 * float64(checkpointPeriod) * rand.Float64()))

	return &ArchiveTrie{
		directory:             directory,
		head:                  state,
		forest:                forest,
		roots:                 roots,
		checkpointCoordinator: coordinator,
		checkpointInterval:    checkpointInterval,
		checkpointPeriod:      checkpointPeriod,
		lastCheckpointTime:    lastCheckpointTime,
	}, nil
}

// VerifyArchiveTrie validates file-based archive stored in the given directory.
// If the test passes, the data stored in the respective directory
// can be considered a valid archive database of the given configuration.
func VerifyArchiveTrie(ctx context.Context, directory string, config MptConfig, observer VerificationObserver) (res error) {
	roots, err := loadRoots(directory)
	if err != nil {
		return err
	}
	defer func() { res = errors.Join(res, roots.Close()) }()
	if roots.length() == 0 {
		return nil
	}
	rootsSeq, err := roots.Iterate()
	if err != nil {
		return err
	}
	return VerifyMptState(ctx, directory, config, rootsSeq, observer)
}

func (a *ArchiveTrie) Add(block uint64, update common.Update, hint any) error {
	if err := a.CheckErrors(); err != nil {
		return err
	}

	precomputedHashes, _ := hint.(*NodeHashes)

	a.addMutex.Lock()
	defer a.addMutex.Unlock()

	a.rootsMutex.Lock()
	previousRootsLength := a.roots.length()
	if uint64(previousRootsLength) > block {
		a.rootsMutex.Unlock()
		return fmt.Errorf("block %d already present", block)
	}

	// Mark skipped blocks as having no changes.
	if uint64(a.roots.length()) < block {
		lastHash, err := a.head.GetHash()
		if err != nil {
			a.rootsMutex.Unlock()
			return a.addError(err)
		}
		for uint64(a.roots.length()) < block {
			if err := a.roots.append(Root{a.head.Root(), lastHash}); err != nil {
				a.rootsMutex.Unlock()
				return a.addError(err)
			}
		}
	}
	a.rootsMutex.Unlock()

	// Apply all the changes of the update.
	if err := update.ApplyTo(a.head); err != nil {
		return a.addError(err)
	}

	// Freeze new state.
	root := a.head.Root()
	if err := a.forest.Freeze(&root); err != nil {
		return a.addError(err)
	}

	// Refresh hashes.
	var err error
	var hash common.Hash
	if precomputedHashes == nil {
		var hashes *NodeHashes
		hash, hashes, err = a.head.UpdateHashes()
		if hashes != nil {
			hashes.Release()
		}
	} else {
		err = a.head.setHashes(precomputedHashes)
		if err == nil {
			hash, err = a.head.GetHash()
		}
	}
	if err != nil {
		return a.addError(err)
	}

	// Save new root node.
	a.rootsMutex.Lock()
	if err := a.roots.append(Root{a.head.Root(), hash}); err != nil {
		a.rootsMutex.Unlock()
		return a.addError(err)
	}
	a.rootsMutex.Unlock()

	// Create a new checkpoint if we crossed an interval boundary.
	shouldCheckpoint := false
	if previousRootsLength == 0 {
		shouldCheckpoint = block >= uint64(a.checkpointInterval)
	} else {
		oldCheckpointInterval := (previousRootsLength - 1) / a.checkpointInterval
		newCheckpointInterval := int(block) / a.checkpointInterval
		shouldCheckpoint = oldCheckpointInterval != newCheckpointInterval
	}
	shouldCheckpoint = shouldCheckpoint || time.Since(a.lastCheckpointTime) > a.checkpointPeriod
	if shouldCheckpoint {
		if err := a.createCheckpoint(); err != nil {
			return err
		}
	}

	return nil
}

func (a *ArchiveTrie) GetBlockRoot(block uint64) (NodeId, error) {
	if block >= uint64(a.roots.length()) {
		return EmptyId(), fmt.Errorf("block %d not present in archive", block)
	}
	root, err := a.roots.get(block)
	if err != nil {
		return EmptyId(), err
	}
	return root.NodeRef.id, nil
}

func (a *ArchiveTrie) GetBlockHeight() (block uint64, empty bool, err error) {
	a.rootsMutex.Lock()
	length := uint64(a.roots.length())
	a.rootsMutex.Unlock()
	if length == 0 {
		return 0, true, nil
	}
	return length - 1, false, nil
}

func (a *ArchiveTrie) Exists(block uint64, account common.Address) (exists bool, err error) {
	view, err := a.getView(block)
	if err != nil {
		return false, err
	}
	_, exists, err = view.GetAccountInfo(account)
	if err != nil {
		return false, a.addError(err)
	}
	return exists, err
}

func (a *ArchiveTrie) GetBalance(block uint64, account common.Address) (balance amount.Amount, err error) {
	view, err := a.getView(block)
	if err != nil {
		return amount.New(), err
	}
	info, _, err := view.GetAccountInfo(account)
	if err != nil {
		return amount.New(), a.addError(err)
	}
	return info.Balance, nil
}

func (a *ArchiveTrie) GetCode(block uint64, account common.Address) (code []byte, err error) {
	view, err := a.getView(block)
	if err != nil {
		return nil, err
	}
	info, _, err := view.GetAccountInfo(account)
	if err != nil {
		return nil, a.addError(err)
	}
	return a.GetCodeForHash(info.CodeHash)
}

func (a *ArchiveTrie) GetCodeForHash(hash common.Hash) ([]byte, error) {
	return a.head.GetCodeForHash(hash)
}

func (a *ArchiveTrie) GetCodes() (iter_utils.ResultSeq2[common.Hash, []byte], error) {
	return a.head.GetCodes()
}

func (a *ArchiveTrie) GetAccountInfo(block uint64, account common.Address) (info AccountInfo, exists bool, err error) {
	view, err := a.getView(block)
	if err != nil {
		return AccountInfo{}, false, err
	}
	info, exists, err = view.GetAccountInfo(account)
	return info, exists, a.addError(err)
}

func (a *ArchiveTrie) GetNonce(block uint64, account common.Address) (nonce common.Nonce, err error) {
	view, err := a.getView(block)
	if err != nil {
		return common.Nonce{}, err
	}
	info, _, err := view.GetAccountInfo(account)
	if err != nil {
		return common.Nonce{}, a.addError(err)
	}
	return info.Nonce, nil
}

func (a *ArchiveTrie) GetStorage(block uint64, account common.Address, slot common.Key) (value common.Value, err error) {
	view, err := a.getView(block)
	if err != nil {
		return common.Value{}, err
	}
	value, err = view.GetValue(account, slot)
	return value, a.addError(err)
}

func (a *ArchiveTrie) GetAccountHash(block uint64, account common.Address) (common.Hash, error) {
	return common.Hash{}, fmt.Errorf("not implemented")
}

func (a *ArchiveTrie) GetHash(block uint64) (hash common.Hash, err error) {
	a.rootsMutex.Lock()
	length := uint64(a.roots.length())
	if block >= length {
		a.rootsMutex.Unlock()
		return common.Hash{}, fmt.Errorf("invalid block: %d >= %d", block, length)
	}
	res, err := a.roots.get(block)
	a.rootsMutex.Unlock()
	if err != nil {
		return common.Hash{}, err
	}
	return res.Hash, nil
}

func (a *ArchiveTrie) CreateWitnessProof(block uint64, address common.Address, keys ...common.Key) (witness.Proof, error) {
	if !a.forest.getConfig().UseHashedPaths {
		return nil, archive.ErrWitnessProofNotSupported
	}
	view, err := a.getView(block)
	if err != nil {
		return nil, err
	}
	proof, err := view.CreateWitnessProof(address, keys...)
	return proof, a.addError(err)
}

// HasEmptyStorage returns true if account has empty storage in a certain block.
func (a *ArchiveTrie) HasEmptyStorage(block uint64, addr common.Address) (bool, error) {
	view, err := a.getView(block)
	if err != nil {
		return false, err
	}
	empty, err := view.HasEmptyStorage(addr)
	return empty, a.addError(err)
}

// GetDiff computes the difference between the given source and target blocks.
func (a *ArchiveTrie) GetDiff(srcBlock, trgBlock uint64) (Diff, error) {
	a.rootsMutex.Lock()
	if srcBlock >= uint64(a.roots.length()) {
		a.rootsMutex.Unlock()
		return Diff{}, fmt.Errorf("source block %d not present in archive, highest block is %d", srcBlock, a.roots.length()-1)
	}
	if trgBlock >= uint64(a.roots.length()) {
		a.rootsMutex.Unlock()
		return Diff{}, fmt.Errorf("target block %d not present in archive, highest block is %d", trgBlock, a.roots.length()-1)
	}
	before, err := a.roots.get(srcBlock)
	if err != nil {
		a.rootsMutex.Unlock()
		return Diff{}, err
	}
	after, err := a.roots.get(trgBlock)
	if err != nil {
		a.rootsMutex.Unlock()
		return Diff{}, err
	}
	a.rootsMutex.Unlock()
	return GetDiff(a.forest, &before.NodeRef, &after.NodeRef)
}

// GetDiffForBlock computes the diff introduced by the given block compared to its
// predecessor. Note that this enables access to the changes introduced by block 0.
func (a *ArchiveTrie) GetDiffForBlock(block uint64) (Diff, error) {
	if block == 0 {
		a.rootsMutex.Lock()
		if a.roots.length() == 0 {
			a.rootsMutex.Unlock()
			return Diff{}, fmt.Errorf("archive is empty, no diff present for block 0")
		}
		after, err := a.roots.get(0)
		if err != nil {
			a.rootsMutex.Unlock()
			return Diff{}, err
		}
		a.rootsMutex.Unlock()
		return GetDiff(a.forest, &emptyNodeReference, &after.NodeRef)
	}
	return a.GetDiff(block-1, block)
}

func (a *ArchiveTrie) GetMemoryFootprint() *common.MemoryFootprint {
	mf := common.NewMemoryFootprint(unsafe.Sizeof(*a))
	mf.AddChild("head", a.head.GetMemoryFootprint())
	a.rootsMutex.Lock()
	mf.AddChild("roots", a.roots.GetMemoryFootprint())
	a.rootsMutex.Unlock()
	return mf
}

func (a *ArchiveTrie) Check() error {
	a.rootsMutex.Lock()
	defer a.rootsMutex.Unlock()

	it, err := a.roots.Iterate()
	if err != nil {
		return err
	}
	return errors.Join(
		a.CheckErrors(),
		a.forest.CheckAll(iter_utils.MapOk(iter_utils.DropKeysOk2(it), func(v Root) *NodeReference {
			return &v.NodeRef
		}),
		),
	)
}

func (a *ArchiveTrie) Flush() error {
	a.rootsMutex.Lock()
	defer a.rootsMutex.Unlock()
	err := errors.Join(
		a.CheckErrors(),
		a.head.Flush(),
		a.roots.storeRoots(),
	)
	return a.addError(err)
}

func (a *ArchiveTrie) VisitTrie(block uint64, mode AccessMode, visitor NodeVisitor) error {
	view, err := a.getView(block)
	if err != nil {
		return err
	}
	return a.addError(view.VisitTrie(mode, visitor))
}

func (a *ArchiveTrie) VisitAccountStorage(
	block uint64,
	address common.Address,
	mode AccessMode,
	visitor NodeVisitor,
) error {
	view, err := a.getView(block)
	if err != nil {
		return err
	}

	return a.addError(view.VisitAccountStorage(address, mode, visitor))
}

func (a *ArchiveTrie) Close() error {
	return errors.Join(
		a.CheckErrors(),
		a.head.closeWithError(a.Flush()),
		a.roots.Close(),
	)
}

func (a *ArchiveTrie) createCheckpoint() error {
	// Before the checkpoint can be created, all data needs
	// to be flushed to the underlying storage.
	if err := a.Flush(); err != nil {
		return err
	}
	// The creation of the checkpoint makes the current
	// state recoverable in case of a crash.
	_, err := a.checkpointCoordinator.CreateCheckpoint()
	if err == nil {
		a.lastCheckpointTime = time.Now()
	}
	return a.addError(err)
}

func GetCheckpointBlock(dir string) (uint64, error) {
	checkpointDir := filepath.Join(dir, fileNameArchiveCheckpointDirectory)
	coordinator, err := checkpoint.NewCoordinator(checkpointDir)
	if err != nil {
		return 0, err
	}
	cp := coordinator.GetCurrentCheckpoint()
	restorer := getRootListRestorer(dir)
	numRoots, err := restorer.getNumRootsInCheckpoint(cp)
	return uint64(numRoots - 1), err
}

func RestoreBlockHeight(directory string, config MptConfig, block uint64) (retErr error) {

	// Make sure access to the directory is exclusive.
	lock, err := LockDirectory(directory)
	if err != nil {
		return fmt.Errorf("failed to get exclusive access to directory: %v", err)
	}
	defer func() { retErr = errors.Join(retErr, lock.Release()) }()

	// Check available block height -- stop recovery if there are not enough blocks.
	checkpointHeight, err := GetCheckpointBlock(directory)
	if err != nil {
		return fmt.Errorf("failed to get checkpoint height: %v", err)
	}
	if block > uint64(checkpointHeight) {
		return fmt.Errorf("block %d is beyond the last checkpoint height of %d", block, checkpointHeight)
	}

	// Mark this directory as dirty at least for the duration of the recovery.
	if err := markDirty(directory); err != nil {
		return fmt.Errorf("failed to mark directory %s as dirty: %w", directory, err)
	}
	defer func() {
		// Only remove dirty flag is the recovery was successful.
		if retErr == nil {
			retErr = markClean(directory)
		}
	}()

	// Restore the last checkpoint created by the archive.
	rootRestorer := getRootListRestorer(directory)
	accountsDir, branchesDir, extensionsDir, valuesDir := getForestDirectories(directory)
	restorers := []checkpoint.Restorer{
		file.GetRestorer(accountsDir),
		file.GetRestorer(branchesDir),
		file.GetRestorer(extensionsDir),
		file.GetRestorer(valuesDir),
		getCodeRestorer(directory),
		rootRestorer,
	}

	checkpointDir := filepath.Join(directory, "checkpoint")
	if err := checkpoint.Restore(checkpointDir, restorers...); err != nil {
		return fmt.Errorf("failed to restore checkpoint: %w", err)
	}

	// After the checkpoint, restore the block height and make sure the meta-data file
	// is a correct JSON file. Although the meta data is not used by the archive, its
	// absence or corruption would prevent the archive from being opened. The content
	// is irrelevant, since after loading it is replaced by the latest root.
	metaDataFile := getLiveTrieMetadataPath(directory)
	return errors.Join(
		rootRestorer.truncate(int(block+1)),
		utils.WriteJsonFile(metaDataFile, metadata{}),
	)
}

func (a *ArchiveTrie) getView(block uint64) (*LiveTrie, error) {
	if err := a.CheckErrors(); err != nil {
		return nil, err
	}

	a.rootsMutex.Lock()
	length := uint64(a.roots.length())
	if block >= length {
		a.rootsMutex.Unlock()
		return nil, fmt.Errorf("invalid block: %d >= %d", block, length)
	}
	root, err := a.roots.get(block)
	if err != nil {
		a.rootsMutex.Unlock()
		return nil, err
	}
	a.rootsMutex.Unlock()
	return getTrieView(root.NodeRef, a.forest), nil
}

// CheckErrors returns a non-nil error should any error
// happen during any operation in this archive.
// In particular, updating this archive or getting
// values out of it may fail, and in this case,
// the error is stored and returned in this method.
// Further calls to this archive produce the same
// error as this method returns.
func (a *ArchiveTrie) CheckErrors() error {
	a.errorMutex.RLock()
	defer a.errorMutex.RUnlock()
	return a.archiveError
}

func (a *ArchiveTrie) addError(err error) error {
	a.errorMutex.Lock()
	defer a.errorMutex.Unlock()
	a.archiveError = errors.Join(a.archiveError, err)
	return a.archiveError
}

// Directory returns the directory where the archive is stored on disk.
func (a *ArchiveTrie) Directory() string {
	return a.directory
}

// GetConfig returns the configuration of the archive.
func (a *ArchiveTrie) GetConfig() MptConfig {
	return a.forest.getConfig()
}

// ---- Reading and Writing Root Node ID Lists ----

// rootList is a utility type managing an in-memory copy of the list of roots
// of an archive and its synchronization with an on-disk file copy.
//
// The file stores one fixed-size record [<node-id>, <state-hash>] per block.
// Records are flushed asynchronously and in arbitrary block order, so after a
// crash the region beyond the last checkpoint may be sparse, with unwritten
// records reading as zeros. Records up to the last checkpoint are always
// dense.
type rootList struct {
	roots     kv_file.KVFileWithMemoryFootprint[uint64, Root] // < the in-memory copy of the roots list
	filename  string                                          // < the file storing the list of roots
	directory string                                          // < the directory for checkpoint data

	numRoots int // < total number of roots

	// numRootsInFile is a lower bound on the number of roots persisted in the
	// file: it is only synchronised by storeRoots, while the underlying cache
	// may flush entries to disk on its own at any time. It must therefore only
	// be consumed after a storeRoots call, as done by Prepare.
	numRootsInFile int

	checkpoint checkpoint.Checkpoint
}

func (l *rootList) length() int {
	return l.numRoots
}

func (l *rootList) get(block uint64) (Root, error) {
	root, err := l.roots.Get(block)
	if err != nil {
		return Root{}, err
	}
	if root == nil {
		return Root{}, fmt.Errorf("root for block %d not found", block)
	}
	return *root, nil
}

func (l *rootList) append(r Root) error {
	if err := l.roots.Set(uint64(l.numRoots), r); err != nil {
		return err
	}
	l.numRoots++
	return nil
}

// Close releases the resources backing the root list. After Close, the list
// must not be used any further.
func (l *rootList) Close() error {
	if l.roots == nil {
		return nil
	}
	err := l.roots.Close()
	l.roots = nil
	return err
}

func loadRoots(archiveDirectory string) (_ *rootList, retErr error) {
	filename := filepath.Join(archiveDirectory, fileNameArchiveRoots)
	directory := filepath.Join(archiveDirectory, fileNameArchiveRootsCheckpointDirectory)

	// Create the directory for commit files if it does not exist.
	if err := os.MkdirAll(directory, 0700); err != nil {
		return nil, err
	}

	committedCheckpointFile := filepath.Join(directory, fileNameArchiveRootsCommittedCheckpoint)
	checkpointData, err := readRootListCheckpointData(committedCheckpointFile)
	if err != nil {
		return nil, err
	}

	// OpenOrderedFile creates the underlying file if it does not yet exist,
	// so a fresh archive directory yields an empty root list.
	entrySize := uint64(NodeIdEncoder{}.GetEncodedSize() + 32)
	roots, err := kv_file.OpenOrderedFile(filename, entrySize, readRoot, writeRoot)
	if err != nil {
		return nil, err
	}
	kvFile, err := kv_file.OpenKVCachedFile[uint64, Root](roots, 10000, 1000)
	if err != nil {
		return nil, errors.Join(err, roots.Close())
	}

	size, err := kvFile.FileSize()
	if err != nil {
		return nil, errors.Join(err, kvFile.Close())
	}
	size /= entrySize
	if int(size) < checkpointData.NumRoots {
		return nil, errors.Join(
			fmt.Errorf("root list file is corrupted: expected at least %d roots, but found only %d", checkpointData.NumRoots, size),
			kvFile.Close(),
		)
	}

	return &rootList{
		roots:          kvFile,
		filename:       filename,
		directory:      directory,
		numRoots:       int(size),
		numRootsInFile: int(size),
		checkpoint:     checkpointData.Checkpoint,
	}, nil
}

// readRoot reads a single root from the given reader.
// Key is skipped, as the file it's ordered
func readRoot(reader io.Reader) (uint64, Root, error) {
	encoder := NodeIdEncoder{}
	buffer := make([]byte, encoder.GetEncodedSize())
	var hash common.Hash
	if _, err := io.ReadFull(reader, buffer); err != nil {
		return 0, Root{}, fmt.Errorf("invalid root file format: %v", err)
	}

	if _, err := io.ReadFull(reader, hash[:]); err != nil {
		return 0, Root{}, fmt.Errorf("invalid root file format: %v", err)
	}

	var id NodeId
	encoder.Load(buffer, &id)
	return 0, Root{NewNodeReference(id), hash}, nil
}

// writeRoot writes a single root to the given writer with the format
// [<node-id>, <state-hash>].
func writeRoot(writer io.Writer, pos uint64, root Root) error {
	// Format: [<node-id>, <state-hash>]*
	encoder := NodeIdEncoder{}
	buffer := make([]byte, encoder.GetEncodedSize())
	encoder.Store(buffer, &root.NodeRef.id)
	if _, err := writer.Write(buffer[:]); err != nil {
		return err
	}
	if _, err := writer.Write(root.Hash[:]); err != nil {
		return err
	}
	return nil
}

// Iterate returns a sequence of all (block, root) pairs in the list.
func (l *rootList) Iterate() (iter_utils.ResultSeq2[uint64, Root], error) {
	return l.roots.Iterate()
}

func StoreRoots(filename string, rootsToWrite iter.Seq[Root]) (err error) {
	// loadRoots derives the roots file path from the directory and the
	// canonical filename. Reject inputs that would otherwise cause writes to
	// silently target a different path than the caller requested.
	if base := filepath.Base(filename); base != fileNameArchiveRoots {
		return fmt.Errorf("StoreRoots: unsupported roots filename %q, expected %q", base, fileNameArchiveRoots)
	}
	roots, err := loadRoots(filepath.Dir(filename))
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(err, roots.Close())
	}()
	for root := range rootsToWrite {
		if err = roots.append(root); err != nil {
			return err
		}
	}
	if err = roots.storeRoots(); err != nil {
		return err
	}
	return nil
}

func (l *rootList) storeRoots() error {
	if l.roots == nil {
		return nil
	}
	if err := l.roots.Flush(); err != nil {
		return err
	}
	l.numRootsInFile = l.numRoots
	return nil
}

func (l *rootList) GetMemoryFootprint() *common.MemoryFootprint {
	mf := common.NewMemoryFootprint(unsafe.Sizeof(*l))
	mf.AddChild("roots", l.roots.GetMemoryFootprint())
	return mf
}

func (l *rootList) GuaranteeCheckpoint(checkpoint checkpoint.Checkpoint) error {
	if l.checkpoint == checkpoint {
		return nil
	}
	if l.checkpoint+1 == checkpoint {
		pendingFile := filepath.Join(l.directory, fileNameArchiveRootsPreparedCheckpoint)
		if _, err := os.Stat(pendingFile); err == nil {
			return l.Commit(checkpoint)
		}
	}
	return fmt.Errorf("unable to guarantee checkpoint %v, current checkpoint is %v", checkpoint, l.checkpoint)
}

func (l *rootList) Prepare(checkpoint checkpoint.Checkpoint) error {
	if l.checkpoint+1 != checkpoint {
		return fmt.Errorf("checkpoint mismatch, expected %v, got %v", l.checkpoint+1, checkpoint)
	}
	if err := l.storeRoots(); err != nil {
		return err
	}
	pendingFile := filepath.Join(l.directory, fileNameArchiveRootsPreparedCheckpoint)
	return writeRootListCheckpointData(pendingFile, rootListCheckpointData{
		Checkpoint: checkpoint,
		NumRoots:   l.numRootsInFile,
	})
}

func (l *rootList) Commit(checkpoint checkpoint.Checkpoint) error {
	if l.checkpoint+1 != checkpoint {
		return fmt.Errorf("checkpoint mismatch, expected %v, got %v", l.checkpoint+1, checkpoint)
	}
	committedFile := filepath.Join(l.directory, fileNameArchiveRootsCommittedCheckpoint)
	pendingFile := filepath.Join(l.directory, fileNameArchiveRootsPreparedCheckpoint)
	meta, err := readRootListCheckpointData(pendingFile)
	if err != nil {
		return err
	}
	if meta.Checkpoint != checkpoint {
		return fmt.Errorf("checkpoint mismatch, prepared %v, committed %v", meta.Checkpoint, checkpoint)
	}
	l.checkpoint = checkpoint
	return os.Rename(pendingFile, committedFile)
}

func (l *rootList) Abort(checkpoint checkpoint.Checkpoint) error {
	if l.checkpoint+1 != checkpoint {
		return fmt.Errorf("checkpoint mismatch, expected %v, got %v", l.checkpoint+1, checkpoint)
	}
	pendingFile := filepath.Join(l.directory, fileNameArchiveRootsPreparedCheckpoint)
	return os.Remove(pendingFile)
}

type rootListRestorer struct {
	rootsFile string
	directory string
}

func getRootListRestorer(archiveDir string) rootListRestorer {
	return rootListRestorer{
		rootsFile: filepath.Join(archiveDir, fileNameArchiveRoots),
		directory: filepath.Join(archiveDir, fileNameArchiveRootsCheckpointDirectory),
	}
}

func (r rootListRestorer) Restore(checkpoint checkpoint.Checkpoint) error {
	committedFile := filepath.Join(r.directory, fileNameArchiveRootsCommittedCheckpoint)
	meta, err := readRootListCheckpointData(committedFile)
	if err != nil {
		return err
	}

	// If the given checkpoint is one step in the future, check whether there is a pending checkpoint.
	if meta.Checkpoint+1 == checkpoint {
		pendingFile := filepath.Join(r.directory, fileNameArchiveRootsPreparedCheckpoint)
		pending, err := readRootListCheckpointData(pendingFile)
		if err == nil && pending.Checkpoint == checkpoint {
			meta = pending
			if err := os.Rename(pendingFile, committedFile); err != nil {
				return err
			}
		}
	}

	if meta.Checkpoint != checkpoint {
		return fmt.Errorf("unknown checkpoint, have %v, wanted %v", meta.Checkpoint, checkpoint)
	}

	return truncateRootsFile(r.rootsFile, meta.NumRoots)
}

func (r rootListRestorer) getNumRootsInCheckpoint(checkpoint checkpoint.Checkpoint) (int, error) {
	meta, err := utils.ReadJsonFile[rootListCheckpointData](filepath.Join(r.directory, fileNameArchiveRootsCommittedCheckpoint))
	if err != nil {
		return 0, err
	}
	if meta.Checkpoint == checkpoint {
		return meta.NumRoots, nil
	}
	if meta.Checkpoint+1 == checkpoint {
		pending, err := utils.ReadJsonFile[rootListCheckpointData](filepath.Join(r.directory, fileNameArchiveRootsPreparedCheckpoint))
		if err == nil && pending.Checkpoint == checkpoint {
			return pending.NumRoots, nil
		}
	}
	return 0, fmt.Errorf("checkpoint %v not found", checkpoint)
}

func (r rootListRestorer) truncate(length int) error {
	committed := filepath.Join(r.directory, fileNameArchiveRootsCommittedCheckpoint)
	meta, err := utils.ReadJsonFile[rootListCheckpointData](committed)
	if err != nil {
		return err
	}
	if meta.NumRoots < length {
		return fmt.Errorf("cannot truncate to %d, only %d roots available", length, meta.NumRoots)
	}
	meta.NumRoots = length
	return errors.Join(
		writeRootListCheckpointData(committed, meta),
		truncateRootsFile(r.rootsFile, length),
	)
}

func truncateRootsFile(path string, length int) error {
	state, err := os.Stat(path)
	if err != nil {
		return err
	}
	entrySize := int64(NodeIdEncoder{}.GetEncodedSize() + 32)
	sourceLength := state.Size()
	targetLength := int64(length) * entrySize
	if sourceLength < targetLength {
		return fmt.Errorf("cannot truncate root file to %d elements, only %d elements available", targetLength/entrySize, sourceLength/entrySize)
	}
	return os.Truncate(path, targetLength)
}

type rootListCheckpointData struct {
	Checkpoint checkpoint.Checkpoint
	NumRoots   int
}

func readRootListCheckpointData(file string) (rootListCheckpointData, error) {
	_, err := os.Stat(file)
	if os.IsNotExist(err) {
		return rootListCheckpointData{}, nil
	}
	return utils.ReadJsonFile[rootListCheckpointData](file)
}

func writeRootListCheckpointData(file string, data rootListCheckpointData) error {
	return utils.WriteJsonFile(file, data)
}
