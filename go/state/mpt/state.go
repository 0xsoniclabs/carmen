package mpt

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"io"
	"os"
	"sync"
	"unsafe"

	"github.com/Fantom-foundation/Carmen/go/backend"
	"github.com/Fantom-foundation/Carmen/go/common"
	"golang.org/x/crypto/sha3"
)

// MptState implementation of a state utilizes an MPT based data structure. While
// functionally equivalent to the Ethereum State MPT, hashes are computed using
// a configurable algorithm.
//
// The main role of the MptState is to provide an adapter between a LiveTrie and
// Carmen's State interface. Also, it retains an index of contract codes.
type MptState struct {
	directory string
	lock      common.LockFile
	trie      *LiveTrie
	code      map[common.Hash][]byte
	codeMutex sync.Mutex
	codefile  string
	hasher    hash.Hash
}

// The capacity of an MPT's node cache must be at least as large as the maximum
// number of nodes modified in a block. Evaluations show that most blocks
// modify less than 2000 nodes. However, one block, presumably the one handling
// the opera fork at ~4.5M, modifies 434.589 nodes. Thus, the cache size of a
// MPT processing Fantom's history should be at least ~500.000 nodes.
const DefaultMptStateCapacity = 10_000_000
const MinMptStateCapacity = 2_000

var emptyCodeHash = common.GetHash(sha3.NewLegacyKeccak256(), []byte{})

func newMptState(directory string, lock common.LockFile, trie *LiveTrie) (*MptState, error) {
	codefile := directory + "/codes.dat"
	codes, err := readCodes(codefile)
	if err != nil {
		return nil, err
	}
	return &MptState{
		directory: directory,
		lock:      lock,
		trie:      trie,
		code:      codes,
		codefile:  codefile,
	}, nil
}

func openStateDirectory(directory string) (common.LockFile, error) {
	lock, err := LockDirectory(directory)
	if err != nil {
		return nil, err
	}
	if err := tryMarkDirty(directory); err != nil {
		return nil, errors.Join(err, lock.Release())
	}

	return lock, nil
}

func tryMarkDirty(directory string) error {
	dirty, err := isDirty(directory)
	if err != nil {
		return err
	}
	if dirty {
		return fmt.Errorf("unable to open %s, content is dirty, likely corrupted", directory)
	}
	return markDirty(directory)
}

// OpenGoMemoryState loads state information from the given directory and
// creates a Trie entirely retained in memory.
func OpenGoMemoryState(directory string, config MptConfig, cacheCapacity int) (*MptState, error) {
	lock, err := openStateDirectory(directory)
	if err != nil {
		return nil, err
	}
	trie, err := OpenInMemoryLiveTrie(directory, config, cacheCapacity)
	if err != nil {
		return nil, err
	}
	return newMptState(directory, lock, trie)
}

func OpenGoFileState(directory string, config MptConfig, cacheCapacity int) (*MptState, error) {
	lock, err := openStateDirectory(directory)
	if err != nil {
		return nil, err
	}
	trie, err := OpenFileLiveTrie(directory, config, cacheCapacity)
	if err != nil {
		return nil, err
	}
	return newMptState(directory, lock, trie)
}

func (s *MptState) CreateAccount(address common.Address) (err error) {
	_, exists, err := s.trie.GetAccountInfo(address)
	if err != nil {
		return err
	}
	if exists {
		// For existing accounts, only clear the storage, preserve the rest.
		return s.trie.ClearStorage(address)
	}
	// Create account with hash of empty code.
	return s.trie.SetAccountInfo(address, AccountInfo{
		CodeHash: emptyCodeHash,
	})
}

func (s *MptState) Exists(address common.Address) (bool, error) {
	_, exists, err := s.trie.GetAccountInfo(address)
	if err != nil {
		return false, err
	}
	return exists, nil
}

func (s *MptState) DeleteAccount(address common.Address) error {
	return s.trie.SetAccountInfo(address, AccountInfo{})
}

func (s *MptState) GetBalance(address common.Address) (balance common.Balance, err error) {
	info, exists, err := s.trie.GetAccountInfo(address)
	if !exists || err != nil {
		return common.Balance{}, err
	}
	return info.Balance, nil
}

func (s *MptState) SetBalance(address common.Address, balance common.Balance) (err error) {
	info, exists, err := s.trie.GetAccountInfo(address)
	if err != nil {
		return err
	}
	if info.Balance == balance {
		return nil
	}
	info.Balance = balance
	if !exists {
		info.CodeHash = emptyCodeHash
	}
	return s.trie.SetAccountInfo(address, info)
}

func (s *MptState) GetNonce(address common.Address) (nonce common.Nonce, err error) {
	info, _, err := s.trie.GetAccountInfo(address)
	if err != nil {
		return common.Nonce{}, err
	}
	return info.Nonce, nil
}

func (s *MptState) SetNonce(address common.Address, nonce common.Nonce) (err error) {
	info, exists, err := s.trie.GetAccountInfo(address)
	if err != nil {
		return err
	}
	if info.Nonce == nonce {
		return nil
	}
	info.Nonce = nonce
	if !exists {
		info.CodeHash = emptyCodeHash
	}
	return s.trie.SetAccountInfo(address, info)
}

func (s *MptState) GetStorage(address common.Address, key common.Key) (value common.Value, err error) {
	return s.trie.GetValue(address, key)
}

func (s *MptState) SetStorage(address common.Address, key common.Key, value common.Value) error {
	return s.trie.SetValue(address, key, value)
}

func (s *MptState) GetCode(address common.Address) (value []byte, err error) {
	info, exists, err := s.trie.GetAccountInfo(address)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, nil
	}
	s.codeMutex.Lock()
	res := s.code[info.CodeHash]
	s.codeMutex.Unlock()
	return res, nil
}

func (s *MptState) GetCodeForHash(hash common.Hash) []byte {
	s.codeMutex.Lock()
	res := s.code[hash]
	s.codeMutex.Unlock()
	return res
}

func (s *MptState) GetCodeSize(address common.Address) (size int, err error) {
	code, err := s.GetCode(address)
	if err != nil {
		return 0, err
	}
	return len(code), err
}

func (s *MptState) SetCode(address common.Address, code []byte) (err error) {
	var codeHash common.Hash
	if s.hasher == nil {
		s.hasher = sha3.NewLegacyKeccak256()
	}
	codeHash = common.GetHash(s.hasher, code)

	info, exists, err := s.trie.GetAccountInfo(address)
	if err != nil {
		return err
	}
	if !exists && len(code) == 0 {
		return nil
	}
	if info.CodeHash == codeHash {
		return nil
	}
	info.CodeHash = codeHash
	s.codeMutex.Lock()
	s.code[codeHash] = code
	s.codeMutex.Unlock()
	return s.trie.SetAccountInfo(address, info)
}

func (s *MptState) GetCodeHash(address common.Address) (hash common.Hash, err error) {
	info, exists, err := s.trie.GetAccountInfo(address)
	if !exists || err != nil {
		return emptyCodeHash, err
	}
	return info.CodeHash, nil
}

func (s *MptState) GetRootId() NodeId {
	return s.trie.root.Id()
}

func (s *MptState) GetHash() (hash common.Hash, err error) {
	hash, hints, err := s.trie.UpdateHashes()
	if hints != nil {
		hints.Release()
	}
	return hash, err
}

func (s *MptState) Apply(block uint64, update common.Update) (archiveUpdateHints common.Releaser, err error) {
	if err := update.ApplyTo(s); err != nil {
		return nil, err
	}
	_, hints, err := s.trie.UpdateHashes()
	return hints, err
}

func (s *MptState) Visit(visitor NodeVisitor) error {
	return s.trie.VisitTrie(visitor)
}

func (s *MptState) GetCodes() (map[common.Hash][]byte, error) {
	return s.code, nil
}

func (s *MptState) Flush() error {
	// Flush codes and state trie.
	return errors.Join(
		errors.Join(s.trie.forest.GetEncounteredIssues()...),
		writeCodes(s.code, s.codefile),
		s.trie.Flush(),
	)
}

func (s *MptState) Close() error {
	return s.closeWithError(nil)
}

func (s *MptState) closeWithError(externalError error) error {
	// Only if the state can be successfully closed, the directory is to
	// be marked as clean. Otherwise, the dirty flag needs to be retained.
	err := errors.Join(
		externalError,
		s.Flush(),
		s.trie.Close(),
	)
	if err == nil {
		err = markClean(s.directory)
	}
	return errors.Join(
		err,
		s.lock.Release(),
	)
}

func (s *MptState) GetSnapshotableComponents() []backend.Snapshotable {
	//panic("not implemented")
	return nil
}

func (s *MptState) RunPostRestoreTasks() error {
	//panic("not implemented")
	return nil
}

// GetMemoryFootprint provides sizes of individual components of the state in the memory
func (s *MptState) GetMemoryFootprint() *common.MemoryFootprint {
	mf := common.NewMemoryFootprint(unsafe.Sizeof(*s))
	mf.AddChild("trie", s.trie.GetMemoryFootprint())
	// TODO: add code store
	return mf
}

// readCodes parses the content of the given file if it exists or returns
// a an empty code collection if there is no such file.
func readCodes(filename string) (map[common.Hash][]byte, error) {
	// If there is no file, initialize and return an empty code collection.
	if _, err := os.Stat(filename); err != nil {
		return map[common.Hash][]byte{}, nil
	}

	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	reader := bufio.NewReader(file)
	return parseCodes(reader)
}

func parseCodes(reader io.Reader) (map[common.Hash][]byte, error) {
	// If the file exists, parse it and return its content.
	res := map[common.Hash][]byte{}
	// The format is simple: [<key>, <length>, <code>]*
	var hash common.Hash
	var length [4]byte
	for {
		if _, err := io.ReadFull(reader, hash[:]); err != nil {
			if err == io.EOF {
				return res, nil
			}
			return nil, err
		}
		if _, err := io.ReadFull(reader, length[:]); err != nil {
			return nil, err
		}
		size := binary.BigEndian.Uint32(length[:])
		code := make([]byte, size)
		if _, err := io.ReadFull(reader, code[:]); err != nil {
			return nil, err
		}
		res[hash] = code
	}
}

// writeCodes write the given map of codes to the given file.
func writeCodes(codes map[common.Hash][]byte, filename string) (err error) {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	writer := bufio.NewWriter(file)
	return errors.Join(
		writeCodesTo(codes, writer),
		writer.Flush(),
		file.Close())
}

func writeCodesTo(codes map[common.Hash][]byte, writer io.Writer) (err error) {
	// The format is simple: [<key>, <length>, <code>]*
	for key, code := range codes {
		if _, err := writer.Write(key[:]); err != nil {
			return err
		}
		var length [4]byte
		binary.BigEndian.PutUint32(length[:], uint32(len(code)))
		if _, err := writer.Write(length[:]); err != nil {
			return err
		}
		if _, err := writer.Write(code); err != nil {
			return err
		}
	}
	return nil
}
