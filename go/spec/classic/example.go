package classic

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/0xsoniclabs/carmen/go/common"
)

type ExampleDB struct {
	backend       Backend
	blockContexts map[BlockContext]*blockContext
	txContexts    map[TransactionContext]*txContext
}

type blockContext struct {
	base                  State
	hasRunningTransaction bool
	update                Update
	errors                []error
	finished              bool
}

type txContext struct {
	blockContext *blockContext
	parent       *txContext
	active       bool
	update       Update
}

func NewExampleDB() *ExampleDB {
	return &ExampleDB{backend: &ExampleBackend{}}
}

func (db *ExampleDB) HasState(state State) (bool, error) {
	return db.backend.HasState(state)
}

func (db *ExampleDB) GetHash(state State) (Hash, error) {
	return db.backend.GetHash(state)
}

func (db *ExampleDB) BeginBlock(base State) (BlockContext, error) {
	found, err := db.backend.HasState(base)
	if err != nil {
		return 0, err
	}
	if !found {
		return 0, fmt.Errorf("no such state")
	}
	if db.blockContexts == nil {
		db.blockContexts = map[BlockContext]*blockContext{}
	}
	next := BlockContext(len(db.blockContexts))
	db.blockContexts[next] = &blockContext{
		base: base,
	}
	return next, nil
}

func (db *ExampleDB) CommitBlock(ctxt BlockContext) (State, error) {
	context, found := db.blockContexts[ctxt]
	if !found {
		return 0, fmt.Errorf("unknown block context")
	}
	if context.hasRunningTransaction {
		return 0, fmt.Errorf("unable to commit block with running transaction")
	}
	if context.finished {
		return 0, fmt.Errorf("unable to commit block that has already been committed")
	}
	context.finished = true
	db.blockContexts[ctxt] = context
	if len(context.errors) > 0 {
		return 0, errors.Join(context.errors...)
	}
	return db.backend.Apply(context.base, context.update)
}

func (db *ExampleDB) RevertBlock(ctxt BlockContext) error {
	context, found := db.blockContexts[ctxt]
	if !found {
		return fmt.Errorf("unknown block context")
	}
	if context.hasRunningTransaction {
		return fmt.Errorf("unable to revert block with running transaction")
	}
	if context.finished {
		return fmt.Errorf("unable to revert block that has already been committed")
	}
	context.finished = true
	db.blockContexts[ctxt] = context
	if len(context.errors) > 0 {
		return errors.Join(context.errors...)
	}
	return nil
}

func (db *ExampleDB) BeginTransaction(ctxt BlockContext) TransactionContext {
	context := db.blockContexts[ctxt]
	if context == nil {
		panic("unknown block context")
	}
	if context.hasRunningTransaction {
		panic("unable to run multiple transactions concurrently")
	}
	context.hasRunningTransaction = true
	next := TransactionContext(len(db.txContexts))
	if db.txContexts == nil {
		db.txContexts = map[TransactionContext]*txContext{}
	}
	db.txContexts[next] = &txContext{
		blockContext: context,
		parent:       nil,
		active:       true,
	}
	return next
}

func (db *ExampleDB) CreateContract(TransactionContext, Address) {
	panic("not implemented")
}

func (db *ExampleDB) SelfDestruct(TransactionContext, Address) {
	panic("not implemented")
}

func (db *ExampleDB) HasSelfDestructed(TransactionContext, Address) {
	panic("not implemented")
}

func (db *ExampleDB) GetBalance(ctxt TransactionContext, addr Address) Value {
	context := db.txContexts[ctxt]
	if context == nil {
		panic("unknown transaction context")
	}
	if !context.active {
		panic("given transaction context is not active")
	}
	blockContext := context.blockContext
	for context != nil {
		if balance, found := context.update.Balance[addr]; found {
			return balance
		}
		context = context.parent
	}

	if balance, found := blockContext.update.Balance[addr]; found {
		return balance
	}

	balance, err := db.backend.GetBalance(blockContext.base, addr)
	if err != nil {
		blockContext.errors = append(blockContext.errors, err)
		return Value{}
	}
	return balance
}

func (db *ExampleDB) SetBalance(ctxt TransactionContext, addr Address, balance Value) {
	context := db.txContexts[ctxt]
	if context == nil {
		panic("unknown transaction context")
	}
	if !context.active {
		panic("given transaction context is not active")
	}
	if context.update.Balance == nil {
		context.update.Balance = map[Address]Value{}
	}
	context.update.Balance[addr] = balance
}

func (db *ExampleDB) GetNonce(TransactionContext, Address) Nonce {
	panic("not implemented")
}

func (db *ExampleDB) SetNonce(TransactionContext, Address, Nonce) {
	panic("not implemented")
}

func (db *ExampleDB) GetCode(TransactionContext, Address) Code {
	panic("not implemented")
}

func (db *ExampleDB) SetCode(TransactionContext, Address, Code) {
	panic("not implemented")
}

func (db *ExampleDB) GetStorage(TransactionContext, Address, Key) Value {
	panic("not implemented")
}

func (db *ExampleDB) GetCommittedStorage(TransactionContext, Address, Key) Value {
	panic("not implemented")
}

func (db *ExampleDB) SetStorage(TransactionContext, Address, Key, Value) {
	panic("not implemented")
}

func (db *ExampleDB) BeginNested(ctxt TransactionContext) TransactionContext {
	parent := db.txContexts[ctxt]
	if parent == nil {
		panic("unknown transaction context")
	}
	if !parent.active {
		panic("given transaction context is not active")
	}
	next := TransactionContext(len(db.txContexts))
	db.txContexts[next] = &txContext{
		blockContext: parent.blockContext,
		parent:       parent,
		active:       true,
	}
	parent.active = false
	return next
}

func (db *ExampleDB) Commit(ctxt TransactionContext) error {
	context := db.txContexts[ctxt]
	if context == nil {
		panic("unknown transaction context")
	}
	if !context.active {
		panic("given transaction context is not active")
	}
	var target *Update
	if context.parent == nil {
		target = &context.blockContext.update
	} else {
		target = &context.parent.update
	}
	applyUpdate(target, context.update)
	return db.Revert(ctxt)
}

func (db *ExampleDB) Revert(ctxt TransactionContext) error {
	context := db.txContexts[ctxt]
	if context == nil {
		panic("unknown transaction context")
	}
	if !context.active {
		panic("given transaction context is not active")
	}
	context.active = false
	if context.parent != nil {
		context.parent.active = true
	} else {
		context.blockContext.hasRunningTransaction = false
	}
	return errors.Join(context.blockContext.errors...)
}

func (db *ExampleDB) Flush() error {
	panic("not implemented")
}

func (db *ExampleDB) Close() error {
	panic("not implemented")
}

// ---- Example Backend Implementation ----

type ExampleBackend struct {
	roots []*node
}

func (r *ExampleBackend) dump(base State) {
	dumpNode(r.roots[base])
}

func (r *ExampleBackend) getRoot(state State) (*node, bool) {
	if state == 0 {
		return nil, true
	}
	pos := int(state) - 1
	if pos < 0 || pos >= len(r.roots) {
		return nil, false
	}
	return r.roots[pos], true
}

func (r *ExampleBackend) HasState(state State) (bool, error) {
	return int(state) <= len(r.roots), nil
}

func (r *ExampleBackend) GetHash(state State) (Hash, error) {
	root, found := r.getRoot(state)
	if !found {
		return Hash{}, fmt.Errorf("no such state")
	}
	return hashNode(root), nil
}

func (r *ExampleBackend) GetBalance(state State, addr Address) (Value, error) {
	root, found := r.getRoot(state)
	if !found {
		return Value{}, fmt.Errorf("no such state")
	}
	return getAccount(root, addr[:]).balance, nil
}

func (r *ExampleBackend) GetNonce(state State, addr Address) (Nonce, error) {
	root, found := r.getRoot(state)
	if !found {
		return 0, fmt.Errorf("no such state")
	}
	return getAccount(root, addr[:]).nonce, nil
}

func (r *ExampleBackend) GetCode(state State, addr Address) (Code, error) {
	root, found := r.getRoot(state)
	if !found {
		return nil, fmt.Errorf("no such state")
	}
	account := getAccount(root, addr[:])
	return bytes.Clone(account.code), nil
}

func (r *ExampleBackend) GetStorage(state State, addr Address, key Key) (Word, error) {
	root, found := r.getRoot(state)
	if !found {
		return Word{}, fmt.Errorf("no such state")
	}
	account := getAccount(root, addr[:])
	return getValue(account.storage, key[:]), nil
}

func (r *ExampleBackend) Apply(state State, update Update) (State, error) {
	root, found := r.getRoot(state)
	if !found {
		return 0, fmt.Errorf("no such state")
	}

	for addr, balance := range update.Balance {
		account := getAccount(root, addr[:])
		account.balance = balance
		root = setAccount(root, addr[:], &account)
	}

	for addr, nonce := range update.Nonce {
		account := getAccount(root, addr[:])
		account.nonce = nonce
		root = setAccount(root, addr[:], &account)
	}

	for addr, code := range update.Code {
		account := getAccount(root, addr[:])
		account.code = bytes.Clone(code)
		root = setAccount(root, addr[:], &account)
	}

	for addr, changes := range update.Storage {
		account := getAccount(root, addr[:])
		for key, word := range changes {
			account.storage = setValue(account.storage, key[:], word)
		}
		root = setAccount(root, addr[:], &account)
	}

	newState := State(len(r.roots) + 1)
	r.roots = append(r.roots, root)
	return newState, nil
}

type node struct {
	children [256]*node
	account  *account
}

type storageNode struct {
	children [256]*storageNode
	word     *Word
}

type account struct {
	balance Value
	nonce   Nonce
	code    Code
	storage *storageNode
}

func (a *account) isEmpty() bool {
	return a.balance == Value{} && a.nonce == 0 &&
		len(a.code) == 0 && a.storage == nil
}

var emptyHash = Hash(common.Keccak256(nil))

func hashNode(n *node) Hash {
	if n == nil {
		return emptyHash
	}
	data := []byte{}
	for _, child := range n.children {
		hash := hashNode(child)
		data = append(data, hash[:]...)
	}

	if account := n.account; account != nil {
		data = append(data, account.balance[:]...)
		data = binary.BigEndian.AppendUint64(data, uint64(account.nonce))
		data = append(data, account.code...)
		if account.storage == nil {
			data = append(data, emptyHash[:]...)
		} else {
			hash := hashStorageNode(account.storage)
			data = append(data, hash[:]...)
		}
	}

	return Hash(common.Keccak256(data))
}

func hashStorageNode(n *storageNode) Hash {
	if n == nil {
		return emptyHash
	}
	data := []byte{}
	for _, child := range n.children {
		hash := hashStorageNode(child)
		data = append(data, hash[:]...)
	}

	if word := n.word; word != nil {
		data = append(data, word[:]...)
	}
	return Hash(common.Keccak256(data))
}

func dumpNode(n *node) {
	if n == nil {
		fmt.Printf("empty\n")
	} else {
		dumpNodeInternal(n, "", -1)
	}
}

func dumpNodeInternal(n *node, prefix string, index int) {
	fmt.Print(prefix)
	if index >= 0 {
		fmt.Printf("%2x: ", index)
	}
	hash := hashNode(n)
	fmt.Printf("Node %x", hash[0:4])
	if n.account != nil {
		fmt.Printf(": N=%d, B=%x", n.account.nonce, n.account.balance)
		dumpStorageNodeInternal(n.account.storage, prefix+"  ", -1)
	}
	fmt.Printf("\n")
	for i, next := range n.children {
		if next == nil {
			continue
		}
		dumpNodeInternal(next, prefix+"  ", i)
	}
}

/*
func dumpStorageNode(n *storageNode) {
	if n == nil {
		fmt.Printf("empty\n")
	} else {
		dumpStorageNodeInternal(n, "", -1)
	}
}
*/

func dumpStorageNodeInternal(n *storageNode, prefix string, index int) {
	if n == nil {
		return
	}
	fmt.Print(prefix)
	if index >= 0 {
		fmt.Printf("%2x: ", index)
	}
	hash := hashStorageNode(n)
	fmt.Printf("Node %x", hash[0:4])
	if n.word != nil {
		fmt.Printf(": W=%x", n.word)
	}
	fmt.Printf("\n")
	for i, next := range n.children {
		if next == nil {
			continue
		}
		dumpStorageNodeInternal(next, prefix+"  ", i)
	}
}

func getAccount(n *node, path []byte) account {
	if n == nil {
		return account{}
	}
	if len(path) == 0 {
		return *n.account
	}
	return getAccount(n.children[path[0]], path[1:])
}

func setAccount(n *node, path []byte, account *account) *node {
	res := &node{}
	if n == nil {
		if account.isEmpty() {
			return nil
		}
	} else {
		*res = *n
	}
	if len(path) == 0 {
		if account.isEmpty() {
			return nil
		}
		res.account = account
		return res
	}

	new := setAccount(res.children[path[0]], path[1:], account)
	res.children[path[0]] = new
	if new == nil && res.children == [256]*node{} {
		return nil
	}
	return res
}

func getValue(n *storageNode, path []byte) Word {
	if n == nil {
		return Word{}
	}
	if len(path) == 0 {
		return *n.word
	}
	return getValue(n.children[path[0]], path[1:])
}

func setValue(n *storageNode, path []byte, word Word) *storageNode {
	res := &storageNode{}
	if n == nil {
		if word == (Word{}) {
			return nil
		}
	} else {
		*res = *n
	}
	if len(path) == 0 {
		if word == (Word{}) {
			return nil
		}
		res.word = &word
		return res
	}

	new := setValue(res.children[path[0]], path[1:], word)
	res.children[path[0]] = new
	if new == nil && res.children == [256]*storageNode{} {
		return nil
	}
	return res
}

func applyUpdate(dst *Update, src Update) {
	if len(src.Balance) > 0 && dst.Balance == nil {
		dst.Balance = map[Address]Value{}
	}
	if len(src.Nonce) > 0 && dst.Nonce == nil {
		dst.Nonce = map[Address]Nonce{}
	}
	if len(src.Code) > 0 && dst.Code == nil {
		dst.Code = map[Address]Code{}
	}
	if len(src.Storage) > 0 && dst.Storage == nil {
		dst.Storage = map[Address]map[Key]Word{}
	}

	for addr, balance := range src.Balance {
		dst.Balance[addr] = balance
	}
	for addr, nonce := range src.Nonce {
		dst.Nonce[addr] = nonce
	}
	for addr, code := range src.Code {
		dst.Code[addr] = code
	}
	for addr, updates := range src.Storage {
		storage := dst.Storage[addr]
		if storage == nil {
			dst.Storage[addr] = map[Key]Word{}
			storage = dst.Storage[addr]
		}
		for key, value := range updates {
			storage[key] = value
		}
	}
}
