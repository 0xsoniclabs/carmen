package classic

type Database interface {
	// --- State Handling ---
	HasState(State) (bool, error)
	GetHash(State) (Hash, error)

	// --- Block Handling ---
	BeginBlock(State) (BlockContext, error)
	CommitBlock(BlockContext) (State, error)
	RevertBlock(BlockContext) error

	BeginTransaction(BlockContext) TransactionContext

	// --- Transaction Context ---
	CreateContract(TransactionContext, Address)
	SelfDestruct(TransactionContext, Address)
	HasSelfDestructed(TransactionContext, Address)

	GetBalance(TransactionContext, Address) Value
	SetBalance(TransactionContext, Address, Value)

	GetNonce(TransactionContext, Address) Nonce
	SetNonce(TransactionContext, Address, Nonce)

	GetCode(TransactionContext, Address) Code
	SetCode(TransactionContext, Address, Code)

	GetStorage(TransactionContext, Address, Key) Value
	GetCommittedStorage(TransactionContext, Address, Key) Value
	SetStorage(TransactionContext, Address, Key, Value)

	BeginNested(TransactionContext) TransactionContext

	Commit(TransactionContext) error
	Revert(TransactionContext) error

	// --- File System Operations ---
	Flush() error
	Close() error
}

type Backend interface {
	HasState(State) (bool, error)
	GetHash(State) (Hash, error)

	GetBalance(State, Address) (Value, error)
	GetNonce(State, Address) (Nonce, error)
	GetCode(State, Address) (Code, error)
	GetStorage(State, Address, Key) (Word, error)

	Apply(State, Update) (State, error)
}

type Update struct {
	Balance map[Address]Value
	Nonce   map[Address]Nonce
	Code    map[Address]Code
	Storage map[Address]map[Key]Word
}

const EmptyState State = 0

type State uint64
type Nonce uint64

type Address [20]byte
type Key [32]byte
type Word [32]byte

type Value [32]byte

type Code []byte

type Hash [32]byte

// Helper Types

type BlockContext int
type TransactionContext int
type QueryContext int
