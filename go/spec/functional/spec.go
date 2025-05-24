package functional

type Database interface {
	HasState(State) (bool, error)
	GetHash(State) (Hash, error)

	QueryBlock(root State, query func(BlockContext) error) error
	AddBlock(base State, run func(BlockContext) error) (State, error)

	Flush() error
	Close() error
}

type BlockContext interface {
	RunTransaction(run func(TransactionContext) error) error
}

type TransactionContext interface {
	CreateContract(Address)
	SelfDestruct(Address)
	HasSelfDestructed(Address)

	GetBalance(Address) Value
	SetBalance(Address, Value)

	GetNonce(Address) Nonce
	SetNonce(Address, Nonce)

	GetCode(Address) Code
	SetCode(Address, Code)

	GetStorage(Address, Key) Word
	GetCommittedStorage(Address, Key) Word
	SetStorage(Address, Key, Word)

	RunNested(run func(TransactionContext) error)
}

// --- Backend ---

type Backend interface {
	HasState(State) (bool, error)
	GetHash(State) (Hash, error)

	QueryBlock(state State, query func(QueryContext)) error
	AddBlock(state State, run func(NewBlockContext) error) error

	Flush() error
	Close() error
}

type QueryContext interface {
	GetBalance(Address) Value
	GetNonce(Address) Nonce
	GetCode(Address) Code
	GetStorage(Address, Key) Word
}

type NewBlockContext interface {
	QueryContext
	Apply(Update)
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
