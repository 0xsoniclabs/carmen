package functional

import "github.com/0xsoniclabs/carmen/go/spec/classic"

type ExampleDB struct {
	db classic.Database
}

func NewExampleDB() *ExampleDB {
	return &ExampleDB{db: classic.NewExampleDB()}
}

func (db *ExampleDB) HasState(state State) (bool, error) {
	return db.db.HasState(classic.State(state))
}

func (db *ExampleDB) GetHash(state State) (Hash, error) {
	hash, err := db.db.GetHash(classic.State(state))
	return Hash(hash), err
}

func (db *ExampleDB) QueryBlock(root State, query func(BlockContext) error) error {
	panic("not implemented")
}

func (db *ExampleDB) AddBlock(base State, run func(BlockContext) error) (State, error) {
	panic("not implemented")
}

func (db *ExampleDB) Flush() error {
	panic("not implemented")
}

func (db *ExampleDB) Close() error {
	panic("not implemented")
}
