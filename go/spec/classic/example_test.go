package classic

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

var _ Database = &ExampleDB{}
var _ Backend = &ExampleBackend{}

func TestExampleDB_ProcessBlocks(t *testing.T) {
	require := require.New(t)
	db := NewExampleDB()

	found, err := db.HasState(EmptyState)
	require.NoError(err)
	require.True(found)

	// -- Block 1 --

	block1, err := db.BeginBlock(EmptyState)
	require.NoError(err)

	tx1 := db.BeginTransaction(block1)
	db.SetBalance(tx1, Address{1}, Value{1})
	db.Commit(tx1)

	tx2 := db.BeginTransaction(block1)
	db.SetBalance(tx2, Address{2}, Value{2})
	db.Commit(tx2)

	state1, err := db.CommitBlock(block1)
	require.NoError(err)

	require.True(db.HasState(state1))

	// -- Query Block 0 --
	query0, err := db.BeginBlock(EmptyState)
	require.NoError(err)
	tx1 = db.BeginTransaction(query0)
	balance := db.GetBalance(tx1, Address{1})
	require.Equal(Value{0}, balance)
	db.Revert(tx1)
	db.RevertBlock(query0)

	// -- Query Block 1 --
	query1, err := db.BeginBlock(state1)
	require.NoError(err)
	tx1 = db.BeginTransaction(query1)
	balance = db.GetBalance(tx1, Address{1})
	require.Equal(Value{1}, balance)
	balance = db.GetBalance(tx1, Address{2})
	require.Equal(Value{2}, balance)
	db.Revert(tx1)
	db.RevertBlock(query1)

	// -- Check Hashes --
	hash0, err := db.GetHash(EmptyState)
	require.NoError(err)
	hash1, err := db.GetHash(state1)
	require.NoError(err)
	require.NotEqual(hash0, hash1)
}

func TestExampleDB_InsertElement(t *testing.T) {
	var node *node

	dumpNode(node)
	fmt.Printf("%x\n", hashNode(node))

	node = setAccount(node, []byte{1, 2, 3}, &account{
		nonce: 123,
	})

	dumpNode(node)
	fmt.Printf("%x\n", hashNode(node))

	node = setAccount(node, []byte{2, 1, 3}, &account{
		nonce: 45,
	})

	dumpNode(node)
	fmt.Printf("%x\n", hashNode(node))

	node = setAccount(node, []byte{2, 1, 3}, &account{})
	dumpNode(node)
	fmt.Printf("%x\n", hashNode(node))

	node = setAccount(node, []byte{1, 2, 3}, &account{})

	dumpNode(node)
	fmt.Printf("%x\n", hashNode(node))

	//t.Fail()
}

func TestExampleBackend_Updates(t *testing.T) {
	require := require.New(t)
	db := ExampleBackend{}

	addr1 := Address{1}

	balance0 := Value{0}
	balance1 := Value{1}

	empty := EmptyState
	balance, err := db.GetBalance(empty, addr1)
	require.NoError(err)
	require.Equal(balance0, balance)

	block1, err := db.Apply(empty, Update{
		Balance: map[Address]Value{
			addr1: balance1,
		},
	})
	require.NoError(err)
	require.NotEqual(empty, block1)

	block2, err := db.Apply(block1, Update{
		Balance: map[Address]Value{
			addr1: balance0,
		},
	})
	require.NoError(err)
	require.NotEqual(empty, block2)

	balance, err = db.GetBalance(empty, addr1)
	require.NoError(err)
	require.Equal(balance0, balance)

	balance, err = db.GetBalance(block1, addr1)
	require.NoError(err)
	require.Equal(balance1, balance)

	balance, err = db.GetBalance(block2, addr1)
	require.NoError(err)
	require.Equal(balance0, balance)
}
