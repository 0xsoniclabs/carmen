package functional

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestExampleDB_ProcessBlocks(t *testing.T) {
	require := require.New(t)
	db := NewExampleDB()

	found, err := db.HasState(EmptyState)
	require.NoError(err)
	require.True(found)

	// -- Block 1 --

	state1, err := db.AddBlock(EmptyState, func(context BlockContext) error {
		context.RunTransaction(func(tx TransactionContext) error {
			tx.SetBalance(Address{1}, Value{1})
			return nil
		})
		context.RunTransaction(func(tx TransactionContext) error {
			tx.SetBalance(Address{2}, Value{2})
			return nil
		})
		return nil
	})
	require.NoError(err)
	require.True(db.HasState(state1))

	// -- Query Block 0 --

	require.NoError(db.QueryBlock(EmptyState, func(query BlockContext) error {
		require.NoError(query.RunTransaction(func(tx TransactionContext) error {
			require.Equal(Value{0}, tx.GetBalance(Address{1}))
			require.Equal(Value{0}, tx.GetBalance(Address{2}))
			return nil
		}))
		return nil
	}))

	require.NoError(db.QueryBlock(state1, func(query BlockContext) error {
		require.NoError(query.RunTransaction(func(tx TransactionContext) error {
			require.Equal(Value{1}, tx.GetBalance(Address{1}))
			require.Equal(Value{2}, tx.GetBalance(Address{2}))
			return nil
		}))
		return nil
	}))

	// -- Check Hashes --
	hash0, err := db.GetHash(EmptyState)
	require.NoError(err)
	hash1, err := db.GetHash(state1)
	require.NoError(err)
	require.NotEqual(hash0, hash1)
}
