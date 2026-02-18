package geth2

import (
	"testing"

	"github.com/stretchr/testify/require"
)

var _ NodeStore = (*levelDbStore)(nil)

func TestLevelDbStore_CanKeepDataPersistent(t *testing.T) {
	key1 := []byte("key1")
	value1 := []byte("value1")
	key2 := []byte("key2")
	value2 := []byte("value2")

	dir := t.TempDir()

	store, err := newLevelDbStore(dir)
	require.NoError(t, err)

	require.NoError(t, store.Set(key1, value1))
	require.NoError(t, store.Set(key2, value2))

	err = store.Close()
	require.NoError(t, err)

	store2, err := newLevelDbStore(dir)
	require.NoError(t, err)

	val, err := store2.Get(key1)
	require.NoError(t, err)
	require.Equal(t, value1, val)

	val, err = store2.Get(key2)
	require.NoError(t, err)
	require.Equal(t, value2, val)

	require.NoError(t, store2.Close())
}

func TestLevelDbStore_ReturnsNotFoundForMissingKey(t *testing.T) {
	store, err := newLevelDbStore(t.TempDir())
	require.NoError(t, err)

	_, err = store.Get([]byte("nonexistent"))
	require.ErrorIs(t, err, ErrNotFound)

	require.NoError(t, store.Close())
}

func TestMemoryDbStore_CanSetAndGet(t *testing.T) {
	require := require.New(t)
	store := newMemoryDbStore()

	key1 := []byte("key1")
	value1 := []byte("value1")
	key2 := []byte("key2")
	value2 := []byte("value2")

	require.NoError(store.Set(key1, value1))
	require.NoError(store.Set(key2, value2))
	val, err := store.Get(key1)
	require.NoError(err)
	require.Equal(value1, val)

	val, err = store.Get(key2)
	require.NoError(err)
	require.Equal(value2, val)
}

func TestMemoryDbStore_ReturnsNotFoundForMissingKey(t *testing.T) {
	store := newMemoryDbStore()
	_, err := store.Get([]byte("nonexistent"))
	require.ErrorIs(t, err, ErrNotFound)
	require.NoError(t, store.Close())
}
