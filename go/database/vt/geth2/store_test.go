package geth2

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLevelDb_CanKeepDataPersistent(t *testing.T) {
	tests := map[string]func(string) (*levelDbStore, error){
		"live":    newLevelDbLiveStore,
		"archive": newLevelDbArchiveStore,
	}

	for name, factory := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)

			key1 := []byte("key1")
			value1 := []byte("value1")
			key2 := []byte("key2")
			value2 := []byte("value2")

			dir := t.TempDir()

			// --- first reincarnation ---
			store, err := factory(dir)
			require.NoError(err)
			require.Zero(store.NextBlock())

			// First block.
			require.NoError(store.AddBlock(
				0, []Entry{
					{Path: key1, Blob: value1},
					{Path: key2, Blob: value2},
				},
			))

			require.EqualValues(1, store.NextBlock())
			head := store.HeadState()
			val, err := head.GetNode(key1)
			require.NoError(err)
			require.Equal(value1, val)

			val, err = head.GetNode(key2)
			require.NoError(err)
			require.Equal(value2, val)

			// Second block.
			require.NoError(store.AddBlock(
				1, []Entry{
					{Path: key1, Blob: value2},
					{Path: key2, Blob: value1},
				},
			))

			require.EqualValues(2, store.NextBlock())
			head = store.HeadState()
			val, err = head.GetNode(key1)
			require.NoError(err)
			require.Equal(value2, val)

			val, err = head.GetNode(key2)
			require.NoError(err)
			require.Equal(value1, val)

			require.NoError(store.Close())

			// --- second reincarnation ---
			store2, err := factory(dir)
			require.NoError(err)
			require.EqualValues(2, store2.NextBlock())

			// Check head state.
			val, err = store2.HeadState().GetNode(key1)
			require.NoError(err)
			require.Equal(value2, val)

			val, err = store2.HeadState().GetNode(key2)
			require.NoError(err)
			require.Equal(value1, val)

			require.NoError(store2.Close())
		})
	}
}

func TestLiveDb_ReturnsErrorForHistoricState(t *testing.T) {
	store, err := newLevelDbLiveStore(t.TempDir())
	require.NoError(t, err)
	_, err = store.HistoricState(0)
	require.ErrorIs(t, err, ErrNoArchive)
	require.NoError(t, store.Close())
}

func TestArchiveDb_TracksHistory(t *testing.T) {
	require := require.New(t)
	store, err := newLevelDbArchiveStore(t.TempDir())
	require.NoError(err)

	key1 := []byte("key1")
	key2 := []byte("key2")
	value1 := []byte("value1")
	value2 := []byte("value2")

	// First block.
	require.NoError(store.AddBlock(
		0, []Entry{{Path: key1, Blob: value1}},
	))

	// Second block is empty.
	require.NoError(store.AddBlock(1, nil))

	// Third block updates key1 and adds key2.
	require.NoError(store.AddBlock(
		2, []Entry{
			{Path: key1, Blob: value2},
			{Path: key2, Blob: value2},
		},
	))

	// Fourth block is empty.
	require.NoError(store.AddBlock(3, nil))

	tests := map[string]struct {
		block uint64
		key   []byte
		value []byte
	}{
		"block 0, key1": {block: 0, key: key1, value: value1},
		"block 0, key2": {block: 0, key: key2, value: nil},
		"block 1, key1": {block: 1, key: key1, value: value1},
		"block 1, key2": {block: 1, key: key2, value: nil},
		"block 2, key1": {block: 2, key: key1, value: value2},
		"block 2, key2": {block: 2, key: key2, value: value2},
		"block 3, key1": {block: 3, key: key1, value: value2},
		"block 3, key2": {block: 3, key: key2, value: value2},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			historicState, err := store.HistoricState(tc.block)
			require.NoError(err)

			val, err := historicState.GetNode(tc.key)
			if tc.value == nil {
				require.ErrorIs(err, ErrNotFound)
				return
			}
			require.NoError(err)
			require.Equal(tc.value, val)
		})
	}

	// Can not request historic state for future block.
	_, err = store.HistoricState(4)
	require.ErrorContains(err, "requested block 4 is in the future")

	require.NoError(store.Close())
}
