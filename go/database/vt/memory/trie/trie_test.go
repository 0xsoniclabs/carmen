package trie

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestTrie_InitialTrieIsEmpty(t *testing.T) {
	require := require.New(t)

	zero := Value{}
	trie := &Trie{}
	require.Equal(zero, trie.Get(Key{1}))
	require.Equal(zero, trie.Get(Key{2}))
	require.Equal(zero, trie.Get(Key{3}))
}

func TestTrie_ValuesCanBeSetAndRetrieved(t *testing.T) {
	require := require.New(t)

	trie := &Trie{}

	require.Equal(Value{}, trie.Get(Key{1}))
	require.Equal(Value{}, trie.Get(Key{2}))
	require.Equal(Value{}, trie.Get(Key{0, 31: 1}))
	require.Equal(Value{}, trie.Get(Key{0, 31: 2}))

	trie.Set(Key{1}, Value{1})

	require.Equal(Value{1}, trie.Get(Key{1}))
	require.Equal(Value{}, trie.Get(Key{2}))
	require.Equal(Value{}, trie.Get(Key{0, 31: 1}))
	require.Equal(Value{}, trie.Get(Key{0, 31: 2}))

	trie.Set(Key{2}, Value{2})

	require.Equal(Value{1}, trie.Get(Key{1}))
	require.Equal(Value{2}, trie.Get(Key{2}))
	require.Equal(Value{}, trie.Get(Key{0, 31: 1}))
	require.Equal(Value{}, trie.Get(Key{0, 31: 2}))

	trie.Set(Key{0, 31: 1}, Value{3})

	require.Equal(Value{1}, trie.Get(Key{1}))
	require.Equal(Value{2}, trie.Get(Key{2}))
	require.Equal(Value{3}, trie.Get(Key{0, 31: 1}))
	require.Equal(Value{}, trie.Get(Key{0, 31: 2}))

	trie.Set(Key{0, 31: 2}, Value{4})

	require.Equal(Value{1}, trie.Get(Key{1}))
	require.Equal(Value{2}, trie.Get(Key{2}))
	require.Equal(Value{3}, trie.Get(Key{0, 31: 1}))
	require.Equal(Value{4}, trie.Get(Key{0, 31: 2}))
}

func TestTrie_ValuesCanBeUpdated(t *testing.T) {
	require := require.New(t)

	trie := &Trie{}

	key := Key{1}
	require.Equal(Value{}, trie.Get(key))
	trie.Set(key, Value{1})
	require.Equal(Value{1}, trie.Get(key))
	trie.Set(key, Value{2})
	require.Equal(Value{2}, trie.Get(key))
	trie.Set(key, Value{3})
	require.Equal(Value{3}, trie.Get(key))
}

func TestTrie_ManyValuesCanBeSetAndRetrieved(t *testing.T) {
	const N = 1000
	require := require.New(t)

	toKey := func(i int) Key {
		return Key{byte(i >> 8 & 0x0F), byte(i >> 4 & 0x0F), 31: byte(i & 0x0F)}
	}

	trie := &Trie{}
	for i := range N {
		for j := range N {
			want := Value{}
			if j < i {
				want = Value{byte(j)}
			}
			got := trie.Get(toKey(j))
			require.Equal(want, got, "In round %d Get(%d) should return %v, got %v", i, j, want, got)
		}
		trie.Set(toKey(i), Value{byte(i)})
	}
}

func TestTrie_CommitmentForExampleTries(t *testing.T) {
	require := require.New(t)

	// Example trie from the documentation.
	trie := &Trie{}
	trie.Set(Key{1}, Value{1})
	trie.Set(Key{2}, Value{2})
	trie.Set(Key{3}, Value{3})

	commitment := trie.Commit()
	require.True(commitment.IsValid(), "Commitment should be valid")

	// TODO: check the value of the commitment against a known value.
}
