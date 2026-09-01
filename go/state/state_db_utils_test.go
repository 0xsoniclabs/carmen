package state_test

import (
	"testing"

	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/stretchr/testify/require"
)

func Test_CartesianTriple(t *testing.T) {
	t.Parallel()

	input := []string{"a", "b"}
	var triples [][]string
	for t := range state.CartesianProductTriple(input) {
		triples = append(triples, t)
	}

	expected := [][]string{
		{"a", "a", "a"},
		{"a", "a", "b"},
		{"a", "b", "a"},
		{"a", "b", "b"},
		{"b", "a", "a"},
		{"b", "a", "b"},
		{"b", "b", "a"},
		{"b", "b", "b"},
	}

	require.Equal(t, expected, triples)
}

func Test_OrderedPartitions(t *testing.T) {
	t.Parallel()

	input := []int{1, 2, 3}
	var partitions [][][]int
	for p := range state.OrderedPartitions(input) {
		partitions = append(partitions, p)
	}

	expected := [][][]int{
		{{1, 2, 3}},
		{{1}, {2, 3}},
		{{1, 2}, {3}},
		{{1}, {2}, {3}},
	}

	require.Equal(t, expected, partitions)
}
