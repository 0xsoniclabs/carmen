// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package commit

import (
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/stretchr/testify/require"
)

func TestCommitment_DefaultIsNotValid(t *testing.T) {
	commitment := Commitment{}
	require.False(t, commitment.IsValid(), "Default commitment should not be valid")
}

func TestCommitment_IdentityIsValid(t *testing.T) {
	commitment := Identity()
	require.True(t, commitment.IsValid(), "Identity commitment should be valid")
}

func TestCommitment_IdentityToValue_ReturnsZeroValue(t *testing.T) {
	value := Identity().ToValue()
	require.Equal(t, Value{}, value, "Identity commitment should convert to zero value")
}

func TestCommitment_Hash_IdentityReturnsZeroHash(t *testing.T) {
	hash := Identity().Hash()
	require.Equal(t, common.Hash{}, hash, "Hash of identity commitment should be zero")
}

func TestCommitment_Hash_CommitmentToNonZeroValuesHasNonZeroHash(t *testing.T) {
	values := [VectorSize]Value{NewValue(12)}

	commitment := Commit(values)
	require.True(t, commitment.IsValid(), "Commitment should be valid")

	hash := commitment.Hash()
	require.NotEqual(t, common.Hash{}, hash, "Hash of non-identity commitment should not be zero")
}

func TestCommitment_Compress_IdentityReturnsZero(t *testing.T) {
	compressed := Identity().Compress()
	require.Equal(t, [32]byte{}, compressed, "Compressed form of identity commitment should be zero")
}

func TestCommitment_Compress_CommitmentToNonZeroValuesHasNonZeroCompressedForm(t *testing.T) {
	values := [VectorSize]Value{NewValue(12)}

	commitment := Commit(values)
	require.True(t, commitment.IsValid(), "Commitment should be valid")

	compressed := commitment.Compress()
	require.NotEqual(t, [32]byte{}, compressed, "Compressed form of non-identity commitment should not be zero")
}

func TestCommitment_UpdateChangesIndividualElements(t *testing.T) {
	require := require.New(t)

	values := [VectorSize]Value{}

	original := Commit(values)
	require.True(original.IsValid())

	old := values[1]
	values[1] = NewValue(42)
	new := values[1]
	recomputed := Commit(values)
	modified := original.Update(1, old, new)

	require.True(modified.IsValid())
	require.True(modified.Equal(recomputed))
}

func TestCommitment_Add_SumsUpCommitments(t *testing.T) {
	require := require.New(t)

	valuesA := [VectorSize]Value{}
	valuesA[0] = NewValue(10)
	valuesA[1] = NewValue(15)
	commitment1 := Commit(valuesA)
	require.True(commitment1.IsValid())

	valuesB := [VectorSize]Value{}
	valuesB[0] = NewValue(25)
	valuesB[2] = NewValue(5)
	commitment2 := Commit(valuesB)
	require.True(commitment2.IsValid())

	sumCommitment := commitment1
	sumCommitment.Add(commitment2)
	require.True(sumCommitment.IsValid())

	expectedValues := [VectorSize]Value{}
	expectedValues[0] = NewValue(35) // 10 + 25
	expectedValues[1] = NewValue(15) // 15 + 0
	expectedValues[2] = NewValue(5)  // 0 + 5
	expectedCommitment := Commit(expectedValues)
	require.True(sumCommitment.Equal(expectedCommitment))
}

func TestCommitment_Add_UsesZeroAsNeutralElement(t *testing.T) {
	require := require.New(t)

	values := [VectorSize]Value{}
	values[0] = NewValue(10)
	values[1] = NewValue(15)
	commitment := Commit(values)
	require.True(commitment.IsValid())

	sum := Commitment{}
	sum.Add(commitment)
	require.True(sum.IsValid())

	require.True(sum.Equal(commitment))
}
