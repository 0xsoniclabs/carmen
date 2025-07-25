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
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCommitment_CommittedValuesCanBeUsedToProofValues(t *testing.T) {
	values := [VectorSize]Value{}
	for i := range uint64(VectorSize) {
		values[i] = NewValue(i + 1)
	}

	commitment := Commit(values)
	require.True(t, commitment.IsValid())

	for i, value := range values {
		t.Run(fmt.Sprintf("pos=%d", i), func(t *testing.T) {
			t.Parallel()
			require := require.New(t)
			pos := byte(i)
			opening, err := Open(commitment, values, pos)
			require.NoError(err, "Opening should not return an error")

			// Verify that the opening can verify the committed value.
			valid, err := opening.Verify(commitment, pos, value)
			require.NoError(err, "Verification should not return an error")
			require.True(valid, "Opening should verify for committed value")

			// Verify that the opening does not verify another value.
			valid, err = opening.Verify(commitment, pos, NewValue(uint64(i+2)))
			require.NoError(err, "Verification should not return an error")
			require.False(valid, "Opening should not verify for different value")
		})
	}
}
