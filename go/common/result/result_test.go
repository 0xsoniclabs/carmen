// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package result

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestResult_Ok_ProducesResultWithValue(t *testing.T) {
	result := Ok[int](42)
	value, err := result.Get()
	require.NoError(t, err)
	require.Equal(t, 42, value)
}

func TestResult_Err_ProducesResultWithError(t *testing.T) {
	issue := fmt.Errorf("test error")
	result := Err[int](issue)
	value, err := result.Get()
	require.ErrorIs(t, err, issue)
	require.Zero(t, value)
}
