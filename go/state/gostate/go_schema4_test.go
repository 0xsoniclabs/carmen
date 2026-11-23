// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package gostate

import (
	"testing"

	"github.com/0xsoniclabs/carmen/go/state"
	"github.com/stretchr/testify/require"
)

func TestScheme4_NewState_LimitsNodeCacheToLiveCache(t *testing.T) {
	const liveCache = 1 << 27

	tests := map[string]struct {
		variant state.Variant
		factory state.StateFactory
	}{
		"memory": {
			variant: VariantGoMemory,
			factory: newGoMemoryS4State,
		},
		"file": {
			variant: VariantGoFile,
			factory: newGoFileS4State,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)
			s, err := test.factory(state.Parameters{
				Directory: t.TempDir(),
				Variant:   test.variant,
				Schema:    4,
				LiveCache: liveCache,
			})
			require.NoError(err)
			defer func() {
				require.NoError(s.Close())
			}()

			cache := s.GetMemoryFootprint().GetChild("live").GetChild("trie").GetChild("forest").GetChild("cache")
			require.NotNil(cache)
			require.LessOrEqual(cache.Total(), uintptr(liveCache))
		})
	}
}
