// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package archive_test

import (
	"math/rand"
	"testing"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/stretchr/testify/require"
)

const (
	bmAddressToCreate                = 100
	bmBlocksToInsert                 = 1_000
	bmAddressToUseParBlock           = 20
	bmKeysToInsertParAddressAndBlock = 50
)

func BenchmarkAdding(b *testing.B) {
	for _, factory := range getArchiveFactories(b) {
		a := factory.getArchive(b.TempDir())
		defer func() { require.NoError(b, a.Close()) }()

		// initialize
		var update common.Update
		for i := range byte(bmAddressToCreate) {
			update.AppendBalanceUpdate(common.Address{i}, amount.New(uint64(i)))
		}
		if err := a.Add(1, update, nil); err != nil {
			b.Fatalf("failed to add block; %s", err)
		}

		var block uint64 = 2
		b.Run(factory.label, func(b *testing.B) {
			for range bmBlocksToInsert {
				var update common.Update
				for range bmAddressToUseParBlock {
					addr := byte(rand.Intn(bmAddressToCreate))
					for range bmKeysToInsertParAddressAndBlock {
						key := byte(rand.Intn(0xFF))
						update.AppendSlotUpdate(common.Address{addr}, common.Key{key}, common.Value{addr + key})
					}
				}
				if err := update.Normalize(); err != nil {
					b.Fatalf("failed to normalize update; %s", err)
				}
				if err := a.Add(block, update, nil); err != nil {
					b.Fatalf("failed to add block; %s", err)
				}
				block++
			}
			// add flush here if parallel archives are implemented
		})
	}
}
