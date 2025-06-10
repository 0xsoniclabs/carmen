// Copyright (c) 2024 Fantom Foundation
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at fantom.foundation/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package state

import (
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"time"

	"testing"
)

// To run this benchmarks, use the following command:
// go test ./state -run none -bench BenchmarkFlushGoState --benchtime 10s

func BenchmarkFlushGoState(b *testing.B) {
	b.StopTimer()
	dir := b.TempDir()
	state, err := NewState(Parameters{
		Variant:   "go-file",
		Schema:    5,
		Archive:   S5Archive,
		Directory: dir,
	})
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := state.Close(); err != nil {
			b.Fatalf("failed to close state %v", err)
		}
	}()

	for n := uint64(0); n < uint64(b.N); n++ {
		update := common.Update{}
		for i := 0; i < 10_000; i++ {
			update.AppendBalanceUpdate(common.Address{
				byte(n >> 24), byte(n >> 16), byte(n >> 8), byte(n),
				byte(i >> 24), byte(i >> 16), byte(i >> 8), byte(i),
			}, amount.New(n))
		}
		err := state.Apply(n, update)
		if err != nil {
			b.Fatal(err)
		}

		b.StartTimer()
		err = state.Flush()
		b.StopTimer()

		if err != nil {
			b.Fatal(err)
		}
	}
}

// Run, for example, with:
// go test . -cpuprofile=cpu.prof -run none -bench Benchmark_Long_vs_Short_Flush_Period/short  --benchtime 10s
// go test . -cpuprofile=cpu.prof -run none -bench Benchmark_Long_vs_Short_Flush_Period/long  --benchtime 10s
// and observe CPU time consumed by 'tryFlushDirtyNodes'
func Benchmark_Long_vs_Short_Flush_Period(b *testing.B) {
	tests := map[string]struct {
		period time.Duration
	}{
		"short": {time.Millisecond},
		"long":  {time.Second},
	}

	for name, config := range tests {
		b.Run(name, func(b *testing.B) {

			parameters := Parameters{
				Variant:               "go-file",
				Schema:                5,
				Archive:               NoArchive,
				LiveCache:             1 << 29, // 0.5 GB
				BackgroundFlushPeriod: config.period,
			}

			s, err := NewState(parameters)
			if err != nil {
				b.Fatalf("failed to initialize state %s; %s", name, err)
			}
			db := CreateStateDBUsing(s)

			db.BeginBlock()
			db.BeginTransaction()

			// Create accounts.
			for i := 0; i < b.N; i++ {
				addr := common.Address{byte(i), byte(i >> 8), byte(i >> 16), byte(i >> 24)}
				db.CreateAccount(addr)
				db.SetNonce(addr, 12)
			}

			db.EndTransaction()
			db.EndBlock(12)

			if err := db.Check(); err != nil {
				b.Errorf("update failed with unexpected error: %v", err)
			}

			if err := db.Close(); err != nil {
				b.Errorf("failed to close DB: %v", err)
			}
		})
	}
}
