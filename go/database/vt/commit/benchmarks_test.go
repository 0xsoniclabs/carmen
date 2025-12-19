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
	"sync"
	"testing"

	"github.com/crate-crypto/go-ipa/banderwagon"
	"github.com/crate-crypto/go-ipa/ipa"
	"github.com/ethereum/go-ethereum/common/hexutil"
)

// To run the benchmarks in this package use:
//
//  go test ./database/vt/commit -run='^$' -bench=.
//
// To get aggregated statistics, install benchstat using
//
//  go install golang.org/x/perf/cmd/benchstat@latest
//
// and then run
//
//  go test ./database/vt/commit -run='^$' -bench=. -count 10 > bench.txt
//
// and finally
//
//  benchstat bench.txt
//
// Sample output:
//
//                                     │  bench.txt  │
//                                     │   sec/op    │
//      _PolySingleUpdate/index=0-12     7.911µ ± 5%
//      _PolySingleUpdate/index=1-12     7.891µ ± 1%
//      _PolySingleUpdate/index=2-12     7.871µ ± 4%
//      _PolySingleUpdate/index=3-12     7.861µ ± 5%
//      _PolySingleUpdate/index=4-12     7.811µ ± 4%
//      _PolySingleUpdate/index=5-12     15.40µ ± 7%
//      _PolySingleUpdate/index=32-12    15.32µ ± 7%
//      _PolySingleUpdate/index=64-12    15.40µ ± 7%
//      _PolySingleUpdate/index=128-12   15.25µ ± 7%
//      _PolySingleUpdate/index=255-12   15.37µ ± 7%
//      geomean                          10.99µ
//

func Benchmark_PolySingleUpdate(b *testing.B) {
	var random banderwagon.Fr
	random.SetBytesLE(hexutil.MustDecode("0x46123387734d09ce6f083425adf3b9d8bf70359cdae94686794a4a23ac478000"))
	config := getConfig() // < the polynomial commit "engine"

	for _, i := range []int{0, 4, 32, 64, 255} {
		b.Run(fmt.Sprintf("index=%d", i), func(b *testing.B) {
			// All 0, but one point is set to `random`
			poly := make([]banderwagon.Fr, VectorSize)
			poly[i] = random

			for b.Loop() {
				config.Commit(poly)
			}
		})
	}
}

func Benchmark_PolyMultiCommit(b *testing.B) {
	var random banderwagon.Fr
	random.SetBytesLE(hexutil.MustDecode("0x46123387734d09ce6f083425adf3b9d8bf70359cdae94686794a4a23ac478000"))
	config := getConfig() // < the polynomial commit "engine"

	for _, i := range []int{64, 256} {
		b.Run(fmt.Sprintf("values=%d", i), func(b *testing.B) {
			poly := make([]banderwagon.Fr, VectorSize)
			for j := 0; j < i; j++ {
				poly[j] = random
			}
			for b.Loop() {
				config.Commit(poly)
			}
		})
	}
}

func Benchmark_CommitAdd(b *testing.B) {
	var random banderwagon.Fr
	random.SetBytesLE(hexutil.MustDecode("0x46123387734d09ce6f083425adf3b9d8bf70359cdae94686794a4a23ac478000"))
	config := getConfig() // < the polynomial commit "engine"

	poly := make([]banderwagon.Fr, VectorSize)
	for i := 0; i < VectorSize; i++ {
		poly[i] = random
	}
	c1 := config.Commit(poly)

	var c1v banderwagon.Fr
	c1.MapToScalarField(&c1v)
	for i := 0; i < VectorSize; i++ {
		poly[i] = c1v
	}
	c2 := config.Commit(poly)

	for b.Loop() {
		var res banderwagon.Element
		res.Add(&c1, &c2)
	}
}

var (
	config  *ipa.IPAConfig
	onceCfg sync.Once
)

func getConfig() *ipa.IPAConfig {
	onceCfg.Do(func() {
		cfg, err := ipa.NewIPASettings()
		if err != nil {
			panic(err)
		}
		config = cfg
	})
	return config
}
