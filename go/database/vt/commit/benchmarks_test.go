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

var _random = func() banderwagon.Fr {
	var fr banderwagon.Fr
	fr.SetBytes(hexutil.MustDecode("0x8ace54a66ae992faf22d3eedb0edecff16ded1e168c474263519eb3b388008b4"))
	return fr
}()

func Benchmark_Commit256Elements(b *testing.B) {
	config := getConfig()

	// All elements are set to `random`
	poly := make([]banderwagon.Fr, VectorSize)
	for i := range poly {
		poly[i] = _random
	}
	for b.Loop() {
		config.Commit(poly)
	}
}

func Benchmark_PolySingleUpdate(b *testing.B) {
	for _, i := range []int{0, 1, 2, 3, 4, 5, 32, 64, 128, 255} {
		b.Run(fmt.Sprintf("index=%d", i), func(b *testing.B) {
			benchmark_SinglePoint(b, i, _random)
		})
	}

}

func benchmark_SinglePoint(
	b *testing.B,
	index int,
	value banderwagon.Fr,
) {
	config := getConfig()

	// All 0, but one point is set to `value`
	poly := make([]banderwagon.Fr, VectorSize)
	poly[index] = value
	for b.Loop() {
		config.Commit(poly)
	}
}

func BenchmarkPolySingleUpdateWrapped(b *testing.B) {
	random := Value{scalar: _random}
	for _, i := range []int{0, 1, 2, 3, 4, 5, 32, 64, 128, 255} {
		b.Run(fmt.Sprintf("index=%d", i), func(b *testing.B) {
			benchmarkSinglePointWrapped(b, i, random)
		})
	}

}

func benchmarkSinglePointWrapped(
	b *testing.B,
	index int,
	value Value,
) {
	// All 0, but one point is set to `value`
	poly := [VectorSize]Value{}
	poly[index] = value
	for b.Loop() {
		Commit(poly)
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
