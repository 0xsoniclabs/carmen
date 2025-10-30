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

func Benchmark_PolySingleUpdate(b *testing.B) {
	var random banderwagon.Fr
	random.SetBytes(hexutil.MustDecode("0x8ace54a66ae992faf22d3eedb0edecff16ded1e168c474263519eb3b388008b4"))
	for _, i := range []int{0, 1, 2, 3, 4, 5, 32, 64, 128, 255} {
		b.Run(fmt.Sprintf("index=%d", i), func(b *testing.B) {
			benchmark_SinglePoint(b, i, random)
		})
	}

}

func benchmark_SinglePoint(
	b *testing.B,
	index int,
	value banderwagon.Fr,
) {
	config := getConfig() // < the polynomial commit "engine"

	// All 0, but one point is set to `value`
	poly := make([]banderwagon.Fr, VectorSize)
	poly[index] = value
	for b.Loop() {
		config.Commit(poly)
	}
}

func BenchmarkPolySingleUpdateWrapped(b *testing.B) {
	bytes := hexutil.MustDecode("0x8ace54a66ae992faf22d3eedb0edecff16ded1e168c474263519eb3b388008b4")
	random := NewValueFromLittleEndianBytes(bytes)
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
