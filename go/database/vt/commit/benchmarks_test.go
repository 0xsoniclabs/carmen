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
