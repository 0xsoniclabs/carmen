// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use carmen_rust::database::verkle::crypto::{Commitment, Scalar};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sha3::{Digest, Keccak256};

fn commitment_benchmark(c: &mut Criterion) {
    let random_bytes =
        &const_hex::decode("46123387734d09ce6f083425adf3b9d8bf70359cdae94686794a4a23ac478000")
            .unwrap();
    let random = Scalar::from_le_bytes(random_bytes);

    let mut g = c.benchmark_group("keccak_vs_pedersen/single_value");
    g.bench_function(BenchmarkId::from_parameter("pedersen"), |b| {
        let mut scalars = vec![Scalar::zero(); 64];
        scalars[63] = random;
        b.iter(|| {
            std::hint::black_box(Commitment::new(&scalars));
        });
    });
    g.bench_function(BenchmarkId::from_parameter("keccak"), |b| {
        b.iter(|| {
            let mut hasher = Keccak256::new();
            hasher.update(random_bytes);
            std::hint::black_box(hasher.finalize());
        });
    });
    drop(g);

    // Compare hashing a full Verkle node (256 children) vs a full MPT node (16 children)
    let mut g = c.benchmark_group("keccak_vs_pedersen/full_node");
    g.bench_function(BenchmarkId::from_parameter("verkle"), |b| {
        let scalars = vec![random; 256];
        b.iter(|| {
            std::hint::black_box(Commitment::new(&scalars));
        });
    });
    g.bench_function(BenchmarkId::from_parameter("mpt"), |b| {
        let node_bytes: Vec<u8> = (0..16).flat_map(|_| random_bytes).copied().collect();
        b.iter(|| {
            let mut hasher = Keccak256::new();
            hasher.update(&node_bytes);
            std::hint::black_box(hasher.finalize());
        });
    });
}

criterion_group!(name = benches;  config = Criterion::default(); targets = commitment_benchmark);
criterion_main!(benches);
