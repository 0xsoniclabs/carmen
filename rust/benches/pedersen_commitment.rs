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

fn commitment_benchmark(c: &mut Criterion) {
    let random = Scalar::from_le_bytes(
        &const_hex::decode("8ace54a66ae992faf22d3eedb0edecff16ded1e168c474263519eb3b388008b4")
            .unwrap(),
    );

    // The first five (0..=4) indices are particularly important for key computations
    // (state embedding) and typically use lookup tables with larger window sizes.
    let scenarios = [
        ("0=zero", vec![(0, Scalar::zero())]),
        ("0=random", vec![(0, random)]),
        ("4=random", vec![(4, random)]),
        ("32=random", vec![(32, random)]),
        ("64=random", vec![(64, random)]),
        ("255=random", vec![(255, random)]),
        (
            "0,1,8,16,64,128=random",
            vec![
                (0, random),
                (1, random),
                (8, random),
                (16, random),
                (64, random),
                (128, random),
            ],
        ),
        (
            "0..64=random",
            (0..64).map(|i| (i, random)).collect::<Vec<_>>(),
        ),
        (
            "0..256=random",
            (0..=255).map(|i| (i, random)).collect::<Vec<_>>(),
        ),
    ];

    let mut reused_commit = Commitment::default();

    for (name, index_values) in scenarios {
        let mut g = c.benchmark_group(format!("pedersen_commitment/{name}"));

        g.bench_with_input(
            BenchmarkId::from_parameter("new"),
            &index_values,
            |b, index_values| {
                let max_index = index_values.iter().map(|(slot, _)| *slot).max().unwrap();
                let mut scalars = vec![Scalar::zero(); max_index as usize + 1];
                for (slot, new) in index_values.iter() {
                    scalars[*slot as usize] = *new;
                }
                b.iter(|| {
                    Commitment::new(&scalars);
                });
            },
        );

        g.bench_with_input(
            BenchmarkId::from_parameter("update"),
            &index_values,
            |b, index_values| {
                b.iter(|| {
                    let mut commit = Commitment::default();
                    for (slot, new) in index_values.iter() {
                        commit.update(*slot, Scalar::zero(), *new);
                    }
                });
            },
        );

        g.bench_with_input(
            BenchmarkId::from_parameter("update existing"),
            &index_values,
            |b, index_values| {
                b.iter(|| {
                    for (slot, new) in index_values.iter() {
                        reused_commit.update(*slot, Scalar::zero(), *new);
                    }
                });
            },
        );
    }
}

criterion_group!(name = benches;  config = Criterion::default(); targets = commitment_benchmark);
criterion_main!(benches);
