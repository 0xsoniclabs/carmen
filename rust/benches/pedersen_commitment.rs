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

// TODO:
// - Benchmark default committer, aggressive lut, psalz fork
//      - For different indices (0, 64, 256)
// - Also benchmark addition
// - Unrelated, but also: Benchmark Keccak-256

fn commitment_benchmark(c: &mut Criterion) {
    let random = Scalar::from_le_bytes(
        &const_hex::decode("46123387734d09ce6f083425adf3b9d8bf70359cdae94686794a4a23ac478000")
            .unwrap(),
    );

    // The first five (0..=4) indices are particularly important for key computations
    // (state embedding) and typically use lookup tables with larger window sizes.
    let scenarios = [
        ("0=zero", vec![(0, Scalar::zero())], None),
        (
            "0=random",
            vec![(0, random)],
            Some("364393d8d69646260335a452e38b49fdce6d32850d712faf6a2982b9b98c255a"),
        ),
        (
            "4=random",
            vec![(4, random)],
            Some("2fc94f4eb058035b01978a324241946a766816598af7bb866c86fb51a27895e0"),
        ),
        (
            "32=random",
            vec![(32, random)],
            Some("54a4e9c0286695a9c4478d5cc65342f368aa2098b6ca568d16935897c99d87ec"),
        ),
        (
            "64=random",
            vec![(64, random)],
            Some("11128a4e312db963fb47b5ff6f3ec627ea1b0874195f1468128089d144f2a177"),
        ),
        (
            "255=random",
            vec![(255, random)],
            Some("1df1156afcbe469b05ac7e64dcfcdb1d7815723a57f2278c62effe446cfccb12"),
        ),
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
            None,
        ),
        (
            "0..64=random",
            (0..64).map(|i| (i, random)).collect::<Vec<_>>(),
            None,
        ),
        (
            "0..256=random",
            (0..=255).map(|i| (i, random)).collect::<Vec<_>>(),
            None,
        ),
    ];

    let mut reused_commit = Commitment::default();

    for (name, index_values, expected) in scenarios {
        let mut g = c.benchmark_group(format!("pedersen_commitment/{name}"));

        g.bench_with_input(
            BenchmarkId::from_parameter("new"),
            &index_values,
            |b, index_values| {
                let max_index = index_values.iter().map(|(index, _)| *index).max().unwrap();
                let mut scalars = vec![Scalar::zero(); max_index as usize + 1];
                for (index, value) in index_values.iter() {
                    scalars[*index as usize] = *value;
                }
                b.iter(|| {
                    let commit = Commitment::new(&scalars);
                    // if let Some(expected) = expected {
                    //     assert_eq!(const_hex::encode(commit.compress()), expected);
                    // }
                });
            },
        );

        g.bench_with_input(
            BenchmarkId::from_parameter("update"),
            &index_values,
            |b, index_values| {
                b.iter(|| {
                    let mut commit = Commitment::default();
                    for (index, value) in index_values.iter() {
                        commit.update(*index, Scalar::zero(), *value);
                    }
                    // if let Some(expected) = expected {
                    //     assert_eq!(const_hex::encode(commit.compress()), expected);
                    // }
                });
            },
        );

        // reused_commit = Commitment::default();
        // g.bench_with_input(
        //     BenchmarkId::from_parameter("update existing"),
        //     &index_values,
        //     |b, index_values| {
        //         b.iter(|| {
        //             for (index, value) in index_values.iter() {
        //                 reused_commit.update(*index, Scalar::zero(), *value);
        //             }
        //             if let Some(expected) = expected {
        //                 assert_eq!(const_hex::encode(reused_commit.compress()), expected);
        //             }
        //         });
        //     },
        // );
    }
}

// TODO: It may be interesting to compare commitments that have zero values for some of the points
fn commitment_add_benchmark(c: &mut Criterion) {
    let random = Scalar::from_le_bytes(
        &const_hex::decode("46123387734d09ce6f083425adf3b9d8bf70359cdae94686794a4a23ac478000")
            .unwrap(),
    );
    let c1 = Commitment::new(&(0..=255).map(|_| random).collect::<Vec<_>>());
    let c2 = Commitment::new(&(0..=255).map(|_| c1.to_scalar()).collect::<Vec<_>>());

    c.bench_function("commitment_add", |b| {
        b.iter(|| {
            let c3 = c1 + c2;

            // Generated with Go reference implementation
            // let expected = "278d27abd1c02de16b63968d075a3c4d5cca8abe3567b4ed3d4afca5962a2b28";
            // assert_eq!(const_hex::encode(c3.compress()), expected);

            std::hint::black_box(c3);
        });
    });
}

criterion_group!(name = benches;  config = Criterion::default(); targets = commitment_benchmark, commitment_add_benchmark);
criterion_main!(benches);
