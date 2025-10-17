// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{fs::OpenOptions, path::Path, sync::Arc, time::Instant};

use carmen_rust::{
    BalanceUpdate, CarmenDb, Update,
    database::verkle::crypto::{Commitment, Scalar},
};
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main,
    measurement::WallTime,
};

// TODO: Add multi-threaded access patterns
// TODO: Add lookup patterns
// TODO: Write to different fields (for now it doesn't matter since we do not optimize leaf
//       commitment updates)
// fn carmen_state_benchmark_matrix(c: &mut Criterion) {
//     let plot_config =
// PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic);

//     let num_keys = [1, 100, 1_000, 10_000, 100_000];

//     for nk in num_keys {
//         let mut group = c.benchmark_group(format!("carmen_state/store/{nk}_keys",));
//         group.plot_config(plot_config.clone());
//         carmen_state_benchmark(&mut group, "S6InMemory", create_s6_in_memory_db, nk);
//     }
// }

// TODO: Write benchmarks for very small components
// - Tree with 3 accounts, only update balance for a single account over and over again
//  - With and without commitment update
// - Or only get_basic_data_key (and other functions)
// - Or only commitment computations
// - etc
//
// Flamegraph: Can we skip criterion stack frames?

// fn carmen_state_benchmark(
//     g: &mut BenchmarkGroup<'_, WallTime>,
//     db_name: &'static str,
//     create_db_fn: DatabaseCreationFn,
//     num_keys: usize,
// ) {
//     let mut options = OpenOptions::new();
//     options.create(true).read(true).write(true);

//     // g.throughput(Throughput::Bytes(chunk_size as u64)); // TODO ?
//     g.bench_with_input(
//         BenchmarkId::from_parameter(db_name),
//         // these are passed though [criterion::black_box]
//         &(num_keys),
//         |b, num_keys| {
//             b.iter_custom(|iterations| {
//                 let _span = tracy_client::span!("carmen_state_benchmark");
//                 let tempdir = tempfile::tempdir().unwrap();
//                 let db = create_db_fn(tempdir.path()).unwrap();
//                 let state = db.get_live_state().unwrap();

//                 eprintln!("iterations: {iterations}");
//                 let before = Instant::now();
//                 for i in 0..iterations {
//                     let k = i % *num_keys as u64;
//                     let mut addr = [0u8; 20];
//                     addr[..8].copy_from_slice(&k.to_le_bytes());
//                     // eprintln!("Updating address {addr:x?} with balance {i}");
//                     let update = Update {
//                         balances: &[BalanceUpdate {
//                             addr,
//                             balance: [i as u8; 32],
//                         }],
//                         ..Default::default()
//                     };
//                     state.apply_block_update(0, update).unwrap();
//                     state.get_hash().unwrap(); // <- this should not be outside the loop!
//                 }
//                 // => If we want to test commitment updates on larger trees we need to update
//                 //    multiple keys in a single iteration
//                 let after = Instant::now();
//                 eprintln!(
//                     "Tree depth: {}, node count: {}",
//                     state.depth(),
//                     state.node_count()
//                 );
//                 after.duration_since(before)
//             });
//         },
//     );
// }

// Different inputs: zero, one, random string (other commitment)
// Update different slots: 1, 2, 4, 8, ...
fn commitment_update_benchmark(c: &mut Criterion) {
    let empty_commitment = Commitment::default().to_scalar();
    // let random_commitment = Commitment::new(&[Scalar::from(1234); 256]).to_scalar();
    let random_commitment = Scalar::from(43);

    let value_index = [0u8, 4u8, 15, 63, 255];
    let previous_commitments = [empty_commitment, random_commitment];

    for i in value_index {
        let mut g = c.benchmark_group(format!("Commitment::update/index={i}"));

        for prev in &previous_commitments {
            g.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "prev={}",
                    if *prev == empty_commitment {
                        "empty"
                    } else {
                        "random"
                    }
                )),
                &(i, prev),
                |b, (slot, prev)| {
                    b.iter(|| {
                        let mut c = Commitment::default();
                        let new = Scalar::from(42);
                        c.update(*slot, **prev, new);
                    });
                },
            );
        }
    }
}

// criterion_group!(name = benches;  config = Criterion::default().sample_size(10); targets =
// carmen_state_benchmark_matrix);
criterion_group!(name = benches;  config = Criterion::default(); targets = commitment_update_benchmark);
criterion_main!(benches);
