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

use carmen_rust::{BalanceUpdate, CarmenDb, Update};
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main,
    measurement::WallTime,
};

// TODO: Use another error type here..?
pub type DatabaseCreationFn = fn(&Path) -> std::io::Result<Arc<dyn CarmenDb>>;

fn create_s6_in_memory_db(path: &Path) -> std::io::Result<Arc<dyn CarmenDb>> {
    let db = carmen_rust::open_carmen_db(
        6,
        carmen_rust::LiveImpl::Memory,
        carmen_rust::ArchiveImpl::None,
        path.to_str().unwrap().as_bytes(),
    )
    .map_err(|e| std::io::Error::other(format!("{e:?}")))?;
    Ok(Arc::from(db))
}

// TODO: Add multi-threaded access patterns
// TODO: Add lookup patterns
// TODO: Write to different fields (for now it doesn't matter since we do not optimize leaf
//       commitment updates)
fn carmen_state_benchmark_matrix(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic);

    let num_keys = [1, 100, 1_000, 10_000, 100_000];

    for nk in num_keys {
        let mut group = c.benchmark_group(format!("carmen_state/store/{nk}_keys",));
        group.plot_config(plot_config.clone());
        carmen_state_benchmark(&mut group, "S6InMemory", create_s6_in_memory_db, nk);
    }
}

// TODO: Write benchmarks for very small components
// - Tree with 3 accounts, only update balance for a single account over and over again
//  - With and without commitment update
// - Or only get_basic_data_key (and other functions)
// - Or only commitment computations
// - etc
//
// Flamegraph: Can we skip criterion stack frames?

fn carmen_state_benchmark(
    g: &mut BenchmarkGroup<'_, WallTime>,
    db_name: &'static str,
    create_db_fn: DatabaseCreationFn,
    num_keys: usize,
) {
    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true);

    // g.throughput(Throughput::Bytes(chunk_size as u64)); // TODO ?
    g.bench_with_input(
        BenchmarkId::from_parameter(db_name),
        // these are passed though [criterion::black_box]
        &(num_keys),
        |b, num_keys| {
            b.iter_custom(|iterations| {
                let _span = tracy_client::span!("carmen_state_benchmark");
                let tempdir = tempfile::tempdir().unwrap();
                let db = create_db_fn(tempdir.path()).unwrap();
                let state = db.get_live_state().unwrap();

                eprintln!("iterations: {iterations}");
                let before = Instant::now();
                for i in 0..iterations {
                    let k = i % *num_keys as u64;
                    let mut addr = [0u8; 20];
                    addr[..8].copy_from_slice(&k.to_le_bytes());
                    // eprintln!("Updating address {addr:x?} with balance {i}");
                    let update = Update {
                        balances: &[BalanceUpdate {
                            addr,
                            balance: [i as u8; 32],
                        }],
                        ..Default::default()
                    };
                    state.apply_block_update(0, update).unwrap();
                    state.get_hash().unwrap(); // <- this should not be outside the loop!
                }
                // => If we want to test commitment updates on larger trees we need to update
                //    multiple keys in a single iteration
                let after = Instant::now();
                eprintln!(
                    "Tree depth: {}, node count: {}",
                    state.depth(),
                    state.node_count()
                );
                after.duration_since(before)
            });
        },
    );
}

criterion_group!(name = benches;  config = Criterion::default().sample_size(10); targets = carmen_state_benchmark_matrix);
criterion_main!(benches);
