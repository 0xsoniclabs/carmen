// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use carmen_rust::{
    CarmenState, Update,
    database::{
        CrateCryptoInMemoryVerkleTrie, SimpleInMemoryVerkleTrie, verkle::VerkleTrieCarmenState,
    },
    types::SlotUpdate,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use utils::{bench_single_call, execute_with_threads};

use crate::utils::pow_2_threads;

mod utils;

/// An enum representing the different CarmenState implementations to benchmark.
#[derive(Debug, Copy, Clone)]
enum CarmenStateKind {
    CrateCryptoInMemoryVerkleTrie,
    SimpleInMemoryVerkleTrie,
}

impl CarmenStateKind {
    /// Constructs the corresponding CarmenState instance.
    fn make_carmen_state(self) -> Box<dyn CarmenState> {
        match self {
            CarmenStateKind::SimpleInMemoryVerkleTrie => {
                Box::new(VerkleTrieCarmenState::<SimpleInMemoryVerkleTrie>::new())
                    as Box<dyn CarmenState>
            }
            CarmenStateKind::CrateCryptoInMemoryVerkleTrie => {
                Box::new(VerkleTrieCarmenState::<CrateCryptoInMemoryVerkleTrie>::new())
                    as Box<dyn CarmenState>
            }
        }
    }

    fn variants() -> &'static [CarmenStateKind] {
        &[
            CarmenStateKind::SimpleInMemoryVerkleTrie,
            CarmenStateKind::CrateCryptoInMemoryVerkleTrie,
        ]
    }
}

/// An enum representing the initial size of the carmen state.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum InitialState {
    Empty,
    Small,
    Large,
    Test,
}

impl InitialState {
    fn num_accounts(self) -> usize {
        match self {
            InitialState::Empty => 0,
            InitialState::Small => 1000,
            InitialState::Large => 10000,
            InitialState::Test => 1,
        }
    }

    fn num_storage_keys(self) -> usize {
        match self {
            InitialState::Empty => 0,
            InitialState::Small => 5,
            InitialState::Large => 10,
            InitialState::Test => 1,
        }
    }

    /// Initializes the Carmen state with the current initial state.
    /// Accounts and keys are incrementally generated.
    fn init(self, carmen_state: &dyn CarmenState) {
        if matches!(self, InitialState::Empty) {
            return;
        }
        let num_accounts = self.num_accounts();
        let num_storage_keys = self.num_storage_keys();
        let buffer_size = usize::max(1, usize::min(num_accounts / 10, 10000));
        for i in (0..num_accounts).step_by(buffer_size) {
            let mut slots_update = vec![];
            for account_index in i..usize::min(i + buffer_size, num_accounts) {
                let account_address = {
                    let mut addr = [0u8; 20];
                    addr[0..8].copy_from_slice(&account_index.to_be_bytes());
                    addr
                };
                for storage_index in 0..num_storage_keys {
                    let storage_key = {
                        let mut key = [0u8; 32];
                        key[0..8].copy_from_slice(&storage_index.to_be_bytes());
                        key
                    };
                    slots_update.push(SlotUpdate {
                        addr: account_address,
                        key: storage_key,
                        value: [1u8; 32],
                    });
                }
            }

            carmen_state
                .apply_block_update(
                    0,
                    Update {
                        slots: &slots_update,
                        ..Default::default()
                    },
                )
                .expect("Failed to initialize Carmen state");
        }
    }

    fn variants() -> &'static [InitialState] {
        &[
            InitialState::Empty,
            InitialState::Small,
            InitialState::Large,
        ]
    }
}

/// Benchmark reading storage values from the Carmen state.
/// It varies:
/// - Initial state size (Small, Large)
/// - Whether to read existing storage keys or non-existing ones
/// - Number of threads
fn state_read_benchmark(c: &mut criterion::Criterion) {
    let initial_states = if cfg!(debug_assertions) {
        vec![InitialState::Test]
    } else {
        vec![InitialState::Small, InitialState::Large]
    };

    for initial_state in initial_states {
        for existing in [true, false] {
            let mut group = c.benchmark_group(format!(
                "carmen_state_read/{initial_state:?}/in_storage={existing}"
            ));
            for state_type in CarmenStateKind::variants() {
                let carmen_state = state_type.make_carmen_state();
                initial_state.init(&*carmen_state);
                for num_threads in pow_2_threads() {
                    let mut completed_iterations = 0u64;
                    group.bench_with_input(
                        BenchmarkId::from_parameter(format!("{state_type:?}/{num_threads}threads")),
                        &num_threads,
                        |b, &num_threads| {
                            let num_accounts = initial_state.num_accounts() as u64;
                            let num_storage_keys = initial_state.num_storage_keys() as u64;
                            b.iter_custom(|iters| {
                                execute_with_threads(
                                    num_threads as u64,
                                    iters,
                                    &mut completed_iterations,
                                    || (),
                                    |iter, _| {
                                        let account_index = iter % num_accounts;
                                        let storage_index = if existing {
                                            iter % num_storage_keys
                                        } else {
                                            iter + num_storage_keys
                                        };
                                        let account_address = {
                                            let mut addr = [0u8; 20];
                                            addr[0..8]
                                                .copy_from_slice(&account_index.to_be_bytes());
                                            addr
                                        };
                                        let storage_key = {
                                            let mut key = [0u8; 32];
                                            key[0..8].copy_from_slice(&storage_index.to_be_bytes());
                                            key
                                        };
                                        let _value = carmen_state
                                            .get_storage_value(&account_address, &storage_key)
                                            .unwrap();
                                    },
                                )
                            });
                        },
                    );
                }
            }
        }
    }
}

/// Benchmark updating storage values in the Carmen state.
/// It varies:
/// - Initial state size (Empty, Small, Large)
/// - Whether to update existing storage keys or new ones
/// - Number of batches (how many updates share the same address)
fn state_update_benchmark(c: &mut criterion::Criterion) {
    let (num_key_to_update, num_batches, initial_states) = if cfg!(debug_assertions) {
        (10, vec![1, 10], vec![InitialState::Test])
    } else {
        (
            1_000_000,                                // Total number of storage keys to update
            vec![1, 10, 100, 1_000, 10_000, 100_000], // Number of batches
            InitialState::variants().to_vec(),        // Initial states to test
        )
    };

    for initial_state in initial_states {
        let mut group = c.benchmark_group(format!("carmen_state_update/{initial_state:?}"));
        group.sample_size(10); // This is the minimum allowed by criterion
        for num_batch in num_batches.clone() {
            for existing in [true, false] {
                if initial_state == InitialState::Empty && existing {
                    continue;
                }
                for state_type in CarmenStateKind::variants() {
                    // TODO: we could keep a copy of the initial state and clone it here instead of
                    // re-initializing, but it would basically double the memory usage.
                    let init = move || {
                        let carmen_state = state_type.make_carmen_state();
                        initial_state.init(&*carmen_state);
                        let num_accounts = initial_state.num_accounts();
                        let num_storage_keys = initial_state.num_storage_keys() as u64;
                        let mut slots_update = Vec::with_capacity(num_key_to_update as usize);
                        let keys_per_batch = num_key_to_update / num_batch;
                        for (account, _) in (0..num_key_to_update)
                            .step_by(keys_per_batch as usize)
                            .enumerate()
                        {
                            let account_idx = if existing {
                                account % num_accounts
                            } else {
                                account + num_accounts
                            };
                            let mut address = [0u8; 20];
                            address[0..8].copy_from_slice(&account_idx.to_be_bytes());
                            for key_idx in 0..keys_per_batch {
                                let key_idx = if existing {
                                    key_idx % num_storage_keys
                                } else {
                                    key_idx + num_storage_keys
                                };
                                let mut key = [0u8; 32];
                                key[0..8].copy_from_slice(&key_idx.to_be_bytes());
                                slots_update.push(SlotUpdate {
                                    addr: address,
                                    key,
                                    value: [1u8; 32],
                                });
                            }
                        }
                        (carmen_state, slots_update)
                    };
                    bench_single_call(
                        &mut group,
                        format!("{state_type:?}/{num_batch}batches/{existing}_existing").as_str(),
                        init,
                        |(carmen_state, slots_update)| {
                            carmen_state
                                .apply_block_update(
                                    0,
                                    Update {
                                        slots: slots_update,
                                        ..Default::default()
                                    },
                                )
                                .unwrap();
                        },
                    );
                }
            }
        }
    }
}

criterion_group!(name = carmen_state;  config = Criterion::default(); targets = state_read_benchmark, state_update_benchmark);
criterion_main!(carmen_state);
