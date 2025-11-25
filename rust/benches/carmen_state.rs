// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{cell::LazyCell, sync::Arc};

use carmen_rust::{
    CarmenState, Update,
    database::{
        CrateCryptoInMemoryVerkleTrie, SimpleInMemoryVerkleTrie,
        verkle::{
            VerkleTrieCarmenState,
            variants::managed::{VerkleNode, VerkleNodeId},
        },
    },
    node_manager::in_memory_node_manager::InMemoryNodeManager,
    types::SlotUpdate,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use utils::{bench_expensive_call_with_state_mutation, execute_with_threads};

use crate::utils::pow_2_threads;

mod utils;

/// An enum representing the different CarmenState implementations to benchmark.
#[derive(Debug, Copy, Clone, PartialEq)]
#[allow(clippy::enum_variant_names)]
enum CarmenStateKind {
    CrateCryptoInMemoryVerkleTrie,
    SimpleInMemoryVerkleTrie,
    ManagedInMemoryVerkleTrie,
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
            CarmenStateKind::ManagedInMemoryVerkleTrie => {
                let mem_node_manager = Arc::new(
                    InMemoryNodeManager::<VerkleNodeId, VerkleNode>::new(200_000),
                );
                Box::new(
                    VerkleTrieCarmenState::<
                        carmen_rust::database::ManagedVerkleTrie<
                            InMemoryNodeManager<VerkleNodeId, VerkleNode>,
                        >,
                    >::try_new(mem_node_manager)
                    .unwrap(),
                ) as Box<dyn CarmenState>
            }
        }
    }

    fn variants() -> &'static [CarmenStateKind] {
        &[
            CarmenStateKind::ManagedInMemoryVerkleTrie,
            CarmenStateKind::SimpleInMemoryVerkleTrie,
            CarmenStateKind::CrateCryptoInMemoryVerkleTrie,
        ]
    }
}

/// An enum representing the initial size of the carmen state.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum InitialState {
    Small,
    Large,
    Test,
}

impl InitialState {
    fn num_accounts(self) -> usize {
        match self {
            InitialState::Small => 1000,
            InitialState::Large => 10000,
            InitialState::Test => 1,
        }
    }

    fn num_storage_keys(self) -> usize {
        match self {
            InitialState::Small => 5,
            InitialState::Large => 10,
            InitialState::Test => 1,
        }
    }

    /// Initializes the Carmen state with the current initial state.
    /// Accounts and keys are incrementally generated.
    fn init(self, carmen_state: &dyn CarmenState) {
        let num_accounts = self.num_accounts();
        let num_storage_keys = self.num_storage_keys();
        let mut slots_update = vec![];
        for i in 0..num_accounts {
            let account_address = {
                let mut addr = [0u8; 20];
                addr[0..8].copy_from_slice(&i.to_be_bytes());
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
            .expect("failed to initialize Carmen state");
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
        let mut group = c.benchmark_group(format!("carmen_state_read/{initial_state:?}"));
        for state_type in CarmenStateKind::variants() {
            let carmen_state = LazyCell::new(|| {
                let state = state_type.make_carmen_state();
                initial_state.init(&*state);
                state
            });
            for num_threads in pow_2_threads(Some(4)) {
                let mut completed_iterations = 0u64;
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!("{state_type:?}/{num_threads:02}threads")),
                    &num_threads,
                    |b, &num_threads| {
                        let num_accounts = initial_state.num_accounts() as u64;
                        let num_storage_keys = initial_state.num_storage_keys() as u64;
                        let carmen_state = &*carmen_state;
                        b.iter_custom(|iters| {
                            execute_with_threads(
                                num_threads as u64,
                                iters,
                                &mut completed_iterations,
                                || (),
                                |iter, _| {
                                    let account_index = iter % num_accounts;
                                    let storage_index = iter % num_storage_keys;
                                    let account_address = {
                                        let mut addr = [0u8; 20];
                                        addr[0..8].copy_from_slice(&account_index.to_be_bytes());
                                        addr
                                    };
                                    let storage_key = {
                                        let mut key = [0u8; 32];
                                        key[0..8].copy_from_slice(&storage_index.to_be_bytes());
                                        key
                                    };
                                    std::hint::black_box(
                                        carmen_state
                                            .get_storage_value(&account_address, &storage_key)
                                            .unwrap(),
                                    );
                                },
                            )
                        });
                    },
                );
            }
        }
    }
}

/// Benchmark that creates a Carmen state from scratch, applying a number of slot updates.
/// It varies:
/// - Batch size of slot updates (whether all updates are applied in one batch or multiple)
/// - Whether to compute the commitment hash after all updates
fn state_update_benchmark(c: &mut criterion::Criterion) {
    const NUM_KEYS: usize = 10_000;
    const NUM_ACCOUNTS: usize = 100;

    let batch_sizes = if cfg!(debug_assertions) {
        vec![1, 10]
    } else {
        vec![10_000, 100]
    };

    let all_slot_updates = {
        let mut updates = Vec::with_capacity(NUM_KEYS);
        for account_idx in 0..NUM_ACCOUNTS {
            let mut address = [0u8; 20];
            address[0..8].copy_from_slice(&account_idx.to_be_bytes());

            for key_idx in 0..(NUM_KEYS / NUM_ACCOUNTS) {
                let mut key = [0u8; 32];
                key[0..8].copy_from_slice(&(key_idx as u64).to_be_bytes());
                updates.push(SlotUpdate {
                    addr: address,
                    key,
                    value: [1u8; 32],
                });
            }
        }
        updates
    };

    for compute_commitment in [true, false] {
        let mut group = c.benchmark_group(format!(
            "carmen_state_insert/{NUM_KEYS}keys/commit={compute_commitment:?}"
        ));
        group.sample_size(10); // This is the minimum allowed by criterion
        for batch_size in batch_sizes.clone() {
            for state_type in CarmenStateKind::variants() {
                if !compute_commitment
                    && *state_type == CarmenStateKind::CrateCryptoInMemoryVerkleTrie
                {
                    // Crate crypto always computes commitments during update, so we skip this case.
                    continue;
                }

                let all_slot_updates = &all_slot_updates;
                let init = move || {
                    let carmen_state = state_type.make_carmen_state();
                    let mut batches = Vec::with_capacity(NUM_KEYS / batch_size);
                    for b in 0..(NUM_KEYS / batch_size) {
                        let mut batch = Vec::with_capacity(batch_size);
                        // We select slot updates using a stride to ensure that smaller batches
                        // contain updates spread across all accounts.
                        for u in all_slot_updates
                            .iter()
                            .skip(b)
                            .step_by(NUM_KEYS / batch_size)
                        {
                            batch.push(u.clone());
                        }
                        batches.push(batch);
                    }
                    (carmen_state, batches)
                };
                bench_expensive_call_with_state_mutation(
                    &mut group,
                    format!("{state_type:?}/batch_size={batch_size}").as_str(),
                    init,
                    |(carmen_state, batches)| {
                        for b in batches {
                            carmen_state
                                .apply_block_update(
                                    0,
                                    Update {
                                        slots: b,
                                        ..Default::default()
                                    },
                                )
                                .unwrap();
                        }
                        if compute_commitment {
                            std::hint::black_box(carmen_state.get_hash().unwrap());
                        }
                    },
                );
            }
        }
    }
}

criterion_group!(name = carmen_state;  config = Criterion::default(); targets = state_read_benchmark, state_update_benchmark);
criterion_main!(carmen_state);
