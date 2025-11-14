// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

/// Returns true with the given probability (in percent).
#[allow(dead_code)]
pub fn with_prob(prob: u8) -> bool {
    fastrand::f32() < (prob as f32) / 100.0
}

/// Returns an iterator over powers of 2 up to the number of available threads.
#[allow(dead_code)]
pub fn pow_2_threads() -> impl Iterator<Item = usize> {
    (1..=thread::available_parallelism().unwrap().get()).filter(|x| x.is_power_of_two())
}

/// Executes the given operation in parallel using the specified number of threads.
///
/// Each thread will perform up to iter / num_threads operations, and will receive a unique index
/// starting from `offset`.
/// Thread-specific data can be created using the `op_data` closure, which is passed to the `op`
/// closure.
/// `offset` is updated to reflect the total number of iterations completed.
/// Returns the time elapsed from the first thread starting to the last thread finishing.
pub fn execute_with_threads<T>(
    num_threads: u64,
    iters: u64,
    offset: &mut u64,
    op_data: impl Fn() -> T + Send + Sync,
    op: impl Fn(u64, &mut T) + Send + Sync,
) -> Duration {
    let start_toggle = AtomicBool::new(false);
    thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads as usize);
        for thread_id in 0..num_threads {
            let start_toggle = &start_toggle;
            let completed_iterations = *offset;
            let op_data = &op_data;
            let op = &op;
            handles.push(s.spawn(move || {
                let mut data = op_data();
                while !start_toggle.load(Ordering::Acquire) {}
                let start = Instant::now();
                for iter in ((completed_iterations + thread_id)..(completed_iterations + iters))
                    .step_by(num_threads as usize)
                {
                    op(iter, &mut data);
                }
                let end = Instant::now();
                (start, end)
            }));
        }
        start_toggle.store(true, Ordering::Release);
        let times: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        let first_start = times.iter().map(|(start, _)| *start).min().unwrap();
        let last_end = times.iter().map(|(_, end)| *end).max().unwrap();
        *offset += iters;
        last_end.duration_since(first_start)
    })
}
