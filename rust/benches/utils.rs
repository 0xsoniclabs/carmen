use std::{
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

/// Executes the given operation in parallel using the specified number of threads.
/// Thread-local data can be created using the `op_data` closure, and iteration id can be customized
/// using the `get_id` closure.
/// Each thread will perform its share of iterations, starting only after all threads are ready.
/// Returns the total duration taken to complete all iterations across all threads.
pub fn execute_with_threads<T>(
    num_threads: u64,
    iters: u64,
    get_id: impl Fn(&u64) -> u64 + Send + Sync,
    completed_iterations: &mut u64,
    op_data: impl Fn() -> T + Send + Sync,
    op: impl Fn(u64, &mut T) + Send + Sync,
) -> Duration {
    let start_toggle = AtomicBool::new(false);
    thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads as usize);
        for thread_id in 0..num_threads {
            let start_toggle = &start_toggle;
            let completed_iterations = *completed_iterations;
            let get_id = &get_id;
            let op_data = &op_data;
            let op = &op;
            handles.push(s.spawn(move || {
                let mut data = op_data();
                while !start_toggle.load(Ordering::Acquire) {}
                let start = Instant::now();
                for iter in ((completed_iterations + thread_id)..(completed_iterations + iters))
                    .step_by(num_threads as usize)
                {
                    let iter = get_id(&iter);
                    op(iter, &mut data);
                }
                let end = Instant::now();
                (start, end)
            }));
        }
        start_toggle.store(true, Ordering::Release);
        // Get the time between the first start and the last end
        let times: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        let first_start = times.iter().map(|(start, _)| *start).min().unwrap();
        let last_end = times.iter().map(|(_, end)| *end).max().unwrap();
        *completed_iterations += iters;
        last_end.duration_since(first_start)
    })
}
