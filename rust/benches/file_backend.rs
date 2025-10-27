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
    fmt::Display,
    fs::{self, OpenOptions},
    io::Write,
    ops::Deref,
    path::Path,
    sync::Arc,
};

#[cfg(unix)]
use carmen_rust::storage::file::{MultiPageCachedFile, NoSeekFile, PageCachedFile};
use carmen_rust::{
    error::BTResult,
    storage::file::{FileBackend, SeekFile},
};
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main, measurement::WallTime,
};

use crate::utils::execute_with_threads;

mod utils;

const ONE_GB: usize = 1024 * 1024 * 1024;
const FILE_SIZE_MEMORY_MULTIPLIER: u64 = 4;
const FILE: &str = "benchmark_file_backend.bin";

/// Defines the access pattern for the benchmark.
#[derive(Debug, Clone, Copy)]
enum AccessPattern {
    /// Always use the offset 0.
    Static,
    /// Advances the offset by the chunk size.
    Linear,
    /// Use a random offset.
    Random,
}

impl AccessPattern {
    /// Returns all access patterns.
    fn variants() -> impl IntoIterator<Item = AccessPattern> {
        [
            AccessPattern::Static,
            AccessPattern::Linear,
            AccessPattern::Random,
        ]
    }

    /// Returns the offset for the given iteration and chunk size.
    fn offset(self, iter: u64, chunk_size: usize, file_size: usize) -> u64 {
        match self {
            AccessPattern::Static => 0,
            AccessPattern::Linear => {
                ((iter * chunk_size as u64) % file_size as u64) / chunk_size as u64
                    * chunk_size as u64
            }
            AccessPattern::Random => {
                // splitmix64
                let rand = iter + 0x9e3779b97f4a7c15;
                let rand = (rand ^ (rand >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                let rand = (rand ^ (rand >> 27)).wrapping_mul(0x94d049bb133111eb);
                let rand = rand ^ (rand >> 31);
                ((rand.wrapping_mul(chunk_size as u64)) % file_size as u64) / chunk_size as u64
                    * chunk_size as u64
            }
        }
    }
}

impl Display for AccessPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccessPattern::Static => write!(f, "static"),
            AccessPattern::Linear => write!(f, "linear"),
            AccessPattern::Random => write!(f, "random"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Operation {
    Read,
    Write,
    Mixed,
}

impl Operation {
    // The number of reads performed for each write in the "mixed" operation.
    const MIXED_WRITE_RATIO: u64 = 10;

    fn variants() -> impl IntoIterator<Item = Operation> {
        [Operation::Read, Operation::Write, Operation::Mixed]
    }

    fn execute(self, backend: &dyn FileBackend, data: &mut [u8], offset: u64, iter: u64) {
        match self {
            Operation::Read => backend.read_exact_at(data, offset).unwrap(),
            Operation::Write => backend.write_all_at(data, offset).unwrap(),
            Operation::Mixed => {
                if iter.is_multiple_of(Self::MIXED_WRITE_RATIO) {
                    backend.write_all_at(data, offset).unwrap();
                } else {
                    backend.read_exact_at(data, offset).unwrap();
                }
            }
        }
    }
}

impl Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operation::Read => write!(f, "read"),
            Operation::Write => write!(f, "write"),
            Operation::Mixed => write!(f, "mixed-{}%write", 100.0 / Self::MIXED_WRITE_RATIO as f64),
        }
    }
}

/// A type alias for a function that opens a `FileBackend` implementation.
/// The function takes a [`Path`] and [`OpenOptions`] and returns a tuple of the opened backend
/// and a string identifying the backend.
pub type BackendOpenFn =
    fn(&Path, OpenOptions, usize) -> BTResult<(Arc<dyn FileBackend>, &'static str), std::io::Error>;

/// Returns an iterator over functions that open different `FileBackend` implementations.
/// Each function returns a tuple of the opened backend and a string identifying the backend.
pub fn backend_open_fns() -> impl Iterator<Item = BackendOpenFn> {
    [
        (|path, options, chunk_size| {
            <SeekFile as FileBackend>::open(path, options, chunk_size)
                .map(|f| (Arc::new(f) as Arc<dyn FileBackend>, "SeekFile"))
        }) as BackendOpenFn,
        #[cfg(unix)]
        {
            (|path, options, chunk_size| {
                <NoSeekFile as FileBackend>::open(path, options, chunk_size)
                    .map(|f| (Arc::new(f) as Arc<dyn FileBackend>, "NoSeekFile"))
            }) as BackendOpenFn
        },
        #[cfg(unix)]
        {
            (|path, options, chunk_size| {
                <PageCachedFile<SeekFile, true> as FileBackend>::open(path, options, chunk_size)
                    .map(|f| {
                        (
                            Arc::new(f) as Arc<dyn FileBackend>,
                            "PageCachedFile<SeekFile, true>",
                        )
                    })
            }) as BackendOpenFn
        },
        #[cfg(unix)]
        {
            (|path, options, chunk_size| {
                <PageCachedFile<NoSeekFile, true> as FileBackend>::open(path, options, chunk_size)
                    .map(|f| {
                        (
                            Arc::new(f) as Arc<dyn FileBackend>,
                            "PageCachedFile<NoSeekFile, true>",
                        )
                    })
            }) as BackendOpenFn
        },
        #[cfg(unix)]
        {
            (|path, options, chunk_size| {
                <PageCachedFile<SeekFile, false> as FileBackend>::open(path, options, chunk_size)
                    .map(|f| {
                        (
                            Arc::new(f) as Arc<dyn FileBackend>,
                            "PageCachedFile<SeekFile, false>",
                        )
                    })
            }) as BackendOpenFn
        },
        #[cfg(unix)]
        {
            (|path, options, chunk_size| {
                <PageCachedFile<NoSeekFile, false> as FileBackend>::open(path, options, chunk_size)
                    .map(|f| {
                        (
                            Arc::new(f) as Arc<dyn FileBackend>,
                            "PageCachedFile<NoSeekFile, false>",
                        )
                    })
            }) as BackendOpenFn
        },
        #[cfg(unix)]
        {
            (|path, options, chunk_size| {
                <MultiPageCachedFile<8, SeekFile, true> as FileBackend>::open(
                    path, options, chunk_size,
                )
                .map(|f| {
                    (
                        Arc::new(f) as Arc<dyn FileBackend>,
                        "MultiPageCachedFile<8, SeekFile, true>",
                    )
                })
            }) as BackendOpenFn
        },
        #[cfg(unix)]
        {
            (|path, options, chunk_size| {
                <MultiPageCachedFile<8, NoSeekFile, true> as FileBackend>::open(
                    path, options, chunk_size,
                )
                .map(|f| {
                    (
                        Arc::new(f) as Arc<dyn FileBackend>,
                        "MultiPageCachedFile<8, NoSeekFile, true>",
                    )
                })
            }) as BackendOpenFn
        },
        #[cfg(unix)]
        {
            (|path, options, chunk_size| {
                <MultiPageCachedFile<8, SeekFile, false> as FileBackend>::open(
                    path, options, chunk_size,
                )
                .map(|f| {
                    (
                        Arc::new(f) as Arc<dyn FileBackend>,
                        "MultiPageCachedFile<8, SeekFile, false>",
                    )
                })
            }) as BackendOpenFn
        },
        #[cfg(unix)]
        {
            (|path, options, chunk_size| {
                <MultiPageCachedFile<8, NoSeekFile, false> as FileBackend>::open(
                    path, options, chunk_size,
                )
                .map(|f| {
                    (
                        Arc::new(f) as Arc<dyn FileBackend>,
                        "MultiPageCachedFile<8, NoSeekFile, false>",
                    )
                })
            }) as BackendOpenFn
        },
    ]
    .into_iter()
}

fn file_backend_benchmark_matrix(c: &mut Criterion) {
    let meminfo = fs::read_to_string("/proc/meminfo").unwrap();
    let mem_total = meminfo
        .lines()
        .find(|l| l.starts_with("MemTotal:"))
        .unwrap();
    let memory_kb: u64 = mem_total
        .strip_prefix("MemTotal:")
        .unwrap()
        .strip_suffix("kB")
        .unwrap()
        .trim()
        .parse()
        .unwrap();

    // Because we do not use the default test harness, cfg!(test) is true also when running with
    // `cargo bench`, but cfg(bench) is not. To detect if we are running the benchmarks as tests, we
    // use cfg!(debug_assertions) as a proxy because by default tests are run in debug mode and
    // benchmarks are not.
    let target_file_size = if cfg!(debug_assertions) {
        100_000
    } else {
        let size = memory_kb * 1024 * FILE_SIZE_MEMORY_MULTIPLIER;
        eprintln!(
            "Using benchmark file of size {} * main memory size ({} GiB) = {} GiB to limit effects of OS page cache",
            FILE_SIZE_MEMORY_MULTIPLIER,
            memory_kb / 1024 / 1024,
            size / ONE_GB as u64
        );
        size
    };

    let path = Path::new(FILE);

    if !fs::exists(path).unwrap() {
        eprintln!("Creating benchmark file at {path:?}");
    }
    let mut options = OpenOptions::new();
    options.create(true).truncate(false).write(true);
    let mut file = options.open(path).unwrap();

    if file.metadata().unwrap().len() < target_file_size {
        eprintln!(
            "Expanding benchmark file {path:?} to {:.1}GB",
            target_file_size as f64 / ONE_GB as f64
        );

        let data_1gb = vec![0; ONE_GB];
        for _ in 0..(target_file_size.div_ceil(ONE_GB as u64)) {
            file.write_all(&data_1gb).unwrap();
        }
        // Note: Using File::set_len creates sparse files on some file systems which results in
        // unrealistic read performance.
    }

    let plot_config = PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic);

    for operation in Operation::variants() {
        for access in AccessPattern::variants() {
            for chunk_size in [32, 4096] {
                for threads in [1, 4, 16] {
                    let mut group = c.benchmark_group(format!(
                        "file_backend/{operation}/{access}/{chunk_size}B/{threads}threads",
                    ));
                    group.plot_config(plot_config.clone());
                    for backend_fn in backend_open_fns() {
                        file_backend_benchmark(
                            &mut group,
                            path,
                            target_file_size,
                            operation,
                            access,
                            chunk_size,
                            threads,
                            backend_fn,
                        );
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn file_backend_benchmark(
    g: &mut BenchmarkGroup<'_, WallTime>,
    path: &Path,
    file_size: u64,
    operation: Operation,
    access: AccessPattern,
    chunk_size: usize,
    threads: usize,
    backend_fn: BackendOpenFn,
) {
    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true);

    let (backend, backend_name) = backend_fn(path, options.clone(), chunk_size).unwrap();

    let mut completed_iterations = 0u64;
    g.throughput(Throughput::Bytes(chunk_size as u64));
    g.bench_with_input(
        BenchmarkId::from_parameter(backend_name),
        // these are passed though [criterion::black_box]
        &(operation, access, chunk_size, threads, backend),
        |b, (operation, access, chunk_size, threads, backend)| {
            let chunk_size = *chunk_size;
            let threads = *threads;
            b.iter_custom(|iterations| {
                execute_with_threads(
                    threads as u64,
                    iterations,
                    &mut completed_iterations,
                    || vec![0u8; chunk_size],
                    |iter, data| {
                        let offset = access.offset(iter, chunk_size, file_size as usize);
                        operation.execute(backend.deref(), data, offset, iter);
                    },
                )
            });
        },
    );
}

criterion_group!(name = benches;  config = Criterion::default(); targets = file_backend_benchmark_matrix);
criterion_main!(benches);
