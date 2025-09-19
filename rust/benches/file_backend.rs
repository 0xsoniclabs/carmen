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
    fs::{File, OpenOptions},
    path::Path,
    sync::Arc,
};

use carmen_rust::storage::file::{FileBackend, NoSeekFile, PageCachedFile, SeekFile};
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::Measurement,
};

const FILE_SIZE: usize = 10 * 1024 * 1024 * 1024; // 10GB

/// Defines the access pattern for the benchmark.
#[derive(Debug, Clone, Copy)]
enum AccessPattern {
    /// Always use the offset 0.
    Static,
    /// Always advances the offset by the chunk size.
    Linear,
}

impl AccessPattern {
    /// Returns all access patterns.
    fn variants() -> [AccessPattern; 2] {
        [AccessPattern::Static, AccessPattern::Linear]
    }

    /// Returns the offset for the given iteration and chunk size.
    fn offset(self, iter: usize, chunk_size: usize) -> u64 {
        match self {
            AccessPattern::Static => 0,
            AccessPattern::Linear => ((iter * chunk_size) % FILE_SIZE) as u64,
        }
    }
}

impl Display for AccessPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccessPattern::Static => write!(f, "static"),
            AccessPattern::Linear => write!(f, "linear"),
        }
    }
}

/// A type alias for a function that opens a `FileBackend` implementation.
/// The function takes a [`Path`] and [`OpenOptions`] and returns a tuple of the opened backend
/// and a string identifying the backend.
pub type BackendOpenFn =
    fn(&Path, OpenOptions) -> std::io::Result<(Arc<dyn FileBackend>, &'static str)>;

/// Returns an iterator over functions that open different `FileBackend` implementations.
/// Each function returns a tuple of the opened backend and a string identifying the backend.
pub fn backend_open_fns() -> impl Iterator<Item = BackendOpenFn> {
    [
        (|path, options| {
            <SeekFile as FileBackend>::open(path, options)
                .map(|f| (Arc::new(f) as Arc<dyn FileBackend>, "SeekFile"))
        }) as BackendOpenFn,
        (|path, options| {
            <NoSeekFile as FileBackend>::open(path, options)
                .map(|f| (Arc::new(f) as Arc<dyn FileBackend>, "NoSeekFile"))
        }) as BackendOpenFn,
        #[cfg(unix)]
        {
            (|path, options| {
                <PageCachedFile<SeekFile> as FileBackend>::open(path, options).map(|f| {
                    (
                        Arc::new(f) as Arc<dyn FileBackend>,
                        "PageCachedFile<SeekFile>",
                    )
                })
            }) as BackendOpenFn
        },
        #[cfg(unix)]
        {
            (|path, options| {
                <PageCachedFile<NoSeekFile> as FileBackend>::open(path, options).map(|f| {
                    (
                        Arc::new(f) as Arc<dyn FileBackend>,
                        "PageCachedFile<NoSeekFile>",
                    )
                })
            }) as BackendOpenFn
        },
    ]
    .into_iter()
}

fn write_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("file_backend/write");
    for access in AccessPattern::variants() {
        for chunk_size in [32, 4096] {
            for backend_fn in backend_open_fns() {
                write(&mut group, access, chunk_size, backend_fn);
            }
        }
    }
}

fn write<M: Measurement>(
    g: &mut BenchmarkGroup<'_, M>,
    access: AccessPattern,
    chunk_size: usize,
    backend_fn: BackendOpenFn,
) {
    // Note: At least on Ubuntu, reading and writing to a file which is located directly in `/tmp`
    // is slower than with a file in a subdirectory of `/tmp`.
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().join("data.bin");
    let path = path.as_path();

    {
        let file = File::create(path).unwrap();
        file.set_len(FILE_SIZE as u64).unwrap();
    }

    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true);

    let (backend, backend_name) = backend_fn(path, options.clone()).unwrap();

    let id = BenchmarkId::from_parameter(format!("{access}/{chunk_size}B/{backend_name}"));

    g.throughput(Throughput::Bytes(chunk_size as u64));
    g.bench_with_input(
        id,
        &(access, chunk_size, backend), // these are passed though a [criterion::black_box]
        |b, (access, chunk_size, backend)| {
            let data = vec![0; *chunk_size];
            let mut iter = 0;
            b.iter(|| {
                let offset = access.offset(iter, *chunk_size);
                backend.write_all_at(&data, offset).unwrap();
                iter += 1;
            });
        },
    );
}

criterion_group!(name = benches;  config = Criterion::default(); targets = write_benchmarks);
criterion_main!(benches);
