# Carmen NG (Rust)

Next generation Carmen DB implementation written in Rust.

The Rust code encompasses a full database implementation using Verkle Tries (S6), tests and benchmarks.
It is intended to be used as a shared library from Go.

## Usage

### Building and Linking against the Shared Library

To build the shared library either run `make` in the root directory which will build both the C++ and the Rust code, or run in the `rust` directory
```sh
cargo build --release
```

The shared library called `libcarmen_rust.so` can then be found in [`rust/target/release`](target/release).

The Go code will link against the library automatically when the [`external`](../go/state/externalstate) package is included. (Or the [`experimental`](../go/experimental) packages which includes the `external` package.)

### Running Rust Tests

```sh
cargo test
```

### Running Rust Benchmarks

```sh
cargo bench
```

### Collecting Node Statistics

There are currently two ways of collecting node statistics, either by traversing the tree or by linearly scanning the file.
The former one provides also information about the depth of the nodes in the tree but only works for live DBs.
The latter works also for archive DB and is significantly faster.

```sh
❯ cargo run --release -- --help
A command line tool to print statistics about a Carmen DB (Rust variants only). The tool reads a path containing a Carmen DB and prints various statistics

Usage: node_stats [OPTIONS] --db-path <DB_PATH>

Options:
  -d, --db-path <DB_PATH>
          Path to the DB directory. This must be either the `live` or the `archive` directory

  -n, --node-lookup <NODE_LOOKUP>
          Possible values:
          - linear-scan:    Perform node lookups via a linear scan of the node files. This is fast, but does not provide any information about the depth of the node in the tree
          - tree-traversal: Perform node lookups via a tree traversal of the node files. This is slower, but provides more information about the structure of the tree[default: linear-scan]

  -f, --format <FORMAT>...
          Output format(s) to use

          [default: pretty]
          [possible values: pretty, csv]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

## Carmen & Bertha Integration

[Bertha](https://github.com/0xsoniclabs/bertha) can be used to replay the history with Carmen.
This can be used for multiple purposes including testing and profiling.

Bertha provides wrapper scripts to run with a local Carmen repository and optionally enable Tracy profiling: `go-run-with-carmen.sh` and `go-run-with-carmen-and-tracy.sh` which can be found in `bertha/block-db-go-bindings`.
For it to work, Bertha and Carmen must be in the same parent directory.
The scripts can be provided with arguments which are passed to `go run` in Bertha.

The a full list of possible arguments run:
```sh
go run ./cmd/block-db replay --help
```

### Testing using History Replay

Replaying the history can be used as a testing tool to verify the resulting state root hashes after each block.

For a history run using the Carmen Rust implementation, run in `bertha/block-db-go-bindings`:

```sh
# Record state root hashes for S6 (Verkle Tries) using the Go in-memory variant
go run ./cmd/block-db replay \
    --json-genesis <path to genesis.json> \
    -db <path to blockdb> \
    --db-schema 6 --db-variant go-memory \
    --overwrite-state-roots

# Replay using the Carmen Rust implementation and verify state root hashes.
./go-run-with-carmen.sh ./cmd/block-db replay \
    --json-genesis <path to genesis.json> \
    -db <path to blockdb> \
    --db-schema 6 --db-variant rust-file
```

### Profiling using History Replay with Tracy

Both Bertha and Carmen have [Tracy](https://github.com/wolfpld/tracy) integration for profiling.

For a history run with Tracy profiling enabled and using the Carmen Rust implementation, run in `bertha/block-db-go-bindings`:

```sh
./go-run-with-carmen-and-tracy.sh ./cmd/block-db replay \
    --json-genesis <path to genesis.json> \
    -db <path to blockdb> \
    --db-schema 6 --db-variant rust-file \
    --use-pipeline=false
```


To view the profiling data, open the Tracy profiler and connect to the running Bertha process.
