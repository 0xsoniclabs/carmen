// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

//! A command line tool to print storage statistics for a Carmen database.
//! The tool reads a storage path containing a Carmen database a prints various
//! statistics about the storage using the specified output formatters.

use std::{
    io::stdout,
    path::{Path, PathBuf},
    sync::Arc,
};

use carmen_rust::{
    database::{
        self,
        verkle::variants::managed::{
            FullLeafNode, InnerNode, SparseLeafNode, VerkleNode, VerkleNodeFileStorageManager,
        },
        visitor::AcceptVisitor,
    },
    node_manager::cached_node_manager::CachedNodeManager,
    statistics::{
        PrintStatistic,
        formatters::{
            StatisticsFormatter, csv_writer::CSVWriter, human_readable_writer::HumanReadableWriter,
        },
        node_count::NodeCountVisitor,
    },
    storage::{
        Storage,
        file::{NoSeekFile, NodeFileStorage},
    },
};
use clap::{Parser, ValueEnum};

/// An enum representing the available output formatters. It must match the available
/// implementations of [`StatisticsFormatter`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Formatter {
    Stdout,
    Csv,
}

impl Formatter {
    /// Instantiate the formatter corresponding to this enum variant.
    fn to_formatter(self) -> Box<dyn StatisticsFormatter> {
        match self {
            Formatter::Stdout => Box::new(HumanReadableWriter::new(stdout())),
            Formatter::Csv => Box::new(CSVWriter {}),
        }
    }
}

/// Storage Statistics Tool
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the DB directory
    #[arg(short, long)]
    db_path: PathBuf,

    /// Output format(s) to use
    #[arg(short, long, value_enum, num_args = 1.., default_value = "stdout")]
    formatter: Vec<Formatter>,
}

fn main() {
    type FileStorage = VerkleNodeFileStorageManager<
        NodeFileStorage<InnerNode, NoSeekFile>,
        NodeFileStorage<SparseLeafNode<2>, NoSeekFile>,
        NodeFileStorage<FullLeafNode, NoSeekFile>,
    >;

    let args = Args::parse();
    let storage_path = Path::new(&args.db_path);

    // TODO: We should have a way of opening the DB in read-only mode.
    if !std::fs::read_dir(storage_path)
        .unwrap()
        .filter_map(Result::ok)
        .any(|entry| entry.file_type().unwrap().is_file() && entry.file_name() == "metadata.bin")
    {
        eprintln!("error: the specified path does not appear to contain a valid Carmen database");
        std::process::exit(1);
    }

    let is_pinned = |_n: &VerkleNode| false; // We don't care about the pinned status for stats
    let storage = FileStorage::open(storage_path).unwrap();
    let manager = Arc::new(CachedNodeManager::new(100_000, storage, is_pinned));
    let mut formatters: Vec<_> = args
        .formatter
        .into_iter()
        .map(Formatter::to_formatter)
        .collect();

    // NOTE: the `ManagedVerkleTrie` must be dropped before closing the DB, hence the
    // inner scope.
    {
        let managed_trie = database::ManagedVerkleTrie::<_>::try_new(manager.clone()).unwrap();
        let mut count_visitor = NodeCountVisitor::default();
        managed_trie.accept(&mut count_visitor).unwrap();
        count_visitor.node_count.print(&mut formatters).unwrap();
    }

    // Close the DB
    Arc::into_inner(manager).unwrap().close().unwrap();
}
