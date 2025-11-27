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

use std::{io::stdout, path::Path, sync::Arc};

use carmen_rust::{
    CarmenDb, CarmenS6FileBasedDb,
    database::{
        self, AcceptVisitor, VerkleTrieCarmenState,
        verkle::variants::managed::{
            FullLeafNode, InnerNode, SparseLeafNode, VerkleNode, VerkleNodeFileStorageManager,
        },
    },
    node_manager::cached_node_manager::CachedNodeManager,
    statistics::{
        PrintStatistic,
        formatters::{
            StatisticsFormatter, csv_writer::CSVWriter,
            writer_with_indentation::WriterWithIndentation,
        },
        trie_count::TrieCountVisitor,
    },
    storage::{
        Storage,
        file::{NoSeekFile, NodeFileStorage},
        storage_with_flush_buffer::StorageWithFlushBuffer,
    },
};
use clap::{Parser, ValueEnum};

/// An enum representing the available output formatters. It must match the available
/// implementations of `StatisticsFormatter`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Formatter {
    Stdout,
    Csv,
}

impl Formatter {
    /// Instantiate the formatter corresponding to this enum variant
    fn to_formatter(self) -> Box<dyn StatisticsFormatter> {
        match self {
            Formatter::Stdout => Box::new(WriterWithIndentation::new(stdout())),
            Formatter::Csv => Box::new(CSVWriter {}),
        }
    }
}

/// Command line arguments
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the storage directory
    #[arg(short, long)]
    storage_path: String,

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

    let storage_path = Path::new(&args.storage_path);
    let is_pinned = |_n: &VerkleNode| false; // We don't care about the pinned status for stats
    let storage = StorageWithFlushBuffer::<FileStorage>::open(storage_path).unwrap();
    let manager = Arc::new(CachedNodeManager::new(1_000_000, storage, is_pinned));
    let managed_trie = database::ManagedVerkleTrie::<_>::try_new(manager.clone()).unwrap();

    let formatters_to_use = args.formatter;
    let mut formatters = Vec::new();
    for formatter in formatters_to_use {
        formatters.push(formatter.to_formatter());
    }

    let mut count_visitor = TrieCountVisitor::default();
    managed_trie.accept(&mut count_visitor).unwrap();
    count_visitor.trie_count.print(&mut formatters).unwrap();

    // Close the DB
    let _ = &CarmenS6FileBasedDb::new(manager, VerkleTrieCarmenState::from(managed_trie))
        .close()
        .unwrap();
}
