// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::path::PathBuf;

use clap::Parser;

mod node_statistics;

use std::io;

use carmen_rust::statistics::{
    PrintStatistic, Statistic,
    formatters::{
        StatisticsFormatter, csv_writer::CSVWriter, human_readable_writer::HumanReadableWriter,
    },
    node_count::NodeCountStatistic,
};
use clap::ValueEnum;

use crate::node_statistics::{linear_scan_stats, tree_traversal_stats};

/// An enum representing the available node lookup strategies.
#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum NodeLookup {
    /// Perform node lookups via a linear scan of the node files. This is fast, but does not provide
    /// any information about the depth of the node in the tree.
    LinearScan,
    /// Perform node lookups via a tree traversal of the node files. This is slower, but provides
    /// more information about the structure of the tree.
    TreeTraversal,
}

/// An enum representing the available output formatters. It must match the available
/// implementations of [`StatisticsFormatter`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Format {
    Pretty,
    Csv,
}

impl Format {
    /// Instantiate the formatter corresponding to this enum variant.
    fn to_formatter(self) -> Box<dyn StatisticsFormatter> {
        match self {
            Format::Pretty => Box::new(HumanReadableWriter::new(io::stdout())),
            Format::Csv => Box::new(CSVWriter {}),
        }
    }
}

/// A command line tool to print statistics about a Carmen DB (Rust variants only).
/// The tool reads a path containing a Carmen DB and prints various statistics.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the DB directory. This must be either the `live` or the `archive` directory.
    #[arg(short, long)]
    db_path: PathBuf,

    #[arg(short, long, value_enum, default_value_t = NodeLookup::LinearScan)]
    node_lookup: NodeLookup,

    /// Output format(s) to use
    #[arg(short, long, value_enum, num_args = 1.., default_value = "pretty")]
    format: Vec<Format>,
}

fn main() {
    let args = Args::parse();
    let dir = args.db_path;
    let mut formatters: Vec<_> = args.format.into_iter().map(Format::to_formatter).collect();
    match args.node_lookup {
        NodeLookup::LinearScan => {
            let stats = linear_scan_stats::linear_scan_stats(&dir);
            let stats = Statistic::NodeCount(NodeCountStatistic::NodeCountsByKind(stats));

            for mut formatter in formatters {
                formatter.write_statistic(&stats).unwrap();
            }
        }
        NodeLookup::TreeTraversal => {
            let stats = tree_traversal_stats::tree_traversal_stats(&dir);
            stats.print(&mut formatters).unwrap();
        }
    }
}
