// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::io::Write;

use crate::statistics::{
    Statistic,
    formatters::StatisticsFormatter,
    trie_count::{NodeDepthStatistic, NodeSizeStatistic, TrieCountStatistics},
};

/// A statistics formatter that writes statistics to a writer with indentation support.
/// It implements the [`StatisticsFormatter`] trait, and creates a different CSV file for each
/// statistic type.
pub struct WriterWithIndentation<W: Write> {
    pub writer: W,
    pub indentation: Indentation,
}

impl<W: Write> WriterWithIndentation<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            indentation: Indentation::default(),
        }
    }

    pub fn inc(&mut self) {
        self.indentation.inc();
    }

    pub fn dec(&mut self) {
        self.indentation.dec();
    }

    pub fn reset(&mut self) {
        self.indentation.reset();
    }

    /// Writes a newline character while skipping indentation.
    pub fn newline(&mut self) -> std::io::Result<()> {
        self.writer.write_all(b"\n")
    }

    fn print_node_size_per_tree(&mut self, stat: &NodeSizeStatistic) -> std::io::Result<()> {
        self.reset();
        self.write_all(b"--- Node distribution ---\n")?;
        self.write_all(format!("Total nodes: {}\n", stat.total_nodes).as_bytes())?;
        self.inc();
        self.write_all(b"Node types:\n")?;
        self.inc();
        for (type_name, stats) in &stat.aggregated_node_statistics {
            let node_count = stats.size_count.values().sum::<u64>();
            self.write_all(format!("{type_name} nodes: {node_count}\n").as_bytes())?;
            self.inc();
            let mut kinds = stats.size_count.keys().collect::<Vec<_>>();
            kinds.sort();
            for kind in kinds {
                let kind_count = stats.size_count[kind];
                self.write_all(format!("{type_name}_{kind}: {kind_count}\n").as_bytes())?;
            }
            self.dec();
        }
        self.newline()?;
        Ok(())
    }

    fn print_node_depth(&mut self, item: &NodeDepthStatistic) -> std::io::Result<()> {
        self.reset();
        self.write_all(b"Node depth distribution:\n")?;
        for (level, count) in &item.node_depth {
            self.write_all(format!("{level}, {count}\n").as_bytes())?;
        }
        self.newline()?;
        Ok(())
    }
}

impl<W: Write> Write for WriterWithIndentation<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let indented_buf = format!("{}{}", self.indentation, String::from_utf8_lossy(buf));
        let written_bytes = self.writer.write(indented_buf.as_bytes())?;
        if written_bytes < indented_buf.len() {
            Err(std::io::Error::other("Could not write all indented bytes"))
        } else {
            Ok(buf.len())
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

impl<W: Write> StatisticsFormatter for WriterWithIndentation<W> {
    fn print_statistic(&mut self, distribution: &Statistic) -> std::io::Result<()> {
        match distribution {
            Statistic::TrieCount(trie_count_statistics) => match trie_count_statistics {
                TrieCountStatistics::NodeSizePerTree(stat) => self.print_node_size_per_tree(stat),
                TrieCountStatistics::NodeDepth(stat) => self.print_node_depth(stat),
            },
        }
    }
}

/// A utility to manage indentation levels for formatted output.
#[derive(Clone, Debug)]
pub struct Indentation {
    pub level: usize,
    size: usize,
}

impl Indentation {
    pub fn inc(&mut self) {
        self.level += 1;
    }

    pub fn dec(&mut self) {
        if self.level > 0 {
            self.level -= 1;
        }
    }

    pub fn reset(&mut self) {
        let default = Indentation::default();
        self.level = default.level;
        self.size = default.size;
    }
}

impl Default for Indentation {
    fn default() -> Self {
        Self { level: 0, size: 4 } // 4 spaces
    }
}

impl std::fmt::Display for Indentation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.level {
            let mut size = self.size;
            if i == 0 {
                write!(f, "╰")?;
                size = size.saturating_sub(1);
            }
            write!(f, "{}", "╴".repeat(size))?;
        }
        Ok(())
    }
}
