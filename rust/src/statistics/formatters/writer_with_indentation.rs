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
    node_count::{NodeCountStatistic, NodeCountsByKindStatistic, NodeCountsByLevel},
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

    /// Increases the indentation level by one.
    fn inc(&mut self) {
        self.indentation.inc();
    }

    /// Decreases the indentation level by one.
    fn dec(&mut self) {
        self.indentation.dec();
    }

    /// Resets the indentation level.
    fn reset(&mut self) {
        self.indentation.reset();
    }

    /// Writes a newline character without indenting the next line.
    fn newline(&mut self) -> std::io::Result<()> {
        self.writer.write_all(b"\n")
    }

    fn print_node_size_per_tree(
        &mut self,
        stat: &NodeCountsByKindStatistic,
    ) -> std::io::Result<()> {
        self.reset();
        self.write_with_indentation("--- Node distribution ---\n")?;
        self.write_with_indentation(format!("Total nodes: {}\n", stat.total_nodes))?;
        self.inc();
        self.write_with_indentation("Node types:\n")?;
        self.inc();
        for (type_name, stats) in &stat.aggregated_node_statistics {
            let node_count = stats.size_count.values().sum::<u64>();
            self.write_with_indentation(format!("{type_name}: {node_count}\n"))?;
            self.inc();
            let mut kinds = stats.size_count.keys().collect::<Vec<_>>();
            kinds.sort();
            for kind in kinds {
                let kind_count = stats.size_count[kind];
                self.write_with_indentation(format!("{type_name}_{kind}: {kind_count}\n"))?;
            }
            self.dec();
        }
        self.newline()?;
        Ok(())
    }

    fn print_node_depth(&mut self, item: &NodeCountsByLevel) -> std::io::Result<()> {
        self.reset();
        self.write_with_indentation("Node depth distribution:\n")?;
        for (level, count) in &item.node_depth {
            self.write_with_indentation(format!("{level}, {count}\n"))?;
        }
        self.newline()?;
        Ok(())
    }

    /// Writes a string to [`Self::writer`] with the current indentation.
    fn write_with_indentation(&mut self, string: impl AsRef<str>) -> std::io::Result<()> {
        self.writer
            .write_all(format!("{}{}", self.indentation, string.as_ref()).as_bytes())
    }
}

impl<W: Write> StatisticsFormatter for WriterWithIndentation<W> {
    fn print_statistic(&mut self, distribution: &Statistic) -> std::io::Result<()> {
        match distribution {
            Statistic::NodeCount(node_count_statistics) => match node_count_statistics {
                NodeCountStatistic::NodeCountsByKind(stat) => self.print_node_size_per_tree(stat),
                NodeCountStatistic::NodeCountsByLevel(stat) => self.print_node_depth(stat),
            },
        }
    }
}

/// A utility to manage indentation levels for formatted output.
#[derive(Clone, Debug)]
pub struct Indentation {
    level: usize,
    size: usize,
}

impl Indentation {
    pub fn inc(&mut self) {
        self.level += 1;
    }

    fn dec(&mut self) {
        if self.level > 0 {
            self.level -= 1;
        }
    }

    fn reset(&mut self) {
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
