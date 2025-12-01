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
    node_count::{NodeCountStatistic, NodeCountsByKindStatistic, NodeCountsByLevelStatistic},
};

/// A statistics formatter that writes statistics to a writer with indentation support.
pub struct WriterWithIndentation<W: Write> {
    writer: W,
    indentation: Indentation,
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

    /// Writes a string to [`Self::writer`] with the current indentation.
    fn write_with_indentation(&mut self, string: impl AsRef<str>) -> std::io::Result<()> {
        self.writer
            .write_all(format!("{}{}", self.indentation, string.as_ref()).as_bytes())
    }

    // -------------- Statistics writing methods --------------

    /// Writes the [`NodeCountsByKindStatistic`] statistic to [`Self::writer`] in a human-readable
    /// format with indentation.
    fn write_node_counts_by_kind(
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

    /// Writes the [`NodeCountsByLevelStatistic`] statistic to [`Self::writer`] in a human-readable
    /// format with indentation.
    fn write_node_counts_by_level(
        &mut self,
        item: &NodeCountsByLevelStatistic,
    ) -> std::io::Result<()> {
        self.reset();
        self.write_with_indentation("Node depth distribution:\n")?;
        for (level, count) in &item.node_depth {
            self.write_with_indentation(format!("{level}, {count}\n"))?;
        }
        self.newline()?;
        Ok(())
    }
}

impl<W: Write> StatisticsFormatter for WriterWithIndentation<W> {
    fn write_statistic(&mut self, distribution: &Statistic) -> std::io::Result<()> {
        match distribution {
            Statistic::NodeCount(node_count_statistics) => match node_count_statistics {
                NodeCountStatistic::NodeCountsByKind(stat) => self.write_node_counts_by_kind(stat),
                NodeCountStatistic::NodeCountsByLevelStatistic(stat) => {
                    self.write_node_counts_by_level(stat)
                }
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
    const SIZE: usize = 4; // spaces

    /// Increases the indentation level by one.
    pub fn inc(&mut self) {
        self.level += 1;
    }

    /// Decreases the indentation level by one.
    fn dec(&mut self) {
        if self.level > 0 {
            self.level -= 1;
        }
    }

    /// Resets the indentation level.
    fn reset(&mut self) {
        let _ = std::mem::take(self);
    }
}

impl Default for Indentation {
    fn default() -> Self {
        Self {
            level: 0,
            size: Indentation::SIZE,
        }
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::statistics::node_count::NodeCountBySize;

    #[test]
    fn write_node_counts_by_kind_writes_correctly() {
        let mut output = Vec::new();
        let mut writer = WriterWithIndentation::new(&mut output);

        let mut node_counts_by_kind_statistic = NodeCountsByKindStatistic {
            total_nodes: 4,
            aggregated_node_statistics: std::collections::BTreeMap::new(),
        };
        node_counts_by_kind_statistic
            .aggregated_node_statistics
            .insert(
                "Leaf",
                NodeCountBySize {
                    size_count: BTreeMap::from([(1, 3)]),
                },
            );
        node_counts_by_kind_statistic
            .aggregated_node_statistics
            .insert(
                "Inner",
                NodeCountBySize {
                    size_count: BTreeMap::from([(2, 1)]),
                },
            );

        writer
            .write_statistic(&Statistic::NodeCount(NodeCountStatistic::NodeCountsByKind(
                node_counts_by_kind_statistic,
            )))
            .unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "--- Node distribution ---\n\
            Total nodes: 4\n\
            ╰╴╴╴Node types:\n\
            ╰╴╴╴╴╴╴╴Inner: 1\n\
            ╰╴╴╴╴╴╴╴╴╴╴╴Inner_2: 1\n\
            ╰╴╴╴╴╴╴╴Leaf: 3\n\
            ╰╴╴╴╴╴╴╴╴╴╴╴Leaf_1: 3\n\
            \n"
        );
    }

    #[test]
    fn write_node_counts_by_level_writes_correctly() {
        let mut output = Vec::new();
        let mut writer = WriterWithIndentation::new(&mut output);
        let node_counts_by_level = NodeCountsByLevelStatistic {
            node_depth: BTreeMap::from([(0, 2), (1, 3), (2, 5)]),
        };

        writer
            .write_statistic(&Statistic::NodeCount(
                NodeCountStatistic::NodeCountsByLevelStatistic(node_counts_by_level),
            ))
            .unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "Node depth distribution:\n\
            0, 2\n\
            1, 3\n\
            2, 5\n\
            \n"
        );
    }

    #[test]
    fn write_with_indentation_uses_indentation() {
        let mut output = Vec::new();
        let mut writer = WriterWithIndentation::new(&mut output);

        writer.write_with_indentation("Level 0\n").unwrap();
        writer.inc();
        writer.write_with_indentation("Level 1\n").unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "Level 0\n\
            ╰╴╴╴Level 1\n"
        );
    }

    #[test]
    fn formats_indentation_correctly() {
        let mut output = Vec::new();
        let mut writer = WriterWithIndentation::new(&mut output);

        writer.write_with_indentation("Level 0\n").unwrap();
        writer.inc();
        writer.write_with_indentation("Level 1\n").unwrap();
        writer.inc();
        writer.write_with_indentation("Level 2\n").unwrap();
        writer.dec();
        writer.write_with_indentation("Back to Level 1\n").unwrap();
        writer.dec();
        writer.write_with_indentation("Back to Level 0\n").unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "Level 0\n\
            ╰╴╴╴Level 1\n\
            ╰╴╴╴╴╴╴╴Level 2\n\
            ╰╴╴╴Back to Level 1\n\
            Back to Level 0\n"
        );
    }

    #[test]
    fn newline_writes_newline_without_indentation() {
        let mut output = Vec::new();
        let mut writer = WriterWithIndentation::new(&mut output);

        writer.inc();
        writer.newline().unwrap();
        writer.write_with_indentation("After newline\n").unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "\n\
            ╰╴╴╴After newline\n"
        );
    }

    #[test]
    fn indentation_formats_correctly() {
        let mut indentation = Indentation::default();
        assert_eq!(indentation.to_string(), "");

        indentation.inc();
        assert_eq!(indentation.to_string(), "╰╴╴╴");

        indentation.inc();
        assert_eq!(indentation.to_string(), "╰╴╴╴╴╴╴╴");

        indentation.dec();
        assert_eq!(indentation.to_string(), "╰╴╴╴");

        indentation.dec();
        assert_eq!(indentation.to_string(), "");

        indentation.dec(); // should not go below 0
        assert_eq!(indentation.to_string(), "");

        indentation.reset();
        assert_eq!(indentation.to_string(), "");
    }
}
