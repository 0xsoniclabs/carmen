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

/// A statistics formatter that writes statistics to a writer in a human-readable format.
pub struct HumanReadableWriter<W: Write> {
    writer: W,
}

impl<W: Write> HumanReadableWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    fn write(&mut self, string: impl AsRef<str>) -> std::io::Result<()> {
        self.writer.write_all(string.as_ref().as_bytes())
    }

    // -------------- Statistics writing methods --------------

    /// Writes the [`NodeCountsByKindStatistic`] statistic to [`Self::writer`] in a human-readable
    /// format using a tree-like structure.
    fn write_node_counts_by_kind(
        &mut self,
        stat: &NodeCountsByKindStatistic,
    ) -> std::io::Result<()> {
        self.write(format!("Total nodes: {}\n", stat.total_nodes))?;
        for (i, (node_kind, stats)) in stat.aggregated_node_statistics.iter().enumerate() {
            let last_kind = i == stat.aggregated_node_statistics.len() - 1;
            if !last_kind {
                self.write("├──")?;
            } else {
                self.write("╰──")?;
            }

            let node_count = stats.size_count.values().sum::<u64>();
            self.write(format!(" {node_kind}: {node_count}\n"))?;
            let mut sizes = stats.size_count.keys().collect::<Vec<_>>();
            sizes.sort();
            for (i, size) in sizes.iter().enumerate() {
                if !last_kind {
                    self.write("│  ")?;
                } else {
                    self.write("   ")?;
                }
                if i < sizes.len() - 1 {
                    self.write("├──")?;
                } else {
                    self.write("╰──")?;
                }

                let size_count = stats.size_count[size];
                self.write(format!(" {size}: {size_count}\n"))?;
            }
        }
        self.write("\n")?;
        Ok(())
    }

    /// Writes the [`NodeCountsByLevelStatistic`] statistic to [`Self::writer`] in a human-readable
    /// format.
    fn write_node_counts_by_level(
        &mut self,
        item: &NodeCountsByLevelStatistic,
    ) -> std::io::Result<()> {
        self.write("Node counts by level:\n")?;
        for (level, count) in &item.node_depth {
            self.write(format!("{level}, {count}\n"))?;
        }
        self.write("\n")?;
        Ok(())
    }
}

impl<W: Write> StatisticsFormatter for HumanReadableWriter<W> {
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::statistics::node_count::NodeCountBySize;

    #[test]
    fn write_node_counts_by_kind_writes_correctly() {
        let mut output = Vec::new();
        let mut writer = HumanReadableWriter::new(&mut output);

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
            "Total nodes: 4\n\
            ├── Inner: 1\n\
            │  ╰── 2: 1\n\
            ╰── Leaf: 3\n"
                .to_owned()
                + "   ╰── 1: 3\n\
            \n"
        );
    }

    #[test]
    fn write_node_counts_by_level_writes_correctly() {
        let mut output = Vec::new();
        let mut writer = HumanReadableWriter::new(&mut output);
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
            "Node counts by level:\n\
            0, 2\n\
            1, 3\n\
            2, 5\n\
            \n"
        );
    }
}
