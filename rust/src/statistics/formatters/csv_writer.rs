// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{fs::File, io::Write, path::Path};

use crate::statistics::{
    Statistic, StatisticsFormatter,
    node_count::{NodeCountStatistic, NodeCountsByKindStatistic, NodeCountsByLevelStatistic},
};

/// A statistics formatter that writes statistics in CSV format.
pub struct CSVWriter;

impl CSVWriter {
    /// Initialize a CSV writer for the given statistic suffix.
    fn init(path: &Path, suffix: &str) -> std::io::Result<File> {
        File::create(path.join(format!("carmen_stats_{suffix}.csv")))
    }
}

/// Writes [`NodeCountsByKindStatistic`] in CSV format.
fn write_node_counts_by_kind(stat: &NodeCountsByKindStatistic, path: &Path) -> std::io::Result<()> {
    let mut wtr = CSVWriter::init(path, "node_counts_by_kind")?;
    wtr.write_all(b"Node Kind,Node Size,Count\n")?;
    for (node_kind, stats) in &stat.aggregated_node_statistics {
        let mut sizes = stats.size_count.keys().collect::<Vec<_>>();
        sizes.sort();
        for size in sizes {
            let size_count = stats.size_count[size];
            wtr.write_all(format!("{node_kind},{size},{size_count}\n").as_bytes())?;
        }
    }
    Ok(())
}

/// Writes [`NodeCountsByLevelStatistic`] in CSV format.
fn write_node_counts_by_level(
    stat: &NodeCountsByLevelStatistic,
    path: &Path,
) -> std::io::Result<()> {
    let mut wtr = CSVWriter::init(path, "node_counts_by_level")?;
    wtr.write_all(b"Level,Count\n")?;
    for (level, count) in &stat.node_depth {
        wtr.write_all(format!("{level},{count}\n").as_bytes())?;
    }
    Ok(())
}

impl StatisticsFormatter for CSVWriter {
    fn write_statistic(&mut self, statistic: &Statistic) -> std::io::Result<()> {
        match statistic {
            Statistic::NodeCount(node_count_statistics) => match node_count_statistics {
                NodeCountStatistic::NodeCountsByKind(stat) => {
                    write_node_counts_by_kind(stat, Path::new("."))
                }
                NodeCountStatistic::NodeCountsByLevelStatistic(stat) => {
                    write_node_counts_by_level(stat, Path::new("."))
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
    fn new_creates_csv_writer_with_suffix() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let writer = CSVWriter::init(tmp_dir.path(), "test_suffix");
        assert!(writer.is_ok());
        drop(writer);
        assert!(tmp_dir.path().join("carmen_stats_test_suffix.csv").exists());
    }

    #[test]
    fn write_node_counts_by_kind_writes_correctly() {
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

        let tmp_dir = tempfile::tempdir().unwrap();
        write_node_counts_by_kind(&node_counts_by_kind_statistic, tmp_dir.path()).unwrap();
        let content =
            std::fs::read_to_string(tmp_dir.path().join("carmen_stats_node_counts_by_kind.csv"))
                .unwrap();
        assert_eq!(
            content,
            "Node Kind,Node Size,Count\n\
             Inner,2,1\n\
             Leaf,1,3\n"
        );
    }

    #[test]
    fn write_node_counts_by_level_writes_correctly() {
        let node_counts_by_level_statistic = NodeCountsByLevelStatistic {
            node_depth: BTreeMap::from([(0, 2), (1, 3), (2, 5)]),
        };

        let tmp_dir = tempfile::tempdir().unwrap();
        write_node_counts_by_level(&node_counts_by_level_statistic, tmp_dir.path()).unwrap();
        let content =
            std::fs::read_to_string(tmp_dir.path().join("carmen_stats_node_counts_by_level.csv"))
                .unwrap();
        assert_eq!(
            content,
            "Level,Count\n\
             0,2\n\
             1,3\n\
             2,5\n"
        );
    }
}
