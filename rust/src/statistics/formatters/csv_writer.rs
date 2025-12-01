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
    fs::File,
    io::Write,
    ops::{Deref, DerefMut},
};

use crate::statistics::{
    Statistic, StatisticsFormatter,
    node_count::{NodeCountStatistic, NodeCountsByKindStatistic, NodeCountsByLevelStatistic},
};

/// A statistics formatter that writes statistics in CSV format.
pub struct CSVWriter {
    file: File,
}

impl CSVWriter {
    /// Initialize a CSV writer for the given statistic suffix.
    fn new(suffix: &str) -> std::io::Result<Self> {
        Ok(Self {
            file: File::create(format!("carmen_stats_{suffix}.csv"))?,
        })
    }
}

impl Deref for CSVWriter {
    type Target = File;

    fn deref(&self) -> &Self::Target {
        &self.file
    }
}

impl DerefMut for CSVWriter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.file
    }
}

/// Writes [`NodeCountsByKindStatistic`] in CSV format.
fn write_node_counts_by_kind(stat: &NodeCountsByKindStatistic) -> std::io::Result<()> {
    let mut wtr = CSVWriter::new("node_counts_by_kind")?;
    wtr.write_all(b"Node Type,Node Subtype,Node Count\n")?;
    for (type_name, stats) in &stat.aggregated_node_statistics {
        let node_count = stats.size_count.values().sum::<u64>();
        wtr.write_all(format!("{type_name},{type_name},{node_count}\n").as_bytes())?;
        let mut kinds = stats.size_count.keys().collect::<Vec<_>>();
        kinds.sort();
        for kind in kinds {
            let kind_count = stats.size_count[kind];
            wtr.write_all(format!("{type_name},{type_name}_{kind},{kind_count}\n").as_bytes())?;
        }
    }
    Ok(())
}

/// Writes [`NodeCountsByLevel`] in CSV format.
fn write_node_counts_by_level(stat: &NodeCountsByLevel) -> std::io::Result<()> {
    let mut wtr = CSVWriter::new("node_counts_by_level")?;
    wtr.write_all(b"Level,Node Count\n")?;
    for (level, count) in &stat.node_depth {
        wtr.write_all(format!("{level},{count}\n").as_bytes())?;
    }
    Ok(())
}

impl StatisticsFormatter for CSVWriter {
    fn write_statistic(&mut self, statistic: &Statistic) -> std::io::Result<()> {
        match statistic {
            Statistic::NodeCount(node_count_statistics) => match node_count_statistics {
                NodeCountStatistic::NodeCountsByKind(stat) => write_node_counts_by_kind(stat),
                NodeCountStatistic::NodeCountsByLevel(stat) => write_node_counts_by_level(stat),
            },
        }
    }
}
