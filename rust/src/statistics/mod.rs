use std::{collections::BTreeMap, ops::AddAssign};

use crate::statistics::formatters::StatisticsFormatter;

pub mod formatters;
mod utils;

pub trait TrieStatistics {
    fn get_statistics(&self) -> Statistics;
}

pub trait TrieVisitor<N> {
    fn visit(&mut self, node: &N, level: u8);
}

#[derive(Default, Clone, Debug)]
pub struct Statistics {
    pub level_statistics: BTreeMap<u8, LevelStatistics>,
    pub byte_size: u64,
}

#[derive(Default, Clone, Debug)]
pub struct LevelStatistics {
    pub node_count: u64,
    pub byte_size: u64,
    pub node_statistics: BTreeMap<String, NodeStatistics>,
}

#[derive(Default, Clone, Debug)]
pub struct NodeStatistics {
    pub node_count: u64,
    pub byte_size: u64,
    pub node_kinds: BTreeMap<String, u64>,
}

impl AddAssign<&NodeStatistics> for NodeStatistics {
    fn add_assign(&mut self, other: &NodeStatistics) {
        self.node_count += other.node_count;
        for (kind, count) in &other.node_kinds {
            *self.node_kinds.entry(kind.clone()).or_insert(0) += count;
        }
    }
}

impl Statistics {
    pub fn print(&self, writers: &mut [Box<dyn StatisticsFormatter>]) -> std::io::Result<()> {
        let node_distribution = Distributions::Node(NodeDistribution::new(&self.level_statistics));
        let node_depth_distribution =
            Distributions::NodePerLevel(NodePerLevelDistribution::new(&self.level_statistics));
        let node_type_distribution = Distributions::NodeTypePerLevel(
            NodeTypePerLevelDistribution::new(&self.level_statistics),
        );
        for writer in writers {
            writer.print_distribution(&node_distribution)?;
            writer.print_distribution(&node_depth_distribution)?;
            writer.print_distribution(&node_type_distribution)?;
        }
        Ok(())
    }
}

pub enum Distributions {
    Node(NodeDistribution),
    NodePerLevel(NodePerLevelDistribution),
    NodeTypePerLevel(NodeTypePerLevelDistribution),
}

#[derive(Clone, Debug)]
pub struct NodeDistribution {
    aggregated_node_statistics: BTreeMap<String, NodeStatistics>,
    total_nodes: u64,
}

impl NodeDistribution {
    fn new(level_statistics: &BTreeMap<u8, LevelStatistics>) -> Self {
        let node_statistics = level_statistics.values().fold(
            BTreeMap::<String, NodeStatistics>::default(),
            |mut acc, stats| {
                for (type_name, type_stats) in &stats.node_statistics {
                    let entry = acc.entry(type_name.clone()).or_default();
                    *entry += type_stats;
                }
                acc
            },
        );
        let total_nodes = node_statistics
            .values()
            .map(|stats| stats.node_count)
            .sum::<u64>();
        Self {
            aggregated_node_statistics: node_statistics,
            total_nodes,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NodePerLevelDistribution {
    distribution: BTreeMap<u8, u64>,
}

impl NodePerLevelDistribution {
    fn new(level_statistics: &BTreeMap<u8, LevelStatistics>) -> Self {
        let mut distribution = BTreeMap::new();
        for (level, stats) in level_statistics {
            distribution.insert(*level, stats.node_count);
        }
        Self { distribution }
    }
}

#[derive(Clone, Debug)]
pub struct NodeTypePerLevelDistribution {
    level_statistics: BTreeMap<u8, LevelStatistics>,
}

impl NodeTypePerLevelDistribution {
    fn new(level_statistics: &BTreeMap<u8, LevelStatistics>) -> Self {
        Self {
            level_statistics: level_statistics.clone(),
        }
    }
}
