// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::collections::BTreeMap;

/// Counts of different node kinds, organized by level.
/// This can be used to answer questions such as "how many leaf nodes
/// with 7 values are on level 3?".
#[derive(Default, Clone, Debug)]
pub struct NodeCountsByLevelAndKind {
    levels_count: Vec<BTreeMap<&'static str, NodeCountBySize>>,
}

/// A count of how many nodes there are that contain a certain number of values / children.
#[derive(Default, Clone, Debug)]
pub struct NodeCountBySize {
    size_count: BTreeMap<u64, u64>,
}

/// A visitor implementation that counts the number of nodes within a trie,
/// as well as their sizes, on a level-by-level basis.
#[derive(Default)]
pub struct NodeCountVisitor {
    node_count: NodeCountsByLevelAndKind,
}

impl NodeCountVisitor {
    /// Records the occurrence of a node of the given type at the given level,
    pub fn count_node(&mut self, level: u64, type_name: &'static str, num_children: Option<u64>) {
        while self.node_count.levels_count.len() <= level as usize {
            self.node_count.levels_count.push(BTreeMap::new());
        }
        let level_entry = &mut self.node_count.levels_count[level as usize];
        let node_entry = level_entry.entry(type_name).or_default();
        if let Some(num_children) = num_children {
            *node_entry.size_count.entry(num_children).or_insert(0) += 1;
        }
    }
}

/// Counts of nodes, organized by kind.
#[derive(Clone, Debug)]
pub struct NodeCountsByKindStatistic {
    pub aggregated_node_statistics: BTreeMap<&'static str, NodeCountBySize>,
    pub total_nodes: u64,
}

impl NodeCountsByKindStatistic {
    #[cfg_attr(not(test), expect(unused))]
    fn new(counts: &NodeCountsByLevelAndKind) -> Self {
        let node_count = counts.levels_count.iter().fold(
            BTreeMap::<&'static str, NodeCountBySize>::default(),
            |mut acc, stats| {
                for (kind, count) in stats.iter() {
                    let acc_entry = acc.entry(*kind).or_default();
                    for (size, size_count) in count.size_count.iter() {
                        *acc_entry.size_count.entry(*size).or_insert(0) += size_count;
                    }
                }
                acc
            },
        );
        let total_nodes = node_count
            .values()
            .map(|stats| stats.size_count.values().sum::<u64>())
            .sum::<u64>();
        Self {
            aggregated_node_statistics: node_count,
            total_nodes,
        }
    }
}

/// Counts of nodes, organized by level.
#[derive(Clone, Debug)]
pub struct NodeCountsByLevel {
    pub node_depth: BTreeMap<usize, u64>,
}

impl NodeCountsByLevel {
    #[cfg_attr(not(test), expect(unused))]
    fn new(node_count: &NodeCountsByLevelAndKind) -> Self {
        let mut node_depth = BTreeMap::new();
        for (level, stats) in node_count.levels_count.iter().enumerate() {
            node_depth.insert(
                level,
                stats
                    .values()
                    .map(|ns| ns.size_count.values().sum::<u64>())
                    .sum(),
            );
        }
        Self { node_depth }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_node_counts() -> NodeCountsByLevelAndKind {
        let mut node_count = NodeCountsByLevelAndKind::default();

        // Level 0
        node_count.levels_count.push(BTreeMap::new());
        node_count.levels_count[0].insert(
            "Inner",
            NodeCountBySize {
                size_count: BTreeMap::from([(2, 3), (3, 1)]),
            },
        );
        node_count.levels_count[0].insert(
            "Leaf",
            NodeCountBySize {
                size_count: BTreeMap::from([(1, 5)]),
            },
        );

        // Level 1
        node_count.levels_count.push(BTreeMap::new());
        node_count.levels_count[1].insert(
            "Inner",
            NodeCountBySize {
                size_count: BTreeMap::from([(2, 2), (4, 3)]),
            },
        );
        node_count.levels_count[1].insert(
            "Leaf",
            NodeCountBySize {
                size_count: BTreeMap::from([(1, 5)]),
            },
        );

        node_count
    }

    #[test]
    fn node_count_visitor_records_node_statistics_records_statistics_correctly() {
        let mut visitor = NodeCountVisitor::default();

        let node1 = TestNode { children: 1 };
        let node2 = TestNode { children: 2 };
        let node3 = TestNode { children: 3 };

        visitor.count_node(0, "Inner", Some(node1.children));
        visitor.count_node(0, "Inner", Some(node1.children));
        visitor.count_node(0, "Inner", Some(node2.children));
        visitor.count_node(1, "Leaf", Some(node3.children));

        let node_count = &visitor.node_count;

        assert_eq!(node_count.levels_count.len(), 2);
        let level0 = &node_count.levels_count[0];
        let inner_stats = level0.get("Inner").unwrap();
        assert_eq!(inner_stats.size_count.get(&1), Some(&2));
        assert_eq!(inner_stats.size_count.get(&2), Some(&1));

        let level1 = &node_count.levels_count[1];
        let leaf_stats = level1.get("Leaf").unwrap();
        assert_eq!(leaf_stats.size_count.get(&3), Some(&1));
    }

    #[test]
    fn node_counts_by_kind_statistic_aggregates_node_sizes_correctly() {
        let statistic = NodeCountsByKindStatistic::new(&create_sample_node_counts());

        assert_eq!(statistic.total_nodes, 19);

        let inner_stats = statistic.aggregated_node_statistics.get("Inner").unwrap();
        assert_eq!(inner_stats.size_count.get(&2), Some(&5));
        assert_eq!(inner_stats.size_count.get(&3), Some(&1));
        assert_eq!(inner_stats.size_count.get(&4), Some(&3));

        let leaf_stats = statistic.aggregated_node_statistics.get("Leaf").unwrap();
        assert_eq!(leaf_stats.size_count.get(&1), Some(&10));
    }

    #[test]
    fn node_counts_by_level_statistic_computes_depths_correctly() {
        let statistic = NodeCountsByLevel::new(&create_sample_node_counts());
        assert_eq!(statistic.node_depth.get(&0), Some(&9));
        assert_eq!(statistic.node_depth.get(&1), Some(&10));
    }

    struct TestNode {
        children: u64,
    }
}
