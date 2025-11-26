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

/// A component to collect statistics on the whole tree, organized per level.
#[derive(Default, Clone, Debug)]
pub struct TrieCount {
    pub levels_count: Vec<BTreeMap<String, NodeSize>>,
}

/// A component to collect statistics for a single node type and its subkinds.
#[derive(Default, Clone, Debug)]
pub struct NodeSize {
    pub size_count: BTreeMap<u64, u64>,
}

/// A visitor implementation that collects statistics about nodes.
#[derive(Default)]
pub struct TrieCountVisitor {
    pub trie_count: TrieCount,
}

impl TrieCountVisitor {
    /// Records statistics for the given node.
    /// It takes an optional function to count the number of children of the node and compute
    /// additional statistics over it.
    pub fn record_node_statistics<N>(
        &mut self,
        node: &N,
        level: u64,
        type_name: &str,
        count_num_children: Option<impl Fn(&N) -> u64>,
    ) {
        while self.trie_count.levels_count.len() <= level as usize {
            self.trie_count.levels_count.push(BTreeMap::new());
        }
        let level_entry = &mut self.trie_count.levels_count[level as usize];
        let node_entry = level_entry.entry(type_name.to_string()).or_default();
        if let Some(get_children_count) = count_num_children {
            let count = get_children_count(node);
            *node_entry.size_count.entry(count).or_insert(0) += 1;
        }
    }
}

/// Statistic about the node size distribution in the tree.
#[derive(Clone, Debug)]
pub struct NodeSizePerTreeStatistic {
    pub aggregated_node_statistics: BTreeMap<String, NodeSize>,
    pub total_nodes: u64,
}

impl NodeSizePerTreeStatistic {
    #[cfg_attr(not(test), expect(unused))]
    fn new(trie_count: &TrieCount) -> Self {
        let node_count = trie_count.levels_count.iter().fold(
            BTreeMap::<String, NodeSize>::default(),
            |mut acc, stats| {
                for (kind, count) in stats.iter() {
                    let new_entry = acc.entry(kind.clone()).or_default();
                    for (size, size_count) in count.size_count.iter() {
                        *new_entry.size_count.entry(*size).or_insert(0) += size_count;
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

/// Statistic about the node depth distribution in the tree.
#[derive(Clone, Debug)]
pub struct NodeDepthStatistic {
    pub node_depth: BTreeMap<usize, u64>,
}

impl NodeDepthStatistic {
    #[cfg_attr(not(test), expect(unused))]
    fn new(trie_count: &TrieCount) -> Self {
        let mut node_depth = BTreeMap::new();
        for (level, stats) in trie_count.levels_count.iter().enumerate() {
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

    fn create_sample_trie_count() -> TrieCount {
        let mut trie_count = TrieCount::default();

        // Level 0
        trie_count.levels_count.push(BTreeMap::new());
        trie_count.levels_count[0].insert(
            "Inner".to_string(),
            NodeSize {
                size_count: BTreeMap::from([(2, 3), (3, 1)]),
            },
        );
        trie_count.levels_count[0].insert(
            "Leaf".to_string(),
            NodeSize {
                size_count: BTreeMap::from([(1, 5)]),
            },
        );

        // Level 1
        trie_count.levels_count.push(BTreeMap::new());
        trie_count.levels_count[1].insert(
            "Inner".to_string(),
            NodeSize {
                size_count: BTreeMap::from([(2, 2), (4, 3)]),
            },
        );
        trie_count.levels_count[1].insert(
            "Leaf".to_string(),
            NodeSize {
                size_count: BTreeMap::from([(1, 5)]),
            },
        );

        trie_count
    }

    #[test]
    fn trie_count_visitor_records_node_statistics_records_statistics_correctly() {
        let mut visitor = TrieCountVisitor::default();

        let count_children = |node: &TestNode| node.children;

        let node1 = TestNode { children: 1 };
        let node2 = TestNode { children: 2 };
        let node3 = TestNode { children: 3 };

        visitor.record_node_statistics(&node1, 0, "Inner", Some(count_children));
        visitor.record_node_statistics(&node1, 0, "Inner", Some(count_children));
        visitor.record_node_statistics(&node2, 0, "Inner", Some(count_children));
        visitor.record_node_statistics(&node3, 1, "Leaf", Some(count_children));

        let trie_count = &visitor.trie_count;

        assert_eq!(trie_count.levels_count.len(), 2);
        let level0 = &trie_count.levels_count[0];
        let inner_stats = level0.get("Inner").unwrap();
        assert_eq!(inner_stats.size_count.get(&1), Some(&2));
        assert_eq!(inner_stats.size_count.get(&2), Some(&1));

        let level1 = &trie_count.levels_count[1];
        let leaf_stats = level1.get("Leaf").unwrap();
        assert_eq!(leaf_stats.size_count.get(&3), Some(&1));
    }

    #[test]
    fn node_size_per_tree_statistic_aggregates_node_sizes_correctly() {
        let statistic = NodeSizePerTreeStatistic::new(&create_sample_trie_count());

        assert_eq!(statistic.total_nodes, 19);

        let inner_stats = statistic.aggregated_node_statistics.get("Inner").unwrap();
        assert_eq!(inner_stats.size_count.get(&2), Some(&5));
        assert_eq!(inner_stats.size_count.get(&3), Some(&1));
        assert_eq!(inner_stats.size_count.get(&4), Some(&3));

        let leaf_stats = statistic.aggregated_node_statistics.get("Leaf").unwrap();
        assert_eq!(leaf_stats.size_count.get(&1), Some(&10));
    }

    #[test]
    fn node_depth_statistic_computes_depths_correctly() {
        let statistic = NodeDepthStatistic::new(&create_sample_trie_count());
        assert_eq!(statistic.node_depth.get(&0), Some(&9));
        assert_eq!(statistic.node_depth.get(&1), Some(&10));
    }

    struct TestNode {
        children: u64,
    }
}
