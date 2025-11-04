use std::io::Write;

use crate::statistics::{
    Distributions, NodeDistribution, NodePerLevelDistribution, NodeTypePerLevelDistribution,
    StatisticsFormatter, utils::natural_key,
};

pub struct CSVWriter {}

impl CSVWriter {
    fn init(suffix: &str) -> std::io::Result<impl Write> {
        std::fs::File::create(format!("carmen_stats_{suffix}.csv"))
    }

    fn print_node_distribution(distribution: &NodeDistribution) -> std::io::Result<()> {
        let mut wtr = Self::init("node_distribution")?;
        wtr.write_all(b"Node Type,Node Subtype,Node Count\n")?;
        for (type_name, stats) in &distribution.aggregated_node_statistics {
            wtr.write_all(
                format!("{},{},{}\n", type_name, type_name, stats.node_count).as_bytes(),
            )?;
            let mut kinds = stats.node_kinds.keys().collect::<Vec<_>>();
            kinds.sort_by_key(|k| natural_key(k));
            for kind in kinds {
                let kind_count = stats.node_kinds[kind];
                wtr.write_all(format!("{type_name},{kind},{kind_count}\n").as_bytes())?;
            }
        }
        Ok(())
    }

    fn print_node_level_distribution(dist: &NodePerLevelDistribution) -> std::io::Result<()> {
        let mut wtr = Self::init("node_per_level_distribution")?;
        wtr.write_all(b"Level,Node Count\n")?;
        for (level, count) in &dist.distribution {
            wtr.write_all(format!("{level},{count}\n").as_bytes())?;
        }
        Ok(())
    }

    fn print_node_type_per_level_distribution(
        dist: &NodeTypePerLevelDistribution,
    ) -> std::io::Result<()> {
        let mut wtr = Self::init("node_type_per_level_distribution")?;
        wtr.write_all(b"Level,Total Node Count,Node Type,Node Subtype,Node Count\n")?;
        let mut levels = dist.level_statistics.keys().collect::<Vec<_>>();
        levels.sort();
        for level in levels {
            let level_stats = &dist.level_statistics[level];
            for (type_name, node_stats) in &level_stats.node_statistics {
                wtr.write_all(
                    format!(
                        "{},{},{},{},{}\n",
                        level, level_stats.node_count, type_name, type_name, node_stats.node_count
                    )
                    .as_bytes(),
                )?;
                let mut sub_node_kinds = node_stats.node_kinds.keys().collect::<Vec<_>>();
                sub_node_kinds.sort_by_key(|k| natural_key(k));
                for kind in sub_node_kinds {
                    let kind_count = node_stats.node_kinds[kind];
                    wtr.write_all(
                        format!(
                            "{},{},{},{},{}\n",
                            level, level_stats.node_count, type_name, kind, kind_count
                        )
                        .as_bytes(),
                    )?;
                }
            }
        }
        Ok(())
    }
}
impl StatisticsFormatter for CSVWriter {
    fn print_distribution(&mut self, distribution: &Distributions) -> std::io::Result<()> {
        match distribution {
            Distributions::Node(item) => Self::print_node_distribution(item),
            Distributions::NodePerLevel(item) => Self::print_node_level_distribution(item),
            Distributions::NodeTypePerLevel(item) => {
                Self::print_node_type_per_level_distribution(item)
            }
        }
    }
}
