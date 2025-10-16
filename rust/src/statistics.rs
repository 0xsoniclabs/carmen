use std::{
    collections::{BTreeMap, HashMap},
    io::Write,
};

pub trait TrieStatistics {
    fn get_statistics(&self) -> Statistics;
}

#[derive(Default, Clone, Debug)]
pub struct Statistics {
    pub level_statistics: HashMap<u8, LevelStatistics>,
}

#[derive(Default, Clone, Debug)]
pub struct LevelStatistics {
    pub node_count: u64,
    pub node_statistics: BTreeMap<String, NodeStatistics>,
}

#[derive(Default, Clone, Debug)]
pub struct NodeStatistics {
    pub node_count: u64,
    pub node_kinds: BTreeMap<String, u64>,
}

impl std::ops::AddAssign<&NodeStatistics> for NodeStatistics {
    fn add_assign(&mut self, other: &NodeStatistics) {
        self.node_count += other.node_count;
        for (kind, count) in &other.node_kinds {
            *self.node_kinds.entry(kind.clone()).or_insert(0) += count;
        }
    }
}

impl Statistics {
    fn node_depth_distribution(&self) -> BTreeMap<u8, u64> {
        let mut distribution = BTreeMap::new();
        for (level, stats) in &self.level_statistics {
            distribution.insert(*level, stats.node_count);
        }
        distribution
    }

    pub fn print(&self, writer: &mut impl Write) -> std::io::Result<()> {
        writeln!(writer, "--- Node Statistics ---")?;
        let node_statistics = self.level_statistics.values().fold(
            HashMap::<String, NodeStatistics>::default(),
            |mut acc, stats| {
                for (type_name, type_stats) in &stats.node_statistics {
                    let entry = acc.entry(type_name.clone()).or_default();
                    *entry += type_stats;
                }
                acc
            },
        );
        writeln!(writer, "Node types:")?;
        for (type_name, stats) in node_statistics {
            writeln!(writer, "{} nodes, {}", type_name, stats.node_count)?;
        }

        writeln!(writer, "Node depth distribution:")?;
        for (level, count) in self.node_depth_distribution() {
            writeln!(writer, "{level}, {count}")?;
        }

        writeln!(writer, "--- Node type distribution ---")?;
        let mut levels = self.level_statistics.keys().collect::<Vec<_>>();
        levels.sort();
        for level in levels {
            let mut indentation = Indentation::default();
            let level_stats = &self.level_statistics[level];
            writeln!(writer, "Level {level}: ")?;
            indentation.inc();
            writeln!(
                writer,
                "{indentation}Total nodes: {}",
                level_stats.node_count
            )?;
            for (type_name, node_stats) in &level_stats.node_statistics {
                writeln!(
                    writer,
                    "{indentation}{} nodes: {}",
                    type_name, node_stats.node_count
                )?;
                indentation.inc();
                let mut kinds = node_stats.node_kinds.keys().collect::<Vec<_>>();
                kinds.sort_by_key(|k| natural_key(k));
                for kind in kinds {
                    let kind_count = node_stats.node_kinds[kind];
                    writeln!(writer, "{indentation}{kind}: {kind_count}")?;
                }
                indentation.dec();
            }
        }
        Ok(())
    }
}

pub trait Visitor<N> {
    fn visit(&mut self, node: &N, level: u8);
}

#[derive(Clone, Debug)]
struct Indentation {
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

impl Default for Indentation {
    fn default() -> Self {
        Self { level: 0, size: 4 } // 4 spaces
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Part {
    Text(String),
    Number(u64),
}

fn natural_key(s: &str) -> Vec<Part> {
    let mut parts = Vec::new();
    let mut buf = String::new();
    let mut in_number = false;

    for ch in s.chars() {
        if ch.is_ascii_digit() {
            if !in_number {
                if !buf.is_empty() {
                    parts.push(Part::Text(buf.clone()));
                    buf.clear();
                }
                in_number = true;
            }
            buf.push(ch);
        } else {
            if in_number {
                let num: u64 = buf.parse().unwrap_or(0);
                parts.push(Part::Number(num));
                buf.clear();
                in_number = false;
            }
            buf.push(ch);
        }
    }
    if !buf.is_empty() {
        if in_number {
            let num: u64 = buf.parse().unwrap_or(0);
            parts.push(Part::Number(num));
        } else {
            parts.push(Part::Text(buf));
        }
    }
    parts
}
