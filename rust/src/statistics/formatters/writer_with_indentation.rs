use std::io::Write;

use crate::statistics::{
    NodeDepthDistribution, NodeDistribution, formatters::StatisticsFormatter, utils::natural_key,
};

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

    pub fn inc(&mut self) {
        self.indentation.inc();
    }

    pub fn dec(&mut self) {
        self.indentation.dec();
    }

    pub fn reset(&mut self) {
        self.indentation.reset();
    }
}

impl<W: Write> Write for WriterWithIndentation<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let indented_buf = format!("{}{}", self.indentation, String::from_utf8_lossy(buf));
        self.writer
            .write(indented_buf.as_bytes())
            .map(|_| buf.len()) //TODO: why the hell I need this?
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

impl<W: Write> StatisticsFormatter for WriterWithIndentation<W> {
    fn header(&mut self) -> std::io::Result<()> {
        self.reset();
        self.write_all(b"--- Trie Statistics ---")
    }

    fn print_node_distribution(&mut self, item: &NodeDistribution) -> std::io::Result<()> {
        self.reset();
        self.write_fmt(format_args!("Total nodes: {}", item.total_nodes))?;
        self.inc();
        self.write_all(b"Node types:")?;
        self.inc();
        for (type_name, stats) in &item.aggregated_node_statistics {
            self.write_fmt(format_args!("{} nodes: {}", type_name, stats.node_count))?;
            self.inc();
            let mut kinds = stats.node_kinds.keys().collect::<Vec<_>>();
            kinds.sort_by_key(|k| natural_key(k));
            for kind in kinds {
                let kind_count = stats.node_kinds[kind];
                self.write_fmt(format_args!("{kind}: {kind_count}"))?;
            }
            self.dec();
        }
        Ok(())
    }

    fn print_node_depth_distribution(
        &mut self,
        item: &NodeDepthDistribution,
    ) -> std::io::Result<()> {
        self.reset();
        self.write_all(b"Node depth distribution:")?;
        for (level, count) in &item.distribution {
            self.write_fmt(format_args!("{level}, {count}"))?;
        }
        Ok(())
    }

    fn print_node_type_distribution(
        &mut self,
        item: &super::NodeTypeDistribution,
    ) -> std::io::Result<()> {
        self.reset();
        self.write_all(b"--- Node type distribution ---")?;
        let mut levels = item.level_statistics.keys().collect::<Vec<_>>();
        levels.sort();
        for level in levels {
            let mut ind = Indentation::default();
            let level_stats = &item.level_statistics[level];
            self.write_all(b"Level {level}: ")?;
            ind.inc();
            self.write_fmt(format_args!("{ind}Total nodes: {}", level_stats.node_count))?;
            ind.inc();
            for (type_name, node_stats) in &level_stats.node_statistics {
                self.write_fmt(format_args!(
                    "{ind}{} nodes: {}",
                    type_name, node_stats.node_count
                ))?;
                ind.inc();
                let mut kinds = node_stats.node_kinds.keys().collect::<Vec<_>>();
                kinds.sort_by_key(|k| natural_key(k));
                for kind in kinds {
                    let kind_count = node_stats.node_kinds[kind];
                    self.write_fmt(format_args!("{ind}{kind}: {kind_count}"))?;
                }
                ind.dec();
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Indentation {
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

    pub fn reset(&mut self) {
        let default = Indentation::default();
        self.level = default.level;
        self.size = default.size;
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
