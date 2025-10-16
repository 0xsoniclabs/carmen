use crate::statistics::{NodeDepthDistribution, NodeDistribution, NodeTypeDistribution};

pub mod csv_writer;
pub mod writer_with_indentation;

pub trait StatisticsFormatter {
    fn header(&mut self) -> std::io::Result<()>;
    fn print_node_distribution(&mut self, item: &NodeDistribution) -> std::io::Result<()>;
    fn print_node_depth_distribution(
        &mut self,
        item: &NodeDepthDistribution,
    ) -> std::io::Result<()>;
    fn print_node_type_distribution(&mut self, item: &NodeTypeDistribution) -> std::io::Result<()>;
}
