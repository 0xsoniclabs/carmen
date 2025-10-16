use std::io::Write;

use crate::statistics::StatisticsFormatter;

pub struct CSVWriter<W: Write> {
    pub writer: W,
}

impl<W: Write> Write for CSVWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.writer.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

impl<W: Write> StatisticsFormatter for CSVWriter<W> {
    fn print_node_distribution(&mut self, item: &super::NodeDistribution) -> std::io::Result<()> {
        todo!()
    }

    fn print_node_depth_distribution(
        &mut self,
        item: &super::NodeDepthDistribution,
    ) -> std::io::Result<()> {
        todo!()
    }

    fn print_node_type_distribution(
        &mut self,
        item: &super::NodeTypeDistribution,
    ) -> std::io::Result<()> {
        todo!()
    }

    fn header(&mut self) -> std::io::Result<()> {
        todo!()
    }
}
