use crate::statistics::Distributions;

pub mod csv_writer;
pub mod writer_with_indentation;

pub trait StatisticsFormatter {
    fn print_distribution(&mut self, distribution: &Distributions) -> std::io::Result<()>;
}
