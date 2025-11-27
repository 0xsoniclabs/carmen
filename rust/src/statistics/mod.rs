// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use crate::statistics::{formatters::StatisticsFormatter, node_count::NodeCountStatistic};

pub mod formatters;
pub mod node_count;
pub mod storage;

/// A trait for printing the available statistics on the implementing type with the provided
/// [`StatisticsFormatter`]s.
pub trait PrintStatistic {
    /// Prints all the available `Statistics` using the provided [`StatisticsFormatter`]s.
    fn print(&self, writers: &mut [Box<dyn StatisticsFormatter>]) -> std::io::Result<()>;
}

/// Union of all available statistics distributions.
#[derive(Debug)]
pub enum Statistic {
    NodeCount(NodeCountStatistic),
}
