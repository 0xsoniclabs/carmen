// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

mod file_backend;
mod file_storage_manager;
#[cfg(all(test, target_os = "linux"))]
mod io_uring_file;
mod node_file_storage;
#[cfg(test)]
mod other;
#[cfg(unix)]
mod page_cached_file;
#[cfg(unix)]
mod page_utils;

pub use file_backend::*;
#[cfg(test)]
pub use file_storage_manager::{FileStorageManager, MockFileStorageManager};
#[cfg(all(test, target_os = "linux"))]
use io_uring_file::IoUringFile;
pub use node_file_storage::NodeFileStorage;
#[cfg(all(test, unix))]
pub use page_cached_file::PageCachedFile;
