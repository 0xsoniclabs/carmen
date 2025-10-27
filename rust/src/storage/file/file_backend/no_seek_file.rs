// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{fs::OpenOptions, os::unix::fs::FileExt, path::Path};

use crate::{error::BTResult, storage::file::FileBackend};

/// A wrapper around [`std::fs::File`] that implements [`FileBackend`] using the Unix-specific file
/// operations `pread` and `pwrite` which do not modify the file offset. This avoids the syscall for
/// seeking and allows for concurrent access without needing to manage a cursor.
pub struct NoSeekFile(std::fs::File);

impl FileBackend for NoSeekFile {
    fn open(path: &Path, options: OpenOptions) -> BTResult<Self, std::io::Error> {
        let file = options.open(path)?;
        file.try_lock().map_err(std::io::Error::from)?;
        Ok(Self(file))
    }

    fn write_all_at(&self, buf: &[u8], offset: u64) -> BTResult<(), std::io::Error> {
        self.0.write_all_at(buf, offset).map_err(Into::into)
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> BTResult<(), std::io::Error> {
        self.0.read_exact_at(buf, offset).map_err(Into::into)
    }

    fn flush(&self) -> BTResult<(), std::io::Error> {
        self.0.sync_all().map_err(Into::into)
    }

    fn len(&self) -> BTResult<u64, std::io::Error> {
        self.0.metadata().map(|m| m.len()).map_err(Into::into)
    }
}

// Note: The tests for `NoSeekFile as FileBackend` are in `file_backend.rs`.
