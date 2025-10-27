// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{
    fs::OpenOptions,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    sync::Mutex,
};

use crate::{error::BTResult, storage::file::FileBackend};

/// A wrapper around [`std::fs::File`] that implements [`FileBackend`] using a mutex to ensure
/// exclusive access to the file. This is suitable for platforms where `pread` and `pwrite` are not
/// available or where seeking is required for other reasons.
pub struct SeekFile(Mutex<std::fs::File>);

impl FileBackend for SeekFile {
    fn open(path: &Path, options: OpenOptions) -> BTResult<Self, std::io::Error> {
        let file = options.open(path)?;
        file.try_lock().map_err(std::io::Error::from)?;
        Ok(Self(Mutex::new(file)))
    }

    fn write_all_at(&self, buf: &[u8], offset: u64) -> BTResult<(), std::io::Error> {
        let mut file = self.0.lock().unwrap();
        file.seek(SeekFrom::Start(offset))?;
        file.write_all(buf).map_err(Into::into)
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> BTResult<(), std::io::Error> {
        let mut file = self.0.lock().unwrap();
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(buf).map_err(Into::into)
    }

    fn flush(&self) -> BTResult<(), std::io::Error> {
        self.0.lock().unwrap().sync_all().map_err(Into::into)
    }

    fn len(&self) -> BTResult<u64, std::io::Error> {
        self.0
            .lock()
            .unwrap()
            .metadata()
            .map(|m| m.len())
            .map_err(Into::into)
    }
}

// Note: The tests for `SeekFile as FileBackend` are in `file_backend.rs`.
