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

use crate::storage::file::FileBackend;

/// A wrapper around [`std::fs::File`] that implements [`FileBackend`] using a mutex to ensure
/// exclusive access to the file. This is suitable for platforms where `pread` and `pwrite` are not
/// available or where seeking is required for other reasons.
pub struct SeekFile(Mutex<std::fs::File>);

impl FileBackend for SeekFile {
    fn open(path: &Path, options: OpenOptions) -> std::io::Result<Self> {
        let file = options.open(path)?;
        file.try_lock()?;
        Ok(Self(Mutex::new(file)))
    }

    fn write_all_at(&self, buf: &[u8], offset: u64) -> std::io::Result<()> {
        let mut file = self.0.lock().unwrap();
        file.seek(SeekFrom::Start(offset))?;
        file.write_all(buf)
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        let mut file = self.0.lock().unwrap();
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(buf)
    }

    fn flush(&self) -> std::io::Result<()> {
        self.0.lock().unwrap().sync_all()
    }

    fn len(&self) -> std::io::Result<u64> {
        self.0.lock().unwrap().metadata().map(|m| m.len())
    }

    fn set_len(&self, len: u64) -> std::io::Result<()> {
        self.0.lock().unwrap().set_len(len)
    }
}

// Note: The tests for `SeekFile as FileBackend` are in `file_backend.rs`.
