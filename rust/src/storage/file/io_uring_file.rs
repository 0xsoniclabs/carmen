use std::{
    fs::{File, OpenOptions},
    os::fd::AsRawFd,
    path::Path,
    sync::{Condvar, Mutex},
};

use io_uring::{IoUring, opcode, types};

use crate::storage::file::file_backend::FileBackend;

pub struct IoUringFile {
    file: File,
    io_uring_pool: Mutex<Vec<IoUring>>,
    cond: Condvar,
}

impl FileBackend for IoUringFile {
    fn open(path: &Path, options: OpenOptions) -> std::io::Result<Self> {
        let file = options.open(path)?;
        file.try_lock()?;

        let io_uring_pool = Mutex::new(
            (0..10)
                .map(|_| {
                    IoUring::builder()
                        // .setup_sqpoll(u32::MAX)
                        .build(1)
                        .map_err(std::io::Error::other)
                })
                .collect::<Result<_, _>>()?,
        );

        Ok(Self {
            file,
            io_uring_pool,
            cond: Condvar::new(),
        })
    }

    fn write_all_at(&self, buf: &[u8], offset: u64) -> std::io::Result<()> {
        let write_e = opcode::Write::new(
            types::Fd(self.file.as_raw_fd()),
            buf.as_ptr(),
            buf.len() as _,
        )
        .offset(offset)
        .build();
        // .user_data(0x1);

        let mut io_uring = self.get_io_uring()?;

        let mut sq = io_uring.submission();
        // SAFETY:
        // The contents of `write_e` are valid for the duration of the operation, because both the
        // buffer `buf` and the file descriptor are borrowed for the duration of the function and
        // the operation is waited on in this function.
        unsafe {
            sq.push(&write_e).expect("submission queue is full");
        }
        drop(sq);

        io_uring.submit_and_wait(1)?;

        let cqe = io_uring
            .completion()
            .next()
            .expect("completion queue is empty");

        self.release_io_uring(io_uring);

        // assert_eq!(cqe.user_data(), 0x1);
        match cqe.result() {
            -1 => return Err(std::io::Error::other("io_uring write failed")),
            0 => return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof)),
            l @ 1.. if l == buf.len() as i32 => {}
            _ => {
                return Err(std::io::Error::other(format!(
                    "io_uring write: expected {} bytes, got {}",
                    buf.len(),
                    cqe.result()
                )));
            }
        }

        Ok(())
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        let read_e = opcode::Read::new(
            types::Fd(self.file.as_raw_fd()),
            buf.as_mut_ptr(),
            buf.len() as _,
        )
        .offset(offset)
        .build();
        // .user_data(0x2);

        let mut io_uring = self.get_io_uring()?;

        let mut sq = io_uring.submission();
        // SAFETY:
        // The contents of `read_e` are valid for the duration of the operation, because both the
        // buffer `buf` and the file descriptor are borrowed for the duration of the function and
        // the operation is waited on in this function.
        unsafe {
            sq.push(&read_e).expect("submission queue is full");
        }
        drop(sq);

        io_uring.submit_and_wait(1)?;

        let cqe = io_uring
            .completion()
            .next()
            .expect("completion queue is empty");

        self.release_io_uring(io_uring);

        // assert_eq!(cqe.user_data(), 0x2);
        match cqe.result() {
            -1 => return Err(std::io::Error::other("io_uring read failed")),
            0 => return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof)),
            l @ 1.. if l == buf.len() as i32 => {}
            _ => {
                return Err(std::io::Error::other(format!(
                    "io_uring read: expected {} bytes, got {}",
                    buf.len(),
                    cqe.result()
                )));
            }
        }

        Ok(())
    }

    fn flush(&self) -> std::io::Result<()> {
        self.file.sync_all()
    }

    fn len(&self) -> Result<u64, std::io::Error> {
        self.file.metadata().map(|m| m.len())
    }

    fn set_len(&self, size: u64) -> std::io::Result<()> {
        self.file.set_len(size)
    }
}

impl IoUringFile {
    fn get_io_uring(&self) -> std::io::Result<IoUring> {
        let mut pool = self.io_uring_pool.lock().unwrap();
        loop {
            match pool.pop() {
                Some(uring) => return Ok(uring),
                None => {
                    // Wait for a connection to become available
                    pool = self.cond.wait(pool).unwrap();
                }
            }
        }
    }

    fn release_io_uring(&self, io_uring: IoUring) {
        let mut pool = self.io_uring_pool.lock().unwrap();
        pool.push(io_uring);
        self.cond.notify_one();
    }
}

// Note: The unit tests for `PageCachedFile<F> as FileBackend` are in `file_backend.rs`.
