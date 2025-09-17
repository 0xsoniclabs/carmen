use std::{fs::OpenOptions, os::fd::AsRawFd};

use io_uring::{IoUring, opcode, types};

use crate::storage::file::file_backend::{FileBackend, NoSeekFile};

const ITER: u64 = 10_000_000;

#[test]
fn write_multiple_no_seek_file() -> std::io::Result<()> {
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().join("test_file1.txt");

    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true);

    let file = NoSeekFile::open(path.as_path(), options).unwrap();

    let buf = [1u8; 32];

    let now = std::time::Instant::now();
    for i in 0..ITER {
        file.write_all_at(&buf, i * 32).unwrap();
    }

    println!("{:?}", now.elapsed());

    Ok(())
}

#[test]
fn read_multiple_no_seek_file() -> std::io::Result<()> {
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().join("test_file1.txt");

    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true);

    std::fs::write(&path, vec![1u8; (ITER * 32) as usize]).unwrap();
    let file = NoSeekFile::open(path.as_path(), options).unwrap();

    let mut buf = [1u8; 32];

    let now = std::time::Instant::now();
    for i in 0..ITER {
        file.read_exact_at(&mut buf, i * 32).unwrap();
    }

    println!("{:?}", now.elapsed());

    Ok(())
}

#[test]
fn write_multiple_io_uring() -> std::io::Result<()> {
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().join("test_file1.txt");

    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true);

    let file = options.open(path)?;

    let size = 10_000;
    let mut io_uring: IoUring = IoUring::builder()
        .setup_sqpoll(u32::MAX)
        .build(size)
        .map_err(std::io::Error::other)?;

    let mut buf = [0; 32];

    // SAFETY:
    unsafe {
        io_uring
            .submitter()
            .register_buffers(&[libc::iovec {
                iov_base: buf.as_mut_ptr() as *mut _,
                iov_len: buf.len(),
            }])
            .unwrap();
    }

    let now = std::time::Instant::now();
    for _ in 0..ITER / size as u64 {
        for i in 0..size {
            // let write_e =
            //     opcode::Write::new(types::Fd(file.as_raw_fd()), buf.as_ptr(), buf.len() as _)
            //         .offset(i as u64 * buf.len() as u64)
            //         .build();
            let write_e = opcode::WriteFixed::new(
                types::Fd(file.as_raw_fd()),
                buf.as_ptr(),
                buf.len() as _,
                0,
            )
            .offset(i as u64 * buf.len() as u64)
            .build();

            let mut sq = io_uring.submission();
            // SAFETY:
            unsafe {
                sq.push(&write_e).expect("submission queue is full");
            }
        }

        io_uring.submit_and_wait(size as usize)?;

        for entry in io_uring.completion() {
            // assert_eq!(cqe.user_data(), 0x1);
            match entry.result() {
                -1 => return Err(std::io::Error::other("io_uring write failed")),
                0 => return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof)),
                l @ 1.. if l == buf.len() as i32 => {}
                l => {
                    return Err(std::io::Error::other(format!(
                        "io_uring write: expected {} bytes, got {}",
                        buf.len(),
                        l
                    )));
                }
            }
        }
    }

    println!("{:?}", now.elapsed());

    Ok(())
}
