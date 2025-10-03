#[cfg(not(feature = "shuttle"))]
pub(crate) use std::sync::{
    Arc, Mutex, RwLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
#[cfg(not(feature = "shuttle"))]
#[cfg(test)]
pub(crate) use std::{
    hint,
    sync::{Barrier, RwLockReadGuard, RwLockWriteGuard, atomic::AtomicUsize},
    thread,
};

#[cfg(feature = "shuttle")]
pub(crate) use shuttle::{
    hint,
    sync::{
        Arc, Barrier, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    thread,
};
