// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

#[cfg(not(feature = "shuttle"))]
#[cfg(test)]
pub(crate) use std::sync::Barrier;
#[cfg(not(feature = "shuttle"))]
pub(crate) use std::{
    hint,
    sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard, atomic},
    thread,
};

#[cfg(feature = "shuttle")]
#[cfg(test)]
pub(crate) use shuttle::sync::Barrier;
#[cfg(feature = "shuttle")]
pub(crate) use shuttle::{
    hint,
    sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard, atomic},
    thread,
};

/// Wraps the `is_finished` method of `std::thread::JoinHandle`, which is not available in shuttle
/// and calls `unimplemented!` instead.
pub fn is_finished(_handle: &thread::JoinHandle<impl Send>) -> bool {
    #[cfg(not(feature = "shuttle"))]
    {
        _handle.is_finished()
    }
    #[cfg(feature = "shuttle")]
    {
        unimplemented!("shuttle threads do not support is_finished")
    }
}
