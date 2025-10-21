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
pub(crate) use std::sync::{
    Arc, Mutex, RwLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
#[cfg(not(feature = "shuttle"))]
#[cfg(test)]
pub(crate) use std::{
    hint,
    sync::{Barrier, RwLockReadGuard, RwLockWriteGuard},
};

#[cfg(feature = "shuttle")]
#[allow(unused_imports)]
pub(crate) use shuttle::{
    hint,
    sync::{
        Arc, Barrier, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
};
