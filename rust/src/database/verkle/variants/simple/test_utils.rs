// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use crate::types::{Key, Value};

pub fn make_key(prefix: &[u8]) -> Key {
    let mut key = [0; 32];
    key[..prefix.len()].copy_from_slice(prefix);
    key
}

pub fn make_leaf_key(prefix: &[u8], suffix: u8) -> Key {
    let mut key = make_key(prefix);
    key[31] = suffix;
    key
}

pub fn make_value(value: u64) -> Value {
    let mut val = [0; 32];
    val[0..8].copy_from_slice(&value.to_le_bytes());
    val
}
