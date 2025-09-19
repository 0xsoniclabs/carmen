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
