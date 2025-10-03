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

/// A utility trait to create an array-like object from a list of index-value pairs.
pub trait FromIndexValues {
    type Value;

    /// Creates a new [`Self`], where the specified values are set at their respective
    /// indices. The remaining indices are set to the `default` value.
    fn from_index_values(default: Self::Value, index_values: &[(usize, Self::Value)]) -> Self;
}

impl<const N: usize, T: Copy> FromIndexValues for [T; N] {
    type Value = T;

    fn from_index_values(default: Self::Value, index_values: &[(usize, Self::Value)]) -> Self {
        let mut result = [default; N];
        for (index, value) in index_values {
            result[*index] = *value;
        }
        result
    }
}

impl<T: Clone> FromIndexValues for Vec<T> {
    type Value = T;

    fn from_index_values(default: Self::Value, index_values: &[(usize, Self::Value)]) -> Self {
        if let Some(max_index) = index_values.iter().map(|(i, _)| *i).max() {
            let mut result = vec![default; max_index + 1];
            for (index, value) in index_values {
                result[*index] = value.clone();
            }
            result
        } else {
            vec![]
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_index_values_creates_array_with_provided_default_and_values() {
        let result = <[u8; 3]>::from_index_values(1, &[]);
        assert_eq!(result, [1, 1, 1]);

        let result = <[u8; 5]>::from_index_values(0, &[(0, 1), (2, 3)]);
        assert_eq!(result, [1, 0, 3, 0, 0]);

        let result = <[u8; 7]>::from_index_values(2, &[(5, 7), (1, 4)]);
        assert_eq!(result, [2, 4, 2, 2, 2, 7, 2]);
    }

    #[test]
    fn from_index_values_creates_vector_with_provided_default_and_values() {
        let result = Vec::<u8>::from_index_values(1, &[]);
        assert_eq!(result, vec![0u8; 0]);

        let result = Vec::<u8>::from_index_values(0, &[(0, 1), (2, 3)]);
        assert_eq!(result, vec![1, 0, 3]);

        let result = Vec::<u8>::from_index_values(2, &[(5, 7), (1, 4)]);
        assert_eq!(result, vec![2, 4, 2, 2, 2, 7]);
    }
}
