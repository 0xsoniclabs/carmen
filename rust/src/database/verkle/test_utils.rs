// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

/// A utility trait to create an array-like object from a list of index-value pairs.
pub trait FromIndexValues {
    type Value;
    type Output;

    /// Creates a new [`Self::Output`], where the specified values are set at their respective
    /// indices. The remaining indices are set to the `default` value.
    fn from_index_values(
        default: Self::Value,
        index_values: &[(usize, Self::Value)],
    ) -> Self::Output;
}

impl<const N: usize> FromIndexValues for [u8; N] {
    type Value = u8;
    type Output = Self;

    fn from_index_values(
        default: Self::Value,
        index_values: &[(usize, Self::Value)],
    ) -> Self::Output {
        let mut result = [default; N];
        for (index, value) in index_values {
            result[*index] = *value;
        }
        result
    }
}

impl<T: Clone> FromIndexValues for Vec<T> {
    type Value = T;
    type Output = Self;

    fn from_index_values(
        default: Self::Value,
        index_values: &[(usize, Self::Value)],
    ) -> Self::Output {
        let max_index = index_values.iter().map(|(i, _)| *i).max().unwrap_or(0);
        let mut result = vec![default; max_index + 1];
        for (index, value) in index_values {
            result[*index] = value.clone();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_index_values_creates_array_with_provided_default_and_values() {
        let result = <[u8; 5]>::from_index_values(0, &[(0, 1), (2, 3)]);
        assert_eq!(result, [1, 0, 3, 0, 0]);

        let result = <[u8; 7]>::from_index_values(2, &[(5, 7), (1, 4)]);
        assert_eq!(result, [2, 4, 2, 2, 2, 7, 2]);
    }

    #[test]
    fn from_index_values_creates_vector_with_provided_default_and_values() {
        let result = Vec::<u8>::from_index_values(0, &[(0, 1), (2, 3)]);
        assert_eq!(result, vec![1, 0, 3]);

        let result = Vec::<u8>::from_index_values(2, &[(5, 7), (1, 4)]);
        assert_eq!(result, vec![2, 4, 2, 2, 2, 7]);
    }
}
