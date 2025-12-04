// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use rayon::prelude::*;

use crate::{
    database::verkle::crypto::{Commitment, Scalar},
    types::Value,
};

/// Computes the commitment of a leaf node.
///
/// This function performs incremental updates to the partial commitments `c1` and `c2`,
/// as well as the final commitment `c`, based on the values changed between the last
/// time the commitment was computed and now. To that end:
/// - `changed_indices` is a bitfield indicating which of the 256 values have changed since the last
///   commitment.
/// - `prev_values` and `cur_values` contain the previous and current values of the leaf node for
///   all indices that have changed. Other indices are ignored.
/// - `stem` is the stem of the leaf node and must remain the same between calls.
/// - `committed_used_indices` is a bitfield indicating which of the 256 values have been set
///   before, at the point of the last commitment. This is used to distinguish between empty values
///   and values that have been explicitly set to zero. This is an in/out parameter, all
///   `changed_indices` are marked as used afterwards.
/// - `c1`, `c2` and `c` are the leaf's two partial commitments and final commitment, respectively,
///   which are updated in place.
///
/// ## Theoretical Background
///
/// Since [`crate::database::verkle::crypto::Scalar`] cannot safely represent 32 bytes,
/// the 256 32-byte values are split into two interleaved sets of 16 byte values, on which
/// commitments C1 and C2 are computed separately:
///
///   C1 = Commit([  v[0][..16]),   v[0][16..]),   v[1][..16]),   v[1][16..]), ...])
///   C2 = Commit([v[128][..16]), v[128][16..]), v[129][..16]), v[129][16..]), ...])
///
/// The final commitment of a leaf node is then computed as follows:
///
///    C = Commit([1, stem, C1, C2])
///
/// For details on the commitment procedure, see
/// <https://blog.ethereum.org/2021/12/02/verkle-tree-structure#commitment-to-the-values-leaf-nodes>
#[allow(clippy::too_many_arguments)]
pub fn compute_leaf_node_commitment(
    changed_indices: [u8; 256 / 8],
    prev_values: &[Value; 256],
    cur_values: &[Value; 256],
    stem: &[u8; 31],
    committed_used_indices: &mut [u8; 256 / 8],
    c1: &mut Commitment,
    c2: &mut Commitment,
    c: &mut Commitment,
) {
    /// Computing a single commitment update is relatively cheap (in the order of ~10us),
    /// which means we have to balance it against the overhead introduced by Rayon workers.
    /// 16 was empirically determined to provide good performance.
    const MIN_UPDATES_PER_THREAD: usize = 16;

    let prev_c1 = *c1;
    let prev_c2 = *c2;

    let update_index = |i: usize| {
        let prev_value = prev_values[i];
        let cur_value = cur_values[i];

        let mut prev_lower = Scalar::from_le_bytes(&prev_value[..16]);
        let prev_upper = Scalar::from_le_bytes(&prev_value[16..]);
        if committed_used_indices[i / 8] & (1 << (i % 8)) != 0 {
            prev_lower.set_bit128();
        }
        let mut lower = Scalar::from_le_bytes(&cur_value[..16]);
        let upper = Scalar::from_le_bytes(&cur_value[16..]);
        lower.set_bit128();

        let mut delta_commitment = Commitment::default();
        delta_commitment.update(((i * 2) % 256) as u8, prev_lower, lower);
        delta_commitment.update(((i * 2 + 1) % 256) as u8, prev_upper, upper);
        delta_commitment
    };

    let c1_indices = (0..128).filter(|i| changed_indices[i / 8] & (1 << (i % 8)) != 0);
    let c2_indices = (128..256).filter(|i| changed_indices[i / 8] & (1 << (i % 8)) != 0);

    let c1_delta = c1_indices
        .collect::<Vec<_>>()
        .into_par_iter()
        .with_min_len(MIN_UPDATES_PER_THREAD)
        .map(update_index)
        .fold(Commitment::default, |acc, c| acc + c)
        .reduce(Commitment::default, |acc, c| acc + c);

    let c2_delta = c2_indices
        .collect::<Vec<_>>()
        .into_par_iter()
        .with_min_len(MIN_UPDATES_PER_THREAD)
        .map(update_index)
        .fold(Commitment::default, |acc, c| acc + c)
        .reduce(Commitment::default, |acc, c| acc + c);

    *c1 = *c1 + c1_delta;
    *c2 = *c2 + c2_delta;

    for i in 0..(256 / 8) {
        committed_used_indices[i] |= changed_indices[i];
    }

    if *c == Commitment::default() {
        let combined = [
            Scalar::from(1),
            Scalar::from_le_bytes(stem),
            c1.to_scalar(),
            c2.to_scalar(),
        ];
        *c = Commitment::new(&combined);
    } else {
        let deltas = [
            Scalar::zero(),
            Scalar::zero(),
            c1.to_scalar() - prev_c1.to_scalar(),
            c2.to_scalar() - prev_c2.to_scalar(),
        ];
        *c = *c + Commitment::new(&deltas);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::verkle::test_utils::FromIndexValues;

    #[test]
    fn compute_leaf_node_commitment_with_empty_values_is_commitment_of_one_and_stem() {
        let values = [Value::default(); 256];
        let stem = <[u8; 31]>::from_index_values(0, &[(0, 1), (1, 2), (2, 3)]);
        let mut commitment = Commitment::default();
        compute_leaf_node_commitment(
            [0; 256 / 8],
            &values,
            &values,
            &stem,
            &mut [0; 256 / 8],
            &mut Commitment::default(),
            &mut Commitment::default(),
            &mut commitment,
        );
        let expected = Commitment::new(&[Scalar::from(1), Scalar::from_le_bytes(&stem)]);
        assert_eq!(commitment, expected);
    }

    #[test]
    fn compute_leaf_node_commitment_produces_expected_values() {
        {
            let value1 = <[u8; 32]>::from_index_values(0, &[(8, 1), (20, 10)]);
            let value2 = <[u8; 32]>::from_index_values(0, &[(8, 2), (20, 20)]);

            let mut values = [Value::default(); 256];
            values[1] = Value::from(value1);
            values[130] = Value::from(value2);
            let mut changed = [0; 256 / 8];
            changed[1 / 8] |= 1 << 1;
            changed[130 / 8] |= 1 << (130 % 8);
            let stem = <[u8; 31]>::from_index_values(0, &[(0, 1), (1, 2), (2, 3)]);
            let mut commitment = Commitment::default();
            compute_leaf_node_commitment(
                changed,
                &[Value::default(); 256],
                &values,
                &stem,
                &mut [0; 256 / 8],
                &mut Commitment::default(),
                &mut Commitment::default(),
                &mut commitment,
            );

            // Value generated with Go reference implementation
            let expected = "0x56889d1fd78e20e2164261c44d1acde0964fe6351be92d7b5a6baf2914bc4c17";
            assert_eq!(const_hex::encode_prefixed(commitment.hash()), expected);
        }

        // Same as before, but we now first commit to two different values and then update them
        {
            let value1a = <[u8; 32]>::from_index_values(0, &[(8, 7), (20, 70)]);
            let value2a = <[u8; 32]>::from_index_values(0, &[(8, 8), (20, 80)]);
            let value1b = <[u8; 32]>::from_index_values(0, &[(8, 1), (20, 10)]);
            let value2b = <[u8; 32]>::from_index_values(0, &[(8, 2), (20, 20)]);

            let mut values = [Value::default(); 256];
            values[1] = Value::from(value1a);
            values[130] = Value::from(value2a);
            let mut changed = [0; 256 / 8];
            changed[1 / 8] |= 1 << 1;
            changed[130 / 8] |= 1 << (130 % 8);
            let stem = <[u8; 31]>::from_index_values(0, &[(0, 1), (1, 2), (2, 3)]);
            let mut committed_used_slots = [0; 256 / 8];
            let mut c1 = Commitment::default();
            let mut c2 = Commitment::default();
            let mut commitment = Commitment::default();
            compute_leaf_node_commitment(
                changed,
                &[Value::default(); 256],
                &values,
                &stem,
                &mut committed_used_slots,
                &mut c1,
                &mut c2,
                &mut commitment,
            );

            let mut new_values = [Value::default(); 256];
            new_values[1] = Value::from(value1b);
            new_values[130] = Value::from(value2b);
            compute_leaf_node_commitment(
                changed, // we can reuse this
                &values,
                &new_values,
                &stem,
                &mut committed_used_slots,
                &mut c1,
                &mut c2,
                &mut commitment,
            );

            // Value generated with Go reference implementation
            let expected = "0x56889d1fd78e20e2164261c44d1acde0964fe6351be92d7b5a6baf2914bc4c17";
            assert_eq!(const_hex::encode_prefixed(commitment.hash()), expected);
        }
    }
}
