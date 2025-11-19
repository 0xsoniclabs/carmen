// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use crate::{
    database::verkle::crypto::{Commitment, Scalar},
    types::Value,
};

/// Computes the commitment of a leaf node.
///
/// Since [`crate::database::verkle::crypto::Scalar`] cannot safely represent 32 bytes,
/// the 256 32-bit values are split into two interleaved sets of 16 byte values, on which
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
pub fn compute_leaf_node_commitment(
    input_values: &[Value; 256],
    used_bits: &[u8; 256 / 8],
    stem: &[u8; 31],
) -> Commitment {
    let mut values = [[Commitment::default().to_scalar(); 256]; 2];
    for (i, value) in input_values.iter().enumerate() {
        let mut lower = Scalar::from_le_bytes(&value[..16]);
        let upper = Scalar::from_le_bytes(&value[16..]);

        if used_bits[i / 8] & (1 << (i % 8)) != 0 {
            lower.set_bit128();
        }

        values[i / 128][(2 * i) % 256] = lower;
        values[i / 128][(2 * i + 1) % 256] = upper;
    }

    let c1 = Commitment::new(&values[0]);
    let c2 = Commitment::new(&values[1]);

    let combined = [
        Scalar::from(1),
        Scalar::from_le_bytes(stem),
        c1.to_scalar(),
        c2.to_scalar(),
    ];
    Commitment::new(&combined)
}

#[allow(clippy::too_many_arguments)]
pub fn compute_leaf_node_commitment_v2(
    old_values: &[Value; 256],
    new_values: &[Value; 256],
    changed: [u8; 256 / 8],
    stem: &[u8; 31],
    used_bits: &mut [u8; 256 / 8],
    c1: &mut Commitment,
    c2: &mut Commitment,
    c: &mut Commitment,
) {
    let changed_count = changed
        .iter()
        .map(|b| b.count_ones() as usize)
        .sum::<usize>();
    let use_batch_update = changed_count > 32;

    let mut deltas_c1 = [Scalar::zero(); 256];
    let mut deltas_c2 = [Scalar::zero(); 256];
    let prev_c1 = *c1;
    let prev_c2 = *c2;
    for (i, value) in old_values.iter().enumerate() {
        if changed[i / 8] & (1 << (i % 8)) == 0 {
            continue;
        }

        let mut prev_lower = Scalar::from_le_bytes(&value[..16]);
        let prev_upper = Scalar::from_le_bytes(&value[16..]);
        if used_bits[i / 8] & (1 << (i % 8)) != 0 {
            prev_lower.set_bit128();
        }
        let mut lower = Scalar::from_le_bytes(&new_values[i][..16]);
        let upper = Scalar::from_le_bytes(&new_values[i][16..]);
        lower.set_bit128();

        if use_batch_update {
            if i < 128 {
                deltas_c1[(i * 2) % 256] = lower - prev_lower;
                deltas_c1[(i * 2 + 1) % 256] = upper - prev_upper;
            } else {
                deltas_c2[(i * 2) % 256] = lower - prev_lower;
                deltas_c2[(i * 2 + 1) % 256] = upper - prev_upper;
            }
        } else {
            let c = if i < 128 { &mut *c1 } else { &mut *c2 };
            c.update(((i * 2) % 256) as u8, prev_lower, lower);
            c.update(((i * 2 + 1) % 256) as u8, prev_upper, upper);
        }
        used_bits[i / 8] |= 1 << (i % 8);
    }

    if use_batch_update {
        *c1 = prev_c1 + Commitment::new(&deltas_c1);
        *c2 = prev_c2 + Commitment::new(&deltas_c2);
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
    fn compute_leaf_node_commitment_produces_expected_values() {
        {
            let values = [Value::default(); 256];
            let used_bits = [0; 256 / 8];
            let stem = [0u8; 31];
            let commitment = compute_leaf_node_commitment(&values, &used_bits, &stem);
            let expected = Commitment::new(&[Scalar::from(1)]);
            assert_eq!(commitment, expected);
        }

        {
            let value1 = <[u8; 32]>::from_index_values(0, &[(8, 1), (20, 10)]);
            let value2 = <[u8; 32]>::from_index_values(0, &[(8, 2), (20, 20)]);

            let mut values = [Value::default(); 256];
            values[1] = Value::from(value1);
            values[130] = Value::from(value2);
            let mut used_bits = [0; 256 / 8];
            used_bits[1 / 8] |= 1 << 1;
            used_bits[130 / 8] |= 1 << (130 % 8);
            let stem = <[u8; 31]>::from_index_values(0, &[(0, 1), (1, 2), (2, 3)]);
            let commitment = compute_leaf_node_commitment(&values, &used_bits, &stem);

            // Value generated with Go reference implementation
            let expected = "0x56889d1fd78e20e2164261c44d1acde0964fe6351be92d7b5a6baf2914bc4c17";
            assert_eq!(const_hex::encode_prefixed(commitment.hash()), expected);
        }
    }

    // TODO: Add test for batched update

    #[test]
    fn compute_leaf_node_commitment_v2_produces_expected_values() {
        {
            let values = [Value::default(); 256];
            let stem = [0u8; 31];
            let mut commitment = Commitment::default();
            compute_leaf_node_commitment_v2(
                &values,
                &values,
                [0; 256 / 8],
                &stem,
                &mut [0; 256 / 8],
                &mut Commitment::default(),
                &mut Commitment::default(),
                &mut commitment,
            );
            let expected = Commitment::new(&[Scalar::from(1)]);
            assert_eq!(commitment, expected);
        }

        {
            let value1 = <[u8; 32]>::from_index_values(0, &[(8, 1), (20, 10)]);
            let value2 = <[u8; 32]>::from_index_values(0, &[(8, 2), (20, 20)]);

            let mut values = [Value::default(); 256];
            values[1] = Value::from(value1);
            values[130] = Value::from(value2);
            let mut changed_slots = [0; 256 / 8];
            changed_slots[1 / 8] |= 1 << 1;
            changed_slots[130 / 8] |= 1 << (130 % 8);
            let stem = <[u8; 31]>::from_index_values(0, &[(0, 1), (1, 2), (2, 3)]);
            let mut commitment = Commitment::default();
            compute_leaf_node_commitment_v2(
                &[Value::default(); 256],
                &values,
                changed_slots,
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
            let mut changed_slots = [0; 256 / 8];
            changed_slots[1 / 8] |= 1 << 1;
            changed_slots[130 / 8] |= 1 << (130 % 8);
            let stem = <[u8; 31]>::from_index_values(0, &[(0, 1), (1, 2), (2, 3)]);
            let mut committed_used_slots = [0; 256 / 8];
            let mut c1 = Commitment::default();
            let mut c2 = Commitment::default();
            let mut commitment = Commitment::default();
            compute_leaf_node_commitment_v2(
                &[Value::default(); 256],
                &values,
                changed_slots,
                &stem,
                &mut committed_used_slots,
                &mut c1,
                &mut c2,
                &mut commitment,
            );

            let mut new_values = [Value::default(); 256];
            new_values[1] = Value::from(value1b);
            new_values[130] = Value::from(value2b);
            compute_leaf_node_commitment_v2(
                &values,
                &new_values,
                changed_slots, // we can reuse this
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
