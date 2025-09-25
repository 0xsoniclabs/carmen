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
/// https://blog.ethereum.org/2021/12/02/verkle-tree-structure#commitment-to-the-values-leaf-nodes
#[cfg_attr(not(test), expect(unused))]
pub fn compute_leaf_node_commitment(
    input_values: &[Value; 256],
    used_bits: &[u8; 256 / 8],
    stem: &[u8; 31],
) -> Commitment {
    let mut values = vec![vec![Commitment::default().to_scalar(); 256]; 2];
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
}
