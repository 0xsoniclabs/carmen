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

    let combined = vec![
        Scalar::from(1),
        Scalar::from_le_bytes(stem),
        c1.to_scalar(),
        c2.to_scalar(),
    ];
    Commitment::new(&combined)
}
