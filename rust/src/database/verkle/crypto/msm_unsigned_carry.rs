//! Windowed MSM with unsigned carry, ported from Go implementation for Verkle trees.
use ark_ec::CurveGroup;
use ark_ed_on_bls12_381_bandersnatch::{EdwardsAffine, EdwardsProjective, Fr};
use ark_ff::{BigInteger256, Field, Zero};
use banderwagon::Element;

const SUPPORTED_MSM_LENGTH: usize = 256;
const WINDOW_16VS8_INDEX_LIMIT: usize = 5;

/// Precomputed table for a single point.
struct PrecompPoint {
    window_size: usize,
    windows: Vec<Vec<EdwardsAffine>>, // [window][entry]
}

/// MSM precomputation engine for fixed basis.
pub struct MSMPrecomp {
    precomp_points: Vec<PrecompPoint>,
}

impl MSMPrecomp {
    /// Create a new MSMPrecomp with precomputed tables.
    pub fn new(points: &[Element]) -> Self {
        assert!(points.len() <= SUPPORTED_MSM_LENGTH);
        let mut precomp_points = Vec::with_capacity(SUPPORTED_MSM_LENGTH);
        for (i, pt) in points.iter().enumerate() {
            let window_size = if i < WINDOW_16VS8_INDEX_LIMIT { 16 } else { 8 };
            precomp_points.push(PrecompPoint::new(&pt.0, window_size));
        }
        Self { precomp_points }
    }

    /// Perform MSM on the fixed basis with given scalars.
    pub fn mul(&self, scalars: &[Fr]) -> Element {
        let mut result = EdwardsProjective::zero();
        for (i, scalar) in scalars.iter().enumerate() {
            if !scalar.is_zero() {
                self.precomp_points[i].scalar_mul(scalar, &mut result);
            }
        }
        Element(result)
    }

    pub fn mul_index(&self, scalar: Fr, index: usize) -> Element {
        let mut result = EdwardsProjective::zero();
        self.precomp_points[index].scalar_mul(&scalar, &mut result);
        Element(result)
    }
}

impl PrecompPoint {
    /// Precompute tables for a point and window size.
    fn new(point: &EdwardsProjective, window_size: usize) -> Self {
        // CurveGroup trait is required for normalize_batch
        if !window_size.is_power_of_two() {
            panic!("window_size must be power of 2");
        }
        let num_windows = 256 / window_size;
        let mut windows = Vec::with_capacity(num_windows);
        // TODO: Do this in parallel using rayon
        for window_index in 0..num_windows {
            // Calculate base = point * 2^{window_size * window_index} safely
            let exp = (window_size * window_index) as u64;
            let scalar = Fr::from(2u64).pow([exp]);
            let base = *point * scalar;
            let mut entries_proj = Vec::with_capacity(1 << (window_size - 1));
            let mut curr = base;
            for _ in 0..(1 << (window_size - 1)) {
                entries_proj.push(curr);
                curr += base;
            }
            // Batch normalize all entries to affine
            let entries_affine = EdwardsProjective::normalize_batch(&entries_proj);
            windows.push(entries_affine);
        }
        Self {
            window_size,
            windows,
        }
    }

    /// Multiply the precomputed point by scalar, pushing carry between windows.
    fn scalar_mul(&self, scalar: &Fr, res: &mut EdwardsProjective) {
        let num_windows_in_limb = 64 / self.window_size;
        // let scalar_repr = scalar.0;
        let scalar_repr: BigInteger256 = (*scalar).into();

        let mut carry = 0u64;
        for (l, limb) in scalar_repr.as_ref().iter().enumerate() {
            for w in 0..num_windows_in_limb {
                let mask = (1u64 << self.window_size) - 1;
                let mut window_value = ((limb >> (self.window_size * w)) & mask) + carry;
                if window_value == 0 {
                    continue;
                }
                carry = 0;
                if window_value > (1 << (self.window_size - 1)) {
                    window_value = (1 << self.window_size) - window_value;
                    if window_value != 0 {
                        let neg =
                            -self.windows[l * num_windows_in_limb + w][window_value as usize - 1];
                        *res += EdwardsProjective::from(neg);
                    }
                    carry = 1;
                } else {
                    *res += EdwardsProjective::from(
                        self.windows[l * num_windows_in_limb + w][window_value as usize - 1],
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use banderwagon::msm_windowed_sign::MSMPrecompWindowSigned;
    use verkle_trie::constants::CRS;

    use super::*;
    use crate::database::verkle::crypto::Scalar;

    #[test]
    fn compare_against_msm_window_sign() {
        let signed = MSMPrecompWindowSigned::new(&CRS.G, 8);
        let unsigned = MSMPrecomp::new(&CRS.G);

        // Test single zero
        let res_signed = signed.mul(&[Fr::from(0u64); 1]);
        let res_unsigned = unsigned.mul(&[Fr::from(0u64); 1]);
        assert_eq!(res_signed, res_unsigned, "MSM(0) mismatch");

        // Test single one
        let res_signed = signed.mul(&[Fr::from(1u64); 1]);
        let res_unsigned = unsigned.mul(&[Fr::from(1u64); 1]);
        assert_eq!(res_signed, res_unsigned, "MSM(1) mismatch");

        // Test single other value
        let res_signed = signed.mul(&[Fr::from(33u64); 1]);
        let res_unsigned = unsigned.mul(&[Fr::from(33u64); 1]);
        assert_eq!(res_signed, res_unsigned, "MSM(33) mismatch");

        // Test multiple values
        let scalars: Vec<Fr> = (0..256).map(|i| Fr::from(i as u64 * 7u64)).collect();
        let res_signed = signed.mul(&scalars);
        let res_unsigned = unsigned.mul(&scalars);
        assert_eq!(res_signed, res_unsigned, "MSM(multiple) mismatch");

        // Test single larger scalar
        let mut bigint: BigInteger256 = BigInteger256::default();
        // TODO: Also try setting other limbs
        // bigint.0[0] = u64::MAX;
        // bigint.0[0] = 1 << 63;
        // bigint.0[0] = 3 << 62;
        bigint.0[1] = u64::MAX;
        let fr = Fr::from(bigint);
        let res_signed = signed.mul(&[fr]);
        let res_unsigned = unsigned.mul(&[fr]);
        assert_eq!(res_signed, res_unsigned, "MSM(large scalar) mismatch");

        // Test very large scalar
        let random = Scalar::from_le_bytes(
            &const_hex::decode("8ace54a66ae992faf22d3eedb0edecff16ded1e168c474263519eb3b388008b4")
                .unwrap(),
        );
        let res_signed = signed.mul(&[random.into()]);
        let res_unsigned = unsigned.mul(&[random.into()]);
        assert_eq!(res_signed, res_unsigned, "MSM(large scalar) mismatch");

        // TODO: Consider doing a test that compares against manual multiplication w/ generators
        // let signed = MSMPrecompWindowSigned::new(&CRS.G[..1], 8);
        // let unsigned = MSMPrecomp::new(&CRS.G[..1]);
        // let res_signed = signed.mul(&[Fr::from(1u64)]);
        // let res_unsigned = unsigned.mul(&[Fr::from(1u64)]);
        // assert_eq!(res_signed, CRS.G[0] * Fr::from(1u64));
        // assert_eq!(res_signed, res_unsigned);
    }
}
