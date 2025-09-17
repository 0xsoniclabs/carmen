// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use banderwagon::Fr;
use ipa_multipoint::{
    lagrange_basis::LagrangeBasis,
    multiproof::{MultiPoint, MultiPointProof, ProverQuery, VerifierQuery},
    transcript::Transcript,
};
use verkle_trie::constants::{CRS, PRECOMPUTED_WEIGHTS};

use crate::database::verkle::crypto::{Commitment, Scalar};

/// A proof demonstrating that a [`Commitment`] contains certain values at specific positions.
///
/// It uses the Inner Product Argument (IPA) proof system for Pedersen commitments.
/// Details: <https://dankradfeist.de/ethereum/2021/07/27/inner-product-arguments.html>
pub struct Opening {
    pub proof: MultiPointProof,
}

#[cfg_attr(not(test), expect(unused))]
impl Opening {
    /// Creates a new opening for a given commitment, proving that it contains the
    /// value at the specified position. The opening can then be verified using
    /// [`Self::verify_single`].
    ///
    /// For the opening to be valid, the `values` slice must contain the same sequence
    /// of scalars that were used to create the commitment.
    ///
    /// NOTE: This is fairly expensive operation (tens of milliseconds).
    /// Prefer to open multiple values or even multiple commitments at once, if possible.
    pub fn for_single(commitment: &Commitment, values: &[Scalar], position: u8) -> Self {
        let mut transcript = Transcript::new(b"vt");
        let query = ProverQuery {
            commitment: commitment.into(),
            point: position as usize,
            result: Fr::from(values[position as usize]),
            poly: LagrangeBasis::new(values.iter().map(|v| Fr::from(*v)).collect()),
        };
        let proof = MultiPoint::open(
            CRS.clone(),
            &PRECOMPUTED_WEIGHTS,
            &mut transcript,
            vec![query],
        );
        Opening { proof }
    }

    /// Like [`Self::for_single`], but creates an opening that proves all values
    /// contained in the commitment at once. The opening can then be verified using
    /// [`Self::verify_all`].
    pub fn for_all(commitment: &Commitment, values: &[Scalar]) -> Self {
        let mut transcript = Transcript::new(b"vt");
        let poly = LagrangeBasis::new(values.iter().map(|v| Fr::from(*v)).collect());
        let queries: Vec<ProverQuery> = values
            .iter()
            .enumerate()
            .map(|(i, value)| ProverQuery {
                commitment: commitment.into(),
                point: i,
                result: Fr::from(*value),
                poly: poly.clone(),
            })
            .collect();
        let proof = MultiPoint::open(CRS.clone(), &PRECOMPUTED_WEIGHTS, &mut transcript, queries);
        Opening { proof }
    }

    /// Verifies that the opening (created with [`Self::for_single`]) proves that the given
    /// commitment contains the specified value at the specified position.
    pub fn verify_single(&self, commitment: &Commitment, position: u8, value: &Scalar) -> bool {
        let mut transcript = Transcript::new(b"vt");
        let query = VerifierQuery {
            commitment: commitment.into(),
            point: Fr::from(position),
            result: Fr::from(*value),
        };
        self.proof
            .check(&CRS, &PRECOMPUTED_WEIGHTS, &[query], &mut transcript)
    }

    /// Verifies that the opening (created with [`Self::for_all`]) proves that the given
    /// commitment contains the specified sequence of values.
    pub fn verify_all(&self, commitment: &Commitment, values: &[Scalar]) -> bool {
        let mut transcript = Transcript::new(b"vt");
        let queries: Vec<VerifierQuery> = values
            .iter()
            .enumerate()
            .map(|(i, value)| VerifierQuery {
                commitment: commitment.into(),
                point: Fr::from(i as u8),
                result: Fr::from(*value),
            })
            .collect();
        self.proof
            .check(&CRS, &PRECOMPUTED_WEIGHTS, &queries, &mut transcript)
    }
}

#[cfg(test)]
mod slow_tests {
    use super::*;

    #[test]
    fn for_single_can_be_used_to_proof_individual_values() {
        let values: Vec<_> = (0..256).map(|i| Scalar::from(i + 1)).collect();
        let commitment = Commitment::new(&values);

        // Since opening is expensive, we only test a few positions.
        for i in [0, 5, 42, 100, 254, 255] {
            let opening = Opening::for_single(&commitment, &values, i);
            assert!(
                opening.verify_single(&commitment, i, &values[i as usize].clone()),
                "failed to verify position {i}"
            );
            assert!(
                !opening.verify_single(&commitment, i, &Scalar::from(i as u64 + 2)),
                "verified wrong value at position {i}"
            );
        }
    }

    #[test]
    fn using_different_values_results_in_invalid_opening() {
        let values_1: Vec<_> = (0..256).map(|i| Scalar::from(i + 1)).collect();
        let values_2: Vec<_> = (0..256).map(|i| Scalar::from(i + 2)).collect();
        let commitment = Commitment::new(&values_1);

        for i in [0, 5, 42, 100, 254, 255] {
            let opening = Opening::for_single(&commitment, &values_2, i);
            assert!(
                !opening.verify_single(&commitment, i, &values_1[i as usize].clone()),
                "verified value using invalid opening at position {i}"
            );
        }
    }

    #[test]
    fn commitment_update_can_be_used_to_proof_modified_values() {
        let mut values: Vec<_> = (0..256).map(|i| Scalar::from(i + 1)).collect();
        let mut commitment = Commitment::new(&values);

        for i in [0, 5, 42, 100, 254, 255] {
            let old_value = values[i];
            values[i] = Scalar::from(i as u64 * 10);
            commitment.update(i as u8, old_value, values[i]);
            let opening = Opening::for_single(&commitment, &values, i as u8);

            assert!(
                opening.verify_single(&commitment, i as u8, &values[i].clone()),
                "failed to verify updated position {i}"
            );

            assert!(
                !opening.verify_single(&commitment, i as u8, &old_value),
                "verified outdated value at position {i}"
            );
        }
    }

    #[test]
    fn for_all_can_be_used_to_proof_all_values() {
        let values: Vec<_> = (0..256).map(|i| Scalar::from(i + 1)).collect();
        let commitment = Commitment::new(&values);
        let opening = Opening::for_all(&commitment, &values);

        assert!(opening.verify_all(&commitment, &values));

        for i in 0..256 {
            let mut wrong_values = values.clone();
            wrong_values[i] = Scalar::from(9999);
            assert!(
                !opening.verify_all(&commitment, &wrong_values),
                "verified wrong value at position {i}"
            );
        }
    }

    #[test]
    fn single_proof_does_not_verify_all() {
        let values: Vec<_> = (0..256).map(|i| Scalar::from(i + 1)).collect();
        let commitment = Commitment::new(&values);
        let opening = Opening::for_single(&commitment, &values, 42);
        assert!(
            !opening.verify_all(&commitment, &values),
            "single proof verified all values"
        );
    }

    #[test]
    fn all_proof_does_not_verify_single() {
        let values: Vec<_> = (0..256).map(|i| Scalar::from(i + 1)).collect();
        let commitment = Commitment::new(&values);
        let opening = Opening::for_all(&commitment, &values);
        assert!(
            !opening.verify_single(&commitment, 42, &values[42].clone()),
            "all proof verified single value"
        );
    }
}
