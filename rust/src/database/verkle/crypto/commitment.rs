// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::sync::LazyLock;

use ark_ff::{BigInteger, PrimeField};
use banderwagon::{Element, Fr};
use ipa_multipoint::committer::{Committer, DefaultCommitter};
use verkle_trie::constants::CRS;
use zerocopy::{FromBytes, Immutable, IntoBytes, Unaligned};

use crate::database::verkle::crypto::Scalar;

/// A vector commitment to a sequence of 256 scalar values, using the Pedersen commitment scheme
/// on the Banderwagon curve.
///
/// Commitments can be de-/serialized from/to bytes using zerocopy.
/// Note that not all byte arrays correspond to valid commitments, i.e., points in the Banderwagon
/// prime subgroup. Performing any operations on invalid commitments will produce indeterminate
/// results (garbage in, garbage out).
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned)]
#[repr(C)]
pub struct Commitment {
    // We store a byte representation to allow for easy serialization using zerocopy.
    // TODO: Store `Element` directly once we have separate on-disk representations.
    // TODO: Consider using compressed 32-byte on-disk representation to save space.
    // https://github.com/0xsoniclabs/sonic-admin/issues/373
    point_bytes: [u8; 64],
}

// Creating the committer is very expensive (in the order of seconds!), so we cache it.
static COMMITTER: LazyLock<DefaultCommitter> = LazyLock::new(|| DefaultCommitter::new(&CRS.G));

impl Commitment {
    /// Creates a commitment to the given sequence of scalar values.
    /// At most 256 values are used, any additional values are ignored.
    /// If fewer than 256 values are provided, the remaining values are equal to
    /// the return value of [`Scalar::zero`].
    pub fn new(values: &[Scalar]) -> Self {
        // Note: The compiler should be able to eliminate this allocation, because `Fr::from` is a
        // no-op. If this is not the case and performance critical, Scalar could be marked
        // `#[repr(transparent)]` and then `iter-map-collect` and be replaced by a `transmute`.
        let point =
            COMMITTER.commit_lagrange(&values.iter().map(|v| Fr::from(*v)).collect::<Vec<Fr>>());
        Commitment {
            point_bytes: point.to_bytes_uncompressed(),
        }
    }

    /// Updates the commitment by changing the value at the given index.
    pub fn update(&mut self, index: u8, old_value: Scalar, new_value: Scalar) {
        let delta = Fr::from(new_value) - Fr::from(old_value);
        let delta_commitment = COMMITTER.scalar_mul(delta, index as usize);
        self.point_bytes = (self.as_element() + delta_commitment).to_bytes_uncompressed();
    }

    /// Maps the commitment point to the Banderwagon scalar field,
    /// allowing it to be used as input to other commitments.
    pub fn to_scalar(self) -> Scalar {
        Scalar::from(self.as_element().map_to_scalar_field())
    }

    /// Returns a hash corresponding to the commitment.
    /// Used for computing keys during Verkle trie state embedding.
    pub fn hash(&self) -> [u8; 32] {
        let scalar = self.as_element().map_to_scalar_field();
        scalar
            .into_bigint()
            .to_bytes_le()
            .try_into()
            .expect("scalar field element should be 32 bytes")
    }

    /// Returns a compressed 32-byte representation of the commitment.
    /// Used as the state root commitment in Verkle tries.
    pub fn compress(&self) -> [u8; 32] {
        self.as_element().to_bytes()
    }

    fn as_element(&self) -> Element {
        Element::from_bytes_unchecked_uncompressed(self.point_bytes)
    }
}

impl From<Element> for Commitment {
    fn from(element: Element) -> Self {
        Commitment {
            point_bytes: element.to_bytes_uncompressed(),
        }
    }
}

impl From<&Commitment> for Element {
    fn from(commitment: &Commitment) -> Self {
        commitment.as_element()
    }
}

impl Default for Commitment {
    fn default() -> Self {
        Commitment {
            point_bytes: Element::zero().to_bytes_uncompressed(),
        }
    }
}

#[cfg(test)]
mod slow_tests {
    use super::*;
    use crate::database::verkle::test_utils::FromIndexValues;

    #[test]
    fn default_is_commitment_to_zero_values() {
        let default_commitment = Commitment::default();
        let zero_values = vec![Scalar::from(0); 256];
        let zero_commitment = Commitment::new(&zero_values);
        assert_eq!(default_commitment, zero_commitment);
    }

    #[test]
    fn commitments_to_different_values_are_different() {
        let c1_values: Vec<_> = (0..256).map(Scalar::from).collect();
        let c2_values: Vec<_> = (0..256).map(|i| Scalar::from(i * 2)).collect();
        let c1 = Commitment::new(&c1_values);
        let c2 = Commitment::new(&c2_values);
        assert_eq!(c1, c1);
        assert_ne!(c1, c2);
    }

    #[test]
    fn new_commits_to_up_to_256_values() {
        let values: Vec<_> = (0..1024).map(Scalar::from).collect();
        let c1 = Commitment::new(&values);

        let values: Vec<_> = (0..512).map(Scalar::from).collect();
        let c2 = Commitment::new(&values);

        let values: Vec<_> = (0..256).map(Scalar::from).collect();
        let c3 = Commitment::new(&values);

        assert_eq!(c1, c2);
        assert_eq!(c1, c3);

        let values: Vec<_> = (0..255).map(Scalar::from).collect();
        let c4 = Commitment::new(&values);

        assert_ne!(c3, c4);

        let values = Vec::from_index_values(Scalar::zero(), &[(0, Scalar::from(42))]);
        let c5 = Commitment::new(&values);

        let values = Vec::from_index_values(
            Scalar::zero(),
            &[(0, Scalar::from(42)), (255, Scalar::from(0))],
        );
        let c6 = Commitment::new(&values);

        assert_eq!(c5, c6);
    }

    #[test]
    fn update_allows_modifying_individual_values() {
        let mut values: Vec<_> = (0..256).map(Scalar::from).collect();
        let mut commitment = Commitment::new(&values);

        for i in 0..256 {
            let old_value = values[i];
            values[i] = Scalar::from(i as u64 * 10);
            commitment.update(i as u8, old_value, values[i]);
            let expected_commitment = Commitment::new(&values);
            assert_eq!(commitment, expected_commitment);
        }
    }

    #[test]
    fn to_scalar_maps_point_to_scalar_field() {
        let c1 = Commitment::default();
        assert_eq!(c1.to_scalar(), Scalar::zero());
        let c2 = Commitment::new(&[Scalar::from(42)]);
        assert_ne!(c2.to_scalar(), Scalar::zero());
        assert_ne!(c2.to_scalar(), Scalar::from(42));
    }

    #[test]
    fn hash_works_as_expected() {
        let hash = Commitment::default().hash();
        let expected = [0u8; 32];
        assert_eq!(hash, expected);

        let values = vec![Scalar::from(12)];
        let hash = Commitment::new(&values).hash();
        // Value generated with Go reference implementation
        let expected = "0xb0852d6ab1ab96f7a08e042125eb4e2c11fb3a11a63cbe60246e2f8351fe9017";
        assert_eq!(const_hex::encode_prefixed(hash), expected);

        let values = vec![Scalar::from(12), Scalar::from(13), Scalar::from(42)];
        let hash = Commitment::new(&values).hash();
        // Value generated with Go reference implementation
        let expected = "0x029109502e1c90f2306e55eba15f3115f6a78edb3494246bf15b2def1911fb14";
        assert_eq!(const_hex::encode_prefixed(hash), expected);
    }

    #[test]
    fn compress_returns_commitment_in_compressed_form() {
        let values = vec![Scalar::from(12)];
        let commitment = Commitment::new(&values);
        let compressed = commitment.compress();
        assert_eq!(compressed, commitment.as_element().to_bytes());
    }

    #[test]
    fn can_be_serialized_to_and_from_bytes() {
        let c1 = Commitment::new(&vec![Scalar::from(42); 256]);
        let c2 = Commitment::new(&vec![Scalar::from(33); 256]);
        let c1_bytes = c1.as_bytes();
        let c2_bytes = c2.as_bytes();
        assert_ne!(c1_bytes, c2_bytes);
        let c1_from_bytes = Commitment::read_from_bytes(c1_bytes).unwrap();
        let c2_from_bytes = Commitment::read_from_bytes(c2_bytes).unwrap();
        assert_eq!(c1, c1_from_bytes);
        assert_eq!(c2, c2_from_bytes);
    }

    #[test]
    fn invalid_commitment_can_still_be_used() {
        let random_bytes = [0x01; 64];

        // First check that this byte sequence is in fact invalid
        let compressed = Element::from_bytes_unchecked_uncompressed(random_bytes).to_bytes();
        assert!(Element::from_bytes(&compressed).is_none());

        let mut c = Commitment::read_from_bytes(&random_bytes).unwrap();

        // These calls should not panic
        c.update(0, Scalar::from(0), Scalar::from(1));
        let _ = c.as_element();
        let _ = c.to_scalar();
        let _ = c.hash();
        let _ = c.compress();
    }
}
