// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::ops::Sub;

use ark_ff::{BigInteger, fields::PrimeField};
use banderwagon::Fr;

/// A value from the Banderwagon scalar field, which uses a 253-bit prime.
/// One or more scalars can be committed to by creating a
/// [`crate::database::verkle::crypto::Commitment`].
/// Scalars can safely commit to values of at most 252 bits (slightly more than 31 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scalar(Fr);

impl Scalar {
    /// Returns the scalar value representing zero.
    /// Same as [`Scalar::from`]`(0)`.
    pub fn zero() -> Self {
        Scalar(Fr::from(0))
    }

    /// Creates a new scalar from the given little-endian byte slice.
    /// At most 31 bytes are read from the slice, shorter slices are right-padded with zeros.
    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        Scalar(Fr::from_le_bytes_mod_order(&bytes[..bytes.len().min(31)]))
    }

    /// Sets the 128th bit of the scalar to 1.
    /// This is a convenience method used for computing the commitment of a
    /// leaf node in Ethereum Verkle tries, where the 128th bit indicates
    /// that a value has been set before (to distinguish it from the default
    /// value, which is needed for future state expiry schemes).
    pub fn set_bit128(&mut self) {
        let mut bytes = self.0.into_bigint().to_bytes_le();
        bytes[16] |= 0x01;
        self.0 = Fr::from_le_bytes_mod_order(&bytes);
    }
}

impl Sub<Self> for Scalar {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Scalar(self.0 - rhs.0)
    }
}

impl From<u64> for Scalar {
    fn from(value: u64) -> Self {
        Scalar(Fr::from(value))
    }
}

impl From<Scalar> for Fr {
    fn from(scalar: Scalar) -> Self {
        scalar.0
    }
}

impl From<Fr> for Scalar {
    fn from(fr: Fr) -> Self {
        Scalar(fr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_returns_zero_scalar() {
        let zero = Scalar::zero();
        assert_eq!(zero.0, Fr::from(0));
    }

    #[test]
    fn can_be_created_from_u64() {
        let values = [0, 1, 42, u64::MAX - 1, u64::MAX];
        for value in values {
            assert_eq!(Scalar::from(value).0, Fr::from(value));
        }
    }

    #[test]
    fn can_be_converted_to_and_from_banderwagon_fr() {
        let values = [0, 1, 42, u64::MAX - 1, u64::MAX];
        for value in values {
            let scalar = Scalar::from(value);
            let fr = Fr::from(scalar);
            assert_eq!(fr, Fr::from(value));

            let scalar2 = Scalar::from(fr);
            assert_eq!(scalar, scalar2);
        }
    }

    #[test]
    fn can_be_compared() {
        let scalar1 = Scalar::from(42);
        let scalar2 = Scalar::from(42);
        let scalar3 = Scalar::from(43);

        assert_eq!(scalar1, scalar1);
        assert_eq!(scalar1, scalar2);
        assert_ne!(scalar1, scalar3);
    }

    #[test]
    fn from_le_bytes_interprets_slice_as_little_endian() {
        let scalar = Scalar::from_le_bytes(&[]);
        assert_eq!(scalar, Scalar::from(0));

        let scalar = Scalar::from_le_bytes(&[0x01]);
        assert_eq!(scalar, Scalar::from(1));

        let scalar = Scalar::from_le_bytes(&[0x01, 0x02]);
        assert_eq!(scalar, Scalar::from(0x0201));

        let scalar = Scalar::from_le_bytes(&[0x01, 0x02, 0x03]);
        assert_eq!(scalar, Scalar::from(0x030201));
    }

    #[test]
    fn from_le_bytes_uses_at_most_31_bytes() {
        let bytes: Vec<u8> = (0..40).collect();
        let scalar = Scalar::from_le_bytes(&bytes);
        let mut expected_bytes = [0u8; 32];
        expected_bytes[..31].copy_from_slice(&bytes[..31]);
        let expected_scalar = Scalar(Fr::from_le_bytes_mod_order(&expected_bytes));
        assert_eq!(scalar, expected_scalar);
    }

    #[test]
    fn set_bit128_sets_correct_bit() {
        // Starting from zero
        let mut scalar = Scalar::from(0);
        assert_eq!(scalar.0.into_bigint().to_bytes_be(), &[0; 32]);

        scalar.set_bit128();
        let mut expected = [0; 32];
        expected[15] = 0x01;
        assert_eq!(scalar.0.into_bigint().to_bytes_be(), expected);

        scalar.set_bit128();
        assert_eq!(scalar.0.into_bigint().to_bytes_be(), expected);

        // Starting from non-zero value
        let mut scalar = Scalar::from(0xf1f2f3f4f5f6f7f8);
        let mut expected = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xf1, 0xf2,
            0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        ];
        assert_eq!(scalar.0.into_bigint().to_bytes_be(), expected);

        scalar.set_bit128();
        expected[15] = 0x01;
        assert_eq!(scalar.0.into_bigint().to_bytes_be(), expected);

        scalar.set_bit128();
        assert_eq!(scalar.0.into_bigint().to_bytes_be(), expected);
    }

    #[test]
    fn scalars_can_be_subtracted() {
        let scalar1 = Scalar::from(100);
        let scalar2 = Scalar::from(42);
        let result = scalar1 - scalar2;
        assert_eq!(result, Scalar::from(58));
    }
}
