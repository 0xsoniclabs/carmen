use ark_ff::{BigInteger, fields::PrimeField};
use banderwagon::{Element, Fr};
use ipa_multipoint::{
    committer::{Committer, DefaultCommitter},
    lagrange_basis::LagrangeBasis,
    multiproof::{MultiPoint, ProverQuery, VerifierQuery},
    transcript::Transcript,
};
use verkle_trie::constants::{CRS, PRECOMPUTED_WEIGHTS, new_crs};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Value {
    scalar: Fr,
}

impl Value {
    pub fn new(scalar: u64) -> Self {
        Value {
            scalar: Fr::from(scalar),
        }
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        // TODO: Handle longer slices
        let mut arr = [0u8; 32];
        arr[0..bytes.len()].copy_from_slice(bytes);
        Value {
            // TODO: Is _mod_order what we want?
            scalar: Fr::from_le_bytes_mod_order(&arr),
        }
    }

    pub fn set_bit128(&mut self) {
        let mut bytes = self.scalar.into_bigint().to_bytes_be();
        bytes[15] |= 0x01;
        self.scalar = Fr::from_be_bytes_mod_order(&bytes);
    }
}

// TODO: Implement default value - should be the same (I think?!) than Commitment::default().to_value()
//  => Use in place of the latter in trie.rs

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commitment {
    point: Element,
}

impl Commitment {
    // TODO: Restrict size?
    pub fn new(values: &[Value]) -> Self {
        let committer = DefaultCommitter::new(&new_crs().G);
        let point =
            committer.commit_lagrange(&values.iter().map(|v| v.scalar).collect::<Vec<Fr>>());
        Commitment { point }
    }

    // pub fn is_valid(&self) -> bool {
    //     // self.point.
    // }

    pub fn to_value(&self) -> Value {
        Value {
            scalar: self.point.map_to_scalar_field(),
        }
    }

    // TODO: Naming - hash trait?
    pub fn hash(&self) -> [u8; 32] {
        let scalar = self.point.map_to_scalar_field();
        scalar
            .0
            .0
            // TODO: LE?
            .map(|limb| limb.to_le_bytes())
            .concat()
            .try_into()
            .expect("Expected 32 bytes")
    }

    pub fn compress(&self) -> [u8; 32] {
        self.point.to_bytes()
    }
}

// TODO: Add test
impl Default for Commitment {
    fn default() -> Self {
        Commitment {
            // TODO: Does this make sense?
            point: Element::zero(),
        }
    }
}

pub struct Opening {
    proof: ipa_multipoint::multiproof::MultiPointProof,
}

impl Opening {
    // TODO: Or new?
    pub fn open(commitment: &Commitment, values: &[Value], position: u8) -> Self {
        let mut transcript = Transcript::new(b"vt");
        let query = ProverQuery {
            commitment: commitment.point,
            point: position as usize, // TODO ?
            result: values[position as usize].scalar,
            poly: LagrangeBasis::new(values.iter().map(|v| v.scalar).collect()),
        };
        let proof = MultiPoint::open(
            CRS.clone(),
            &PRECOMPUTED_WEIGHTS,
            &mut transcript,
            vec![query],
        );
        Opening { proof }
    }

    pub fn verify(&self, commitment: &Commitment, position: u8, value: Value) -> bool {
        let mut transcript = Transcript::new(b"vt");
        let query = VerifierQuery {
            commitment: commitment.point,
            point: Fr::from(position),
            result: value.scalar,
        };
        self.proof
            .check(&CRS, &PRECOMPUTED_WEIGHTS, &[query], &mut transcript)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_from_le_bytes_uses_little_endian_and_right_zero_padding() {
        let value = Value::from_le_bytes(&[]);
        assert_eq!(value, Value::new(0));

        let value = Value::from_le_bytes(&[0x01]);
        assert_eq!(value, Value::new(1));

        let value = Value::from_le_bytes(&[0x01, 0x02]);
        assert_eq!(value, Value::new(0x0201));

        let value = Value::from_le_bytes(&[0x01, 0x02, 0x03]);
        assert_eq!(value, Value::new(0x030201));
    }

    #[test]
    fn value_set_bit128_sets_correct_bit() {
        // Starting from zero
        let mut value = Value::new(0);
        assert_eq!(value.scalar.into_bigint().to_bytes_be(), &[0; 32]);

        value.set_bit128();
        let mut expected = [0; 32];
        expected[15] = 0x01;
        assert_eq!(value.scalar.into_bigint().to_bytes_be(), expected);

        value.set_bit128();
        assert_eq!(value.scalar.into_bigint().to_bytes_be(), expected);

        // Starting from non-zero value
        let mut value = Value::new(0xf1f2f3f4f5f6f7f8);
        let mut expected = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xf1, 0xf2,
            0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        ];
        assert_eq!(value.scalar.into_bigint().to_bytes_be(), expected);

        value.set_bit128();
        expected[15] = 0x01;
        assert_eq!(value.scalar.into_bigint().to_bytes_be(), expected);

        value.set_bit128();
        assert_eq!(value.scalar.into_bigint().to_bytes_be(), expected);
    }

    #[test]
    fn commitment_committed_values_can_be_used_to_proof_values() {
        let mut values = Vec::with_capacity(256);
        for i in 0..256 {
            values.push(Value::new(i + 1));
        }
        let commitment = Commitment::new(&values);

        for (i, value) in values.iter().enumerate() {
            let pos = i as u8;
            let opening = Opening::open(&commitment, &values, pos);

            // Verify that the opening can verify the committed value.
            assert!(
                opening.verify(&commitment, pos, value.clone()),
                "Opening should verify for committed value"
            );

            // Verify that the opening does not verify another value.
            let other_value = Value::new((i + 2) as u64);
            assert!(
                !opening.verify(&commitment, pos, other_value),
                "Opening should not verify for different value"
            );
        }
    }
}
