use banderwagon::{Element, Fr};
use ipa_multipoint::{
    committer::{Committer, DefaultCommitter},
    lagrange_basis::LagrangeBasis,
    multiproof::{MultiPoint, ProverQuery, VerifierQuery},
    transcript::Transcript,
};
use verkle_trie::constants::{CRS, PRECOMPUTED_WEIGHTS, new_crs};

#[derive(Clone)]
pub struct Value {
    scalar: Fr,
}

impl Value {
    pub fn new(scalar: u64) -> Self {
        Value {
            scalar: Fr::from(scalar),
        }
    }
}

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
    fn foo() {
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
