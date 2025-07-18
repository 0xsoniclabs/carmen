package commit

import (
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/crate-crypto/go-ipa/banderwagon"
	"github.com/crate-crypto/go-ipa/ipa"
)

// VectorSize is the size of the vector that the commitment is made to.
const VectorSize = 256

// Commitment is a commitment to a vector of 256 values. It is a point on the
// Banderwagon curve, which is used for the Pederson commitment scheme.
type Commitment struct {
	point banderwagon.Element
}

// Identity returns the identity commitment, which is the commitment to a vector
// of zero values. This is the point at infinity on the Banderwagon curve.
func Identity() Commitment {
	return Commitment{point: banderwagon.Identity}
}

// Commit creates a new commitment to a vector of values.
func Commit(values [VectorSize]Value) Commitment {
	// This function creates a commitment to a vector of values using
	// the Pederson hash function and the Banderwagon curve.
	elements := make([]banderwagon.Fr, VectorSize)
	for i, value := range values {
		elements[i] = value.scalar
	}
	return Commitment{point: ipaConfig.Commit(elements)}
}

// IsValid checks if the commitment is valid, i.e., if it is a point on the
// curve. Not all possible instances of Commitment are valid. If instances are
// fetched from an untrusted source, they should be checked for validity.
func (p Commitment) IsValid() bool {
	return p.point.IsOnCurve()
}

// Equal checks if two commitments are equal. This is a point equality check on
// the Banderwagon curve.
func (p Commitment) Equal(other Commitment) bool {
	return p.point.Equal(&other.point)
}

// ToValue converts the commitment to a value. The result is a scalar field value
// that can be used recursively in other commitments -- as it is required for
// the Verkle trie.
func (p Commitment) ToValue() Value {
	var res banderwagon.Fr
	p.point.MapToScalarField(&res)
	return Value{scalar: res}
}

// Hash returns the hash of the commitment, which is used when utilizing the
// Pederson commitment scheme as a hash function. The primary use case in the
// Verkle trie is when computing keys during the state tree embedding.
func (p Commitment) Hash() common.Hash {
	value := p.ToValue()
	return value.scalar.BytesLE()
}

// Compress returns the compressed representation of the commitment. The result
// can be used as a unique identifier for summarizing the root commitment of a
// Verkle trie.
func (p Commitment) Compress() [32]byte {
	return p.point.Bytes()
}

// Update creates a new commitment that is the same as the original, except
// that the value at the given position is updated from old to new.
func (c Commitment) Update(position byte, old, new Value) Commitment {
	// This is a very inefficient way to update a commitment.
	// Creates a new commitment of a 256-element vector containing the
	// difference between the old and the new vector -- which is zero
	// everywhere, except at the position of the change.
	//
	// It then uses the additive homomorphism property of the Pederson commit:
	//   Commit(A+B) = Commit(A) + Commit(B)
	// for vectors A and B, where + is the point addition.
	values := [VectorSize]Value{}
	values[position].scalar.Sub(&new.scalar, &old.scalar)
	diff := Commit(values)

	var res Commitment
	res.point.Add(&c.point, &diff.point)
	return res
}

// The configuration for the Inner Product Argument (IPA) library used in this
// package to create commitments and openings and to verify them. This config
// also contains the generator points and the curve parameters used for the
// Banderwagon curve, which is used for the Pederson commitment scheme.
var ipaConfig = func() *ipa.IPAConfig {
	conf, _ := ipa.NewIPASettings()
	return conf
}()
