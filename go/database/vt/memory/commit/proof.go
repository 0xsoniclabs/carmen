package commit

import (
	"github.com/crate-crypto/go-ipa/bandersnatch/fr"
	"github.com/crate-crypto/go-ipa/common"
	"github.com/crate-crypto/go-ipa/ipa"
)

// Open is a bit of information that is used to verify that a commitment
// contains a specific value at a specific position. It is an Inner Product
// Argument (IPA) proof.
// Details: https://dankradfeist.de/ethereum/2021/07/27/inner-product-arguments.html
type Opening struct {
	proof ipa.IPAProof
}

// Open creates an opening for a commitment at a specific position.
func Open(
	commitment Commitment,
	values [VectorSize]Value,
	position byte,
) (Opening, error) {
	transcript := common.NewTranscript("vt")

	elements := make([]fr.Element, VectorSize)
	for i, value := range values {
		elements[i] = value.scalar
	}

	var pos fr.Element
	pos.SetUint64(uint64(position))
	proof, err := ipa.CreateIPAProof(
		transcript,
		ipaConfig,
		commitment.point,
		elements,
		pos,
	)
	return Opening{proof: proof}, err
}

// Verify checks if the opening verifies the commitment at the given position
// for the given value. It returns true if the opening is valid, false otherwise.
func (o Opening) Verify(
	commitment Commitment,
	position byte,
	value Value,
) (bool, error) {
	transcript := common.NewTranscript("vt")
	var pos fr.Element
	pos.SetUint64(uint64(position))
	return ipa.CheckIPAProof(
		transcript,
		ipaConfig,
		commitment.point,
		o.proof,
		pos,
		value.scalar,
	)
}
