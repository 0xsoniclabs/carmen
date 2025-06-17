package verkle

import (
	"github.com/0xsoniclabs/carmen/go/common"
)

type NodeSource interface {
	common.FlushAndCloser
	// Get retrieves the value associated with the given path.
	// The input is navigation path in three.
	Get(path []byte) ([]byte, error)

	// Set sets the value at the given path.
	Set(path []byte, value []byte) error
}
