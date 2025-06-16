package verkle

type NodeSource interface {
	// Get retrieves the value associated with the given path.
	// The input is navigation path in three.
	Get(path []byte) ([]byte, error)

	// Set sets the value at the given path.
	Set(path []byte, value []byte) error
}
