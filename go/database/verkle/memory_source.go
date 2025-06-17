package verkle

import (
	"github.com/0xsoniclabs/carmen/go/common/immutable"
)

type memorySource struct {
	nodes map[immutable.Bytes][]byte
}

func NewMemorySource() NodeSource {
	return &memorySource{
		nodes: make(map[immutable.Bytes][]byte),
	}
}

func (s *memorySource) Get(path []byte) ([]byte, error) {
	key := immutable.NewBytes(path)
	if node, exists := s.nodes[key]; exists {
		return node, nil
	}
	return nil, nil
}

func (s *memorySource) Set(path []byte, value []byte) error {
	s.nodes[immutable.NewBytes(path)] = value

	return nil
}

func (s *memorySource) Flush() error {
	return nil
}

func (s *memorySource) Close() error {
	return s.Flush()
}
