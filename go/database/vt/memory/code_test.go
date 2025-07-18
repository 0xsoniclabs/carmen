package memory

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSplitAndMerge(t *testing.T) {
	const PUSH2 = 0x61
	const PUSH32 = 0x7f

	tests := map[string]struct {
		code   []byte
		chunks []chunk
	}{
		"empty": {
			code:   []byte{},
			chunks: []chunk{},
		},
		"single chunk": {
			code:   []byte{0x01, 0x02, 0x03},
			chunks: []chunk{{0x00, 0x01, 0x02, 0x03}},
		},
		"multiple chunks": {
			code:   []byte{0x01, 30: 0x02, 31: 0x03, 70: 0x01},
			chunks: []chunk{{0, 0x01, 31: 0x02}, {0, 0x03}, {70%31 + 1: 0x01}},
		},
		"data at boundary": {
			code: []byte{0x01, 0x02, PUSH32, 0x03, 31: 0x04, 80: PUSH2, 0x01},
			chunks: []chunk{
				{0x00, 0x01, 0x02, PUSH32, 0x03},
				{4, 0x04},
				{80%31 + 1: PUSH2, 0x01},
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			require := require.New(t)

			chunks := splitCode(test.code)
			require.Equal(test.chunks, chunks)

			merged := merge(chunks, len(test.code))
			require.Equal(test.code, merged)
		})
	}
}
