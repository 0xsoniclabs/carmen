// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package reference

// chunk is a 31-byte chunk of some EVM code prefixed by the number of bytes
// in the prefix of the chunk that is part of the data section of a push
// instruction.
type chunk [32]byte

// splitCode splits the given EVM code into 31-byte chunks, where each chunk
// is prefixed by the number of bytes in the prefix of the chunk that is part
// of the data section of a push instruction.
func splitCode(code []byte) []chunk {
	const PUSH1 = 0x60
	const PUSH32 = 0x7f
	// Add 30 additional entries to handle the case where the code is shorter than expected
	// based on the last PUSH instruction. 30 covers the maximum number of potentially
	// "missing" bytes: A PUSH32 instruction at the end of a chunk, followed by a single byte
	// of data in the next chunk => 1 + 30 = 31. This way we still mark the correct number of
	// elements in the second chunk as data (as if the code had the expected length).
	isCode := make([]bool, len(code)+30)
	for i := 0; i < len(isCode); i++ {
		isCode[i] = true
		if i < len(code) {
			cur := code[i]
			if PUSH1 <= cur && cur <= PUSH32 {
				i += int(cur-PUSH1) + 1
			}
		}
	}

	chunks := make([]chunk, 0, len(code)/32+1)
	for i := 0; len(code) > 0; i++ {
		next := chunk{}
		for j := 31 * i; j < 31*(i+1) && j < len(isCode) && !isCode[j]; j++ {
			next[0]++
		}
		code = code[copy(next[1:], code):]
		chunks = append(chunks, next)
	}
	return chunks
}

// merge merges the given chunks into a single byte slice restoring the code of
// the given length in bytes. The function does not check for the right number
// of chunks to fit the length of the resulting code. If there are too few
// chunks, the resulting code will be zero-padded. If there are too many, chunks
// will be ignored.
func merge(chunks []chunk, len int) []byte {
	res := make([]byte, len)
	cur := res
	for _, c := range chunks {
		cur = cur[copy(cur, c[1:]):]
	}
	return res
}
