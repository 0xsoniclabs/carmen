package memory

type chunk [32]byte

func splitCode(code []byte) []chunk {
	const PUSH1 = 0x60
	const PUSH32 = 0x7f
	isCode := make([]bool, len(code)+33)
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
		for j := 31 * i; j < len(isCode) && !isCode[j]; j++ {
			next[0]++
		}
		code = code[copy(next[1:], code):]
		chunks = append(chunks, next)
	}
	return chunks
}

func merge(chunks []chunk, len int) []byte {
	res := make([]byte, len)
	cur := res
	for _, c := range chunks {
		cur = cur[copy(cur, c[1:]):]
	}
	return res
}
