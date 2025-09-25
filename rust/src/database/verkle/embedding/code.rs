// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

const PUSH1: u8 = 0x60;
const PUSH32: u8 = 0x7f;

type Chunk = [u8; 32];

/// Splits the given EVM bytecode into 31-byte chunks, each prefixed with a single byte indicating
/// the number of leading data bytes (i.e., non-opcode bytes, part of the data section of a PUSH
/// instruction) in that chunk.
#[cfg_attr(not(test), expect(unused))]
pub fn split_code(code: &[u8]) -> Vec<Chunk> {
    // Add 30 additional entries to handle the case where the code is shorter than expected
    // based on the last PUSH instruction. 30 covers the maximum number of potentially
    // "missing" bytes: A PUSH32 instruction at the end of a chunk, followed by a single byte
    // of data in the next chunk => 1 + 30 = 31. This way we still mark the correct number of
    // elements in the second chunk as data (as if the code had the expected length).
    let mut is_code = vec![false; code.len() + 30];

    let mut i = 0;
    while i < is_code.len() {
        is_code[i] = true;
        if i < code.len() {
            let cur = code[i];
            if (PUSH1..=PUSH32).contains(&cur) {
                i += (cur - PUSH1) as usize + 2;
                continue;
            }
        }
        i += 1;
    }

    let mut chunks = Vec::with_capacity((code.len() / 32) + 1);
    for i in (0..code.len()).step_by(31) {
        let data_size = 31.min(is_code[i..].iter().take_while(|&ic| !ic).count() as u8);
        let code_size = 31.min(code.len() - i);
        let mut next = Chunk::default();
        next[0] = data_size;
        next[1..code_size + 1].copy_from_slice(&code[i..(i + code_size)]);
        chunks.push(next);
    }

    chunks
}

/// Merges the given chunks (created by [`split_code`]) back into the original bytecode.
/// At most `result.len()` bytes are written to `result`.
#[cfg_attr(not(test), expect(unused))]
pub fn merge_code(chunks: &[Chunk], result: &mut [u8]) {
    for (chunk, out) in chunks.iter().zip(result.chunks_mut(31)) {
        out.copy_from_slice(&chunk[1..1 + out.len()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::verkle::test_utils::FromIndexValues;

    #[test]
    fn split_code_marks_data_segments() {
        let code = vec![PUSH32];
        let chunks = split_code(&code);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0][0], 0);

        // PUSH32 at different positions within the first chunk, followed by 32 bytes of data.
        for i in 0..30 {
            let code = Vec::from_index_values(0, &[(i, PUSH32), (i + 32, 0xab)]);
            let chunks = split_code(&code);
            assert_eq!(chunks.len(), 2);
            assert_eq!(chunks[1][0], 2 + i as u8);
        }

        // PUSH32 at the end of the first chunk, followed by 1-32 bytes of data,
        // and up to 32 additional bytes of "opcodes".
        for i in 0..64 {
            let code = Vec::from_index_values(0, &[(30, PUSH32), (i + 31, 0xab)]);
            let chunks = split_code(&code);
            assert_eq!(
                chunks.len(),
                i / 31 + 2,
                "unexpected number of chunks for i={i}"
            );
            // Second chunk should always contain 31 bytes of data.
            assert_eq!(chunks[1][0], 31, "unexpected data length for i={i}");
            // Third chunk (if it exists) should contain 1 byte of data.
            assert!(chunks.len() < 3 || chunks[2][0] == 1);
        }
    }

    #[test]
    fn split_and_merge_works_correctly() {
        const PUSH2: u8 = PUSH1 + 1;
        let cases = [
            ("empty", vec![], vec![]),
            (
                "single chunk",
                vec![0x01, 0x02, 0x03],
                vec![Chunk::from_index_values(
                    0,
                    &[(1, 0x01), (2, 0x02), (3, 0x03)],
                )],
            ),
            (
                "multiple chunks",
                Vec::from_index_values(0, &[(0, 0x01), (30, 0x02), (31, 0x03), (70, 0x01)]),
                vec![
                    Chunk::from_index_values(0, &[(1, 0x01), (31, 0x02)]),
                    Chunk::from_index_values(0, &[(1, 0x03)]),
                    Chunk::from_index_values(0, &[(70 % 31 + 1, 0x01)]),
                ],
            ),
            (
                "data at boundary",
                Vec::from_index_values(
                    0,
                    &[
                        (0, 0x01),
                        (1, 0x02),
                        (2, PUSH32),
                        (3, 0x03),
                        (31, 0x04),
                        (80, PUSH2),
                        (81, 0x01),
                    ],
                ),
                vec![
                    Chunk::from_index_values(
                        0,
                        &[(0, 0), (1, 0x01), (2, 0x02), (3, PUSH32), (4, 0x03)],
                    ),
                    Chunk::from_index_values(0, &[(0, 4), (1, 0x04)]),
                    Chunk::from_index_values(0, &[(80 % 31 + 1, PUSH2), (80 % 31 + 2, 0x01)]),
                ],
            ),
            (
                "data at the end",
                Vec::from_index_values(0, &[(0, 0x03), (1, 0x04), (2, PUSH32), (34, 0x05)]),
                vec![
                    Chunk::from_index_values(0, &[(1, 0x03), (2, 0x04), (3, PUSH32)]),
                    Chunk::from_index_values(0, &[(0, 4), (4, 0x05)]),
                ],
            ),
            (
                "code ending with PUSH32",
                Vec::from_index_values(0, &[(0, 0x03), (30, PUSH32), (62, 0x05)]),
                vec![
                    Chunk::from_index_values(0, &[(1, 0x03), (31, PUSH32)]),
                    Chunk::from_index_values(0, &[(0, 31), (1, 0x00)]),
                    Chunk::from_index_values(0, &[(0, 1), (1, 0x05)]),
                ],
            ),
            (
                "truncated push data at chunk boundary filling full chunk",
                // In this case the last instruction of the first chunk is a PUSH32.
                // For a valid code, 32 bytes of data (two chunks) should follow.
                // Instead, we only have a single byte -- just enough to force the
                // creation of a second chunk. We nevertheless want the entire second
                // chunk to be marked as data.
                Vec::from_index_values(0, &[(0, 0x03), (30, PUSH32), (31, 0x05)]),
                vec![
                    Chunk::from_index_values(0, &[(1, 0x03), (31, PUSH32)]),
                    Chunk::from_index_values(0, &[(0, 31), (1, 0x05)]),
                ],
            ),
            (
                "truncated push data at chunk boundary filling partial chunk",
                // Same as before, but the PUSH32 comes two bytes earlier.
                // This way only 30 bytes of the second chunk should be marked as data.
                Vec::from_index_values(0, &[(0, 0x03), (28, PUSH32), (31, 0x05)]),
                vec![
                    Chunk::from_index_values(0, &[(1, 0x03), (29, PUSH32)]),
                    Chunk::from_index_values(0, &[(0, 30), (1, 0x05)]),
                ],
            ),
        ];

        for (name, code, expected_chunks) in cases {
            let chunks = split_code(&code);
            assert_eq!(chunks, expected_chunks, "split failed for case: {name}");

            let mut merged_code = vec![0u8; code.len()];
            merge_code(&chunks, &mut merged_code);
            assert_eq!(merged_code, code, "merge failed for case: {name}");
        }
    }

    #[test]
    fn merge_code_handles_different_lengths() {
        let chunks = vec![
            Chunk::from_index_values(0, &[(1, 0x01), (2, 0x02)]),
            Chunk::from_index_values(0, &[(1, 0x03)]),
        ];

        let mut merged_code = vec![0x0; 5];
        merge_code(&chunks, &mut merged_code);
        assert_eq!(merged_code, vec![0x01, 0x02, 0x00, 0x00, 0x00]);

        let mut merged_code = vec![0x0; 1];
        merge_code(&chunks, &mut merged_code);
        assert_eq!(merged_code, vec![0x01]);

        let mut merged_code = vec![0x0; 33];
        merge_code(&chunks, &mut merged_code);
        assert_eq!(
            merged_code,
            Vec::from_index_values(0, &[(0, 0x01), (1, 0x02), (31, 0x03), (32, 0x00)])
        );
    }
}
