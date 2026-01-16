// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

pub mod code;

use std::{
    convert::Infallible,
    ops::{Div, Rem},
    sync::atomic::AtomicU64,
};

use crypto_bigint::U256;
use quick_cache::sync::Cache;

use crate::{
    database::verkle::crypto::{Commitment, Scalar},
    types::{Address, Key},
};

const HEADER_STORAGE_OFFSET: u32 = 64;
const CODE_OFFSET: u32 = 128;
const VERKLE_NODE_WIDTH: u32 = 256;
const VERKLE_NODE_WIDTH_LOG2: u64 = 8;

/// Embedding cache for the Verkle trie for basic account data, code, and storage keys.
pub struct VerkleTrieEmbedding {
    cache: Cache<(Address, U256), Key>,
    misses: AtomicU64,
}

impl Default for VerkleTrieEmbedding {
    fn default() -> Self {
        Self::new()
    }
}

impl VerkleTrieEmbedding {
    pub const CACHE_SIZE: usize = 1_000_000;

    pub fn new() -> Self {
        VerkleTrieEmbedding {
            cache: Cache::new(Self::CACHE_SIZE),
            misses: AtomicU64::new(0),
        }
    }

    pub fn cache_misses(&self) -> u64 {
        self.misses.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns the key of the basic data fields (nonce, balance, code size) for the given address.
    pub fn get_basic_data_key(&self, address: &Address) -> Key {
        self.get_trie_key(address, &U256::ZERO, 0)
    }

    /// Returns the key of the code hash field for the given address.
    pub fn get_code_hash_key(&self, address: &Address) -> Key {
        self.get_trie_key(address, &U256::ZERO, 1)
    }

    /// Returns the key of the code chunk with the given chunk number for the given address.
    pub fn get_code_chunk_key(&self, address: &Address, chunk_number: u32) -> Key {
        // Derived from
        // https://github.com/0xsoniclabs/go-ethereum/blob/e563918a84b4104e44935ddc6850f11738dcc3f5/trie/utils/verkle.go#L188

        let chunk_offset = chunk_number + CODE_OFFSET;
        let tree_index = chunk_offset.div(VERKLE_NODE_WIDTH);
        let sub_index = chunk_offset.rem(VERKLE_NODE_WIDTH);
        self.get_trie_key(address, &tree_index.into(), sub_index as u8)
    }

    /// Returns the storage key for the given address and storage key.
    pub fn get_storage_key(&self, address: &Address, key: &Key) -> Key {
        // Derived from
        // https://github.com/0xsoniclabs/go-ethereum/blob/e563918a84b4104e44935ddc6850f11738dcc3f5/trie/utils/verkle.go#L203

        let code_storage_delta = U256::from(CODE_OFFSET - HEADER_STORAGE_OFFSET);
        let mut tree_index = U256::from_be_slice(key);
        let suffix;

        if tree_index < code_storage_delta {
            tree_index += U256::from(HEADER_STORAGE_OFFSET);
            suffix = tree_index.to_le_bytes()[0];
            tree_index = U256::ZERO;
        } else {
            suffix = key[key.len() - 1];
            tree_index >>= 8;
            tree_index += U256::ONE << (248 - VERKLE_NODE_WIDTH_LOG2 as u32);
        }

        self.get_trie_key(address, &tree_index, suffix)
    }

    /// Retries the Verkle trie key for the given address, tree index, and sub index from a cache or
    /// computes it if it is not cached.
    ///
    /// The key is computed by:
    ///   - `C = Commit([2+256*64, address_low, address_high, tree_index_low, tree_index_high])`
    ///   - `H = Hash(C)`
    ///   - `K = append(H[..31], subIndex)`
    fn get_trie_key(&self, address: &Address, tree_index: &U256, sub_index: u8) -> Key {
        fn compute_trie_key(address: &Address, tree_index: &U256, sub_index: u8) -> Key {
            // Inspired by https://github.com/0xsoniclabs/go-ethereum/blob/e563918a84b4104e44935ddc6850f11738dcc3f5/trie/utils/verkle.go#L116

            let mut expanded = [0u8; 32];
            expanded[12..].copy_from_slice(address);

            let mut values = [Scalar::zero(); 5];
            values[0] = Scalar::from(2 + 256 * 64);
            values[1] = Scalar::from_le_bytes(&expanded[..16]);
            values[2] = Scalar::from_le_bytes(&expanded[16..]);

            let index = tree_index.to_le_bytes();
            values[3] = Scalar::from_le_bytes(&index[..16]);
            values[4] = Scalar::from_le_bytes(&index[16..]);

            let hash = Commitment::new(&values).hash();

            let mut result: Key = [0; 32];
            result[..31].copy_from_slice(&hash[..31]);
            result[31] = sub_index;
            result
        }

        let mut result = self
            .cache
            .get_or_insert_with(&(*address, *tree_index), || {
                self.misses
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok::<_, Infallible>(compute_trie_key(address, tree_index, 0))
            })
            .unwrap(); // this cannot fail
        result[31] = sub_index;
        result
    }
}

#[cfg(test)]
mod tests {
    use zerocopy::IntoBytes;

    use super::*;
    use crate::database::verkle::test_utils::FromIndexValues;

    #[test]
    fn get_trie_key_produces_same_values_as_geth() {
        let expected = [
            "0x1a100684fd68185060405f3f160e4bb6e034194336b547bdae323f888d533200",
            "0x1a100684fd68185060405f3f160e4bb6e034194336b547bdae323f888d533201",
            "0x1a100684fd68185060405f3f160e4bb6e034194336b547bdae323f888d533202",
            "0x7c2f303854b78b803e520069bd50708f7924cef74564f861f7abe83edff92a00",
            "0x7c2f303854b78b803e520069bd50708f7924cef74564f861f7abe83edff92a01",
            "0x7c2f303854b78b803e520069bd50708f7924cef74564f861f7abe83edff92a02",
            "0xaebdd778e4ff86bc29f69bb593ccb7921e5e55cbbc171bbb829eea01e8ecb800",
            "0xaebdd778e4ff86bc29f69bb593ccb7921e5e55cbbc171bbb829eea01e8ecb801",
            "0xaebdd778e4ff86bc29f69bb593ccb7921e5e55cbbc171bbb829eea01e8ecb802",
            "0xb05cdc11080c9a04cca1c20480970e313f244b334897f0c33bc3d9784d83c800",
            "0xb05cdc11080c9a04cca1c20480970e313f244b334897f0c33bc3d9784d83c801",
            "0xb05cdc11080c9a04cca1c20480970e313f244b334897f0c33bc3d9784d83c802",
            "0x6c965c02f5d869d6a1aa307d3caa782f26dc39f4c90d7b158a9d96773c90af00",
            "0x6c965c02f5d869d6a1aa307d3caa782f26dc39f4c90d7b158a9d96773c90af01",
            "0x6c965c02f5d869d6a1aa307d3caa782f26dc39f4c90d7b158a9d96773c90af02",
            "0xd510c4c87555adfaa7fcb87f184cc0663be90e414af0a207cc5693dacd901800",
            "0xd510c4c87555adfaa7fcb87f184cc0663be90e414af0a207cc5693dacd901801",
            "0xd510c4c87555adfaa7fcb87f184cc0663be90e414af0a207cc5693dacd901802",
            "0x7e3bddd76381540fbc78a6cd1cd3df0996d3f633fc18682b07de7f564614c200",
            "0x7e3bddd76381540fbc78a6cd1cd3df0996d3f633fc18682b07de7f564614c201",
            "0x7e3bddd76381540fbc78a6cd1cd3df0996d3f633fc18682b07de7f564614c202",
            "0x94fead141dd87b91655b56d5b1b8e42465c918128eef07f603a0dd0fc547ff00",
            "0x94fead141dd87b91655b56d5b1b8e42465c918128eef07f603a0dd0fc547ff01",
            "0x94fead141dd87b91655b56d5b1b8e42465c918128eef07f603a0dd0fc547ff02",
            "0xe90ba1e60076c244ed0fb1c758215729113789a96db3e3ad107bc9a034d2e200",
            "0xe90ba1e60076c244ed0fb1c758215729113789a96db3e3ad107bc9a034d2e201",
            "0xe90ba1e60076c244ed0fb1c758215729113789a96db3e3ad107bc9a034d2e202",
        ];
        let embedding = VerkleTrieEmbedding::new();
        for i in 0..3_u8 {
            for j in 0..3_u8 {
                for k in 0..3_u8 {
                    let address = Address::from_index_values(0, &[(0, i)]);
                    let received = embedding.get_trie_key(&address, &U256::from(j), k);
                    let expected = expected[(i * 9 + j * 3 + k) as usize];
                    assert_eq!(
                        expected,
                        const_hex::encode_prefixed(received.as_bytes()),
                        "i={i}, j={j}, k={k}"
                    );
                }
            }
        }
    }

    #[test]
    fn get_basic_data_key_produces_same_values_as_geth() {
        let cases = [
            (
                Address::default(),
                "0x1a100684fd68185060405f3f160e4bb6e034194336b547bdae323f888d533200",
            ),
            (
                Address::from_index_values(0, &[(0, 1)]),
                "0xb05cdc11080c9a04cca1c20480970e313f244b334897f0c33bc3d9784d83c800",
            ),
            (
                Address::from_index_values(0, &[(0, 2)]),
                "0x7e3bddd76381540fbc78a6cd1cd3df0996d3f633fc18682b07de7f564614c200",
            ),
            (
                Address::from_index_values(0, &[(0, 1), (19, 2)]),
                "0xfed341c6f4a35fc8f498ac0929a1c224c5661a6d9305c86da923f49d2f0dd900",
            ),
        ];

        let embedding = VerkleTrieEmbedding::new();
        for (address, expected) in cases {
            let received = embedding.get_basic_data_key(&address);
            assert_eq!(expected, const_hex::encode_prefixed(received.as_bytes()));
        }
    }

    #[test]
    fn get_code_hash_key_produces_same_values_as_geth() {
        let cases = [
            (
                Address::default(),
                "0x1a100684fd68185060405f3f160e4bb6e034194336b547bdae323f888d533201",
            ),
            (
                Address::from_index_values(0, &[(0, 1)]),
                "0xb05cdc11080c9a04cca1c20480970e313f244b334897f0c33bc3d9784d83c801",
            ),
            (
                Address::from_index_values(0, &[(0, 2)]),
                "0x7e3bddd76381540fbc78a6cd1cd3df0996d3f633fc18682b07de7f564614c201",
            ),
            (
                Address::from_index_values(0, &[(0, 1), (19, 2)]),
                "0xfed341c6f4a35fc8f498ac0929a1c224c5661a6d9305c86da923f49d2f0dd901",
            ),
        ];

        let embedding = VerkleTrieEmbedding::new();
        for (address, expected) in cases {
            let received = embedding.get_code_hash_key(&address);
            assert_eq!(expected, const_hex::encode_prefixed(received.as_bytes()));
        }
    }

    #[test]
    fn get_code_chunk_key_produces_same_values_as_geth() {
        let expected = [
            "0x1a100684fd68185060405f3f160e4bb6e034194336b547bdae323f888d533280",
            "0x1a100684fd68185060405f3f160e4bb6e034194336b547bdae323f888d533281",
            "0x1a100684fd68185060405f3f160e4bb6e034194336b547bdae323f888d533282",
            "0x1a100684fd68185060405f3f160e4bb6e034194336b547bdae323f888d533290",
            "0x1a100684fd68185060405f3f160e4bb6e034194336b547bdae323f888d5332c0",
            "0x7c2f303854b78b803e520069bd50708f7924cef74564f861f7abe83edff92a00",
            "0x7c2f303854b78b803e520069bd50708f7924cef74564f861f7abe83edff92a7f",
            "0x7c2f303854b78b803e520069bd50708f7924cef74564f861f7abe83edff92a80",
            "0x8ebfe52d677db1040ec4257b4a8679e94d3d1f2a4977e43182e3406913373990",
            "0xb05cdc11080c9a04cca1c20480970e313f244b334897f0c33bc3d9784d83c880",
            "0xb05cdc11080c9a04cca1c20480970e313f244b334897f0c33bc3d9784d83c881",
            "0xb05cdc11080c9a04cca1c20480970e313f244b334897f0c33bc3d9784d83c882",
            "0xb05cdc11080c9a04cca1c20480970e313f244b334897f0c33bc3d9784d83c890",
            "0xb05cdc11080c9a04cca1c20480970e313f244b334897f0c33bc3d9784d83c8c0",
            "0x6c965c02f5d869d6a1aa307d3caa782f26dc39f4c90d7b158a9d96773c90af00",
            "0x6c965c02f5d869d6a1aa307d3caa782f26dc39f4c90d7b158a9d96773c90af7f",
            "0x6c965c02f5d869d6a1aa307d3caa782f26dc39f4c90d7b158a9d96773c90af80",
            "0x72d0928a41c46dfc8d5d32885b0421e48ae91404deeffafd382755be88ff2c90",
            "0x7e3bddd76381540fbc78a6cd1cd3df0996d3f633fc18682b07de7f564614c280",
            "0x7e3bddd76381540fbc78a6cd1cd3df0996d3f633fc18682b07de7f564614c281",
            "0x7e3bddd76381540fbc78a6cd1cd3df0996d3f633fc18682b07de7f564614c282",
            "0x7e3bddd76381540fbc78a6cd1cd3df0996d3f633fc18682b07de7f564614c290",
            "0x7e3bddd76381540fbc78a6cd1cd3df0996d3f633fc18682b07de7f564614c2c0",
            "0x94fead141dd87b91655b56d5b1b8e42465c918128eef07f603a0dd0fc547ff00",
            "0x94fead141dd87b91655b56d5b1b8e42465c918128eef07f603a0dd0fc547ff7f",
            "0x94fead141dd87b91655b56d5b1b8e42465c918128eef07f603a0dd0fc547ff80",
            "0xd75be820dcce35587927c307e59e43ba7d7abab1224c302d6c0566fc07a34f90",
        ];

        let embedding = VerkleTrieEmbedding::new();
        for i in 0..3 {
            for (j, offset) in [0, 1, 2, 16, 64, 128, 255, 256, 10_000]
                .into_iter()
                .enumerate()
            {
                let expected = expected[i * 9 + j];
                let received = embedding
                    .get_code_chunk_key(&Address::from_index_values(0, &[(0, i as u8)]), offset);
                assert_eq!(
                    expected,
                    const_hex::encode_prefixed(received.as_bytes()),
                    "i={i}, j={j}, offset={offset}"
                );
            }
        }
    }

    #[test]
    fn get_storage_key_produces_same_values_as_geth() {
        let expected = [
            "0x1a100684fd68185060405f3f160e4bb6e034194336b547bdae323f888d533240",
            "0x833fa63a3cf9f2c4cb680781bde676de1848c7b691f91ec6908a98e50ab97200",
            "0x43739862248cfbde7c028cb4f767b8ea25f1a65d393f7bd3181a1c0c37e32500",
            "0xb05cdc11080c9a04cca1c20480970e313f244b334897f0c33bc3d9784d83c840",
            "0x3d13761f24e12658a6b80b441169b565c7505331c226cff204b5b4231ac09400",
            "0x0bc2ea43dfff9b1588cd3bd0c35f21f2f5cf966d677c4a84bb508a08deadbe00",
            "0x7e3bddd76381540fbc78a6cd1cd3df0996d3f633fc18682b07de7f564614c240",
            "0x3eb8458c5f06ed8e4c9d5e0308eec26f478c55ffcaaee65efe81dee80e222200",
            "0x4b620aca83b506397a4bd7ee689ef0e71cc3b76eb86c12d90b69c59b2e766900",
        ];

        let embedding = VerkleTrieEmbedding::new();
        for i in 0..3 {
            for j in 0..3 {
                let expected = expected[i * 3 + j];
                let address = Address::from_index_values(0, &[(0, i as u8)]);
                let key = Key::from_index_values(0, &[(0, j as u8)]);
                let received = embedding.get_storage_key(&address, &key);
                assert_eq!(
                    expected,
                    const_hex::encode_prefixed(received.as_bytes()),
                    "i={i}, j={j}"
                );
            }
        }
    }
}
