// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::mem::MaybeUninit;

use crate::{
    CarmenState,
    database::verkle::{
        embedding::{
            code, get_basic_data_key, get_code_chunk_key, get_code_hash_key, get_storage_key,
        },
        keyed_update::{KeyedUpdate, KeyedUpdateBatch},
        variants::{CrateCryptoInMemoryVerkleTrie, SimpleInMemoryVerkleTrie},
        verkle_trie::VerkleTrie,
    },
    error::{BTResult, Error},
    types::{Address, Hash, Key, Nonce, U256, Update, Value},
};

pub const EMPTY_CODE_HASH: Hash = [
    197, 210, 70, 1, 134, 247, 35, 60, 146, 126, 125, 178, 220, 199, 3, 192, 229, 0, 182, 83, 202,
    130, 39, 59, 123, 250, 216, 4, 93, 133, 164, 112,
];

/// An implementation of [`CarmenState`] that uses a Verkle trie as the underlying data structure.
pub struct VerkleTrieCarmenState<T: VerkleTrie> {
    trie: T,
}

impl VerkleTrieCarmenState<SimpleInMemoryVerkleTrie> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let trie = SimpleInMemoryVerkleTrie::new();
        Self { trie }
    }
}

impl VerkleTrieCarmenState<CrateCryptoInMemoryVerkleTrie> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let trie = CrateCryptoInMemoryVerkleTrie::new();
        Self { trie }
    }
}

impl<T: VerkleTrie> CarmenState for VerkleTrieCarmenState<T> {
    fn account_exists(&self, addr: &Address) -> BTResult<bool, Error> {
        Ok(self.get_code_hash(addr)? != Hash::default())
    }

    fn get_balance(&self, addr: &Address) -> BTResult<U256, Error> {
        let key = get_basic_data_key(addr);
        let value = self.trie.lookup(&key)?;
        let mut result = U256::default();
        result[16..].copy_from_slice(&value[16..32]);
        Ok(result)
    }

    fn get_nonce(&self, addr: &Address) -> BTResult<Nonce, Error> {
        let key = get_basic_data_key(addr);
        let value = self.trie.lookup(&key)?;
        // Safe to unwrap: Always 8 bytes
        Ok(value[8..16].try_into().unwrap())
    }

    fn get_storage_value(&self, addr: &Address, key: &Key) -> BTResult<Value, Error> {
        let key = get_storage_key(addr, key);
        let value = self.trie.lookup(&key)?;
        Ok(Value::from(value))
    }

    fn get_code(&self, addr: &Address, code_buf: &mut [MaybeUninit<u8>]) -> BTResult<usize, Error> {
        let len = self.get_code_len(addr)?;
        let chunk_count = len / 31 + 1;
        let mut chunks = Vec::with_capacity(chunk_count as usize);
        for i in 0..chunk_count {
            let key = get_code_chunk_key(addr, i);
            let chunk = self.trie.lookup(&key)?;
            chunks.push(chunk);
        }
        let mut code = vec![0x0; len as usize];
        code::merge_code(&chunks, &mut code);
        // This should hopefully compile down to a memcpy.
        for (i, byte) in code.into_iter().enumerate() {
            code_buf[i] = MaybeUninit::new(byte);
        }
        Ok(len as usize)
    }

    fn get_code_hash(&self, addr: &Address) -> BTResult<Hash, Error> {
        let key = get_code_hash_key(addr);
        let value = self.trie.lookup(&key)?;
        Ok(Hash::from(value))
    }

    fn get_code_len(&self, addr: &Address) -> BTResult<u32, Error> {
        let key = get_basic_data_key(addr);
        let value = self.trie.lookup(&key)?;
        // Safe to unwrap - slice is always 4 bytes
        Ok(u32::from_be_bytes(value[4..8].try_into().unwrap()))
    }

    fn get_hash(&self) -> BTResult<Hash, Error> {
        let commitment = self.trie.commit()?;
        Ok(Hash::from(commitment.compress()))
    }

    #[allow(clippy::needless_lifetimes)]
    fn apply_block_update<'u>(&self, _block: u64, update: Update<'u>) -> BTResult<(), Error> {
        if let Ok(updates) = KeyedUpdateBatch::try_from(update) {
            for update in &*updates {
                match update {
                    KeyedUpdate::FullSlot { key, value } => {
                        self.trie.store(key, value)?;
                    }
                    KeyedUpdate::PartialSlot { key, value, mask } => {
                        let existing_value = self.trie.lookup(key)?;
                        let mut new_value = [0u8; 32];
                        for i in 0..32 {
                            new_value[i] = (existing_value[i] & !mask[i]) | (value[i] & mask[i]);
                        }
                        self.trie.store(key, &new_value)?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use sha3::{Digest, Keccak256};

    use super::*;
    use crate::{
        database::verkle::test_utils::FromIndexValues,
        types::{BalanceUpdate, CodeUpdate, NonceUpdate, SlotUpdate},
    };

    #[rstest_reuse::template]
    #[rstest::rstest]
    #[case::simple_in_memory(Box::new(VerkleTrieCarmenState::<SimpleInMemoryVerkleTrie>::new()) as Box<dyn CarmenState>)]
    #[case::crate_crypto(Box::new(VerkleTrieCarmenState::<CrateCryptoInMemoryVerkleTrie>::new()) as Box<dyn CarmenState>)]
    fn all_state_impls(#[case] state: Box<dyn CarmenState>) {}

    #[test]
    fn empty_code_hash_is_keccak256_of_empty_code() {
        let hasher = Keccak256::new();
        let expected = hasher.finalize();
        assert_eq!(EMPTY_CODE_HASH, expected.as_slice());
    }

    #[test]
    fn new_creates_empty_state() {
        let state = VerkleTrieCarmenState::<SimpleInMemoryVerkleTrie>::new();
        assert_eq!(state.get_hash().unwrap(), Hash::default());
    }

    #[rstest_reuse::apply(all_state_impls)]
    fn account_exists_checks_whether_account_has_non_zero_code_hash(
        #[case] state: Box<dyn CarmenState>,
    ) {
        let addr = Address::default();
        assert!(!state.account_exists(&addr).unwrap());
        assert_eq!(state.get_code_hash(&addr).unwrap(), Hash::default());

        set_code(&*state, addr, &[0x01, 0x02, 0x03]);
        assert!(state.account_exists(&addr).unwrap());
    }

    #[rstest_reuse::apply(all_state_impls)]
    fn creating_account_sets_empty_code_hash(#[case] state: Box<dyn CarmenState>) {
        let addr = Address::from_index_values(0, &[(0, 1)]);
        create_account(&*state, addr);
        let code_hash = state.get_code_hash(&addr).unwrap();
        assert_eq!(code_hash, EMPTY_CODE_HASH);
    }

    #[rstest_reuse::apply(all_state_impls)]
    fn creating_account_does_not_overwrite_basic_account_data(#[case] state: Box<dyn CarmenState>) {
        let addr = Address::from_index_values(0, &[(0, 1)]);
        let initial_balance = crypto_bigint::U256::from_u32(42).to_be_bytes();
        let initial_nonce = 7u64.to_be_bytes();

        set_balance(&*state, addr, initial_balance);
        set_nonce(&*state, addr, initial_nonce);

        create_account(&*state, addr);

        let balance = state.get_balance(&addr).unwrap();
        assert_eq!(balance, initial_balance);
        let nonce = state.get_nonce(&addr).unwrap();
        assert_eq!(nonce, initial_nonce);
    }

    #[rstest_reuse::apply(all_state_impls)]
    fn can_store_and_retrieve_nonces(#[case] state: Box<dyn CarmenState>) {
        let addr1 = Address::from_index_values(0, &[(0, 1)]);
        let addr2 = Address::from_index_values(0, &[(0, 2)]);

        let nonce = state.get_nonce(&addr1).unwrap();
        assert_eq!(nonce, Nonce::default());
        let nonce = state.get_nonce(&addr2).unwrap();
        assert_eq!(nonce, Nonce::default());

        set_nonce(&*state, addr1, 42u64.to_be_bytes());
        let nonce = state.get_nonce(&addr1).unwrap();
        assert_eq!(nonce, 42u64.to_be_bytes());

        set_nonce(&*state, addr2, 33u64.to_be_bytes());
        let nonce = state.get_nonce(&addr2).unwrap();
        assert_eq!(nonce, 33u64.to_be_bytes());

        // Nonce for addr1 should remain unchanged
        let nonce = state.get_nonce(&addr1).unwrap();
        assert_eq!(nonce, 42u64.to_be_bytes());

        set_nonce(&*state, addr1, 123u64.to_be_bytes());
        let nonce = state.get_nonce(&addr1).unwrap();
        assert_eq!(nonce, 123u64.to_be_bytes());

        // Nonce for addr2 should remain unchanged
        let nonce = state.get_nonce(&addr2).unwrap();
        assert_eq!(nonce, 33u64.to_be_bytes());
    }

    #[rstest_reuse::apply(all_state_impls)]
    fn can_store_and_retrieve_balances(#[case] state: Box<dyn CarmenState>) {
        let addr1 = Address::from_index_values(0, &[(0, 1)]);
        let addr2 = Address::from_index_values(0, &[(0, 2)]);

        let balance = state.get_balance(&addr1).unwrap();
        assert_eq!(balance, U256::default());
        let balance = state.get_balance(&addr2).unwrap();
        assert_eq!(balance, U256::default());

        let amount1 = crypto_bigint::U256::from_u32(42).to_be_bytes();
        let amount2 = crypto_bigint::U256::from_u32(33).to_be_bytes();
        let amount3 = crypto_bigint::U256::from_u32(123).to_be_bytes();

        set_balance(&*state, addr1, amount1);
        let balance = state.get_balance(&addr1).unwrap();
        assert_eq!(balance, amount1);

        set_balance(&*state, addr2, amount2);
        let balance = state.get_balance(&addr2).unwrap();
        assert_eq!(balance, amount2);

        // Balance for addr1 should remain unchanged
        let balance = state.get_balance(&addr1).unwrap();
        assert_eq!(balance, amount1);

        set_balance(&*state, addr1, amount3);
        let balance = state.get_balance(&addr1).unwrap();
        assert_eq!(balance, amount3);

        // Balance for addr2 should remain unchanged
        let balance = state.get_balance(&addr2).unwrap();
        assert_eq!(balance, amount2);
    }

    #[rstest_reuse::apply(all_state_impls)]
    fn balance_is_stored_as_128bit_int(#[case] state: Box<dyn CarmenState>) {
        let full256 = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x10, 0x11, 0x12, 0x13,
            0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
            0x28, 0x29, 0x30, 0x31,
        ];
        let truncated128 = [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x16, 0x17, 0x18, 0x19, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
            0x28, 0x29, 0x30, 0x31,
        ];

        let addr = Address::from_index_values(0, &[(0, 1)]);

        set_balance(&*state, addr, full256);

        let stored = state.get_balance(&addr).unwrap();
        assert_eq!(stored, truncated128);
    }

    #[rstest_reuse::apply(all_state_impls)]
    fn can_store_and_retrieve_codes(#[case] state: Box<dyn CarmenState>) {
        let cases = [
            ("empty", Vec::from_index_values(0, &[])),
            ("short", vec![0x01, 0x02, 0x03]),
            ("long", Vec::from_index_values(1, &[(10_000, 0x02)])),
            ("1 KiB", vec![0x01; 1024]),
            ("2 KiB", vec![0x02; 2 * 1024]),
            ("4 KiB", vec![0x03; 4 * 1024]),
            ("8 KiB", vec![0x04; 8 * 1024]),
            ("16 KiB", vec![0x05; 16 * 1024]),
            ("32 KiB", vec![0x06; 32 * 1024]),
        ];

        let addr = Address::from_index_values(0, &[(0, 1)]);

        for (name, code) in cases.clone() {
            set_code(&*state, addr, &code);

            let len = state.get_code_len(&addr).unwrap();
            assert_eq!(len as usize, code.len());

            let hash = state.get_code_hash(&addr).unwrap();
            let mut hasher = Keccak256::new();
            hasher.update(&code);
            assert_eq!(hash, Hash::from(hasher.finalize()));

            let mut code_buf = vec![MaybeUninit::uninit(); len as usize];
            let retrieved_len = state.get_code(&addr, &mut code_buf).unwrap();
            assert_eq!(retrieved_len, code.len());

            for (expected, received) in code.iter().zip(code_buf[..retrieved_len].iter()) {
                // SAFETY: `get_code` initializes `len` bytes of `code_buf`.
                unsafe {
                    assert_eq!(
                        expected,
                        &received.assume_init(),
                        "code mismatch for case: {name}"
                    );
                }
            }
        }
    }

    #[rstest_reuse::apply(all_state_impls)]
    fn setting_basic_data_fields_does_not_interfere_with_others(
        #[case] state: Box<dyn CarmenState>,
    ) {
        let addr = Address::from_index_values(0, &[(0, 1)]);
        let balance = crypto_bigint::U256::from_u32(42).to_be_bytes();
        let nonce = 7u64.to_be_bytes();
        let code = vec![0x01, 0x02, 0x03, 0x04, 0x05];

        // Initially, all fields should be zero
        let stored_balance = state.get_balance(&addr).unwrap();
        assert_eq!(stored_balance, U256::default());
        let stored_nonce = state.get_nonce(&addr).unwrap();
        assert_eq!(stored_nonce, Nonce::default());
        let stored_code_len = state.get_code_len(&addr).unwrap();
        assert_eq!(stored_code_len, 0);

        // Set balance
        set_balance(&*state, addr, balance);
        let stored_balance = state.get_balance(&addr).unwrap();
        assert_eq!(stored_balance, balance);
        let stored_nonce = state.get_nonce(&addr).unwrap();
        assert_eq!(stored_nonce, Nonce::default());
        let stored_code_len = state.get_code_len(&addr).unwrap();
        assert_eq!(stored_code_len, 0);

        // Set nonce
        set_nonce(&*state, addr, nonce);
        let stored_balance = state.get_balance(&addr).unwrap();
        assert_eq!(stored_balance, balance);
        let stored_nonce = state.get_nonce(&addr).unwrap();
        assert_eq!(stored_nonce, nonce);
        let stored_code_len = state.get_code_len(&addr).unwrap();
        assert_eq!(stored_code_len, 0);

        // Set code
        set_code(&*state, addr, &code);
        let stored_balance = state.get_balance(&addr).unwrap();
        assert_eq!(stored_balance, balance);
        let stored_nonce = state.get_nonce(&addr).unwrap();
        assert_eq!(stored_nonce, nonce);
        let stored_code_len = state.get_code_len(&addr).unwrap();
        assert_eq!(stored_code_len, code.len() as u32);
    }

    #[rstest_reuse::apply(all_state_impls)]
    fn can_store_and_retrieve_storage_values(#[case] state: Box<dyn CarmenState>) {
        let addr = Address::from_index_values(0, &[(0, 1)]);
        let key = Key::from_index_values(0, &[(0, 42)]);
        let value = Value::from_index_values(0, &[(0, 1), (0, 2), (0, 3)]);

        // Initially, the storage value should be empty
        let retrieved_value = state.get_storage_value(&addr, &key).unwrap();
        assert_eq!(retrieved_value, Value::default());

        set_storage(&*state, addr, key, value);

        let retrieved_value = state.get_storage_value(&addr, &key).unwrap();
        assert_eq!(retrieved_value, value);

        let value2 = Value::from_index_values(0, &[(0, 3), (1, 2), (2, 1)]);
        set_storage(&*state, addr, key, value2);

        let retrieved_value = state.get_storage_value(&addr, &key).unwrap();
        assert_eq!(retrieved_value, value2);
    }

    #[rstest_reuse::apply(all_state_impls)]
    fn state_with_content_has_same_commitment_as_geth(#[case] state: Box<dyn CarmenState>) {
        const PUSH32: u8 = 0x7f;

        let addr1 = Address::from_index_values(0, &[(0, 1)]);
        let addr2 = Address::from_index_values(0, &[(0, 2)]);
        let addr3 = Address::from_index_values(0, &[(0, 3)]);

        let code1 = Vec::from_index_values(0, &[(0, 0x01), (1, 0x02)]);
        // Truncated push data
        let code2 = Vec::from_index_values(0, &[(0, 0x03), (30, PUSH32), (31, 0x05)]);
        // Fills multiple leafs
        let code3 =
            Vec::from_index_values(0, &[(0, 0x06), (1, 0x07), (2, 0x08), (3 * 256 * 32, 0x09)]);

        let update = Update {
            balances: &[
                BalanceUpdate {
                    addr: addr1,
                    balance: crypto_bigint::U256::from_u32(100).to_be_bytes(),
                },
                BalanceUpdate {
                    addr: addr2,
                    balance: crypto_bigint::U256::from_u32(200).to_be_bytes(),
                },
                BalanceUpdate {
                    addr: addr3,
                    balance: crypto_bigint::U256::from_u32(300).to_be_bytes(),
                },
            ],
            nonces: &[
                NonceUpdate {
                    addr: addr1,
                    nonce: 1u64.to_be_bytes(),
                },
                NonceUpdate {
                    addr: addr2,
                    nonce: 2u64.to_be_bytes(),
                },
                NonceUpdate {
                    addr: addr3,
                    nonce: 3u64.to_be_bytes(),
                },
            ],
            codes: vec![
                CodeUpdate {
                    addr: addr1,
                    code: &code1,
                },
                CodeUpdate {
                    addr: addr2,
                    code: &code2,
                },
                CodeUpdate {
                    addr: addr3,
                    code: &code3,
                },
            ],
            slots: &[
                SlotUpdate {
                    addr: addr1,
                    key: Key::from_index_values(0, &[(0, 1)]),
                    value: Value::from_index_values(0, &[(0, 0x05)]),
                },
                SlotUpdate {
                    addr: addr2,
                    key: Key::from_index_values(0, &[(0, 2)]),
                    value: Value::from_index_values(0, &[(0, 0x06)]),
                },
            ],
            ..Default::default()
        };

        state.apply_block_update(0, update.clone()).unwrap();

        let hash = state.get_hash().unwrap();
        let expected = "0x200c4ce1a7807afcff0226ac29f0d1d611e5bee2a500df573fe36c10034660fa";

        assert_eq!(const_hex::encode_prefixed(hash), expected);
    }

    #[rstest_reuse::apply(all_state_impls)]
    fn single_account_has_same_commitment_as_geth(#[case] state: Box<dyn CarmenState>) {
        let addr = Address::from_index_values(0, &[(0, 1)]);

        let update = Update {
            balances: &[BalanceUpdate {
                addr,
                balance: crypto_bigint::U256::from_u32(1).to_be_bytes(),
            }],
            ..Default::default()
        };

        state.apply_block_update(0, update.clone()).unwrap();

        let hash = state.get_hash().unwrap();
        let expected = "0x6b188de48e78866c34d38382b1965ec4908a0525d41cd66d18385390010b707d";

        assert_eq!(const_hex::encode_prefixed(hash), expected);
    }

    fn create_account(state: &dyn CarmenState, addr: Address) {
        state
            .apply_block_update(
                0,
                Update {
                    created_accounts: &[addr],
                    ..Default::default()
                },
            )
            .unwrap();
    }

    fn set_nonce(state: &dyn CarmenState, addr: Address, nonce: Nonce) {
        state
            .apply_block_update(
                0,
                Update {
                    nonces: &[NonceUpdate { addr, nonce }],
                    ..Default::default()
                },
            )
            .unwrap();
    }

    fn set_balance(state: &dyn CarmenState, addr: Address, balance: U256) {
        state
            .apply_block_update(
                0,
                Update {
                    balances: &[BalanceUpdate { addr, balance }],
                    ..Default::default()
                },
            )
            .unwrap();
    }

    fn set_code(state: &dyn CarmenState, addr: Address, code: &[u8]) {
        state
            .apply_block_update(
                0,
                Update {
                    codes: vec![CodeUpdate { addr, code }],
                    ..Default::default()
                },
            )
            .unwrap();
    }

    fn set_storage(state: &dyn CarmenState, addr: Address, key: Key, value: Value) {
        state
            .apply_block_update(
                0,
                Update {
                    slots: &[SlotUpdate { addr, key, value }],
                    ..Default::default()
                },
            )
            .unwrap();
    }
}
