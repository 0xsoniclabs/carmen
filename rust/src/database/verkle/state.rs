// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{mem::MaybeUninit, sync::Arc};

use crate::{
    CarmenState, IsArchive, VerkleStorage,
    database::verkle::{
        KeyedUpdateBatch, ManagedVerkleTrie,
        embedding::{VerkleTrieEmbedding, code},
        variants::{
            CrateCryptoInMemoryVerkleTrie, SimpleInMemoryVerkleTrie,
            managed::{VerkleNode, VerkleNodeId},
        },
        verkle_trie::VerkleTrie,
    },
    error::{BTResult, Error},
    node_manager::{NodeManager, cached_node_manager::CachedNodeManager},
    storage::RootIdProvider,
    sync::Mutex,
    types::{Address, Hash, Key, Nonce, U256, Update, Value},
};

pub const EMPTY_CODE_HASH: Hash = [
    197, 210, 70, 1, 134, 247, 35, 60, 146, 126, 125, 178, 220, 199, 3, 192, 229, 0, 182, 83, 202,
    130, 39, 59, 123, 250, 216, 4, 93, 133, 164, 112,
];

/// The mode of the Verkle trie state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateMode {
    /// A live state that only keeps the state of the latest block.
    Live,
    /// An archive state for a specific block height.
    Archive(u64),
    /// An archive state that increases its block height on each update and emulates a live state.
    EvolvingArchive,
}

/// The current block height of the state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockHeight {
    /// The block height of a live state which is always at the latest block.
    Live,
    /// The block height of an archive state at a specific block height.
    Archive(u64),
    /// The block height of an archive state that increases on each update.
    /// If the state is empty (i.e., no updates have been applied yet), the height is `None`.
    EvolvingArchive(Option<u64>),
}

impl BlockHeight {
    /// Returns whether the state is in archive mode.
    fn is_archive(&self) -> bool {
        matches!(
            self,
            BlockHeight::Archive(_) | BlockHeight::EvolvingArchive(_)
        )
    }
}

/// An implementation of [`CarmenState`] that uses a Verkle trie as the underlying data structure.
pub struct VerkleTrieCarmenState<T: VerkleTrie> {
    trie: T,
    embedding: VerkleTrieEmbedding,
    block_height: Mutex<BlockHeight>,
}

impl VerkleTrieCarmenState<SimpleInMemoryVerkleTrie> {
    pub fn new_live() -> Self {
        Self {
            trie: SimpleInMemoryVerkleTrie::new(),
            embedding: VerkleTrieEmbedding::new(),
            block_height: Mutex::new(BlockHeight::Live),
        }
    }
}

impl VerkleTrieCarmenState<CrateCryptoInMemoryVerkleTrie> {
    pub fn new_live() -> Self {
        Self {
            trie: CrateCryptoInMemoryVerkleTrie::new(),
            embedding: VerkleTrieEmbedding::new(),
            block_height: Mutex::new(BlockHeight::Live),
        }
    }
}

impl VerkleTrieCarmenState<ManagedVerkleTrie<CachedNodeManager<VerkleStorage>>> {
    /// Creates a new [`VerkleTrieCarmenState`] using a managed Verkle trie with the given node
    /// manager. Forwards any errors from [`ManagedVerkleTrie::try_new`].
    pub fn try_new(
        manager: Arc<CachedNodeManager<VerkleStorage>>,
        state_mode: StateMode,
    ) -> BTResult<Self, Error> {
        let block = match state_mode {
            StateMode::Live | StateMode::EvolvingArchive => manager.highest_block_number()?,
            StateMode::Archive(block) => Some(block),
        };
        let trie = match block {
            Some(block) => ManagedVerkleTrie::try_from_block_height(manager, block)?,
            None => ManagedVerkleTrie::try_new(manager)?,
        };
        let block_height = match state_mode {
            StateMode::Live => BlockHeight::Live,
            StateMode::Archive(block) => BlockHeight::Archive(block),
            StateMode::EvolvingArchive => BlockHeight::EvolvingArchive(block),
        };
        Ok(Self {
            trie,
            embedding: VerkleTrieEmbedding::new(),
            block_height: Mutex::new(block_height),
        })
    }
}

impl<T: VerkleTrie> IsArchive for VerkleTrieCarmenState<T> {
    fn is_archive(&self) -> bool {
        self.block_height.lock().unwrap().is_archive()
    }
}

impl<T: VerkleTrie> CarmenState for VerkleTrieCarmenState<T> {
    fn account_exists(&self, addr: &Address) -> BTResult<bool, Error> {
        Ok(self.get_code_hash(addr)? != Hash::default())
    }

    fn get_balance(&self, addr: &Address) -> BTResult<U256, Error> {
        let key = self.embedding.get_basic_data_key(addr);
        let value = self.trie.lookup(&key)?;
        let mut result = U256::default();
        result[16..].copy_from_slice(&value[16..32]);
        Ok(result)
    }

    fn get_nonce(&self, addr: &Address) -> BTResult<Nonce, Error> {
        let key = self.embedding.get_basic_data_key(addr);
        let value = self.trie.lookup(&key)?;
        // Safe to unwrap: Always 8 bytes
        Ok(value[8..16].try_into().unwrap())
    }

    fn get_storage_value(&self, addr: &Address, key: &Key) -> BTResult<Value, Error> {
        let key = self.embedding.get_storage_key(addr, key);
        let value = self.trie.lookup(&key)?;
        Ok(Value::from(value))
    }

    fn get_code(&self, addr: &Address, code_buf: &mut [MaybeUninit<u8>]) -> BTResult<usize, Error> {
        let len = self.get_code_len(addr)?;
        let chunk_count = len / 31 + 1;
        let mut chunks = Vec::with_capacity(chunk_count as usize);
        for i in 0..chunk_count {
            let key = self.embedding.get_code_chunk_key(addr, i);
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
        let key = self.embedding.get_code_hash_key(addr);
        let value = self.trie.lookup(&key)?;
        Ok(Hash::from(value))
    }

    fn get_code_len(&self, addr: &Address) -> BTResult<u32, Error> {
        let key = self.embedding.get_basic_data_key(addr);
        let value = self.trie.lookup(&key)?;
        // Safe to unwrap - slice is always 4 bytes
        Ok(u32::from_be_bytes(value[4..8].try_into().unwrap()))
    }

    fn get_hash(&self) -> BTResult<Hash, Error> {
        let commitment = self.trie.commit()?;
        Ok(Hash::from(commitment.compress()))
    }

    #[allow(clippy::needless_lifetimes)]
    fn apply_block_update<'u>(&self, block: u64, update: Update<'u>) -> BTResult<(), Error> {
        let _span = tracy_client::span!("VerkleTrieCarmenState::apply_block_update");
        let mut block_height = self.block_height.lock().unwrap();
        if block.is_multiple_of(10_000) {
            self.trie.print_size_stats();
        }
        let block = match &mut *block_height {
            BlockHeight::Live => 0, // For the liveDB we always pass block height 0
            BlockHeight::Archive(_) => {
                return Err(Error::UnsupportedOperation(
                    "apply_block_update is not supported on archive states".into(),
                )
                .into());
            }
            BlockHeight::EvolvingArchive(height) => {
                match height {
                    Some(height) => {
                        if block != *height + 1 {
                            return Err(Error::UnsupportedOperation("apply_block_update called on a block that was already updated or with an update that is not for the next block".into()).into());
                        }
                    }
                    None => {
                        if block != 0 {
                            return Err(Error::UnsupportedOperation("apply_block_update called on a block that was already updated or with an update that is not for the next block".into()).into());
                        }
                    }
                }
                *height = Some(block);
                block
            }
        };
        if let Ok(update) = KeyedUpdateBatch::try_from_with_embedding(update, &self.embedding) {
            self.trie.store(&update, block_height.is_archive())?;
        }
        self.trie.after_update(block)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use mockall::predicate::eq;
    use sha3::{Digest, Keccak256};

    use super::*;
    use crate::{
        database::verkle::{test_utils::FromIndexValues, verkle_trie::MockVerkleTrie},
        error::BTError,
        node_manager::in_memory_node_manager::InMemoryNodeManager,
        types::{BalanceUpdate, CodeUpdate, NonceUpdate, SlotUpdate},
    };

    #[rstest_reuse::template]
    #[rstest::rstest]
    #[case::simple_in_memory(Box::new(VerkleTrieCarmenState::<SimpleInMemoryVerkleTrie>::new_live()) as Box<dyn CarmenState>)]
    #[case::crate_crypto(Box::new(VerkleTrieCarmenState::<CrateCryptoInMemoryVerkleTrie>::new_live()) as Box<dyn CarmenState>)]
    #[case::managed_live(Box::new(VerkleTrieCarmenState::<ManagedVerkleTrie<InMemoryNodeManager<VerkleNodeId, VerkleNode>>>::try_new(Arc::new(InMemoryNodeManager::new(100)),StateMode::Live).unwrap()) as Box<dyn CarmenState>)]
    #[case::managed_evolving_archive(Box::new(VerkleTrieCarmenState::<ManagedVerkleTrie<InMemoryNodeManager<VerkleNodeId, VerkleNode>>>::try_new(Arc::new(InMemoryNodeManager::new(100)),StateMode::EvolvingArchive).unwrap()) as Box<dyn CarmenState>)]
    fn all_state_impls(#[case] state: Box<dyn CarmenState>) {}

    #[test]
    fn empty_code_hash_is_keccak256_of_empty_code() {
        let hasher = Keccak256::new();
        let expected = hasher.finalize();
        assert_eq!(EMPTY_CODE_HASH, expected.as_slice());
    }

    #[test]
    fn new_live_creates_empty_state() {
        let state = VerkleTrieCarmenState::<SimpleInMemoryVerkleTrie>::new_live();
        assert_eq!(state.get_hash().unwrap(), Hash::default());

        let state = VerkleTrieCarmenState::<CrateCryptoInMemoryVerkleTrie>::new_live();
        assert_eq!(state.get_hash().unwrap(), Hash::default());
    }

    #[test]
    fn apply_block_update_on_archive_state_returns_unsupported_operation_error() {
        let node_manager = Arc::new(InMemoryNodeManager::new(100));
        let id = node_manager
            .add(VerkleNode::Inner256(Box::default()))
            .unwrap();
        node_manager.set_root_id(2, id).unwrap();

        let state = VerkleTrieCarmenState::try_new(node_manager, StateMode::Archive(2)).unwrap();

        let result = state.apply_block_update(3, Update::default());
        assert_eq!(
            result.map_err(BTError::into_inner),
            Err(Error::UnsupportedOperation(
                "apply_block_update is not supported on archive states".into()
            ))
        );
    }

    #[test]
    fn apply_block_update_on_evolving_archive_state_checks_that_block_is_next_block() {
        let state = VerkleTrieCarmenState::try_new(
            Arc::new(InMemoryNodeManager::new(100)),
            StateMode::EvolvingArchive,
        )
        .unwrap();

        let result = state.apply_block_update(1, Update::default());
        assert_eq!(
            result.map_err(BTError::into_inner),
            Err(Error::UnsupportedOperation("apply_block_update called on a block that was already updated or with an update that is not for the next block".into()))
        );

        let result = state.apply_block_update(0, Update::default());
        assert!(result.is_ok());

        let result = state.apply_block_update(0, Update::default());
        assert_eq!(
            result.map_err(BTError::into_inner),
            Err(Error::UnsupportedOperation("apply_block_update called on a block that was already updated or with an update that is not for the next block".into()))
        );
        let result = state.apply_block_update(2, Update::default());
        assert_eq!(
            result.map_err(BTError::into_inner),
            Err(Error::UnsupportedOperation("apply_block_update called on a block that was already updated or with an update that is not for the next block".into()))
        );

        let result = state.apply_block_update(1, Update::default());
        assert!(result.is_ok());
    }

    #[rstest_reuse::apply(all_state_impls)]
    fn account_exists_checks_whether_account_has_non_zero_code_hash(
        #[case] state: Box<dyn CarmenState>,
    ) {
        let addr = Address::default();
        assert!(!state.account_exists(&addr).unwrap());
        assert_eq!(state.get_code_hash(&addr).unwrap(), Hash::default());

        set_code(&*state, addr, &[0x01, 0x02, 0x03], 0);
        assert!(state.account_exists(&addr).unwrap());
    }

    #[rstest_reuse::apply(all_state_impls)]
    fn creating_account_sets_empty_code_hash(#[case] state: Box<dyn CarmenState>) {
        let addr = Address::from_index_values(0, &[(0, 1)]);
        create_account(&*state, addr, 0);
        let code_hash = state.get_code_hash(&addr).unwrap();
        assert_eq!(code_hash, EMPTY_CODE_HASH);
    }

    #[rstest_reuse::apply(all_state_impls)]
    fn creating_account_does_not_overwrite_basic_account_data(#[case] state: Box<dyn CarmenState>) {
        let addr = Address::from_index_values(0, &[(0, 1)]);
        let initial_balance = crypto_bigint::U256::from_u32(42).to_be_bytes();
        let initial_nonce = 7u64.to_be_bytes();

        set_balance(&*state, addr, initial_balance, 0);
        set_nonce(&*state, addr, initial_nonce, 1);

        create_account(&*state, addr, 2);

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

        set_nonce(&*state, addr1, 42u64.to_be_bytes(), 0);
        let nonce = state.get_nonce(&addr1).unwrap();
        assert_eq!(nonce, 42u64.to_be_bytes());

        set_nonce(&*state, addr2, 33u64.to_be_bytes(), 1);
        let nonce = state.get_nonce(&addr2).unwrap();
        assert_eq!(nonce, 33u64.to_be_bytes());

        // Nonce for addr1 should remain unchanged
        let nonce = state.get_nonce(&addr1).unwrap();
        assert_eq!(nonce, 42u64.to_be_bytes());

        set_nonce(&*state, addr1, 123u64.to_be_bytes(), 2);
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

        set_balance(&*state, addr1, amount1, 0);
        let balance = state.get_balance(&addr1).unwrap();
        assert_eq!(balance, amount1);

        set_balance(&*state, addr2, amount2, 1);
        let balance = state.get_balance(&addr2).unwrap();
        assert_eq!(balance, amount2);

        // Balance for addr1 should remain unchanged
        let balance = state.get_balance(&addr1).unwrap();
        assert_eq!(balance, amount1);

        set_balance(&*state, addr1, amount3, 2);
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

        set_balance(&*state, addr, full256, 0);

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

        for (i, (name, code)) in cases.into_iter().enumerate() {
            set_code(&*state, addr, &code, i as u64);

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
        set_balance(&*state, addr, balance, 0);
        let stored_balance = state.get_balance(&addr).unwrap();
        assert_eq!(stored_balance, balance);
        let stored_nonce = state.get_nonce(&addr).unwrap();
        assert_eq!(stored_nonce, Nonce::default());
        let stored_code_len = state.get_code_len(&addr).unwrap();
        assert_eq!(stored_code_len, 0);

        // Set nonce
        set_nonce(&*state, addr, nonce, 1);
        let stored_balance = state.get_balance(&addr).unwrap();
        assert_eq!(stored_balance, balance);
        let stored_nonce = state.get_nonce(&addr).unwrap();
        assert_eq!(stored_nonce, nonce);
        let stored_code_len = state.get_code_len(&addr).unwrap();
        assert_eq!(stored_code_len, 0);

        // Set code
        set_code(&*state, addr, &code, 2);
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

        set_storage(&*state, addr, key, value, 0);

        let retrieved_value = state.get_storage_value(&addr, &key).unwrap();
        assert_eq!(retrieved_value, value);

        let value2 = Value::from_index_values(0, &[(0, 3), (1, 2), (2, 1)]);
        set_storage(&*state, addr, key, value2, 1);

        let retrieved_value = state.get_storage_value(&addr, &key).unwrap();
        assert_eq!(retrieved_value, value2);
    }

    #[rstest_reuse::apply(all_state_impls)]
    fn state_with_content_has_same_commitment_as_geth(#[case] state: Box<dyn CarmenState>) {
        const PUSH32: u8 = 0x7f;

        let embedding = VerkleTrieEmbedding::new();

        let addr1 = Address::from_index_values(0, &[(0, 0)]);
        let addr2 = Address::from_index_values(0, &[(0, 174)]);
        let addr3 = Address::from_index_values(0, &[(0, 51), (1, 1)]);

        // We choose addresses that have a collision in the first byte of their basic data key.
        // This way we can trigger an inner node to be inserted after the first update, which
        // then covers some edge cases in point-wise commitment updates (depending on trie
        // implementation).
        assert_eq!(
            embedding.get_basic_data_key(&addr1)[0],
            embedding.get_basic_data_key(&addr2)[0]
        );
        assert_eq!(
            embedding.get_basic_data_key(&addr1)[0],
            embedding.get_basic_data_key(&addr3)[0]
        );

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
        let expected = "0x71db64871d05d74a5e5fe5a0b3843147cc0c45d00beb15102cef224f78bae4e2";

        assert_eq!(const_hex::encode_prefixed(hash), expected);
    }

    #[rstest_reuse::apply(all_state_impls)]
    fn incremental_updates_result_in_same_commitments_as_geth(#[case] state: Box<dyn CarmenState>) {
        const PUSH32: u8 = 0x7f;

        let embedding = VerkleTrieEmbedding::new();

        let addr1 = Address::from_index_values(0, &[(0, 0)]);
        let addr2 = Address::from_index_values(0, &[(0, 174)]);
        let addr3 = Address::from_index_values(0, &[(0, 51), (1, 1)]);

        // We choose addresses that have a collision in the first byte of their basic data key.
        // This way we can trigger an inner node to be inserted after the first update, which
        // then covers some edge cases in point-wise commitment updates (depending on trie
        // implementation).
        assert_eq!(
            embedding.get_basic_data_key(&addr1)[0],
            embedding.get_basic_data_key(&addr2)[0]
        );
        assert_eq!(
            embedding.get_basic_data_key(&addr1)[0],
            embedding.get_basic_data_key(&addr3)[0]
        );

        let code1 = Vec::from_index_values(0, &[(0, 0x01), (1, 0x02)]);
        let code2 = Vec::from_index_values(0, &[(0, 0x11), (1, 0x12)]);
        let code3 = Vec::from_index_values(0, &[(0, 0x13), (1, 0x14), (2, PUSH32), (32, 0x15)]);
        let code4 = Vec::from_index_values(0, &[(0, 0x16), (1, 0x17)]);
        let code5 = Vec::from_index_values(0, &[(10_000, 1)]);
        let code6 = Vec::from_index_values(0, &[(0, 0x01), (1, 0x02), (2, 0x03)]);

        let updates = [
            // Initialize data only for addr1 such that the basic fields (balance, nonce, code
            // size) all end up in a single leaf node attached to the root.
            Update {
                balances: &[BalanceUpdate {
                    addr: addr1,
                    balance: crypto_bigint::U256::from_u32(100).to_be_bytes(),
                }],
                nonces: &[NonceUpdate {
                    addr: addr1,
                    nonce: 1u64.to_be_bytes(),
                }],
                codes: vec![CodeUpdate {
                    addr: addr1,
                    code: &code1,
                }],
                ..Default::default()
            },
            // Update fields for addr1 and add some additional data for addr2 and addr3.
            // This will cause an inner node to be created between the root and the leaf
            // for addr1.
            Update {
                balances: &[
                    BalanceUpdate {
                        addr: addr1,
                        balance: crypto_bigint::U256::from_u32(150).to_be_bytes(),
                    },
                    BalanceUpdate {
                        addr: addr2,
                        balance: crypto_bigint::U256::from_u32(250).to_be_bytes(),
                    },
                    BalanceUpdate {
                        addr: addr3,
                        balance: crypto_bigint::U256::from_u32(350).to_be_bytes(),
                    },
                ],
                nonces: &[
                    NonceUpdate {
                        addr: addr1,
                        nonce: 3u64.to_be_bytes(),
                    },
                    NonceUpdate {
                        addr: addr2,
                        nonce: 4u64.to_be_bytes(),
                    },
                    NonceUpdate {
                        addr: addr3,
                        nonce: 5u64.to_be_bytes(),
                    },
                ],
                codes: vec![
                    CodeUpdate {
                        addr: addr1,
                        code: &code2,
                    },
                    CodeUpdate {
                        addr: addr2,
                        code: &code3,
                    },
                    CodeUpdate {
                        addr: addr3,
                        code: &code4,
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
                    SlotUpdate {
                        addr: addr3,
                        key: Key::from_index_values(0, &[(0, 3)]),
                        value: Value::from_index_values(0, &[(0, 0x07)]),
                    },
                ],
                ..Default::default()
            },
            // Set data to zero.
            Update {
                balances: &[BalanceUpdate {
                    addr: addr1,
                    balance: crypto_bigint::U256::from_u32(0).to_be_bytes(),
                }],
                nonces: &[NonceUpdate {
                    addr: addr1,
                    nonce: 0u64.to_be_bytes(),
                }],
                codes: vec![CodeUpdate {
                    addr: addr1,
                    code: &[],
                }],
                slots: &[SlotUpdate {
                    addr: addr1,
                    key: Key::from_index_values(0, &[(0, 1)]),
                    value: Value::from_index_values(0, &[]),
                }],
                ..Default::default()
            },
            // Grow code size.
            Update {
                codes: vec![CodeUpdate {
                    addr: addr1,
                    code: &code5,
                }],
                ..Default::default()
            },
            // Shrink code size.
            Update {
                codes: vec![CodeUpdate {
                    addr: addr1,
                    code: &code6,
                }],
                ..Default::default()
            },
        ];

        let expected_hashes = [
            "0x2d819e0ef65c45cbe82d2417b957be71fcfe38e40167ed7b77e6cb1010b947fa",
            "0x60b4accd38f50d4ec02fe454511d4a742a50b7b6bda5b9ca198584b5df5b0783",
            "0x251c33d75b1e6f00d9308098ef43212cb1923fd1412291b689e467fe58f7cb1f",
            "0x6b163d9b36a6aa8dd30e5a941d6c9660cddb4fe46862202206875c0c4021b9fe",
            "0x5882a33bda9102b7cdaf74009b3eb385ece7bf0f3e09f27a250954025c3f9fc1",
        ];

        for (i, update) in updates.into_iter().enumerate() {
            state.apply_block_update(i as u64, update.clone()).unwrap();
            let hash = state.get_hash().unwrap();
            let expected = expected_hashes[i];
            assert_eq!(
                const_hex::encode_prefixed(hash),
                *expected,
                "mismatch after update {i}",
            );
        }
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

    #[test]
    fn state_calls_after_update_with_correct_block_height_after_applying_update() {
        let block_height = 42;
        let mut trie = MockVerkleTrie::new();
        trie.expect_after_update()
            .with(eq(block_height))
            .times(1)
            .returning(|_| Ok(()));
        let state = VerkleTrieCarmenState {
            trie,
            embedding: VerkleTrieEmbedding::new(),
            block_height: Mutex::new(BlockHeight::EvolvingArchive(Some(block_height - 1))),
        };
        state
            .apply_block_update(block_height, Update::default())
            .unwrap();
    }

    fn create_account(state: &dyn CarmenState, addr: Address, block_number: u64) {
        state
            .apply_block_update(
                block_number,
                Update {
                    created_accounts: &[addr],
                    ..Default::default()
                },
            )
            .unwrap();
    }

    fn set_nonce(state: &dyn CarmenState, addr: Address, nonce: Nonce, block_number: u64) {
        state
            .apply_block_update(
                block_number,
                Update {
                    nonces: &[NonceUpdate { addr, nonce }],
                    ..Default::default()
                },
            )
            .unwrap();
    }

    fn set_balance(state: &dyn CarmenState, addr: Address, balance: U256, block_number: u64) {
        state
            .apply_block_update(
                block_number,
                Update {
                    balances: &[BalanceUpdate { addr, balance }],
                    ..Default::default()
                },
            )
            .unwrap();
    }

    fn set_code(state: &dyn CarmenState, addr: Address, code: &[u8], block_number: u64) {
        state
            .apply_block_update(
                block_number,
                Update {
                    codes: vec![CodeUpdate { addr, code }],
                    ..Default::default()
                },
            )
            .unwrap();
    }

    fn set_storage(
        state: &dyn CarmenState,
        addr: Address,
        key: Key,
        value: Value,
        block_number: u64,
    ) {
        state
            .apply_block_update(
                block_number,
                Update {
                    slots: &[SlotUpdate { addr, key, value }],
                    ..Default::default()
                },
            )
            .unwrap();
    }
}
