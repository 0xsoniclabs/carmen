// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{
    borrow::Cow,
    cmp::Ordering,
    ops::{Deref, Range},
};

use rayon::{self, slice::ParallelSliceMut};
use sha3::{Digest, Keccak256};
use verkle_trie::Key;

use crate::{
    database::verkle::{
        embedding::{VerkleTrieEmbedding, code},
        state::EMPTY_CODE_HASH,
    },
    types::{BalanceUpdate, CodeUpdate, Hash, NonceUpdate, SlotUpdate, Update, Value},
};

type ValueMask = [u8; 32];

/// An update to a Verkle trie slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyedUpdate {
    FullSlot {
        key: Key,
        value: Value,
    },
    PartialSlot {
        key: Key,
        value: Value,
        mask: ValueMask,
    },
}

impl PartialOrd for KeyedUpdate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for KeyedUpdate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key().cmp(other.key())
    }
}

impl KeyedUpdate {
    /// Returns the key associated with the update.
    pub fn key(&self) -> &Key {
        match self {
            Self::FullSlot { key, .. } | Self::PartialSlot { key, .. } => key,
        }
    }

    /// Applies the update to the given value.
    pub fn apply_to_value(&self, orig_value: &mut [u8; 32]) {
        match self {
            KeyedUpdate::FullSlot { value, .. } => {
                *orig_value = *value;
            }
            KeyedUpdate::PartialSlot { value, mask, .. } => {
                for i in 0..32 {
                    orig_value[i] = (orig_value[i] & !mask[i]) | (value[i] & mask[i]);
                }
            }
        }
    }
}

/// A collection of keyed updates with the invariants that they are sorted by key and non-empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyedUpdateBatch<'a>(Cow<'a, [KeyedUpdate]>);

impl<'a> KeyedUpdateBatch<'a> {
    /// Creates [`KeyedUpdateBatch`] from key-value pairs, treating each as a full slot update.
    /// Used for testing.
    #[cfg(test)]
    pub fn from_key_value_pairs(updates: &[(Key, Value)]) -> Self {
        let mut keyed_updates = Vec::with_capacity(updates.len());
        for (key, value) in updates {
            keyed_updates.push(KeyedUpdate::FullSlot {
                key: *key,
                value: *value,
            });
        }
        keyed_updates.sort();
        KeyedUpdateBatch(Cow::Owned(keyed_updates))
    }

    /// Returns the key of the first update.
    pub fn first_key(&self) -> &Key {
        // self is never empty (invariant)
        self.0[0].key()
    }

    /// Returns an iterator that splits the updates into groups sharing the same byte at the given
    /// depth and yields them as [`KeyedUpdateBatch`].
    /// This is used to group updates for insertion into child nodes. It is therefore expected that
    /// all bytes at smaller (shallower) depths are equal. However, this is not enforced.
    ///
    /// For performance reasons, this method should only be called on borrowed data. This can be
    /// ensured by calling `.borrowed()` first.
    pub fn split(self, depth: u8) -> impl Iterator<Item = KeyedUpdateBatch<'a>> {
        SplitIter {
            updates: self,
            depth,
            start: 0,
        }
    }

    /// Checks if all updates share the given stem (first 31 bytes of the key).
    pub fn all_stems_match(&self, stem: &[u8; 31]) -> bool {
        self.0.iter().all(|update| &update.key()[..31] == stem)
    }

    /// Returns a reference to the inner `KeyedUpdateBatch`.
    pub fn borrowed(&self) -> KeyedUpdateBatch<'_> {
        match &self.0 {
            Cow::Borrowed(slice) => KeyedUpdateBatch(Cow::Borrowed(slice)),
            Cow::Owned(vec) => KeyedUpdateBatch(Cow::Borrowed(vec)),
        }
    }
}

impl Deref for KeyedUpdateBatch<'_> {
    type Target = [KeyedUpdate];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// The error type returned when trying to convert an empty `Update` into `KeyedUpdateBatch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyUpdate;

impl KeyedUpdateBatch<'static> {
    pub fn try_from_with_embedding(
        update: Update<'_>,
        embedding: &VerkleTrieEmbedding,
    ) -> Result<Self, EmptyUpdate> {
        let span = tracy_client::span!("try_from_with_embedding");
        let len = update.created_accounts.len()
            + update.balances.len()
            + update.nonces.len()
            + update
                .codes
                .iter()
                .map(|c| 2 + c.code.len().div_ceil(31))
                .sum::<usize>()
            + update.slots.len();
        span.emit_value(len as u64);
        let misses_before = embedding.cache_misses();
        if update.created_accounts.is_empty()
            && update.balances.is_empty()
            && update.nonces.is_empty()
            && update.codes.is_empty()
            && update.slots.is_empty()
        {
            return Err(EmptyUpdate);
        }
        let mut updates = Vec::with_capacity(
            // in practice created_accounts also have other updates, so we don't count them here
            update.balances.len()
                + update.nonces.len()
                + update.codes.len() * 2 // lower bound: code length and code hash
                + update.slots.len(),
        );
        for addr in update.created_accounts {
            // This is just to set the used bit.
            // Because the mask is all zeros, we don't overwrite any data, so we also don't have to
            // account for nonce, balance or code length updates here.
            updates.push(KeyedUpdate::PartialSlot {
                key: embedding.get_basic_data_key(addr),
                value: [0u8; 32],
                mask: mask_for_range(0..0),
            });
            // If we also get a code update for this account, we have to make sure that this does
            // not override the actual code hash. This is checked when processing the code update.
            updates.push(KeyedUpdate::FullSlot {
                key: embedding.get_code_hash_key(addr),
                value: EMPTY_CODE_HASH,
            });
        }
        for BalanceUpdate { addr, balance } in update.balances {
            updates.push(KeyedUpdate::PartialSlot {
                key: embedding.get_basic_data_key(addr),
                value: *balance,
                mask: mask_for_range(16..32),
            });
        }
        for NonceUpdate { addr, nonce } in update.nonces {
            let mut value = [0u8; 32];
            value[8..16].copy_from_slice(nonce);
            updates.push(KeyedUpdate::PartialSlot {
                key: embedding.get_basic_data_key(addr),
                value,
                mask: mask_for_range(8..16),
            });
        }
        for CodeUpdate { addr, code } in update.codes {
            let code_len = code.len() as u32;
            let mut value = [0u8; 32];
            value[4..8].copy_from_slice(&code_len.to_be_bytes());
            updates.push(KeyedUpdate::PartialSlot {
                key: embedding.get_basic_data_key(&addr),
                value,
                mask: mask_for_range(4..8),
            });

            let mut hasher = Keccak256::new();
            hasher.update(code);
            let code_hash = Hash::from(hasher.finalize());
            let key = embedding.get_code_hash_key(&addr);
            let update = KeyedUpdate::FullSlot {
                key,
                value: code_hash,
            };
            // This is needed in case the account was created in this same batch, in which case we
            // already have a FullSlot update for the code hash with value EMPTY_CODE_HASH that we
            // have to override.
            if let Some(u) = updates.iter_mut().find(|u| u.key() == &key) {
                *u = update;
            } else {
                updates.push(update);
            }

            for (i, chunk) in code::split_code(code).into_iter().enumerate() {
                updates.push(KeyedUpdate::FullSlot {
                    key: embedding.get_code_chunk_key(&addr, i as u32),
                    value: chunk,
                });
            }
        }
        for SlotUpdate { addr, key, value } in update.slots {
            updates.push(KeyedUpdate::FullSlot {
                key: embedding.get_storage_key(addr, key),
                value: *value,
            });
        }

        let misses_after = embedding.cache_misses();
        span.emit_value(misses_after - misses_before);

        drop(span);

        if updates.len() > 100_000 {
            updates.par_sort_unstable();
        } else {
            updates.sort_unstable();
        }
        Ok(KeyedUpdateBatch(Cow::Owned(updates)))
    }
}

/// An iterator that splits `KeyedUpdateBatch` into groups sharing the same key byte at a given
/// depth. These groups are yielded in order as `KeyedUpdateBatch`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SplitIter<'a> {
    updates: KeyedUpdateBatch<'a>,
    depth: u8,
    start: usize,
}

impl<'a> Iterator for SplitIter<'a> {
    type Item = KeyedUpdateBatch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.updates.len() {
            return None;
        }
        let start_pos = self.start;
        let current_key = self.updates[start_pos].key()[self.depth as usize];
        let mut end = start_pos + 1;
        while end < self.updates.len()
            && self.updates[end].key()[self.depth as usize] == current_key
        {
            end += 1;
        }
        self.start = end;
        // This avoids going though deref which would tie the lifetime of data to the lifetime of
        // the borrow of self.
        let data = match &self.updates.0 {
            Cow::Borrowed(slice) => Cow::Borrowed(&slice[start_pos..end]),
            Cow::Owned(vec) => Cow::Owned(vec[start_pos..end].to_vec()),
        };
        Some(KeyedUpdateBatch(data))
    }
}

/// Creates [`Value`] to be used as a mask with all bits set to 1 for the bytes in the given range.
fn mask_for_range(range: Range<usize>) -> Value {
    let mut mask = [0u8; 32];
    mask[range].fill(0xff);
    mask
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
    use zerocopy::transmute;

    use super::*;

    #[test]
    fn ord_compares_keys() {
        let key1 = [0u8; 32];
        let key2 = [1u8; 32];

        let update_pairs = [
            (
                KeyedUpdate::FullSlot {
                    key: [0u8; 32],
                    value: [0u8; 32],
                },
                KeyedUpdate::FullSlot {
                    key: [0u8; 32],
                    value: [0u8; 32],
                },
            ),
            (
                KeyedUpdate::FullSlot {
                    key: [0u8; 32],
                    value: [0u8; 32],
                },
                KeyedUpdate::PartialSlot {
                    key: [0u8; 32],
                    value: [0u8; 32],
                    mask: [0u8; 32],
                },
            ),
            (
                KeyedUpdate::PartialSlot {
                    key: [0u8; 32],
                    value: [0u8; 32],
                    mask: [0u8; 32],
                },
                KeyedUpdate::FullSlot {
                    key: [0u8; 32],
                    value: [0u8; 32],
                },
            ),
            (
                KeyedUpdate::PartialSlot {
                    key: [0u8; 32],
                    value: [0u8; 32],
                    mask: [0u8; 32],
                },
                KeyedUpdate::PartialSlot {
                    key: [0u8; 32],
                    value: [0u8; 32],
                    mask: [0u8; 32],
                },
            ),
        ];

        for (mut update1, mut update2) in update_pairs {
            set_key(&mut update1, key1);
            set_key(&mut update2, key2);
            assert_eq!(key1.cmp(&key2), update1.cmp(&update2));

            set_key(&mut update1, key1);
            set_key(&mut update2, key1);
            assert_eq!(key1.cmp(&key1), update1.cmp(&update2));

            set_key(&mut update1, key2);
            set_key(&mut update2, key1);
            assert_eq!(key2.cmp(&key1), update1.cmp(&update2));
        }
    }

    #[test]
    fn key_returns_correct_key() {
        let key = [42u8; 32];

        let update = KeyedUpdate::FullSlot {
            key,
            value: [0u8; 32],
        };
        assert_eq!(update.key(), &key);

        let update = KeyedUpdate::PartialSlot {
            key,
            value: [0u8; 32],
            mask: [0u8; 32],
        };

        assert_eq!(update.key(), &key);
    }

    #[test]
    fn apply_value_modifies_value_by_overwriting_it_with_full_slot_or_applying_mask_with_partial_slot()
     {
        let mut original_value = [0; 32];

        let full_slot_update = KeyedUpdate::FullSlot {
            key: [0; 32],
            value: [1; 32],
        };
        full_slot_update.apply_to_value(&mut original_value);
        assert_eq!(original_value, [1; 32]);

        let mut original_value = [2; 32];
        let partial_slot_update = KeyedUpdate::PartialSlot {
            key: [0u8; 32],
            value: [3; 32],
            mask: mask_for_range(8..16),
        };
        partial_slot_update.apply_to_value(&mut original_value);
        let expected: [u8; 32] = transmute!([[2u8; 8], [3; 8], [2; 8], [2; 8]]);
        assert_eq!(original_value, expected);
    }

    #[test]
    fn from_key_value_pairs_maps_pairs_to_full_slots_and_sorts_them() {
        let updates = [([3; 32], [4; 32]), ([1; 32], [2; 32])];

        let keyed_updates = KeyedUpdateBatch::from_key_value_pairs(&updates);

        assert_eq!(
            keyed_updates,
            KeyedUpdateBatch(Cow::Borrowed(&[
                KeyedUpdate::FullSlot {
                    key: [1; 32],
                    value: [2; 32],
                },
                KeyedUpdate::FullSlot {
                    key: [3; 32],
                    value: [4; 32],
                }
            ]))
        );
    }

    #[test]
    fn first_key_returns_key_of_first_update() {
        let updates = KeyedUpdateBatch(Cow::Borrowed(&[
            KeyedUpdate::FullSlot {
                key: [3; 32],
                value: [4; 32],
            },
            KeyedUpdate::FullSlot {
                key: [1; 32],
                value: [2; 32],
            },
        ]));

        assert_eq!(updates.first_key(), &[3; 32]);
    }

    #[test]
    fn split_returns_iterator_yielding_keyed_updates_for_each_distinct_byte_at_depth() {
        let updates = KeyedUpdateBatch(Cow::Borrowed(&[
            KeyedUpdate::FullSlot {
                key: [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 11, 111,
                ],
                value: [1; 32],
            },
            KeyedUpdate::FullSlot {
                key: [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 11, 112,
                ],
                value: [2; 32],
            },
            KeyedUpdate::FullSlot {
                key: [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 12, 113,
                ],
                value: [3; 32],
            },
            KeyedUpdate::FullSlot {
                key: [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 12, 114,
                ],
                value: [4; 32],
            },
        ]));

        let iter: Vec<_> = updates.clone().split(32 - 1 - 2).collect();
        assert_eq!(iter.len(), 1);
        assert_eq!(
            iter[0].iter().map(|u| u.key()[31]).collect::<Vec<_>>(),
            [111, 112, 113, 114]
        );

        let iter: Vec<_> = updates.clone().split(32 - 1 - 1).collect();
        assert_eq!(iter.len(), 2);
        assert_eq!(
            iter[0].iter().map(|u| u.key()[31]).collect::<Vec<_>>(),
            [111, 112]
        );
        assert_eq!(
            iter[1].iter().map(|u| u.key()[31]).collect::<Vec<_>>(),
            [113, 114]
        );

        let iter: Vec<_> = updates.split(32 - 1).collect();
        assert_eq!(iter.len(), 4);
        assert_eq!(
            iter[0].iter().map(|u| u.key()[31]).collect::<Vec<_>>(),
            [111]
        );
        assert_eq!(
            iter[1].iter().map(|u| u.key()[31]).collect::<Vec<_>>(),
            [112]
        );
        assert_eq!(
            iter[2].iter().map(|u| u.key()[31]).collect::<Vec<_>>(),
            [113]
        );
        assert_eq!(
            iter[3].iter().map(|u| u.key()[31]).collect::<Vec<_>>(),
            [114]
        );
    }

    #[test]
    fn all_stems_eq_checks_if_all_updates_have_specified_stem() {
        let updates = KeyedUpdateBatch(Cow::Borrowed(&[
            KeyedUpdate::FullSlot {
                key: [1; 32],
                value: [2; 32],
            },
            KeyedUpdate::FullSlot {
                key: [3; 32],
                value: [4; 32],
            },
        ]));
        assert!(!updates.all_stems_match(&[1; 31]));
        let updates = KeyedUpdateBatch(Cow::Borrowed(&[
            KeyedUpdate::FullSlot {
                key: [3; 32],
                value: [2; 32],
            },
            KeyedUpdate::FullSlot {
                key: [3; 32],
                value: [4; 32],
            },
        ]));
        assert!(!updates.all_stems_match(&[1; 31]));
        assert!(updates.all_stems_match(&[3; 31]));
    }

    #[test]
    fn try_from_update_converts_balance_and_nonce_and_code_and_slot_updates_and_sorts_updates() {
        let embedding = VerkleTrieEmbedding::new();

        let update = Update {
            created_accounts: &[[1; 20], [2; 20]],
            deleted_accounts: &[[3; 20], [4; 20]], // These will be ignored
            balances: &[
                BalanceUpdate {
                    addr: [5; 20],
                    balance: [6; 32],
                },
                BalanceUpdate {
                    addr: [7; 20],
                    balance: [8; 32],
                },
            ],
            nonces: &[
                NonceUpdate {
                    addr: [9; 20],
                    nonce: [10; 8],
                },
                NonceUpdate {
                    addr: [11; 20],
                    nonce: [12; 8],
                },
            ],
            codes: vec![
                CodeUpdate {
                    addr: [1; 20], // should overwrite the created account code hash
                    code: &[14; 20],
                },
                CodeUpdate {
                    addr: [15; 20],
                    code: &[16; 80],
                },
            ],
            slots: &[
                SlotUpdate {
                    addr: [17; 20],
                    key: [18; 32],
                    value: [19; 32],
                },
                SlotUpdate {
                    addr: [20; 20],
                    key: [21; 32],
                    value: [22; 32],
                },
            ],
        };

        let keyed_updates =
            KeyedUpdateBatch::try_from_with_embedding(update.clone(), &embedding).unwrap();

        // Verify total number of updates
        assert_eq!(
            keyed_updates.len(),
            update.created_accounts.len() * 2
                + update.balances.len()
                + update.nonces.len()
                + update.codes.len() * 2
                - 1 // the overwrite of the created account code hash
                + update
                    .codes
                    .iter()
                    .map(|c| code::split_code(c.code).len())
                    .sum::<usize>()
                + update.slots.len()
        );

        // Verify created accounts
        for addr in update.created_accounts {
            let mut found = false;
            for keyed_update in &*keyed_updates {
                if let KeyedUpdate::PartialSlot { key, value, mask } = keyed_update
                    && *key == embedding.get_basic_data_key(addr)
                    && *value == [0u8; 32]
                    && *mask == mask_for_range(0..0)
                {
                    assert!(!found);
                    found = true;
                }
            }
            assert!(found, "No matching PartialSlot for created account");
        }

        // Verify balances
        for balance_update in update.balances {
            let mut found = false;
            for keyed_update in &*keyed_updates {
                if let KeyedUpdate::PartialSlot { key, value, mask } = keyed_update
                    && *key == embedding.get_basic_data_key(&balance_update.addr)
                    && *value == balance_update.balance
                    && *mask == mask_for_range(16..32)
                {
                    assert!(!found);
                    found = true;
                }
            }
            assert!(found, "No matching PartialSlot found for balance update");
        }

        // Verify nonces
        for nonce_update in update.nonces {
            let mut found = false;
            for keyed_update in &*keyed_updates {
                if let KeyedUpdate::PartialSlot { key, value, mask } = keyed_update {
                    let mut expected_value = [0u8; 32];
                    expected_value[8..16].copy_from_slice(&nonce_update.nonce);
                    if *key == embedding.get_basic_data_key(&nonce_update.addr)
                        && *value == expected_value
                        && *mask == mask_for_range(8..16)
                    {
                        assert!(!found);
                        found = true;
                    }
                }
            }
            assert!(found, "No matching PartialSlot found for nonce update");
        }

        // Verify codes
        for code_update in update.codes {
            let mut found_code_len = false;
            let mut found_code_hash = false;
            let mut found_chunks = vec![false; code::split_code(code_update.code).len()];

            for keyed_update in &*keyed_updates {
                if let KeyedUpdate::PartialSlot { key, value, mask } = keyed_update {
                    let code_len = code_update.code.len() as u32;
                    let mut expected_value = [0u8; 32];
                    expected_value[4..8].copy_from_slice(&code_len.to_be_bytes());
                    if *key == embedding.get_basic_data_key(&code_update.addr)
                        && *value == expected_value
                        && *mask == mask_for_range(4..8)
                    {
                        assert!(!found_code_len);
                        found_code_len = true;
                    }
                } else if let KeyedUpdate::FullSlot { key, value } = keyed_update {
                    if key == &embedding.get_code_hash_key(&code_update.addr) {
                        let mut hasher = Keccak256::new();
                        hasher.update(code_update.code);
                        let expected_hash = Hash::from(hasher.finalize());
                        if *value == expected_hash {
                            assert!(!found_code_hash);
                            found_code_hash = true;
                        }
                    } else {
                        for (i, chunk) in code::split_code(code_update.code).into_iter().enumerate()
                        {
                            if *key == embedding.get_code_chunk_key(&code_update.addr, i as u32)
                                && *value == chunk
                            {
                                assert!(!found_chunks[i]);
                                found_chunks[i] = true;
                            }
                        }
                    }
                }
            }

            assert!(
                found_code_len,
                "No matching PartialSlot found for code length"
            );
            assert!(found_code_hash, "No matching FullSlot found for code hash");
            assert!(
                found_chunks.iter().all(|&found| found),
                "Not all code chunks were found"
            );
        }

        // Verify slots
        for slot_update in update.slots {
            let mut found = false;
            for keyed_update in &*keyed_updates {
                if let KeyedUpdate::FullSlot { key, value } = keyed_update
                    && *key == embedding.get_storage_key(&slot_update.addr, &slot_update.key)
                    && *value == slot_update.value
                {
                    assert!(!found);
                    found = true;
                }
            }
            assert!(found, "No matching FullSlot found for slot update");
        }
    }

    #[test]
    fn mask_for_range_sets_all_bits_for_bytes_in_range() {
        assert_eq!(mask_for_range(0..0), [0; 32]);
        assert_eq!(mask_for_range(0..32), [255; 32]);
        assert_eq!(
            mask_for_range(4..8),
            [
                0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
            ]
        );
    }

    fn set_key(update: &mut KeyedUpdate, new_key: Key) {
        match update {
            KeyedUpdate::FullSlot { key, .. } | KeyedUpdate::PartialSlot { key, .. } => {
                *key = new_key;
            }
        }
    }

    #[test]
    fn bench() {
        let d: [_; 240] = std::array::from_fn(|i| i as u8);

        let embedding = VerkleTrieEmbedding::new();
        embedding.get_basic_data_key(&[0; 20]);

        let embedding = VerkleTrieEmbedding::new();
        let span = tracy_client::span!("seq");
        let start = Instant::now();
        let r1: Vec<_> = d
            .into_iter()
            .map(|d| {
                let mut addr = [0; 20];
                addr[0] = d;
                (
                    std::thread::current().id().as_u64(),
                    embedding.get_basic_data_key(&addr),
                )
            })
            .collect();
        println!("{:?}", start.elapsed());
        drop(span);
        println!("{:?}", r1.into_iter().map(|(k, _)| k).collect::<Vec<_>>());

        let embedding = VerkleTrieEmbedding::new();
        let span = tracy_client::span!("par");
        let start = Instant::now();
        let r1: Vec<_> = d
            .into_par_iter()
            .map(|d| {
                let mut addr = [0; 20];
                addr[0] = d;
                (
                    std::thread::current().id().as_u64(),
                    embedding.get_basic_data_key(&addr),
                )
            })
            .collect();
        println!("{:?}", start.elapsed());
        drop(span);
        println!("{:?}", r1.into_iter().map(|(k, _)| k).collect::<Vec<_>>());

        let embedding = VerkleTrieEmbedding::new();
        let span = tracy_client::span!("par with_min_len 5");
        let start = Instant::now();
        let r1: Vec<_> = d
            .into_par_iter()
            .with_min_len(5)
            .map(|d| {
                let mut addr = [0; 20];
                addr[0] = d;
                (
                    std::thread::current().id().as_u64(),
                    embedding.get_basic_data_key(&addr),
                )
            })
            .collect();
        println!("{:?}", start.elapsed());
        drop(span);
        println!("{:?}", r1.into_iter().map(|(k, _)| k).collect::<Vec<_>>());

        let embedding = VerkleTrieEmbedding::new();
        let span = tracy_client::span!("par with_min_len 15");
        let start = Instant::now();
        let r1: Vec<_> = d
            .into_par_iter()
            .with_min_len(15)
            .map(|d| {
                let mut addr = [0; 20];
                addr[0] = d;
                (
                    std::thread::current().id().as_u64(),
                    embedding.get_basic_data_key(&addr),
                )
            })
            .collect();
        println!("{:?}", start.elapsed());
        drop(span);
        println!("{:?}", r1.into_iter().map(|(k, _)| k).collect::<Vec<_>>());

        let embedding = VerkleTrieEmbedding::new();
        let span = tracy_client::span!("par with_min_len 30");
        let start = Instant::now();
        let r1: Vec<_> = d
            .into_par_iter()
            .with_min_len(30)
            .map(|d| {
                let mut addr = [0; 20];
                addr[0] = d;
                (
                    std::thread::current().id().as_u64(),
                    embedding.get_basic_data_key(&addr),
                )
            })
            .collect();
        println!("{:?}", start.elapsed());
        drop(span);
        println!("{:?}", r1.into_iter().map(|(k, _)| k).collect::<Vec<_>>());

        // let embedding = VerkleTrieEmbedding::new();
        // let start = Instant::now();
        // let r1: Vec<_> = d
        //     .into_par_iter()
        //     .with_min_len(50)
        //     .map(|d| {
        //         let mut addr = [0; 20];
        //         addr[0] = d;
        //         (
        //             std::thread::current().id().as_u64(),
        //             embedding.get_basic_data_key(&addr),
        //         )
        //     })
        //     .chain(d.into_par_iter().with_min_len(50).map(|d| {
        //         let mut addr = [0; 20];
        //         addr[0] = d;
        //         (
        //             std::thread::current().id().as_u64(),
        //             embedding.get_basic_data_key(&addr),
        //         )
        //     }))
        //     .collect();
        // println!("{:?}", start.elapsed());
        // println!("{:?}", r1.into_iter().map(|(k, _)| k).collect::<Vec<_>>());

        // let embedding = VerkleTrieEmbedding::new();
        // let start = Instant::now();
        // let r1: Vec<_> = d
        //     .into_par_iter()
        //     .map(|d| {
        //         let mut addr = [0; 20];
        //         addr[0] = d;
        //         (
        //             std::thread::current().id().as_u64(),
        //             embedding.get_basic_data_key(&addr),
        //         )
        //     })
        //     .chain(d.into_par_iter().map(|d| {
        //         let mut addr = [0; 20];
        //         addr[0] = d;
        //         (
        //             std::thread::current().id().as_u64(),
        //             embedding.get_basic_data_key(&addr),
        //         )
        //     }))
        //     .with_min_len(50)
        //     .collect();
        // println!("{:?}", start.elapsed());
        // println!("{:?}", r1.into_iter().map(|(k, _)| k).collect::<Vec<_>>());

        // let d = [0u8; 100];

        // let start = Instant::now();
        // let r1: Vec<_> = d.into_iter().map(|d| d + 1).collect();
        // println!("{:?}", start.elapsed());
        // println!("{r1:?}");

        // let start = Instant::now();
        // let r2: Vec<_> = d.into_par_iter().map(|d| d + 1).collect();
        // println!("{:?}", start.elapsed());
        // println!("{r2:?}");

        // let d = [0u8; 10];

        // let start = Instant::now();
        // let r1: Vec<_> = d.into_iter().map(|d| d + 1).collect();
        // println!("{:?}", start.elapsed());
        // println!("{r1:?}");

        // let start = Instant::now();
        // let r2: Vec<_> = d.into_par_iter().map(|d| d + 1).collect();
        // println!("{:?}", start.elapsed());
        // println!("{r2:?}");
    }
}
