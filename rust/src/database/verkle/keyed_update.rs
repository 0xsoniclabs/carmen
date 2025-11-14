// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{cmp::Ordering, ops::Range};

use sha3::{Digest, Keccak256};
use verkle_trie::Key;

use crate::{
    database::verkle::{
        embedding::{
            code, get_basic_data_key, get_code_chunk_key, get_code_hash_key, get_storage_key,
        },
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
    fn key(&self) -> &Key {
        match self {
            Self::FullSlot { key, .. } | Self::PartialSlot { key, .. } => key,
        }
    }
}

impl From<Update<'_>> for Vec<KeyedUpdate> {
    fn from(update: Update<'_>) -> Self {
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
                key: get_basic_data_key(addr),
                value: [0u8; 32],
                mask: mask_for_range(0..0),
            });
            // If we also get a code update for this account, we have to make sure that this does
            // not override the actual code hash. This is checked when processing the code update.
            updates.push(KeyedUpdate::FullSlot {
                key: get_code_hash_key(addr),
                value: EMPTY_CODE_HASH,
            });
        }
        for BalanceUpdate { addr, balance } in update.balances {
            updates.push(KeyedUpdate::PartialSlot {
                key: get_basic_data_key(addr),
                value: *balance,
                mask: mask_for_range(16..32),
            });
        }
        for NonceUpdate { addr, nonce } in update.nonces {
            let mut value = [0u8; 32];
            value[8..16].copy_from_slice(nonce);
            updates.push(KeyedUpdate::PartialSlot {
                key: get_basic_data_key(addr),
                value,
                mask: mask_for_range(8..16),
            });
        }
        for CodeUpdate { addr, code } in update.codes {
            let code_len = code.len() as u32;
            let mut value = [0u8; 32];
            value[4..8].copy_from_slice(&code_len.to_be_bytes());
            updates.push(KeyedUpdate::PartialSlot {
                key: get_basic_data_key(&addr),
                value,
                mask: mask_for_range(4..8),
            });

            let mut hasher = Keccak256::new();
            hasher.update(code);
            let code_hash = Hash::from(hasher.finalize());
            let key = get_code_hash_key(&addr);
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
                    key: get_code_chunk_key(&addr, i as u32),
                    value: chunk,
                });
            }
        }
        for SlotUpdate { addr, key, value } in update.slots {
            updates.push(KeyedUpdate::FullSlot {
                key: get_storage_key(addr, key),
                value: *value,
            });
        }
        updates
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
    fn from_update_converts_balance_and_nonce_and_code_and_slot_updates() {
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

        let keyed_updates: Vec<KeyedUpdate> = update.clone().into();

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
            for keyed_update in &keyed_updates {
                if let KeyedUpdate::PartialSlot { key, value, mask } = keyed_update
                    && *key == get_basic_data_key(addr)
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
            for keyed_update in &keyed_updates {
                if let KeyedUpdate::PartialSlot { key, value, mask } = keyed_update
                    && *key == get_basic_data_key(&balance_update.addr)
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
            for keyed_update in &keyed_updates {
                if let KeyedUpdate::PartialSlot { key, value, mask } = keyed_update {
                    let mut expected_value = [0u8; 32];
                    expected_value[8..16].copy_from_slice(&nonce_update.nonce);
                    if *key == get_basic_data_key(&nonce_update.addr)
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

            for keyed_update in &keyed_updates {
                if let KeyedUpdate::PartialSlot { key, value, mask } = keyed_update {
                    let code_len = code_update.code.len() as u32;
                    let mut expected_value = [0u8; 32];
                    expected_value[4..8].copy_from_slice(&code_len.to_be_bytes());
                    if *key == get_basic_data_key(&code_update.addr)
                        && *value == expected_value
                        && *mask == mask_for_range(4..8)
                    {
                        assert!(!found_code_len);
                        found_code_len = true;
                    }
                } else if let KeyedUpdate::FullSlot { key, value } = keyed_update {
                    if key == &get_code_hash_key(&code_update.addr) {
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
                            if *key == get_code_chunk_key(&code_update.addr, i as u32)
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
            for keyed_update in &keyed_updates {
                if let KeyedUpdate::FullSlot { key, value } = keyed_update
                    && *key == get_storage_key(&slot_update.addr, &slot_update.key)
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
}
