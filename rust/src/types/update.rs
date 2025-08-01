// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use zerocopy::{FromBytes, Immutable, transmute_ref};

use crate::types::{Address, Key, Nonce, U256, Value};

/// A new balance for an account.
#[derive(Debug, Clone, Default, PartialEq, Eq, FromBytes, Immutable)]
#[repr(C)]
pub struct BalanceUpdate {
    pub addr: Address,
    pub balance: U256,
}

/// A new code for an account.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CodeUpdate<'d> {
    pub addr: Address,
    pub code: &'d [u8],
}

/// A new nonce for an account.
#[derive(Debug, Clone, Default, PartialEq, Eq, FromBytes, Immutable)]
#[repr(C)]
pub struct NonceUpdate {
    pub addr: Address,
    pub nonce: Nonce,
}

/// A new key-value pair for an account in the state trie.
#[derive(Debug, Clone, Default, PartialEq, Eq, FromBytes, Immutable)]
#[repr(C)]
pub struct SlotUpdate {
    pub addr: Address,
    pub key: Key,
    pub value: Value,
}

/// A block update.
/// This update contains all changes to the state of the Carmen database
/// that happened in a single block.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Update<'d> {
    pub deleted_accounts: &'d [Address],
    pub created_accounts: &'d [Address],
    pub balances: &'d [BalanceUpdate],
    pub codes: Vec<CodeUpdate<'d>>,
    pub nonces: &'d [NonceUpdate],
    pub slots: &'d [SlotUpdate],
}

const VERSION_0: u8 = 0;

// go/common/update.go:ToBytes
// cpp/state/update.h:Update::FromBytes
impl<'d> Update<'d> {
    /// Parses an update from its encoded form.
    pub fn from_encoded(mut bytes: &'d [u8]) -> Result<Self, String> {
        if bytes.len() < 1 + 6 * 4 {
            return Err(format!(
                "encoded update has length {}, but minimum length is 1 + 6 * 4 = 25",
                bytes.len()
            ));
        }

        let bytes = &mut bytes;

        let version = read_slice(bytes, 1)?[0];
        if version != VERSION_0 {
            return Err(format!("invalid version number: {version}"));
        }

        let deleted_accounts_len = u32::from_be_bytes(read_array(bytes)?) as usize;
        let created_accounts_len = u32::from_be_bytes(read_array(bytes)?) as usize;
        let balances_len = u32::from_be_bytes(read_array(bytes)?) as usize;
        let codes_len = u32::from_be_bytes(read_array(bytes)?) as usize;
        let nonces_len = u32::from_be_bytes(read_array(bytes)?) as usize;
        let slots_len = u32::from_be_bytes(read_array(bytes)?) as usize;

        let deleted_accounts = transmute_ref!(read_slice_of_arrays::<{ size_of::<Address>() }>(
            bytes,
            deleted_accounts_len
        )?);
        let created_accounts = transmute_ref!(read_slice_of_arrays::<{ size_of::<Address>() }>(
            bytes,
            created_accounts_len
        )?);
        let balances = transmute_ref!(read_slice_of_arrays::<{ size_of::<BalanceUpdate>() }>(
            bytes,
            balances_len
        )?);
        let mut codes = Vec::with_capacity(codes_len);
        for _ in 0..codes_len {
            let addr = read_array(bytes)?;
            let code_len = u16::from_be_bytes(read_array(bytes)?) as usize;
            let code = read_slice(bytes, code_len)?;
            codes.push(CodeUpdate { addr, code });
        }
        let nonces = transmute_ref!(read_slice_of_arrays::<{ size_of::<NonceUpdate>() }>(
            bytes, nonces_len
        )?);
        let slots = transmute_ref!(read_slice_of_arrays::<{ size_of::<SlotUpdate>() }>(
            bytes, slots_len
        )?);

        Ok(Self {
            deleted_accounts,
            created_accounts,
            balances,
            codes,
            nonces,
            slots,
        })
    }
}

fn read_slice<'d>(bytes: &mut &'d [u8], len: usize) -> Result<&'d [u8], String> {
    if bytes.len() < len {
        return Err("not enough bytes to read".into());
    }
    let (data, rest) = bytes.split_at(len);
    *bytes = rest;
    Ok(data)
}

fn read_array<const N: usize>(bytes: &mut &[u8]) -> Result<[u8; N], String> {
    if bytes.len() < N {
        return Err("not enough bytes to read".into());
    }
    let mut data = [0; N];
    data.copy_from_slice(&bytes[..N]);
    *bytes = &bytes[N..];
    Ok(data)
}

fn read_slice_of_arrays<'d, const N: usize>(
    bytes: &mut &'d [u8],
    len: usize,
) -> Result<&'d [[u8; N]], String> {
    let data = read_slice(bytes, len * N)?;
    let data = data as *const [u8] as *const [u8; N];
    // Safety:
    // - data is not null because it was created from a reference
    // - data has no mutable aliases because it was created from a shared reference
    // - data is valid for reads for `len * size_of::<[u8; N]>() == len * N` bytes because it was
    //   created from a byte slice of this length
    // - `[u8; N]` has the same alignment as `u8`
    let slice_of_arrays = unsafe { std::slice::from_raw_parts(data, len) };
    Ok(slice_of_arrays)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_slice_returns_slice_of_requested_length_and_advances_buffer() {
        let mut bytes = [1, 2, 3, 4, 5].as_slice();
        let slice = read_slice(&mut bytes, 3);
        assert_eq!(slice, Ok([1, 2, 3].as_slice()));
        assert_eq!(bytes, [4, 5]);
    }

    #[test]
    fn read_slice_returns_error_if_buffer_shorter_than_requested_length() {
        let mut bytes = [1, 2, 3, 4, 5].as_slice();
        assert_eq!(
            read_slice(&mut bytes, 6),
            Err("not enough bytes to read".into())
        );
    }

    #[test]
    fn read_array_returns_array_of_requested_length_and_advances_buffer() {
        let mut bytes = [1, 2, 3, 4, 5].as_slice();
        let array = read_array(&mut bytes);
        assert_eq!(array, Ok([1, 2, 3]));
        assert_eq!(bytes, [4, 5]);
    }

    #[test]
    fn read_array_returns_error_if_buffer_shorter_than_requested_length() {
        let mut bytes = [1, 2, 3, 4, 5].as_slice();
        assert_eq!(
            read_array::<6>(&mut bytes),
            Err("not enough bytes to read".into())
        );
    }

    #[test]
    fn read_slice_of_arrays_returns_slice_of_requested_length_and_advances_buffer() {
        let mut bytes = [1, 2, 3, 4, 5].as_slice();
        let slice = read_slice_of_arrays(&mut bytes, 2);
        assert_eq!(slice, Ok([[1, 2], [3, 4]].as_slice()));
        assert_eq!(bytes, [5]);
    }

    #[test]
    fn read_slice_of_arrays_returns_error_if_buffer_shorter_than_requested_length() {
        let mut bytes = [1, 2, 3, 4, 5].as_slice();
        assert_eq!(
            read_slice_of_arrays::<2>(&mut bytes, 3),
            Err("not enough bytes to read".into())
        );
    }

    #[test]
    fn update_from_encoded_can_decode_empty_update() {
        let update = Update {
            deleted_accounts: &[],
            created_accounts: &[],
            balances: &[],
            codes: Vec::new(),
            nonces: &[],
            slots: &[],
        };

        let mut encoded_update = Vec::new();
        encoded_update.push(VERSION_0);
        encoded_update.extend_from_slice(&(update.deleted_accounts.len() as u32).to_be_bytes());
        encoded_update.extend_from_slice(&(update.created_accounts.len() as u32).to_be_bytes());
        encoded_update.extend_from_slice(&(update.balances.len() as u32).to_be_bytes());
        encoded_update.extend_from_slice(&(update.codes.len() as u32).to_be_bytes());
        encoded_update.extend_from_slice(&(update.nonces.len() as u32).to_be_bytes());
        encoded_update.extend_from_slice(&(update.slots.len() as u32).to_be_bytes());

        let decoded_update = Update::from_encoded(&encoded_update);
        assert_eq!(decoded_update, Ok(update));
    }

    #[test]
    fn update_from_encoded_can_decode_non_empty_update() {
        let update = Update {
            deleted_accounts: &[[1; 20], [2; 20]],
            created_accounts: &[[3; 20], [4; 20]],
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
            codes: vec![
                CodeUpdate {
                    addr: [9; 20],
                    code: &[10, 11, 12],
                },
                CodeUpdate {
                    addr: [13; 20],
                    code: &[14, 15],
                },
            ],
            nonces: &[
                NonceUpdate {
                    addr: [16; 20],
                    nonce: [17; 8],
                },
                NonceUpdate {
                    addr: [18; 20],
                    nonce: [19; 8],
                },
            ],
            slots: &[
                SlotUpdate {
                    addr: [20; 20],
                    key: [21; 32],
                    value: [22; 32],
                },
                SlotUpdate {
                    addr: [23; 20],
                    key: [24; 32],
                    value: [25; 32],
                },
            ],
        };

        let mut encoded_update = Vec::new();
        encoded_update.push(VERSION_0);
        encoded_update.extend_from_slice(&(update.deleted_accounts.len() as u32).to_be_bytes());
        encoded_update.extend_from_slice(&(update.created_accounts.len() as u32).to_be_bytes());
        encoded_update.extend_from_slice(&(update.balances.len() as u32).to_be_bytes());
        encoded_update.extend_from_slice(&(update.codes.len() as u32).to_be_bytes());
        encoded_update.extend_from_slice(&(update.nonces.len() as u32).to_be_bytes());
        encoded_update.extend_from_slice(&(update.slots.len() as u32).to_be_bytes());

        encoded_update.extend_from_slice(update.deleted_accounts.concat().as_slice());
        encoded_update.extend_from_slice(update.created_accounts.concat().as_slice());
        encoded_update.extend_from_slice(
            update
                .balances
                .iter()
                .flat_map(|b| [b.addr.as_slice(), b.balance.as_slice()].concat())
                .collect::<Vec<_>>()
                .as_slice(),
        );
        encoded_update.extend_from_slice(
            update
                .codes
                .iter()
                .flat_map(|b| {
                    [
                        b.addr.as_slice(),
                        (b.code.len() as u16).to_be_bytes().as_slice(),
                        b.code,
                    ]
                    .concat()
                })
                .collect::<Vec<_>>()
                .as_slice(),
        );
        encoded_update.extend_from_slice(
            update
                .nonces
                .iter()
                .flat_map(|b| [b.addr.as_slice(), b.nonce.as_slice()].concat())
                .collect::<Vec<_>>()
                .as_slice(),
        );
        encoded_update.extend_from_slice(
            update
                .slots
                .iter()
                .flat_map(|b| [b.addr.as_slice(), b.key.as_slice(), b.value.as_slice()].concat())
                .collect::<Vec<_>>()
                .as_slice(),
        );

        let decoded_update = Update::from_encoded(&encoded_update);
        assert_eq!(decoded_update, Ok(update));
    }

    #[test]
    fn update_from_encoded_returns_error_for_invalid_version() {
        let update = Update {
            deleted_accounts: &[],
            created_accounts: &[],
            balances: &[],
            codes: Vec::new(),
            nonces: &[],
            slots: &[],
        };

        let mut encoded_update = Vec::new();
        encoded_update.push(1); // Invalid version
        encoded_update.extend_from_slice(&(update.deleted_accounts.len() as u32).to_be_bytes());
        encoded_update.extend_from_slice(&(update.created_accounts.len() as u32).to_be_bytes());
        encoded_update.extend_from_slice(&(update.balances.len() as u32).to_be_bytes());
        encoded_update.extend_from_slice(&(update.codes.len() as u32).to_be_bytes());
        encoded_update.extend_from_slice(&(update.nonces.len() as u32).to_be_bytes());
        encoded_update.extend_from_slice(&(update.slots.len() as u32).to_be_bytes());

        let decoded_update = Update::from_encoded(&encoded_update);
        assert_eq!(decoded_update, Err("invalid version number: 1".to_owned()));
    }

    /// This test checks for every read operation in `Update::from_encoded` that in the case the
    /// read fails because the buffer is too short, an error is returned.
    #[test]
    fn update_from_encoded_returns_error_when_buffer_too_short() {
        let update = Update {
            deleted_accounts: &[[1; 20], [2; 20]],
            created_accounts: &[[3; 20], [4; 20]],
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
            codes: vec![
                CodeUpdate {
                    addr: [9; 20],
                    code: &[10, 11, 12],
                },
                CodeUpdate {
                    addr: [13; 20],
                    code: &[14, 15],
                },
            ],
            nonces: &[
                NonceUpdate {
                    addr: [16; 20],
                    nonce: [17; 8],
                },
                NonceUpdate {
                    addr: [18; 20],
                    nonce: [19; 8],
                },
            ],
            slots: &[
                SlotUpdate {
                    addr: [20; 20],
                    key: [21; 32],
                    value: [22; 32],
                },
                SlotUpdate {
                    addr: [23; 20],
                    key: [24; 32],
                    value: [25; 32],
                },
            ],
        };

        // This is a list of all writes that mirror all reads in `Update::from_encoded`.
        let writes = [
            (|encoded_update: &mut Vec<u8>, _update| encoded_update.push(VERSION_0))
                as fn(&mut Vec<u8>, &Update<'_>),
            |encoded_update: &mut Vec<u8>, update| {
                encoded_update
                    .extend_from_slice(&(update.deleted_accounts.len() as u32).to_be_bytes());
            },
            |encoded_update: &mut Vec<u8>, update| {
                encoded_update
                    .extend_from_slice(&(update.created_accounts.len() as u32).to_be_bytes());
            },
            |encoded_update: &mut Vec<u8>, update| {
                encoded_update.extend_from_slice(&(update.balances.len() as u32).to_be_bytes());
            },
            |encoded_update: &mut Vec<u8>, update| {
                encoded_update.extend_from_slice(&(update.codes.len() as u32).to_be_bytes());
            },
            |encoded_update: &mut Vec<u8>, update| {
                encoded_update.extend_from_slice(&(update.nonces.len() as u32).to_be_bytes());
            },
            |encoded_update: &mut Vec<u8>, update| {
                encoded_update.extend_from_slice(&(update.slots.len() as u32).to_be_bytes());
            },
            |encoded_update: &mut Vec<u8>, update| {
                encoded_update.extend_from_slice(update.deleted_accounts.concat().as_slice());
            },
            |encoded_update: &mut Vec<u8>, update| {
                encoded_update.extend_from_slice(update.created_accounts.concat().as_slice());
            },
            |encoded_update: &mut Vec<u8>, update| {
                encoded_update.extend_from_slice(
                    update
                        .balances
                        .iter()
                        .flat_map(|b| [b.addr.as_slice(), b.balance.as_slice()].concat())
                        .collect::<Vec<_>>()
                        .as_slice(),
                );
            },
            |encoded_update: &mut Vec<u8>, update| {
                encoded_update.extend_from_slice(
                    update
                        .codes
                        .iter()
                        .flat_map(|b| {
                            [
                                b.addr.as_slice(),
                                (b.code.len() as u16).to_be_bytes().as_slice(),
                                b.code,
                            ]
                            .concat()
                        })
                        .collect::<Vec<_>>()
                        .as_slice(),
                );
            },
            |encoded_update: &mut Vec<u8>, update| {
                encoded_update.extend_from_slice(
                    update
                        .nonces
                        .iter()
                        .flat_map(|b| [b.addr.as_slice(), b.nonce.as_slice()].concat())
                        .collect::<Vec<_>>()
                        .as_slice(),
                );
            },
            |_encoded_update: &mut Vec<u8>, _update| {
                // This closure corresponds to the last read in Update::from_encoded. If we would
                // put a write operation here and also call it, Update::from_encoded will succeed.
                // But this test is only supposed to test the cases where parsing fails so we make
                // sure this closure is never called.
                panic!("this closure should never be called");
            },
        ];

        // To let every read operation in `Update::from_encoded` fail, we execute only 0
        // writes and then run `Update::from_encoded`, then we execute 1 write and run
        // `Update::from_encoded`, and so on, until we have executed all but the last writes.
        // This way the first read operation will fail, then the first will succeed but the second
        // will fail, and so on until all succeed except for the last one.
        for write_count in 0..writes.len() {
            let mut encoded_update = Vec::new();

            for write in writes.iter().take(write_count) {
                write(&mut encoded_update, &update);
            }

            let decoded_update = Update::from_encoded(&encoded_update);
            assert!(decoded_update.is_err());
        }
    }
}
