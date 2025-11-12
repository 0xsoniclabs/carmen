// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use crate::{
    database::verkle::crypto::Commitment,
    error::{BTResult, Error},
    types::{Key, Value},
};

/// An implementation of the Verkle trie authenticated storage, as specified by Ethereum.
///
/// Verkle tries provide basic [`Key`]-[`Value`] storage of fixed-length keys and values,
/// and the ability to compute a cryptographic commitment of the trie's state using
/// the Pedersen commitment scheme.
///
/// The trait prescribes interior mutability through shared references,
/// allowing for safe concurrent access.
pub trait VerkleTrie: Send + Sync {
    /// Retrieves the value associated with the given key.
    /// Returns the default [`Value`] if the key does not exist.
    fn lookup(&self, key: &Key) -> BTResult<Value, Error>;

    /// Stores the value for the given key.
    fn store(&self, key: &Key, value: &Value) -> BTResult<(), Error>;

    /// Computes and returns the current root commitment of the trie.
    /// The commitment can be used as cryptographic proof of the trie's state,
    /// i.e., all contained key-value pairs.
    fn commit(&self) -> BTResult<Commitment, Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        database::verkle::{
            CrateCryptoInMemoryVerkleTrie, SimpleInMemoryVerkleTrie,
            test_utils::{make_key, make_leaf_key, make_value},
        },
        error::BTError,
    };

    #[rstest_reuse::template]
    #[rstest::rstest]
    #[case::simple_in_memory(Box::new(SimpleInMemoryVerkleTrie::new()) as Box<dyn VerkleTrie>)]
    #[case::crate_crypto_in_memory(Box::new(CrateCryptoInMemoryVerkleTrie::new()) as Box<dyn VerkleTrie>)]
    fn all_trie_impls(#[case] trie: Box<dyn VerkleTrie>) {}

    #[rstest_reuse::apply(all_trie_impls)]
    fn newly_created_trie_is_empty(#[case] trie: Box<dyn VerkleTrie>) {
        assert_eq!(trie.lookup(&make_key(&[1])).unwrap(), Value::default());
        assert_eq!(trie.lookup(&make_key(&[2])).unwrap(), Value::default());
        assert_eq!(trie.lookup(&make_key(&[3])).unwrap(), Value::default());
    }

    #[rstest_reuse::apply(all_trie_impls)]
    fn commitment_of_empty_trie_is_default_commitment(#[case] trie: Box<dyn VerkleTrie>) {
        assert_eq!(
            trie.commit().map_err(BTError::into_inner),
            Ok(Commitment::default())
        );
    }

    #[rstest_reuse::apply(all_trie_impls)]
    fn values_can_be_stored_and_looked_up(#[case] trie: Box<dyn VerkleTrie>) {
        assert_eq!(trie.lookup(&make_key(&[1])).unwrap(), Value::default());
        assert_eq!(trie.lookup(&make_key(&[2])).unwrap(), Value::default());
        assert_eq!(
            trie.lookup(&make_leaf_key(&[0], 1)).unwrap(),
            Value::default()
        );
        assert_eq!(
            trie.lookup(&make_leaf_key(&[0], 2)).unwrap(),
            Value::default()
        );

        trie.store(&make_key(&[1]), &make_value(1)).unwrap();

        assert_eq!(trie.lookup(&make_key(&[1])).unwrap(), make_value(1));
        assert_eq!(trie.lookup(&make_key(&[2])).unwrap(), Value::default());
        assert_eq!(
            trie.lookup(&make_leaf_key(&[0], 1)).unwrap(),
            Value::default()
        );
        assert_eq!(
            trie.lookup(&make_leaf_key(&[0], 2)).unwrap(),
            Value::default()
        );

        trie.store(&make_key(&[2]), &make_value(2)).unwrap();

        assert_eq!(trie.lookup(&make_key(&[1])).unwrap(), make_value(1));
        assert_eq!(trie.lookup(&make_key(&[2])).unwrap(), make_value(2));
        assert_eq!(
            trie.lookup(&make_leaf_key(&[0], 1)).unwrap(),
            Value::default()
        );
        assert_eq!(
            trie.lookup(&make_leaf_key(&[0], 2)).unwrap(),
            Value::default()
        );

        trie.store(&make_leaf_key(&[0], 1), &make_value(3)).unwrap();

        assert_eq!(trie.lookup(&make_key(&[1])).unwrap(), make_value(1));
        assert_eq!(trie.lookup(&make_key(&[2])).unwrap(), make_value(2));
        assert_eq!(trie.lookup(&make_leaf_key(&[0], 1)).unwrap(), make_value(3));
        assert_eq!(
            trie.lookup(&make_leaf_key(&[0], 2)).unwrap(),
            Value::default()
        );

        trie.store(&make_leaf_key(&[0], 2), &make_value(4)).unwrap();

        assert_eq!(trie.lookup(&make_key(&[1])).unwrap(), make_value(1));
        assert_eq!(trie.lookup(&make_key(&[2])).unwrap(), make_value(2));
        assert_eq!(trie.lookup(&make_leaf_key(&[0], 1)).unwrap(), make_value(3));
        assert_eq!(trie.lookup(&make_leaf_key(&[0], 2)).unwrap(), make_value(4));
    }

    #[rstest_reuse::apply(all_trie_impls)]
    fn values_can_be_updated(#[case] trie: Box<dyn VerkleTrie>) {
        let key = make_key(&[1]);
        assert_eq!(trie.lookup(&key).unwrap(), Value::default());
        trie.store(&key, &make_value(1)).unwrap();
        assert_eq!(trie.lookup(&key).unwrap(), make_value(1));
        trie.store(&key, &make_value(2)).unwrap();
        assert_eq!(trie.lookup(&key).unwrap(), make_value(2));
        trie.store(&key, &make_value(3)).unwrap();
        assert_eq!(trie.lookup(&key).unwrap(), make_value(3));
    }

    #[rstest_reuse::apply(all_trie_impls)]
    fn many_values_can_be_stored_and_looked_up(#[case] trie: Box<dyn VerkleTrie>) {
        const N: u32 = 1000;

        let to_key = |i: u32| {
            make_leaf_key(
                &[(i >> 8 & 0x0F) as u8, (i >> 4 & 0x0F) as u8],
                (i & 0x0F) as u8,
            )
        };

        for i in 0..N {
            for j in 0..N {
                let want = if j < i {
                    make_value(j as u64)
                } else {
                    Value::default()
                };
                let got = trie.lookup(&to_key(j)).unwrap();
                assert_eq!(got, want, "mismatch for key: {:?}", to_key(j));
            }
            trie.store(&to_key(i), &make_value(i as u64)).unwrap();
        }
    }
}
