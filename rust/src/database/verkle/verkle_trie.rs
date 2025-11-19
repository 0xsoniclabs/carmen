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
    database::verkle::{crypto::Commitment, keyed_update::KeyedUpdateBatch},
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
#[cfg_attr(test, mockall::automock, allow(clippy::disallowed_types))]
pub trait VerkleTrie: Send + Sync {
    /// Retrieves the value associated with the given key.
    /// Returns the default [`Value`] if the key does not exist.
    fn lookup(&self, key: &Key) -> BTResult<Value, Error>;

    /// Stores the values for the keys of the updates.
    fn store<'u>(&self, updates: KeyedUpdateBatch<'u>) -> BTResult<(), Error>;

    /// Computes and returns the current root commitment of the trie.
    /// The commitment can be used as cryptographic proof of the trie's state,
    /// i.e., all contained key-value pairs.
    fn commit(&self) -> BTResult<Commitment, Error>;

    /// Notifies the trie that all updates for the given block height have been applied.
    fn after_update(&self, block_height: u64) -> BTResult<(), Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        database::verkle::{
            CrateCryptoInMemoryVerkleTrie, KeyedUpdateBatch, ManagedVerkleTrie,
            SimpleInMemoryVerkleTrie,
            test_utils::{make_key, make_leaf_key, make_value},
            variants::managed::{VerkleNode, VerkleNodeId},
        },
        error::BTError,
        node_manager::in_memory_node_manager::InMemoryNodeManager,
        sync::Arc,
    };

    #[rstest_reuse::template]
    #[rstest::rstest]
    #[case::simple_in_memory(Box::new(SimpleInMemoryVerkleTrie::new()) as Box<dyn VerkleTrie>)]
    #[case::crate_crypto_in_memory(Box::new(CrateCryptoInMemoryVerkleTrie::new()) as Box<dyn VerkleTrie>)]
    #[case::managed(Box::new(ManagedVerkleTrie::<InMemoryNodeManager::<VerkleNodeId, VerkleNode>>::try_new(Arc::new(InMemoryNodeManager::new(100))).unwrap()) as Box<dyn VerkleTrie>)]
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

        trie.store(KeyedUpdateBatch::from_key_value_pairs(&[(
            make_key(&[1]),
            make_value(1),
        )]))
        .unwrap();

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

        trie.store(KeyedUpdateBatch::from_key_value_pairs(&[(
            make_key(&[2]),
            make_value(2),
        )]))
        .unwrap();

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

        trie.store(KeyedUpdateBatch::from_key_value_pairs(&[(
            make_leaf_key(&[0], 1),
            make_value(3),
        )]))
        .unwrap();

        assert_eq!(trie.lookup(&make_key(&[1])).unwrap(), make_value(1));
        assert_eq!(trie.lookup(&make_key(&[2])).unwrap(), make_value(2));
        assert_eq!(trie.lookup(&make_leaf_key(&[0], 1)).unwrap(), make_value(3));
        assert_eq!(
            trie.lookup(&make_leaf_key(&[0], 2)).unwrap(),
            Value::default()
        );

        trie.store(KeyedUpdateBatch::from_key_value_pairs(&[(
            make_leaf_key(&[0], 2),
            make_value(4),
        )]))
        .unwrap();

        assert_eq!(trie.lookup(&make_key(&[1])).unwrap(), make_value(1));
        assert_eq!(trie.lookup(&make_key(&[2])).unwrap(), make_value(2));
        assert_eq!(trie.lookup(&make_leaf_key(&[0], 1)).unwrap(), make_value(3));
        assert_eq!(trie.lookup(&make_leaf_key(&[0], 2)).unwrap(), make_value(4));
    }

    #[rstest_reuse::apply(all_trie_impls)]
    fn values_can_be_updated(#[case] trie: Box<dyn VerkleTrie>) {
        let key = make_key(&[1]);
        assert_eq!(trie.lookup(&key).unwrap(), Value::default());
        trie.store(KeyedUpdateBatch::from_key_value_pairs(&[(
            key,
            make_value(1),
        )]))
        .unwrap();
        assert_eq!(trie.lookup(&key).unwrap(), make_value(1));
        trie.store(KeyedUpdateBatch::from_key_value_pairs(&[(
            key,
            make_value(2),
        )]))
        .unwrap();
        assert_eq!(trie.lookup(&key).unwrap(), make_value(2));
        trie.store(KeyedUpdateBatch::from_key_value_pairs(&[(
            key,
            make_value(3),
        )]))
        .unwrap();
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
            trie.store(KeyedUpdateBatch::from_key_value_pairs(&[(
                to_key(i),
                make_value(i as u64),
            )]))
            .unwrap();
        }
    }

    // Regression test for managed trie implementation:
    // Reparenting a newly created leaf node did not properly mark the leaf as dirty in the new
    // parent, thereby excluding it from the initial commitment computation. This was not an
    // issue for the simple in-memory trie, where nodes are queried for their commitment
    // dirty status directly.
    #[rstest_reuse::apply(all_trie_impls)]
    fn reparented_leaf_is_correctly_included_in_inner_node_commitment(
        #[case] trie: Box<dyn VerkleTrie>,
    ) {
        // Insert a single value. This will create an inner -> leaf.
        trie.store(KeyedUpdateBatch::from_key_value_pairs(&[(
            make_leaf_key(&[2], 0),
            make_value(1),
        )]))
        .unwrap();
        // Trigger insertion of another inner node by inserting a key that shares prefix
        // with existing leaf.
        trie.store(KeyedUpdateBatch::from_key_value_pairs(&[(
            make_leaf_key(&[2, 3], 0),
            make_value(1),
        )]))
        .unwrap();
        let received = trie.commit().unwrap();

        // Generated with Go reference implementation
        let expected = "0x551138e437ee637cbe37a28e35c747e0fed4683129f516ec1937a079d3bbba98";
        assert_eq!(const_hex::encode_prefixed(received.compress()), expected);
    }

    // Regression test for both the simple in-memory and managed trie implementation:
    // Very similar to the previous test, except that between the creation of the leaf
    // and the reparenting, the commitment is computed once. This triggered two issues:
    // - The leaf's commitment was not dirty and thus was not included in the new parent's initial
    //   commitment computation.
    // - The grandparent's (root node) point-wise commitment update did not properly compute the
    //   delta between the old and new child commitment, because according to the new inner node,
    //   which replaced the leaf in the root's respective slot, the previous commitment was zero.
    #[rstest_reuse::apply(all_trie_impls)]
    fn incrementally_computed_commitment_over_reparented_leaf_is_correct(
        #[case] trie: Box<dyn VerkleTrie>,
    ) {
        // Insert a single value. This will create an inner -> leaf.
        trie.store(KeyedUpdateBatch::from_key_value_pairs(&[(
            make_leaf_key(&[2], 0),
            make_value(1),
        )]))
        .unwrap();
        let received = trie.commit().unwrap();

        // Generated with Go reference implementation
        let expected = "0x3f9f58afff402c493d568a2b6f373bc462930646028c62acca99a2ee409d13d5";
        assert_eq!(const_hex::encode_prefixed(received.compress()), expected);

        // Trigger insertion of another inner node by inserting a key that shares prefix
        // with existing leaf.
        trie.store(KeyedUpdateBatch::from_key_value_pairs(&[(
            make_leaf_key(&[2, 3], 0),
            make_value(1),
        )]))
        .unwrap();
        let received = trie.commit().unwrap();

        // Generated with Go reference implementation
        let expected = "0x551138e437ee637cbe37a28e35c747e0fed4683129f516ec1937a079d3bbba98";
        assert_eq!(const_hex::encode_prefixed(received.compress()), expected);
    }
}
