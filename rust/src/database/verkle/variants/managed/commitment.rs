// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::collections::HashMap;

use zerocopy::{FromBytes, Immutable, IntoBytes, Unaligned};

use crate::{
    database::{
        managed_trie::{ManagedTrieNode, TrieCommitment, TrieUpdateLog},
        verkle::{
            compute_commitment::compute_leaf_node_commitment,
            crypto::{Commitment, Scalar},
            variants::managed::{VerkleNode, VerkleNodeId},
        },
    },
    error::{BTResult, Error},
    node_manager::NodeManager,
    types::{HasEmptyId, Value},
};

/// The commitment of a managed verkle trie node, together with metadata required to recompute
/// it after the node has been modified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
pub enum VerkleCommitment {
    Inner(VerkleInnerCommitment),
    Leaf(VerkleLeafCommitment),
}

impl VerkleCommitment {
    /// Returns the stored commitment.
    pub fn commitment(&self) -> Commitment {
        match self {
            VerkleCommitment::Inner(inner) => inner.commitment(),
            VerkleCommitment::Leaf(leaf) => leaf.commitment(),
        }
    }

    /// Returns true if the commitment is up-to-date and does not need to be recomputed.
    pub fn is_clean(&self) -> bool {
        match self {
            VerkleCommitment::Inner(inner) => inner.is_clean(),
            VerkleCommitment::Leaf(leaf) => leaf.is_clean(),
        }
    }

    /// Returns true if the value or child at the given index has been changed since the last
    /// commitment computation.
    pub fn index_changed(&self, index: usize) -> bool {
        match self {
            VerkleCommitment::Inner(inner) => inner.index_changed(index),
            VerkleCommitment::Leaf(leaf) => leaf.index_changed(index),
        }
    }

    /// Returns a mutable reference to the contained inner commitment, or an error if this is a leaf
    /// commitment.
    pub fn as_inner(&mut self) -> BTResult<&mut VerkleInnerCommitment, Error> {
        match self {
            VerkleCommitment::Inner(inner) => Ok(inner),
            VerkleCommitment::Leaf(_) => Err(Error::CorruptedState(
                "cannot convert leaf commitment to inner commitment".to_owned(),
            )
            .into()),
        }
    }

    /// Returns a mutable reference to the contained leaf commitment, or an error if this is an
    /// inner commitment.
    pub fn as_leaf(&mut self) -> BTResult<&mut VerkleLeafCommitment, Error> {
        match self {
            VerkleCommitment::Leaf(leaf) => Ok(leaf),
            VerkleCommitment::Inner(_) => Err(Error::CorruptedState(
                "cannot convert inner commitment to leaf commitment".to_owned(),
            )
            .into()),
        }
    }

    /// Consumes the commitment and returns the contained inner commitment, or an error if this is a
    /// leaf commitment.
    pub fn into_inner(self) -> BTResult<VerkleInnerCommitment, Error> {
        match self {
            VerkleCommitment::Inner(inner) => Ok(inner),
            VerkleCommitment::Leaf(_) => Err(Error::CorruptedState(
                "cannot convert leaf commitment to inner commitment".to_owned(),
            )
            .into()),
        }
    }

    /// Consumes the commitment and returns the contained leaf commitment, or an error if this is an
    /// inner commitment.
    pub fn into_leaf(self) -> BTResult<VerkleLeafCommitment, Error> {
        match self {
            VerkleCommitment::Leaf(leaf) => Ok(leaf),
            VerkleCommitment::Inner(_) => Err(Error::CorruptedState(
                "cannot convert inner commitment to leaf commitment".to_owned(),
            )
            .into()),
        }
    }

    /// Marks the commitment as clean.
    pub fn mark_clean(&mut self) {
        match self {
            VerkleCommitment::Inner(inner) => inner.mark_clean(),
            VerkleCommitment::Leaf(leaf) => leaf.mark_clean(),
        }
    }
}

impl TrieCommitment for VerkleCommitment {
    fn modify_child(&mut self, index: usize) {
        match self {
            VerkleCommitment::Inner(inner) => inner.modify_child(index),
            VerkleCommitment::Leaf(leaf) => leaf.modify_child(index),
        }
    }

    fn store(&mut self, index: usize, prev: Value) {
        match self {
            VerkleCommitment::Inner(inner) => inner.store(index, prev),
            VerkleCommitment::Leaf(leaf) => leaf.store(index, prev),
        }
    }
}

/// The status of the commitment, indicating if and how it needs to be recomputed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitmentStatus {
    /// The commitment has never been computed.
    ///
    /// For leaf commitments, this is the same as being dirty.
    ///
    /// For inner commitments, this means a full computation is required (no point-wise updates).
    /// Not being initialized does not imply the commitment being zero,
    /// as it may have been created from an existing commitment using
    /// [`VerkleInnerCommitment::from_leaf`].
    Uninitialized,

    /// The commitment has been computed at least once, but the node or its children have been
    /// modified since. Point-wise updates over dirty children can be used to update the
    /// commitment.
    Dirty,

    /// The commitment is up-to-date.
    Clean,
}

/// The commitment of a managed verkle trie inner node, together with metadata required to recompute
/// it after the node has been modified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerkleInnerCommitment {
    /// The commitment of the node, or of the node previously at this position in the trie.
    commitment: Commitment,

    /// The status of the commitment, indicating if and how it needs to be recomputed.
    status: CommitmentStatus,

    /// A bitfield indicating which children have been changed since the last commitment
    /// computation.
    changed_indices: [u8; 256 / 8],
}

impl VerkleInnerCommitment {
    /// Creates a new commitment that is meant to replace an existing commitment at a certain
    /// position within the trie. The new commitment is considered to be clean and uninitialized,
    /// however copies the existing commitment value. This allows to compute the delta between
    /// the commitment that used to be stored at this position, and the new commitment after
    /// it has been initialized.
    ///
    /// The concrete use case for this is reparenting a leaf node due to a stem mismatch when
    /// inserting a value: A new inner node is created and the existing leaf is attached as
    /// one of its children. The inner node's commitment is initialized from the existing leaf's
    /// commitment, such that the original parent (now grandparent) of the leaf can correctly
    /// update its commitment by computing the delta between the commitment of the leaf and
    /// the commitment of the new inner node.
    ///
    /// The `dirty_index` parameter can be used to indicate that the reparented leaf node currently
    /// has a dirty commitment. The index is expected to reflect the position where the leaf node
    /// (= the only child) is attached. If set, the corresponding index in the new commitment's
    /// `changed_indices` bitfield is marked as changed.
    pub fn from_leaf(existing: &VerkleLeafCommitment, dirty_index: Option<u8>) -> Self {
        let mut changed_indices = [0u8; 256 / 8];
        if let Some(index) = dirty_index {
            changed_indices[index as usize / 8] |= 1 << (index as usize % 8);
        }
        VerkleInnerCommitment {
            commitment: existing.commitment,
            changed_indices,
            ..Default::default()
        }
    }

    pub fn commitment(&self) -> Commitment {
        self.commitment
    }

    pub fn is_clean(&self) -> bool {
        self.status == CommitmentStatus::Clean
    }

    pub fn index_changed(&self, index: usize) -> bool {
        self.changed_indices[index / 8] & (1 << (index % 8)) != 0
    }

    pub fn mark_clean(&mut self) {
        self.status = CommitmentStatus::Clean;
        self.changed_indices.fill(0);
    }
}

impl Default for VerkleInnerCommitment {
    fn default() -> Self {
        Self {
            commitment: Commitment::default(),
            status: CommitmentStatus::Uninitialized,
            changed_indices: [0u8; 256 / 8],
        }
    }
}

impl TrieCommitment for VerkleInnerCommitment {
    fn modify_child(&mut self, index: usize) {
        self.changed_indices[index / 8] |= 1 << (index % 8);
        if self.status == CommitmentStatus::Clean {
            self.status = CommitmentStatus::Dirty;
        }
    }

    fn store(&mut self, _index: usize, _prev: Value) {
        unimplemented!()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, Unaligned, Immutable)]
#[repr(C)]
pub struct OnDiskVerkleInnerCommitment {
    commitment: [u8; 32],
}

impl From<OnDiskVerkleInnerCommitment> for VerkleInnerCommitment {
    fn from(odvc: OnDiskVerkleInnerCommitment) -> Self {
        VerkleInnerCommitment {
            commitment: Commitment::try_from_bytes(odvc.commitment).unwrap(), // FIXME Unwrap
            status: CommitmentStatus::Clean,
            changed_indices: [0u8; 256 / 8],
        }
    }
}

impl From<&VerkleInnerCommitment> for OnDiskVerkleInnerCommitment {
    fn from(value: &VerkleInnerCommitment) -> Self {
        assert_eq!(value.status, CommitmentStatus::Clean);

        OnDiskVerkleInnerCommitment {
            commitment: value.commitment.compress(),
        }
    }
}

/// The commitment of a managed verkle trie leaf node, together with metadata required to recompute
/// it after the node has been modified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerkleLeafCommitment {
    /// The commitment of the node.
    commitment: Commitment,

    /// A bitfield indicating which indices have been used before.
    /// This allows to distinguish between empty indices and indices that have been set to zero.
    committed_used_indices: [u8; 256 / 8],

    /// The status of the commitment, indicating if and how it needs to be recomputed.
    status: CommitmentStatus,

    /// A bitfield indicating which values have been changed since the last commitment computation.
    changed_indices: [u8; 256 / 8],

    /// The two partial commitments used as part of the leaf commitment computation.
    // TODO: Instead of storing these, consider doing a full commitment recomputation
    // after loading a leaf from disk.
    // See https://github.com/0xsoniclabs/sonic-admin/issues/373
    c1: Commitment,
    c2: Commitment,

    /// The values that were committed at the time of the last commitment computation.
    // TODO: This could be avoided by recomputing leaf commitments directly after storing values.
    // https://github.com/0xsoniclabs/sonic-admin/issues/542
    committed_values: [Value; 256],

    /// The commitment has been restored from disk and needs to be recomputed.
    restored: bool,
}

impl VerkleLeafCommitment {
    pub fn commitment(&self) -> Commitment {
        self.commitment
    }

    pub fn is_clean(&self) -> bool {
        self.status == CommitmentStatus::Clean
    }

    pub fn index_changed(&self, index: usize) -> bool {
        self.changed_indices[index / 8] & (1 << (index % 8)) != 0
    }

    pub fn mark_clean(&mut self) {
        self.status = CommitmentStatus::Clean;
        self.changed_indices.fill(0);
    }
}

impl Default for VerkleLeafCommitment {
    fn default() -> Self {
        Self {
            commitment: Commitment::default(),
            committed_used_indices: [0u8; 256 / 8],
            status: CommitmentStatus::Uninitialized,
            c1: Commitment::default(),
            c2: Commitment::default(),
            restored: false,
            committed_values: [Value::default(); 256],
            changed_indices: [0u8; 256 / 8],
        }
    }
}

impl TrieCommitment for VerkleLeafCommitment {
    fn modify_child(&mut self, _index: usize) {
        unimplemented!()
    }

    fn store(&mut self, index: usize, prev: Value) {
        // Since we want to compute the difference to the previously committed value,
        // we only set it the first time a new value is stored at this index.
        if !self.index_changed(index) {
            self.changed_indices[index / 8] |= 1 << (index % 8);
            self.committed_values[index] = prev;
            if self.status == CommitmentStatus::Clean {
                self.status = CommitmentStatus::Dirty;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, Unaligned, Immutable)]
#[repr(C)]
pub struct OnDiskVerkleLeafCommitment {
    commitment: [u8; 32],
    committed_used_indices: [u8; 256 / 8],
}

impl From<OnDiskVerkleLeafCommitment> for VerkleLeafCommitment {
    fn from(odvc: OnDiskVerkleLeafCommitment) -> Self {
        VerkleLeafCommitment {
            commitment: Commitment::try_from_bytes(odvc.commitment).unwrap(), // FIXME Unwrap
            committed_used_indices: odvc.committed_used_indices,
            status: CommitmentStatus::Clean,
            c1: Commitment::default(),
            c2: Commitment::default(),
            restored: true,
            committed_values: [Value::default(); 256],
            changed_indices: [0u8; 256 / 8],
        }
    }
}

impl From<&VerkleLeafCommitment> for OnDiskVerkleLeafCommitment {
    fn from(value: &VerkleLeafCommitment) -> Self {
        assert_eq!(value.status, CommitmentStatus::Clean);

        OnDiskVerkleLeafCommitment {
            commitment: value.commitment.compress(),
            committed_used_indices: value.committed_used_indices,
        }
    }
}

// TODO: Avoid copying all 256 values / children: https://github.com/0xsoniclabs/sonic-admin/issues/384
#[allow(clippy::large_enum_variant)]
#[derive(Debug, PartialEq, Eq)]
pub enum VerkleCommitmentInput {
    Leaf([Value; 256], [u8; 31]),
    Inner([VerkleNodeId; 256]),
}

/// Recomputes the commitments of all dirty nodes recorded in the given update log.
///
/// This function assumes the update log and the node manager to be in a consistent state:
/// - All nodes marked as dirty in the update log must exist in the node manager.
/// - All nodes marked as dirty in the update log must have a dirty [`VerkleCommitment`].
/// - For a dirty inner node on level `L`, all of its children in
///   [`VerkleCommitment::changed_indices`] must be contained in the update log on level `L+1`.
///
/// After successful completion, the update log is cleared.
pub fn update_commitments(
    log: &TrieUpdateLog<VerkleNodeId>,
    manager: &impl NodeManager<Id = VerkleNodeId, Node = VerkleNode>,
) -> BTResult<(), Error> {
    if log.count() == 0 {
        return Ok(());
    }

    let _span = tracy_client::span!("update_commitments");

    let mut previous_commitments = HashMap::new();
    for level in (0..log.levels()).rev() {
        let dirty_nodes_ids = log.dirty_nodes(level);
        for id in dirty_nodes_ids {
            let mut lock = manager.get_write_access(id)?;
            let mut vc = lock.get_commitment();
            assert!(!vc.is_clean());

            previous_commitments.insert(id, vc.commitment());

            match lock.get_commitment_input()? {
                VerkleCommitmentInput::Leaf(values, stem) => {
                    let _span = tracy_client::span!("leaf node");
                    let vc = vc.as_leaf()?;

                    // If this commitment was just restored from disk, C1 and C2 are default
                    // initialized and we need to do a full recomputation over all values.
                    // TODO: Move this logic into VerkleCommitment? But we don't want to do it after
                    // every restore (only if we are going to update)
                    if vc.restored {
                        vc.commitment = Commitment::default();
                        // Reset committed values to default, since we want to compute the delta
                        // against zero.
                        vc.committed_values = [Value::default(); 256];

                        // Mark all previously committed indices as changed, since we need to
                        // commit to them again regardless of their value (could be zero).
                        // vc.changed_indices = vc.committed_used_indices;
                        for (i, indices) in vc.changed_indices.iter_mut().enumerate() {
                            *indices |= vc.committed_used_indices[i];
                        }
                        vc.committed_used_indices.fill(0);
                        vc.restored = false;
                    }

                    compute_leaf_node_commitment(
                        vc.changed_indices,
                        &vc.committed_values,
                        &values,
                        &stem,
                        &mut vc.committed_used_indices,
                        &mut vc.c1,
                        &mut vc.c2,
                        &mut vc.commitment,
                    );
                }
                VerkleCommitmentInput::Inner(children) => {
                    let _span = tracy_client::span!("inner node");
                    let vc = vc.as_inner()?;

                    let mut scalars = [Scalar::zero(); 256];
                    for (i, child_id) in children.iter().enumerate() {
                        if vc.status == CommitmentStatus::Uninitialized {
                            if !child_id.is_empty_id() {
                                scalars[i] = manager
                                    .get_read_access(*child_id)?
                                    .get_commitment()
                                    .commitment()
                                    .to_scalar();
                            }
                            continue;
                        }

                        if !vc.index_changed(i) {
                            continue;
                        }

                        let child_commitment = manager.get_read_access(*child_id)?.get_commitment();
                        assert!(child_commitment.is_clean());
                        vc.commitment.update(
                            i as u8,
                            previous_commitments[child_id].to_scalar(),
                            child_commitment.commitment().to_scalar(),
                        );
                    }

                    if vc.status == CommitmentStatus::Uninitialized {
                        vc.commitment = Commitment::new(&scalars);
                    }
                }
            }

            vc.mark_clean();
            lock.set_commitment(vc)?;
        }
    }
    log.clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        database::verkle::{
            KeyedUpdate,
            compute_commitment::compute_leaf_node_commitment,
            crypto::Scalar,
            test_utils::FromIndexValues,
            variants::managed::{FullInnerNode, nodes::leaf::FullLeafNode},
        },
        node_manager::in_memory_node_manager::InMemoryNodeManager,
        types::{DiskRepresentable, HasEmptyId, Key},
    };

    #[test]
    fn verkle_inner_commitment_from_leaf() {
        let original = VerkleLeafCommitment {
            commitment: Commitment::new(&[Scalar::from(42), Scalar::from(33)]),
            committed_used_indices: [1u8; 256 / 8],
            status: CommitmentStatus::Dirty,
            changed_indices: [7u8; 256 / 8],
            c1: Commitment::new(&[Scalar::from(7)]),
            c2: Commitment::new(&[Scalar::from(11)]),
            committed_values: [[7u8; 32]; 256],
            restored: false,
        };

        let new = VerkleInnerCommitment::from_leaf(&original, None);
        assert_eq!(new.commitment, original.commitment);
        assert_eq!(new.status, CommitmentStatus::Uninitialized);
        assert_eq!(new.changed_indices, [0u8; 256 / 8]);

        let new = VerkleInnerCommitment::from_leaf(&original, Some(10));
        assert_eq!(
            new.changed_indices,
            <[u8; 256 / 8]>::from_index_values(0, &[(1, 0b00000100)])
        );
    }

    #[test]
    fn verkle_commitment__commitment_returns_stored_commitment() {
        let commitment = Commitment::new(&[Scalar::from(42), Scalar::from(33)]);
        let vc = VerkleCommitment::Inner(VerkleInnerCommitment {
            commitment,
            ..Default::default()
        });
        assert_eq!(vc.commitment(), commitment);
        let vc = VerkleCommitment::Leaf(VerkleLeafCommitment {
            commitment,
            ..Default::default()
        });
        assert_eq!(vc.commitment(), commitment);
    }

    #[test]
    fn verkle_commitment_is_clean_returns_correct_value() {
        let vc = VerkleCommitment::Inner(VerkleInnerCommitment {
            status: CommitmentStatus::Clean,
            ..Default::default()
        });
        assert!(vc.is_clean());

        let vc = VerkleCommitment::Inner(VerkleInnerCommitment {
            status: CommitmentStatus::Uninitialized,
            ..Default::default()
        });
        assert!(!vc.is_clean());

        let vc = VerkleCommitment::Inner(VerkleInnerCommitment {
            status: CommitmentStatus::Dirty,
            ..Default::default()
        });
        assert!(!vc.is_clean());

        let vc = VerkleCommitment::Leaf(VerkleLeafCommitment {
            status: CommitmentStatus::Clean,
            ..Default::default()
        });
        assert!(vc.is_clean());

        let vc = VerkleCommitment::Leaf(VerkleLeafCommitment {
            status: CommitmentStatus::Uninitialized,
            ..Default::default()
        });
        assert!(!vc.is_clean());

        let vc = VerkleCommitment::Leaf(VerkleLeafCommitment {
            status: CommitmentStatus::Dirty,
            ..Default::default()
        });
        assert!(!vc.is_clean());
    }

    #[test]
    fn verkle_commitment_index_changed_returns_correct_value() {
        let mut changed_indices = [0u8; 256 / 8];
        changed_indices[18 / 8] |= 1 << (18 % 8);
        changed_indices[231 / 8] |= 1 << (231 % 8);

        let vc = VerkleCommitment::Inner(VerkleInnerCommitment {
            changed_indices,
            ..Default::default()
        });
        for i in 0..256 {
            let expected = i == 18 || i == 231;
            assert_eq!(vc.index_changed(i), expected);
        }

        let vc = VerkleCommitment::Leaf(VerkleLeafCommitment {
            changed_indices,
            ..Default::default()
        });
        for i in 0..256 {
            let expected = i == 18 || i == 231;
            assert_eq!(vc.index_changed(i), expected);
        }
    }

    #[test]
    fn verkle_commitment_can_be_converted_to_inner_variant() {
        assert!(
            VerkleCommitment::Inner(VerkleInnerCommitment::default())
                .as_inner()
                .is_ok()
        );
        assert!(
            VerkleCommitment::Leaf(VerkleLeafCommitment::default())
                .as_inner()
                .is_err()
        );
        assert!(
            VerkleCommitment::Inner(VerkleInnerCommitment::default())
                .as_leaf()
                .is_err()
        );
        assert!(
            VerkleCommitment::Leaf(VerkleLeafCommitment::default())
                .as_leaf()
                .is_ok()
        );

        assert!(
            VerkleCommitment::Inner(VerkleInnerCommitment::default())
                .into_inner()
                .is_ok()
        );
        assert!(
            VerkleCommitment::Leaf(VerkleLeafCommitment::default())
                .into_inner()
                .is_err()
        );
        assert!(
            VerkleCommitment::Inner(VerkleInnerCommitment::default())
                .into_leaf()
                .is_err()
        );
        assert!(
            VerkleCommitment::Leaf(VerkleLeafCommitment::default())
                .into_leaf()
                .is_ok()
        );
    }

    #[rstest::rstest]
    fn verkle_commitment_modify_child_marks_dirty_and_changed(
        #[values(
            CommitmentStatus::Uninitialized,
            CommitmentStatus::Dirty,
            CommitmentStatus::Clean
        )]
        initial_status: CommitmentStatus,
    ) {
        let mut vc = VerkleCommitment::Inner(VerkleInnerCommitment {
            status: initial_status,
            ..Default::default()
        });
        vc.modify_child(42);
        let vc = vc.into_inner().unwrap();
        if initial_status == CommitmentStatus::Uninitialized {
            assert_eq!(vc.status, CommitmentStatus::Uninitialized);
        } else {
            assert_eq!(vc.status, CommitmentStatus::Dirty);
        }
        for i in 0..256 {
            assert_eq!(vc.changed_indices[i / 8] & (1 << (i % 8)) != 0, i == 42);
        }
    }

    #[rstest::rstest]
    fn verkle_commitment_store_marks_dirty_and_changed(
        #[values(
            CommitmentStatus::Uninitialized,
            CommitmentStatus::Dirty,
            CommitmentStatus::Clean
        )]
        initial_status: CommitmentStatus,
    ) {
        let mut vc = VerkleCommitment::Leaf(VerkleLeafCommitment {
            status: initial_status,
            ..Default::default()
        });
        vc.store(42, [0u8; 32]);
        let vc = vc.into_leaf().unwrap();
        if initial_status == CommitmentStatus::Uninitialized {
            assert_eq!(vc.status, CommitmentStatus::Uninitialized);
        } else {
            assert_eq!(vc.status, CommitmentStatus::Dirty);
        }
        for i in 0..256 {
            assert_eq!(vc.changed_indices[i / 8] & (1 << (i % 8)) != 0, i == 42);
        }
    }

    #[test]
    fn verkle_commitment_store_remembers_committed_value() {
        let mut vc = VerkleCommitment::Leaf(VerkleLeafCommitment::default());
        vc.store(42, [1u8; 32]);
        assert_eq!(vc.as_leaf().unwrap().committed_values[42], [1u8; 32]);
        // Only the first previous value (= the committed one) is remembered.
        vc.store(42, [7u8; 32]);
        assert_eq!(vc.as_leaf().unwrap().committed_values[42], [1u8; 32]);
    }

    #[test]
    fn verkle_inner_commitment_default_returns_uninitialized_commitment() {
        let vc = VerkleInnerCommitment::default();
        assert_eq!(vc.commitment, Commitment::default());
        assert_eq!(vc.status, CommitmentStatus::Uninitialized);
        assert_eq!(vc.changed_indices, [0u8; 256 / 8]);
    }

    #[test]
    fn verkle_inner_commitment_can_be_converted_to_and_from_on_disk_representation() {
        let original = VerkleInnerCommitment {
            commitment: Commitment::new(&[Scalar::from(42), Scalar::from(33)]),
            status: CommitmentStatus::Clean,
            changed_indices: [7u8; 256 / 8],
        };
        let on_disk_commitment: OnDiskVerkleInnerCommitment = (&original).into();
        let disk_repr = on_disk_commitment.to_disk_repr();
        let deserialized: VerkleInnerCommitment =
            OnDiskVerkleInnerCommitment::from_disk_repr::<()>(|buf| {
                buf.copy_from_slice(&disk_repr);
                Ok(())
            })
            .unwrap()
            .into();

        assert_eq!(
            deserialized,
            VerkleInnerCommitment {
                commitment: original.commitment,
                status: CommitmentStatus::Clean,
                changed_indices: [0u8; 256 / 8], // not preserved on disk
            }
        );
    }

    #[test]
    fn verkle_leaf_commitment_default_returns_uninitialized_commitment() {
        let vc = VerkleLeafCommitment::default();
        assert_eq!(vc.commitment, Commitment::default());
        assert_eq!(vc.committed_used_indices, [0u8; 256 / 8]);
        assert_eq!(vc.status, CommitmentStatus::Uninitialized);
        assert_eq!(vc.changed_indices, [0u8; 256 / 8]);
        assert_eq!(vc.c1, Commitment::default());
        assert_eq!(vc.c2, Commitment::default());
        assert_eq!(vc.committed_values, [Value::default(); 256]);
    }

    #[test]
    fn verkle_leaf_commitment_can_be_converted_to_and_from_on_disk_representation() {
        let original = VerkleLeafCommitment {
            commitment: Commitment::new(&[Scalar::from(42), Scalar::from(33)]),
            committed_used_indices: [1u8; 256 / 8],
            status: CommitmentStatus::Clean,
            changed_indices: [7u8; 256 / 8],
            c1: Commitment::new(&[Scalar::from(7)]),
            c2: Commitment::new(&[Scalar::from(11)]),
            committed_values: [[7u8; 32]; 256],
            restored: false,
        };
        let on_disk_commitment: OnDiskVerkleLeafCommitment = (&original).into();
        let disk_repr = on_disk_commitment.to_disk_repr();
        let deserialized: VerkleLeafCommitment =
            OnDiskVerkleLeafCommitment::from_disk_repr::<()>(|buf| {
                buf.copy_from_slice(&disk_repr);
                Ok(())
            })
            .unwrap()
            .into();

        assert_eq!(
            deserialized,
            VerkleLeafCommitment {
                commitment: original.commitment,
                committed_used_indices: original.committed_used_indices,
                // c1: original.c1,
                // c2: original.c2,
                status: CommitmentStatus::Clean,
                restored: true,
                // The remaining fields are not preserved on disk
                ..Default::default()
            }
        );
    }

    #[test]
    fn update_commitments_processes_dirty_nodes_from_leaves_to_root() {
        let manager = InMemoryNodeManager::<VerkleNodeId, VerkleNode>::new(10);
        let log = TrieUpdateLog::<VerkleNodeId>::new();

        let key = Key::from_index_values(33, &[(0, 7), (1, 4), (31, 255)]);

        // Set up simple chain: root -> inner -> leaf

        let mut leaf = FullLeafNode {
            stem: key[..31].try_into().unwrap(),
            ..Default::default()
        };
        let update = KeyedUpdate::FullSlot {
            key,
            value: [42u8; 32],
        };
        leaf.store(&update).unwrap();
        leaf.commitment.store(key[31] as usize, [0u8; 32]);

        let expected_leaf_commitment = {
            let mut vc = leaf.commitment;
            compute_leaf_node_commitment(
                vc.changed_indices,
                &[Value::default(); 256],
                &leaf.values,
                &leaf.stem,
                &mut vc.committed_used_indices,
                &mut vc.c1,
                &mut vc.c2,
                &mut vc.commitment,
            );
            vc.commitment
        };
        let leaf_id = manager.add(VerkleNode::Leaf256(Box::new(leaf))).unwrap();
        log.mark_dirty(2, leaf_id);

        let mut inner = FullInnerNode {
            children: {
                let mut children = [VerkleNodeId::empty_id(); 256];
                children[key[1] as usize] = leaf_id;
                children
            },
            ..Default::default()
        };
        inner.commitment.modify_child(key[1] as usize);
        let expected_inner_commitment = {
            let mut scalars = [Scalar::zero(); 256];
            scalars[key[1] as usize] = expected_leaf_commitment.to_scalar();
            Commitment::new(&scalars)
        };
        let inner_id = manager.add(VerkleNode::Inner256(Box::new(inner))).unwrap();
        log.mark_dirty(1, inner_id);

        let mut root = FullInnerNode {
            children: {
                let mut children = [VerkleNodeId::empty_id(); 256];
                children[key[0] as usize] = inner_id;
                children
            },
            ..Default::default()
        };
        root.commitment.modify_child(key[0] as usize);
        let expected_root_commitment = {
            let mut scalars = [Scalar::zero(); 256];
            scalars[key[0] as usize] = expected_inner_commitment.to_scalar();
            Commitment::new(&scalars)
        };
        let root_id = manager.add(VerkleNode::Inner256(Box::new(root))).unwrap();
        log.mark_dirty(0, root_id);

        // Run commitment update
        update_commitments(&log, &manager).unwrap();

        {
            let leaf_node_commitment = manager
                .get_read_access(leaf_id)
                .unwrap()
                .get_commitment()
                .into_leaf()
                .unwrap();
            assert_eq!(leaf_node_commitment.commitment, expected_leaf_commitment);
            assert_eq!(leaf_node_commitment.status, CommitmentStatus::Clean);
            assert!(leaf_node_commitment.changed_indices.iter().all(|&b| b == 0));
        }

        {
            let inner_node_commitment = manager
                .get_read_access(inner_id)
                .unwrap()
                .get_commitment()
                .into_inner()
                .unwrap();
            assert_eq!(inner_node_commitment.commitment, expected_inner_commitment);
            assert_eq!(inner_node_commitment.status, CommitmentStatus::Clean);
            assert!(
                inner_node_commitment
                    .changed_indices
                    .iter()
                    .all(|&b| b == 0)
            );
        }

        {
            let root_node_commitment = manager
                .get_read_access(root_id)
                .unwrap()
                .get_commitment()
                .into_inner()
                .unwrap();
            assert_eq!(root_node_commitment.commitment, expected_root_commitment);
            assert_eq!(root_node_commitment.status, CommitmentStatus::Clean);
            assert!(root_node_commitment.changed_indices.iter().all(|&b| b == 0));
        }

        assert_eq!(log.count(), 0);
    }
}
