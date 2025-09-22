use std::{collections::HashMap, sync::RwLockWriteGuard};

use zerocopy::{FromBytes, Immutable, IntoBytes, Unaligned};

use crate::{
    database::verkle::{compute_commitment::compute_leaf_node_commitment, crypto::Commitment},
    error::Error,
    node_manager::NodeManager,
    types::{Key, Value},
};

pub enum LookupResult<IdType> {
    Value(Value),
    Node(IdType),
}

pub enum CanStoreResult<IdType> {
    Yes,
    Descend(IdType),
    Reparent,
    Transform,
}

// TODO Test: Default commitment cache is clean
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned,
)]
#[repr(C)]
pub struct CachedCommitment {
    commitment: Commitment,
    // bool does not implement FromBytes, so we use u8 instead
    dirty: u8,
}

impl CachedCommitment {
    pub fn commitment(&self) -> Commitment {
        self.commitment
    }
}

// TODO: Make this generic over a "CommitmentScheme" (this is Ethereum Verkle)
#[allow(clippy::large_enum_variant)] // FIXME
pub enum CommitmentInput<IdType> {
    Empty,
    // TODO: Avoid copying these
    Leaf([Value; 256], [u8; 256 / 8], [u8; 31]),
    // TODO: Make sure we don't pass empty nodes here. In general if there is a single empty node
    //       item, there could be lock contention issues?!
    // TODO: Only pass non-empty children (Vec?)
    Inner([IdType; 256]),
}

/// A helper trait to constrain an [`IdTrieNode`] to be its own union type.
pub trait UnionIdTrieNode: IdTrieNode<UnionType = Self> {}

/// A generic interface for working with nodes in an ID-based (as opposed to pointer-based) trie
/// (Verkle, Binary, Merkle-Patricia, ...).
///
/// Besides simple value lookup, the trait specifies a set of lifecycle operations that allow to
/// update/store values in the trie using an iterative algorithm.
///
/// The trait is designed with the following goals in mind:
/// - Decouple nodes from their storage mechanism.
/// - Make nodes agnostic to locking schemes required for concurrent access.
/// - Clearly distinguish between operations that modify the trie structure and those that modify
///   node contents, allowing for accurate tracking of dirty states.
/// - Move shared logic out of the individual node types, such as tree traversal and commitment
///   updates/caching.
///
/// Since not all lifecycle methods make sense for all node types, the trait provides default
/// implementations that return an [`Error::UnsupportedOperation`] for most methods.
///
/// TODO Test default error?
pub trait IdTrieNode {
    /// The union type (enum) that encompasses all node types in the trie.
    type UnionType;

    /// The ID type used to identify nodes.
    type IdType;

    /// TODO: Docblock
    fn lookup(&self, _key: &Key, _depth: u8) -> Result<LookupResult<Self::IdType>, Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::lookup",
            std::any::type_name::<Self>()
        )))
    }

    /// TODO: Docblock
    fn can_store(&self, _key: &Key, _depth: u8) -> Result<CanStoreResult<Self::IdType>, Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::can_store",
            std::any::type_name::<Self>()
        )))
    }

    /// TODO: Docblock
    fn transform(&self, _key: &Key, _depth: u8) -> Result<Self::UnionType, Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::transform",
            std::any::type_name::<Self>()
        )))
    }

    /// TODO: Docblock
    fn reparent(
        &self,
        _key: &Key,
        _depth: u8,
        _self_id: Self::IdType,
    ) -> Result<Self::UnionType, Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::reparent",
            std::any::type_name::<Self>()
        )))
    }

    /// TODO: Docblock
    fn replace_child(&mut self, _key: &Key, _depth: u8, _new: Self::IdType) -> Result<(), Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::replace_child",
            std::any::type_name::<Self>()
        )))
    }

    /// TODO: Docblock
    fn store(&mut self, _key: &Key, _value: &Value) -> Result<(), Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::store",
            std::any::type_name::<Self>()
        )))
    }

    // TODO: Return Result, don't implement for EmptyNode?
    /// TODO: Docblock
    fn get_cached_commitment(&self) -> CachedCommitment;

    /// TODO: Docblock
    fn set_cached_commitment(&mut self, _cache: CachedCommitment) -> Result<(), Error> {
        Err(Error::UnsupportedOperation(format!(
            "{}::set_cached_commitment",
            std::any::type_name::<Self>()
        )))
    }

    /// TODO: Docblock
    fn get_commitment_input(&self) -> CommitmentInput<Self::IdType>;
}

pub fn lookup<T: IdTrieNode>(
    root_id: T::IdType,
    key: &Key,
    manager: &impl NodeManager<Id = T::IdType, NodeType = T>,
) -> Result<Value, Error> {
    let mut current_lock = manager.get_read_access(root_id)?;
    let mut depth = 0;

    loop {
        match current_lock.lookup(key, depth)? {
            LookupResult::Value(v) => return Ok(v),
            LookupResult::Node(node_id) => {
                let next_lock = manager.get_read_access(node_id)?;
                current_lock = next_lock;
                depth += 1;
            }
        }
    }
}

// TODO: Move elsewhere
// TODO: Have to deal with deletions?
pub struct TrieUpdateLog<IdType> {
    dirty_nodes_by_level: Vec<HashMap<IdType, [u8; 32]>>,
}

impl<IdType: Eq + std::hash::Hash> TrieUpdateLog<IdType> {
    pub fn new() -> Self {
        TrieUpdateLog {
            dirty_nodes_by_level: Vec::new(),
        }
    }

    fn add(&mut self, level: u8, id: IdType, child_idx: u8) {
        let level = level as usize;
        self.dirty_nodes_by_level
            .resize_with(self.dirty_nodes_by_level.len().max(level + 1), HashMap::new);
        self.dirty_nodes_by_level[level].entry(id).or_default()[(child_idx / 8) as usize] |=
            1 << (child_idx % 8);
    }

    fn clear(&mut self) {
        self.dirty_nodes_by_level.clear();
    }
}

pub fn store<T: UnionIdTrieNode>(
    mut root_id: RwLockWriteGuard<T::IdType>,
    key: &Key,
    value: &Value,
    manager: &impl NodeManager<Id = T::IdType, NodeType = T>,
    update_log: &mut TrieUpdateLog<T::IdType>,
) -> Result<(), Error>
where
    T::IdType: Copy + Eq + std::hash::Hash,
{
    let mut parent_lock = None;
    let mut current_id = *root_id;
    // TODO Test: Traversing tree with set sets dirty flag (except on leaf in case of split)
    let mut current_lock = manager.get_write_access(current_id)?;
    let mut depth = 0;

    loop {
        match current_lock.can_store(key, depth)? {
            CanStoreResult::Yes => {
                let cache = CachedCommitment {
                    dirty: 1,
                    ..current_lock.get_cached_commitment()
                };
                current_lock.set_cached_commitment(cache)?;
                update_log.add(depth, current_id, key[31]);

                return current_lock.store(key, value);
            }
            CanStoreResult::Descend(new_id) => {
                let cache = CachedCommitment {
                    dirty: 1,
                    ..current_lock.get_cached_commitment()
                };
                current_lock.set_cached_commitment(cache)?;
                update_log.add(depth, current_id, key[depth as usize]);

                parent_lock = Some(current_lock);
                current_lock = manager.get_write_access(new_id)?;
                current_id = new_id;
                depth += 1;
            }
            // TODO TEST: Parent lock is held for entire duration of transform / replace child
            CanStoreResult::Transform => {
                let new_node = current_lock.transform(key, depth)?;
                let new_id = manager.add(new_node).unwrap();
                if let Some(lock) = &mut parent_lock {
                    lock.replace_child(key, depth - 1, new_id)?;
                } else {
                    *root_id = new_id;
                    // TODO: drop lock on root_id here?
                }

                // TODO: Fetching the node again here may interfere with cache eviction (https://github.com/0xsoniclabs/sonic-admin/issues/380)
                current_lock = manager.get_write_access(new_id)?;
                manager.delete(current_id)?;
                current_id = new_id;

                // No need to log the update here, we are visiting the node again next iteration.
            }
            CanStoreResult::Reparent => {
                let new_node = current_lock.reparent(key, depth, current_id)?;
                let new_id = manager.add(new_node).unwrap();
                if let Some(lock) = &mut parent_lock {
                    lock.replace_child(key, depth - 1, new_id)?;
                } else {
                    *root_id = new_id;
                    // TODO: drop lock on root_id here?
                }
                current_lock = manager.get_write_access(new_id)?;
                current_id = new_id;

                // No need to log the update here, we are visiting the node again next iteration.
            }
        }
    }
}

#[allow(dead_code)] // TODO: Add feature flag?
pub fn compute_commitment_uncached_recursive<T: IdTrieNode>(
    id: T::IdType,
    manager: &impl NodeManager<Id = T::IdType, NodeType = T>,
) -> Result<Commitment, Error>
where
    T::IdType: Copy,
{
    let current_lock = manager.get_read_access(id)?;
    match current_lock.get_commitment_input() {
        CommitmentInput::Empty => Ok(Commitment::default()),
        CommitmentInput::Leaf(values, used_bits, stem) => {
            Ok(compute_leaf_node_commitment(&values, &used_bits, &stem))
        }
        CommitmentInput::Inner(children) => {
            let mut child_commitments = vec![Commitment::default().to_scalar(); 256];
            for (i, child_id) in children.iter().enumerate() {
                let child_commitment = compute_commitment_uncached_recursive(*child_id, manager)?;
                child_commitments[i] = child_commitment.to_scalar();
            }
            Ok(Commitment::new(&child_commitments))
        }
    }
}

// TODO Test: Visiting nodes with clean commitment does not set dirty flag
// TODO: What is the locking behavior we actually want?
// Computing commitments only makes sense at block boundaries, so we can probably just
// lock the full tree from top down.
// We may want to traverse the tree concurrently, as computing commitments is expensive.
// Observations:
// - All dirty nodes need to be visited eventually
// - The root lock is released last
// - Is there any point to doing this non-recursively?
#[allow(dead_code)] // TODO: Add feature flag?
pub fn compute_commitment_cached_recursive<T: IdTrieNode>(
    id: T::IdType,
    manager: &impl NodeManager<Id = T::IdType, NodeType = T>,
) -> Result<Commitment, Error>
where
    T::IdType: Copy,
{
    let mut current_lock = manager.get_write_access(id)?;
    let mut cache = current_lock.get_cached_commitment();

    if cache.dirty == 0 {
        return Ok(cache.commitment);
    }

    match current_lock.get_commitment_input() {
        // TODO: It's weird that we return the default commitment from Leaf nodes but then
        //       again have this logic here. Need single source of truth.
        CommitmentInput::Empty => return Ok(Commitment::default()),
        CommitmentInput::Leaf(values, used_bits, stem) => {
            // TODO: Anything to cache here..?
            cache.commitment = compute_leaf_node_commitment(&values, &used_bits, &stem);
        }
        CommitmentInput::Inner(children) => {
            // TODO: Update only changed children
            let mut child_commitments = vec![Commitment::default().to_scalar(); 256];
            for (i, child_id) in children.iter().enumerate() {
                let child_commitment = compute_commitment_uncached_recursive(*child_id, manager)?;
                child_commitments[i] = child_commitment.to_scalar();
            }
            cache.commitment = Commitment::new(&child_commitments);
        }
    }

    cache.dirty = 0;
    let commitment = cache.commitment;
    current_lock.set_cached_commitment(cache)?;
    Ok(commitment)
}

// TODO: I guess we don't even really need the dirty flag on CommitmentCache (or that type, for
//       that matter) any longer. Except for avoiding eviction..?
pub fn update_commitments<T: IdTrieNode>(
    log: &mut TrieUpdateLog<T::IdType>,
    manager: &impl NodeManager<Id = T::IdType, NodeType = T>,
) -> Result<(), Error>
where
    T::IdType: Copy + Eq + std::hash::Hash,
{
    let mut previous_commitments = HashMap::new();
    for dirty_nodes in log.dirty_nodes_by_level.iter().rev() {
        for (id, child_mask) in dirty_nodes.iter() {
            let mut lock = manager.get_write_access(*id)?;
            let mut cache = lock.get_cached_commitment();
            assert_eq!(cache.dirty, 1);

            previous_commitments.insert(*id, cache.commitment());

            match lock.get_commitment_input() {
                // TODO: It's weird that we return the default commitment from Leaf nodes but then
                //       again have this logic here. Need single source of truth.
                CommitmentInput::Empty => {
                    cache.commitment = Commitment::default();
                    panic!("should not happen?");
                }
                CommitmentInput::Leaf(values, used_bits, stem) => {
                    // TODO: Anything to cache here..?
                    cache.commitment = compute_leaf_node_commitment(&values, &used_bits, &stem);
                }
                CommitmentInput::Inner(children) => {
                    // let mut child_commitments = vec![Commitment::default().to_scalar(); 256];
                    for (i, child_id) in children.iter().enumerate() {
                        if child_mask[i / 8] & 1 << (i as u8 % 8) == 0 {
                            continue;
                        }

                        let child_commitment =
                            manager.get_read_access(*child_id)?.get_cached_commitment();
                        assert_eq!(child_commitment.dirty, 0);
                        // child_commitments[i] = child_commitment_cache.commitment().to_scalar();
                        cache.commitment.update(
                            i as u8,
                            previous_commitments[child_id].to_scalar(),
                            child_commitment.commitment().to_scalar(),
                        );
                    }
                    // cache.commitment = Commitment::new(&child_commitments);
                }
            }

            cache.dirty = 0;
            lock.set_cached_commitment(cache)?;
        }
    }
    // TODO: Test
    log.clear();
    Ok(())
}

#[cfg(test)]
mod tests {

    // TODO TEST: Reparenting does not mark child as dirty
    // TODO TEST: Root lock is released after traversing far enough into tree

    // TODO TEST: Concurrent read/write access (instrument an implementation with channels)

    // TODO: Consider having a faux MPT-style node here as well (using hash commitments)
}
