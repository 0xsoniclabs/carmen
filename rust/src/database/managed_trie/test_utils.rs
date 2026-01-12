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
    collections::HashMap,
    ops::{Deref, DerefMut},
    time::Duration,
};

use crate::{
    database::{
        managed_trie::{
            ManagedTrieNode, TrieCommitment,
            managed_trie_node::{LookupResult, StoreAction, UnionManagedTrieNode},
        },
        verkle::{KeyedUpdate, KeyedUpdateBatch},
    },
    error::{BTResult, Error},
    node_manager::NodeManager,
    sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard, atomic::AtomicU32, thread},
    types::{HasEmptyNode, Key, Value},
};

pub type Id = u32;

/// Spins until the provided function `f` returns `Some(R)`, returning the contained `R`.
/// If more than 1 second elapses, this function panics with the provided `timeout_msg`.
pub fn spin_until_some<R>(f: impl Fn() -> Option<R>, timeout_msg: &str) -> R {
    let start = std::time::Instant::now();
    loop {
        if let Some(res) = f() {
            return res;
        }
        if start.elapsed() > Duration::from_secs(1) {
            panic!("{timeout_msg}");
        }
        thread::yield_now();
    }
}

/// A simple single-producer, single-consumer channel for sending items of type `I` across threads.
/// The channel only holds a single item at a time and blocks on both `send` and `receive` until the
/// item has been transferred, or a timeout is reached. It also allows to peek at the value on the
/// receiving side without removing it from the channel, keeping the sender side blocked.
struct RcChannel<I: Send> {
    item: Mutex<Option<I>>,
}

impl<I: Send> RcChannel<I> {
    pub fn new() -> Self {
        Self {
            item: Mutex::new(None),
        }
    }

    /// Sends an item to the channel, blocking until it has been received.
    pub fn send(&self, item: I, debug_ctx: &str) {
        *self.item.lock().unwrap() = Some(item);
        spin_until_some(
            || self.item.lock().unwrap().is_none().then_some(()),
            &format!("{debug_ctx} was not consumed in time"),
        );
    }

    /// Receives an item from the channel, blocking until one is available.
    pub fn receive(&self, debug_ctx: &str) -> I {
        spin_until_some(
            || self.item.lock().unwrap().take(),
            &format!("{debug_ctx} was not received in time"),
        )
    }
}

impl<I: Clone + Send> RcChannel<I> {
    /// Peeks at the next item in the channel without removing it, blocking until one is available.
    pub fn peek(&self, debug_ctx: &str) -> I {
        spin_until_some(
            || self.item.lock().unwrap().clone(),
            &format!("{debug_ctx} was not received in time"),
        )
    }

    /// Removes the next item from the channel without returning it.
    pub fn pop(&self) {
        *self.item.lock().unwrap() = None;
    }
}

/// A simple implementation of `TrieCommitment` for testing purposes.
/// Keeps track of all changed indices and the previous values at those indices.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct TestNodeCommitment(HashMap<usize, Value>);

impl TestNodeCommitment {
    pub fn expected_single(key: usize, value: Value) -> Self {
        Self(HashMap::from([(key, value)]))
    }

    pub fn expected(x: impl Iterator<Item = (usize, Value)>) -> Self {
        Self(x.collect())
    }
}

impl TrieCommitment for TestNodeCommitment {
    fn modify_child(&mut self, index: usize) {
        self.0.insert(index, Value::default());
    }

    fn store(&mut self, index: usize, prev: Value) {
        self.0.insert(index, prev);
    }
}

/// An expectation that can be sent to a [`RcNode`] to control its behavior remotely from another
/// thread. Each variant is an operation that is expected to be called next, containing the expected
/// parameters as well as the result to return, where applicable.
#[derive(Debug)]
pub enum RcNodeExpectation {
    Lookup {
        key: Key,
        depth: u8,
        result: LookupResult<Id>,
    },
    NextStoreAction {
        updates: KeyedUpdateBatch<'static>,
        depth: u8,
        self_id: Id,
        result: StoreAction<'static, Id, RcNode>,
    },
    ReplaceChild {
        key: Key,
        depth: u8,
        new: Id,
    },
    Store {
        update: KeyedUpdate,
        result: Value,
    },
    GetCommitment {
        result: TestNodeCommitment,
    },
    SetCommitment {
        commitment: TestNodeCommitment,
    },
    Clone {
        new_id: Id,
    },
}

/// A "remote-controlled" [`ManagedTrieNode`] implementation for testing purposes.
/// The behavior of the node can be controlled from another thread through the [`RcNodeManager`],
/// by setting up expectations on the operations being called and specifying the results to return.
#[derive(Default)]
pub struct RcNode {
    // The node stores its own id for easier debugging
    id: Id,
    // To make this type default constructible, we wrap the Arc in an Option
    channel: Option<Arc<RcChannel<RcNodeExpectation>>>,
    // A reference back to the manager that created this node.
    // To make this type default constructible, we wrap the Arc in an Option
    manager: Option<Arc<RcNodeManager>>,
}

impl Clone for RcNode {
    /// Clones the node by receiving a `Clone` expectation from the channel.
    /// The id of the new node is taken from the expectation, and the channel is
    /// retrieved from the manager's list of channels.
    fn clone(&self) -> Self {
        match self.channel.as_ref().unwrap().receive("clone") {
            RcNodeExpectation::Clone { new_id } => {
                let manager = self.manager.as_ref().expect("manager to be set");
                let channel = manager
                    .node_channels
                    .lock()
                    .unwrap()
                    .get(new_id as usize)
                    .expect("node to be created using make() before clone() is called")
                    .clone();
                Self {
                    id: new_id,
                    channel: Some(channel),
                    manager: self.manager.clone(),
                }
            }
            e => panic!("expected call to {e:?} on {self:?}, but clone was called instead"),
        }
    }
}

impl HasEmptyNode for RcNode {
    fn is_empty_node(&self) -> bool {
        false
    }

    fn empty_node() -> Self {
        Self::default()
    }
}

impl RcNode {
    pub fn id(&self) -> Id {
        self.id
    }

    /// Clones the node without going through the channel. This is a 1:1 copy with the same ID.
    pub fn clone_non_rc(&self) -> Self {
        Self {
            id: self.id,
            channel: self.channel.clone(),
            manager: self.manager.clone(),
        }
    }
}

impl PartialEq for RcNode {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl std::fmt::Debug for RcNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RcNode").field("id", &self.id).finish()
    }
}

impl ManagedTrieNode for RcNode {
    type Union = RcNode;

    type Id = Id;

    type Commitment = TestNodeCommitment;

    fn lookup(&self, key: &Key, depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {
        match self.channel.as_ref().unwrap().receive("lookup") {
            RcNodeExpectation::Lookup {
                key: k,
                depth: d,
                result,
            } => {
                if (&k, d) != (key, depth) {
                    panic!(
                        "unexpected lookup parameters: ({k:?}, {d}), expected ({key:?}, {depth})"
                    );
                }
                Ok(result)
            }
            e => panic!("expected call to {e:?} on {self:?}, but lookup was called instead"),
        }
    }

    fn next_store_action<'a>(
        &self,
        key: KeyedUpdateBatch<'a>,
        depth: u8,
        self_id: Self::Id,
    ) -> BTResult<StoreAction<'a, Self::Id, Self::Union>, Error> {
        match self
            .channel
            .as_ref()
            .unwrap()
            .receive(&format!("next_store_action for {self_id}"))
        {
            RcNodeExpectation::NextStoreAction {
                updates: k,
                depth: d,
                self_id: sid,
                result,
            } => {
                if (&k, d, sid) != (&key, depth, self_id) {
                    panic!(
                        "unexpected next_store_action parameters ({k:?}, {d}, {sid}), expected ({key:?}, {depth}, {self_id})"
                    );
                }
                Ok(result)
            }
            e => panic!(
                "expected call to {e:?} on {self:?}, but next_store_action({key:?}, {depth}, {self_id}) was called instead"
            ),
        }
    }

    fn replace_child(&mut self, key: &Key, depth: u8, new: Self::Id) -> BTResult<(), Error> {
        match self.channel.as_ref().unwrap().receive("replace_child") {
            RcNodeExpectation::ReplaceChild {
                key: k,
                depth: d,
                new: n,
            } => {
                if (&k, d, n) != (key, depth, new) {
                    panic!(
                        "unexpected replace_child parameters ({k:?}, {d}, {n}), expected ({key:?}, {depth}, {new})"
                    );
                }
                Ok(())
            }
            e => panic!(
                "expected call to {e:?} on {self:?}, but replace_child({key:?}, {depth}, {new}) was called instead"
            ),
        }
    }

    fn store(&mut self, update: &KeyedUpdate) -> BTResult<Value, Error> {
        match self.channel.as_ref().unwrap().receive("store") {
            RcNodeExpectation::Store {
                update: u,
                result: pv,
            } => {
                if u != *update {
                    panic!("unexpected store parameters ({u:?}), expected ({update:?})");
                }
                Ok(pv)
            }
            e => panic!(
                "expected call to {e:?} on {self:?}, but store({update:?}) was called instead"
            ),
        }
    }

    fn get_commitment(&self) -> Self::Commitment {
        match self.channel.as_ref().unwrap().receive("get_commitment") {
            RcNodeExpectation::GetCommitment { result } => result,
            e => {
                panic!("expected call to {e:?} on {self:?}, but get_commitment was called instead")
            }
        }
    }

    fn set_commitment(&mut self, commitment: Self::Commitment) -> BTResult<(), Error> {
        match self.channel.as_ref().unwrap().receive("set_commitment") {
            RcNodeExpectation::SetCommitment { commitment: c } => {
                if c != commitment {
                    panic!("unexpected set_commitment parameter {c:?}, expected {commitment:?}");
                }
                Ok(())
            }
            e => panic!(
                "expected call to {e:?} on {self:?}, but set_commitment({commitment:?}) was called instead"
            ),
        }
    }
}

impl UnionManagedTrieNode for RcNode {
    fn copy_on_write(&self, _id: Self::Id, _changed_children: Vec<u8>) -> Self {
        self.clone()
    }
}

/// A wrapper around an `RcNode` that tracks its dirty status.
/// The node is marked as dirty whenever it is accessed through [`DerefMut`].
#[derive(Default)]
struct NodeWrapper {
    node: RcNode,
    dirty: bool,
}

impl Deref for NodeWrapper {
    type Target = RcNode;

    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

impl DerefMut for NodeWrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.dirty = true;
        &mut self.node
    }
}

/// An expectation for an operation called on the [`RcNodeManager`].
/// Each variant corresponds to a method on the manager, containing
/// the parameters it is expected to be called with.
#[derive(Debug, Clone)]
enum RcNodeManagerExpectation {
    Add { node: RcNode },
    ReadAccess { id: Id, currently_locked: Vec<Id> },
    WriteAccess { id: Id, currently_locked: Vec<Id> },
    Delete { id: Id },
}

/// A "remote-controlled" [`NodeManager`] implementation for testing purposes.
pub struct RcNodeManager {
    /// The nodes managed by this manager (up to 30).
    nodes: [RwLock<NodeWrapper>; 30],
    /// The channels used to communicate with each node.
    /// These also exist inside the nodes, but we keep a separate copy here to be able to send
    /// expectations to nodes without having to acquire locks.
    node_channels: Mutex<Vec<Arc<RcChannel<RcNodeExpectation>>>>,
    /// The next ID to assign to a new node.
    next_id: AtomicU32,
    /// The channel used to communicate expectations for the manager itself.
    expectation: RcChannel<RcNodeManagerExpectation>,
}

impl RcNodeManager {
    pub fn new() -> Self {
        Self {
            nodes: std::array::from_fn(|_| RwLock::new(NodeWrapper::default())),
            node_channels: Mutex::new(Vec::new()),
            next_id: AtomicU32::new(0),
            expectation: RcChannel::new(),
        }
    }

    /// Creates a new `RcNode` but does not register it with the manager.
    pub fn make(self: &Arc<Self>) -> RcNode {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let channel = Arc::new(RcChannel::new());
        self.node_channels.lock().unwrap().push(channel.clone());
        RcNode {
            id,
            channel: Some(channel),
            manager: Some(Arc::clone(self)),
        }
    }

    /// Adds the given `RcNode` to the manager directly and does not mark the
    /// node as dirty. Intended for test setup code.
    pub fn insert(&self, node: RcNode) -> Id {
        let id = node.id;
        *self.nodes[id as usize].write().unwrap() = NodeWrapper { node, dirty: false };
        id
    }

    /// Returns whether the node with the given `id` is marked as dirty.
    #[track_caller]
    pub fn is_dirty(&self, id: Id) -> bool {
        self.nodes[id as usize]
            .try_read()
            .expect("node is locked for writing")
            .dirty
    }

    /// Waits until the node with the given `id` is unlocked for reading and writing.
    pub fn wait_for_unlock(&self, id: Id) {
        spin_until_some(
            || (!self.is_locked(id)).then_some(()),
            &format!("node {id} did not become unlocked in time"),
        );
    }

    /// Sets up an expectation for the next operation called on the node with the given `id`.
    /// The function will block until the expectation has been received in another thread.
    pub fn expect(&self, id: Id, expectation: RcNodeExpectation) {
        let dbg_str = &format!("expect({id}, {expectation:?})");
        self.node_channels.lock().unwrap()[id as usize].send(expectation, dbg_str);
    }

    /// Sets up an expectation for the next operation on the manager to be
    /// a call to [`NodeManager::add`] with the given `node`.
    /// The function will block until the expectation has been received in another thread.
    pub fn expect_add(&self, node: RcNode) {
        let dbg_str = &format!("expect_add({node:?})");
        self.expectation
            .send(RcNodeManagerExpectation::Add { node }, dbg_str);
    }

    /// Sets up an expectation for the next operation on the manager to be
    /// a call to [`NodeManager::get_read_access`] with the given `id`.
    /// `currently_locked` should contain the list of node IDs that are expected to be
    /// locked at the time of the call.
    /// The function will block until the expectation has been received in another thread.
    pub fn expect_read_access(&self, id: Id, currently_locked: Vec<Id>) {
        self.expectation.send(
            RcNodeManagerExpectation::ReadAccess {
                id,
                currently_locked,
            },
            &format!("expect_read_access({id})"),
        );
    }

    /// Like [`Self::expect_read_access`], but for [`NodeManager::get_write_access`].
    pub fn expect_write_access(&self, id: Id, currently_locked: Vec<Id>) {
        self.expectation.send(
            RcNodeManagerExpectation::WriteAccess {
                id,
                currently_locked,
            },
            &format!("expect_write_access({id})"),
        );
    }

    /// Sets up an expectation for the next operation on the manager to be
    /// a call to [`NodeManager::delete`] with the given `id`.
    /// The function will block until the expectation has been received in another thread.
    pub fn expect_delete(&self, id: Id) {
        self.expectation.send(
            RcNodeManagerExpectation::Delete { id },
            &format!("expect_delete({id})"),
        );
    }

    /// Verifies that the currently locked nodes match the provided list of `currently_locked` IDs.
    fn expect_locked(&self, currently_locked: &[Id]) {
        for i in 0..self.nodes.len() {
            if self.is_locked(i as Id) {
                if !currently_locked.contains(&(i as Id)) {
                    panic!("expected id {i} to not be locked");
                }
            } else if currently_locked.contains(&(i as Id)) {
                panic!("expected id {i} to be locked");
            }
        }
    }

    /// Returns whether the node with the given `id` is currently locked for reading or writing.
    fn is_locked(&self, id: Id) -> bool {
        self.nodes[id as usize].try_write().is_err()
    }
}

impl NodeManager for RcNodeManager {
    type Id = Id;
    type Node = RcNode;

    fn add(&self, node: Self::Node) -> BTResult<Self::Id, Error> {
        match self.expectation.receive("add") {
            RcNodeManagerExpectation::Add {
                node: expected_node,
            } => {
                if node != expected_node {
                    panic!("expected Add({expected_node:?}), received Add({node:?})");
                }
                let id = node.id;
                *self.nodes[id as usize].write().unwrap() = NodeWrapper { node, dirty: true };
                Ok(id as Id)
            }
            e => panic!("expected {e:?}, received Add({node:?})"),
        }
    }

    fn get_read_access(
        &self,
        id: Self::Id,
    ) -> BTResult<RwLockReadGuard<'_, impl Deref<Target = Self::Node>>, Error> {
        match self.expectation.peek(&format!("get_read_access({id})")) {
            RcNodeManagerExpectation::ReadAccess {
                id: expected_id,
                currently_locked,
            } => {
                if id != expected_id {
                    panic!("expected ReadAccess({expected_id}), received ReadAccess({id})");
                }
                self.expect_locked(&currently_locked);
            }
            e => panic!("expected {e:?}, received ReadAccess({id})"),
        }
        let guard = self.nodes[id as usize].read().unwrap();
        self.expectation.pop();
        Ok(guard)
    }

    fn get_write_access(
        &self,
        id: Self::Id,
    ) -> BTResult<RwLockWriteGuard<'_, impl DerefMut<Target = Self::Node>>, Error> {
        match self.expectation.peek(&format!("get_write_access({id})")) {
            RcNodeManagerExpectation::WriteAccess {
                id: expected_id,
                currently_locked,
            } => {
                if id != expected_id {
                    panic!("expected WriteAccess({expected_id}), received WriteAccess({id})");
                }
                self.expect_locked(&currently_locked);
            }
            e => panic!("expected {e:?}, received WriteAccess({id})"),
        }
        let guard = self.nodes[id as usize].write().unwrap();
        self.expectation.pop();
        Ok(guard)
    }

    fn delete(&self, id: Self::Id) -> BTResult<(), Error> {
        match self.expectation.receive(&format!("delete({id})")) {
            RcNodeManagerExpectation::Delete { id: expected_id } => {
                if id != expected_id {
                    panic!("expected Delete({expected_id}), received Delete({id})");
                }
                // We don't actually delete anything
                Ok(())
            }
            e => panic!("expected {e:?}, received Delete({id})"),
        }
    }
}
