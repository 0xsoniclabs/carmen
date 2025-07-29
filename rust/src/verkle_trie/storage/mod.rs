// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.
#![allow(dead_code)]

mod file;

use thiserror::Error;
use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::types::Value;

#[derive(Debug, Error)]
enum Error {
    #[error("not found")]
    NotFound,
    #[error("IO error in storage: {0}")]
    IoError(#[from] std::io::Error),
}

#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
#[repr(C)]
struct LeafNode<const N: usize> {
    // commitment: TODO
    // stem: [u8; 31],
    // values: [OptValue; N], // option could also be replaced by separate bitmap
    values: [Value; N],
}

// #[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned)]
// #[repr(C)]
// struct OptValue {
//     value: Value,
//     is_present: u8,
// }

type TwoChildrenLeafNode = LeafNode<2>;
type AllChildrenLeafNode = LeafNode<256>;

#[derive(Debug, Clone, PartialEq, Eq, FromBytes, IntoBytes, Immutable)]
#[repr(C)]
struct InnerNode {
    // commitment: TODO
    // values: [OptValue; 256],
    values: [Value; 256],
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    Inner(Box<InnerNode>),
    TwoChildren(Box<TwoChildrenLeafNode>),
    AllChildren(Box<AllChildrenLeafNode>),
}

enum NodeRef<'a> {
    TwoChildren(&'a [Value; 2]),
    AllChildren(&'a [Value; 256]),
}

type Id = u64;

trait Storage {
    fn read(&self, id: Id) -> Result<Node, Error>;

    fn create(&self, node: &Node) -> Result<Id, Error>;

    fn flush(&self) -> Result<(), Error>;
}

// struct MmapStorage {
//     mmap: MmapMut,
// }

// impl MmapStorage {
//     fn new(file: &File) -> Result<Self, Error> {
//         //let mmap = unsafe { MmapOptions::new().huge(None).populate().map_mut(file) }
//         let mmap = unsafe { MmapMut::map_mut(file) }.map_err(Error::IoError)?;
//         Ok(Self { mmap })
//     }

//     fn flush(&self) -> Result<(), Error> {
//         self.mmap.flush().map_err(Error::IoError)
//     }
// }

// struct MemoryStorage {
//     data: HashMap<Id, Node>,
// }

// impl Storage for MemoryStorage {
//     fn read(&mut self, id: Id) -> Result<Node, Error> {
//         self.data.get(&id).ok_or(Error::NotFound).cloned()
//     }

//     fn create(&mut self, id: Id, node: Node) -> Result<(), Error> {
//         self.data.insert(id, node);
//         Ok(())
//     }
// }
