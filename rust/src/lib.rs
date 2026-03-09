// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.
#![cfg_attr(test, allow(non_snake_case))]
#![cfg_attr(
    feature = "shuttle",
    deny(clippy::disallowed_types, clippy::disallowed_methods)
)]

use std::{collections::HashMap, mem::MaybeUninit, ops::Deref, path::Path, sync::{LazyLock, Mutex}};

#[cfg(feature = "storage-statistics")]
use crate::statistics::storage::StorageOperationLogger;
pub use crate::types::{ArchiveImpl, BalanceUpdate, LiveImpl, Update};
use crate::{
    database::{
        ManagedTrieNode, ManagedVerkleTrie, VerkleTrieCarmenState,
        verkle::{
            StateMode,
            variants::managed::{
                FullInnerNode, FullLeafNode, InnerDeltaNode, LeafDeltaNode, SparseInnerNode,
                SparseLeafNode, VerkleNode, VerkleNodeFileStorageManager, VerkleNodeId, VerkleNodeKind,
            },
        },
    },
    error::{BTResult, Error},
    node_manager::cached_node_manager::CachedNodeManager,
    storage::{
        DbOpenMode, RootIdProvider, Storage,
        file::{NoSeekFile, NodeFileStorage},
        storage_with_flush_buffer::StorageWithFlushBuffer,
    },
    sync::Arc,
    types::*,
};

pub mod database;
pub mod error;
mod ffi;
pub mod node_manager;
pub mod statistics;
pub mod storage;
pub mod sync;
pub mod types;
mod utils;

pub type VerkleStorageManager = VerkleNodeFileStorageManager<
    NodeFileStorage<SparseInnerNode<1>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<2>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<3>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<4>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<5>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<6>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<7>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<8>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<9>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<10>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<11>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<12>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<13>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<14>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<15>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<16>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<17>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<18>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<19>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<20>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<21>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<22>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<23>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<24>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<25>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<26>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<27>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<28>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<29>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<30>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<31>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<32>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<33>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<34>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<35>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<36>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<37>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<38>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<39>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<40>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<41>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<42>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<43>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<44>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<45>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<46>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<47>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<48>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<49>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<50>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<51>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<52>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<53>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<54>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<55>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<56>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<57>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<58>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<59>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<60>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<61>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<62>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<63>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<64>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<65>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<66>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<67>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<68>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<69>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<70>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<71>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<72>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<73>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<74>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<75>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<76>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<77>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<78>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<79>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<80>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<81>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<82>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<83>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<84>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<85>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<86>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<87>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<88>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<89>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<90>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<91>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<92>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<93>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<94>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<95>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<96>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<97>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<98>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<99>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<100>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<101>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<102>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<103>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<104>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<105>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<106>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<107>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<108>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<109>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<110>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<111>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<112>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<113>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<114>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<115>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<116>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<117>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<118>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<119>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<120>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<121>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<122>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<123>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<124>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<125>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<126>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<127>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<128>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<129>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<130>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<131>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<132>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<133>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<134>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<135>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<136>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<137>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<138>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<139>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<140>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<141>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<142>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<143>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<144>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<145>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<146>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<147>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<148>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<149>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<150>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<151>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<152>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<153>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<154>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<155>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<156>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<157>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<158>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<159>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<160>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<161>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<162>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<163>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<164>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<165>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<166>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<167>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<168>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<169>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<170>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<171>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<172>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<173>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<174>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<175>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<176>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<177>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<178>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<179>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<180>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<181>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<182>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<183>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<184>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<185>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<186>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<187>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<188>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<189>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<190>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<191>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<192>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<193>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<194>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<195>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<196>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<197>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<198>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<199>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<200>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<201>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<202>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<203>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<204>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<205>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<206>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<207>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<208>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<209>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<210>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<211>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<212>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<213>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<214>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<215>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<216>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<217>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<218>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<219>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<220>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<221>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<222>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<223>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<224>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<225>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<226>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<227>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<228>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<229>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<230>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<231>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<232>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<233>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<234>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<235>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<236>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<237>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<238>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<239>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<240>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<241>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<242>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<243>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<244>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<245>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<246>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<247>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<248>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<249>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<250>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<251>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<252>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<253>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<254>, NoSeekFile>,
    NodeFileStorage<SparseInnerNode<255>, NoSeekFile>,
    NodeFileStorage<FullInnerNode, NoSeekFile>,
    NodeFileStorage<InnerDeltaNode, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<1>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<2>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<3>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<4>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<5>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<6>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<7>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<8>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<9>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<10>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<11>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<12>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<13>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<14>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<15>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<16>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<17>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<18>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<19>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<20>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<21>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<22>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<23>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<24>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<25>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<26>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<27>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<28>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<29>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<30>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<31>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<32>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<33>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<34>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<35>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<36>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<37>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<38>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<39>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<40>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<41>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<42>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<43>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<44>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<45>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<46>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<47>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<48>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<49>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<50>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<51>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<52>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<53>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<54>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<55>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<56>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<57>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<58>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<59>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<60>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<61>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<62>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<63>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<64>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<65>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<66>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<67>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<68>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<69>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<70>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<71>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<72>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<73>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<74>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<75>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<76>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<77>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<78>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<79>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<80>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<81>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<82>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<83>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<84>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<85>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<86>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<87>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<88>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<89>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<90>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<91>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<92>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<93>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<94>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<95>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<96>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<97>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<98>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<99>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<100>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<101>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<102>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<103>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<104>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<105>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<106>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<107>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<108>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<109>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<110>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<111>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<112>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<113>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<114>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<115>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<116>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<117>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<118>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<119>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<120>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<121>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<122>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<123>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<124>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<125>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<126>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<127>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<128>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<129>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<130>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<131>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<132>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<133>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<134>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<135>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<136>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<137>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<138>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<139>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<140>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<141>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<142>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<143>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<144>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<145>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<146>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<147>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<148>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<149>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<150>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<151>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<152>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<153>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<154>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<155>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<156>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<157>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<158>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<159>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<160>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<161>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<162>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<163>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<164>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<165>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<166>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<167>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<168>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<169>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<170>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<171>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<172>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<173>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<174>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<175>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<176>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<177>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<178>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<179>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<180>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<181>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<182>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<183>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<184>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<185>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<186>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<187>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<188>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<189>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<190>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<191>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<192>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<193>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<194>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<195>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<196>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<197>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<198>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<199>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<200>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<201>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<202>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<203>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<204>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<205>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<206>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<207>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<208>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<209>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<210>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<211>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<212>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<213>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<214>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<215>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<216>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<217>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<218>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<219>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<220>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<221>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<222>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<223>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<224>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<225>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<226>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<227>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<228>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<229>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<230>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<231>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<232>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<233>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<234>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<235>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<236>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<237>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<238>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<239>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<240>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<241>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<242>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<243>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<244>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<245>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<246>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<247>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<248>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<249>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<250>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<251>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<252>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<253>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<254>, NoSeekFile>,
    NodeFileStorage<SparseLeafNode<255>, NoSeekFile>,
    NodeFileStorage<FullLeafNode, NoSeekFile>,
    NodeFileStorage<LeafDeltaNode, NoSeekFile>,
>;

type VerkleStorage = StorageWithFlushBuffer<VerkleStorageManager>;

pub struct SpecializationTransactions {
    pub inner: HashMap<(usize, VerkleNodeKind), usize>,
    pub leaf: HashMap<(usize, VerkleNodeKind), usize>, 
}

pub static SPECIALIZATION_TRANSACTIONS : Mutex<LazyLock<SpecializationTransactions>> = Mutex::new(LazyLock::new(|| {
    SpecializationTransactions {
        inner: HashMap::new(),
        leaf: HashMap::new(),
    }
}));


impl SpecializationTransactions {
    pub fn print(&self, out: &mut dyn std::io::Write) -> BTResult<(), Error> {
        writeln!(out, "Inner node specialization transactions:").map_err(storage::Error::from)?;
        for (record, num_tranformations) in &self.inner {
            self.print_record(record, num_tranformations, "Inner", out)?;
        }

        writeln!(out, "Leaf node specialization transactions:").map_err(storage::Error::from)?;
        for (record, num_transformations) in &self.leaf {
            self.print_record(record,  num_transformations, "Leaf", out)?;
        }
        Ok(())
    }

    pub fn print_record(&self, record : &(usize, VerkleNodeKind), num: &usize, prefix: &str, out: &mut dyn std::io::Write) -> BTResult<(), Error> {
        let (src, dst) = record;
        let dst_str = format!("{dst:?}").strip_prefix(prefix).map(str::to_owned).unwrap_or_else(|| format!("{dst:?}").to_owned());
        writeln!(out, "{} -> {}: {}", src, dst_str, num).map_err(storage::Error::from)?;
        Ok(())
    }

    pub fn get_node_kind_size(node_kind: &VerkleNodeKind, prefix: &str) -> BTResult<String, Error> {
        Ok(format!("{node_kind:?}").strip_prefix(prefix).ok_or(Error::CorruptedState("Expected node kind to start with".to_string()))?.to_owned())
    }
}


/// Opens a new [CarmenDb] database object based on the provided implementation maintaining
/// its data in the given directory. If the directory does not exist, it is
/// created. If it is empty, a new, empty state is initialized. If it contains
/// state information, the information is loaded.
pub fn open_carmen_db(
    schema: u8,
    live_impl: &[u8],
    archive_impl: &[u8],
    directory: &Path,
    db_open_mode: DbOpenMode,
) -> BTResult<Box<dyn CarmenDb>, Error> {
    if schema != 6 {
        return Err(Error::UnsupportedSchema(schema).into());
    }

    match (live_impl, archive_impl) {
        (b"memory", b"none" | b"") => {
            Ok(Box::new(CarmenS6InMemoryDb::new(VerkleTrieCarmenState::<
                database::SimpleInMemoryVerkleTrie,
            >::new_live())))
        }
        (b"crate-crypto-memory", b"none" | b"") => {
            Ok(Box::new(CarmenS6InMemoryDb::new(VerkleTrieCarmenState::<
                database::CrateCryptoInMemoryVerkleTrie,
            >::new_live())))
        }
        (b"file", b"none"| b"") => {
            let live_dir = directory.join("live");
            let storage = VerkleStorage::open(&live_dir, db_open_mode)?;
            #[cfg(feature = "storage-statistics")]
            let storage = StorageOperationLogger::try_new(storage, Path::new("."))?;

            let is_pinned = |node: &VerkleNode| !node.get_commitment().is_clean();
            // TODO: The cache size is arbitrary, base this on a configurable memory limit instead
            // https://github.com/0xsoniclabs/sonic-admin/issues/382
            let manager = Arc::new(CachedNodeManager::new(1_000_000, storage, is_pinned));
            Ok(Box::new(CarmenS6FileBasedDb::new(
                manager.clone(),
                VerkleTrieCarmenState::<ManagedVerkleTrie<_>>::try_new(manager, StateMode::Live)?,
            )))
        }
        (b"file", b"file") => {
            let archive_dir = directory.join("archive");
            let storage = VerkleStorage::open(&archive_dir, db_open_mode)?;
            let is_pinned = |node: &VerkleNode| !node.get_commitment().is_clean();
            // TODO: The cache size is arbitrary, base this on a configurable memory limit instead
            // https://github.com/0xsoniclabs/sonic-admin/issues/382
            let manager = Arc::new(CachedNodeManager::new(1_000_000, storage, is_pinned));
            Ok(Box::new(CarmenS6FileBasedDb::new(
                manager.clone(),
                VerkleTrieCarmenState::<ManagedVerkleTrie<_>>::try_new(manager, StateMode::EvolvingArchive)?
            )))
        }
        _ => Err(Error::UnsupportedImplementation(format!(
            "the combination of live implementation `{}` and archive implementation `{}` is not supported",
            String::from_utf8_lossy(live_impl),
            String::from_utf8_lossy(archive_impl)
        ))
        .into()),
    }
}

/// The safe Carmen database interface.
/// This is the safe interface which gets called from the exported FFI functions.
#[cfg_attr(test, mockall::automock, allow(clippy::disallowed_types))]
pub trait CarmenDb: Send + Sync {
    /// Creates a new checkpoint by persisting all state information to disk to guarantee permanent
    /// storage.
    fn checkpoint(&self) -> BTResult<(), Error>;

    /// Closes this database, releasing all resources and causing its destruction.
    fn close(self: Box<Self>) -> BTResult<(), Error>;

    /// Returns a handle to the live state. The resulting state must be released and must not
    /// outlive the life time of the database.
    fn get_live_state(&self) -> BTResult<Box<dyn CarmenState>, Error>;

    /// Returns a handle to an archive state reflecting the state at the given block height. The
    /// resulting state must be released and must not outlive the life time of the
    /// provided state.
    fn get_archive_state(&self, block: u64) -> BTResult<Box<dyn CarmenState>, Error>;

    /// Retrieves the last block number of the blockchain.
    fn get_archive_block_height(&self) -> BTResult<Option<u64>, Error>;

    /// Returns a summary of the used memory.
    fn get_memory_footprint(&self) -> BTResult<Box<str>, Error>;
}

/// The safe Carmen state interface.
/// This is the safe interface which gets called from the exported FFI functions.
#[cfg_attr(test, mockall::automock, allow(clippy::disallowed_types))]
pub trait CarmenState: Send + Sync {
    /// Checks if the given account exists.
    fn account_exists(&self, addr: &Address) -> BTResult<bool, Error>;

    /// Returns the balance of the given account.
    fn get_balance(&self, addr: &Address) -> BTResult<U256, Error>;

    /// Returns the nonce of the given account.
    fn get_nonce(&self, addr: &Address) -> BTResult<Nonce, Error>;

    /// Returns the value of storage location (addr,key) in the given state.
    fn get_storage_value(&self, addr: &Address, key: &Key) -> BTResult<Value, Error>;

    /// Retrieves the code stored under the given address and stores it in `code_buf`.
    /// Returns the number of bytes written to `code_buf`.
    fn get_code(&self, addr: &Address, code_buf: &mut [MaybeUninit<u8>]) -> BTResult<usize, Error>;

    /// Returns the hash of the code stored under the given address.
    fn get_code_hash(&self, addr: &Address) -> BTResult<Hash, Error>;

    /// Returns the code length stored under the given address.
    fn get_code_len(&self, addr: &Address) -> BTResult<u32, Error>;

    /// Returns a global state hash of the given state.
    fn get_hash(&self) -> BTResult<Hash, Error>;

    /// Applies the provided block update to the maintained state.
    #[allow(clippy::needless_lifetimes)] // using an elided lifetime here breaks automock
    fn apply_block_update<'u>(&self, block: u64, update: Update<'u>) -> BTResult<(), Error>;
}

pub trait IsArchive {
    /// Returns true if this is an archive state.
    fn is_archive(&self) -> bool;
}

/// An implementation of [`CarmenState`] for `Arc<T>` where `T: CarmenState`,
/// required so we can hand out multiple references to a single state instance
/// on [`CarmenDb::get_live_state`].
impl<T: CarmenState> CarmenState for Arc<T> {
    fn account_exists(&self, addr: &Address) -> BTResult<bool, Error> {
        self.deref().account_exists(addr)
    }

    fn get_balance(&self, addr: &Address) -> BTResult<U256, Error> {
        self.deref().get_balance(addr)
    }

    fn get_nonce(&self, addr: &Address) -> BTResult<Nonce, Error> {
        self.deref().get_nonce(addr)
    }

    fn get_storage_value(&self, addr: &Address, key: &Key) -> BTResult<Value, Error> {
        self.deref().get_storage_value(addr, key)
    }

    fn get_code(&self, addr: &Address, code_buf: &mut [MaybeUninit<u8>]) -> BTResult<usize, Error> {
        self.deref().get_code(addr, code_buf)
    }

    fn get_code_hash(&self, addr: &Address) -> BTResult<Hash, Error> {
        self.deref().get_code_hash(addr)
    }

    fn get_code_len(&self, addr: &Address) -> BTResult<u32, Error> {
        self.deref().get_code_len(addr)
    }

    fn get_hash(&self) -> BTResult<Hash, Error> {
        self.deref().get_hash()
    }

    #[allow(clippy::needless_lifetimes)]
    fn apply_block_update<'u>(&self, block: u64, update: Update<'u>) -> BTResult<(), Error> {
        self.deref().apply_block_update(block, update)
    }
}

/// An in-memory `S6` implementation of [`CarmenDb`].
///
/// Does not support closing or checkpointing.
pub struct CarmenS6InMemoryDb<LS: CarmenState> {
    live_state: Arc<LS>,
}

impl<LS: CarmenState> CarmenS6InMemoryDb<LS> {
    /// Creates a new [CarmenS6InMemoryDb] with the provided live state.
    /// The live state is expected to be an in-memory implementation.
    /// No lifecycle methods for closing or checkpointing will be invoked.
    pub fn new(live_state: LS) -> Self {
        Self {
            live_state: Arc::new(live_state),
        }
    }
}

impl<LS: CarmenState + 'static> CarmenDb for CarmenS6InMemoryDb<LS> {
    fn checkpoint(&self) -> BTResult<(), Error> {
        // No-op for in-memory state
        Ok(())
    }

    fn close(self: Box<Self>) -> BTResult<(), Error> {
        // No-op for in-memory state
        Ok(())
    }

    fn get_live_state(&self) -> BTResult<Box<dyn CarmenState>, Error> {
        Ok(Box::new(self.live_state.clone()))
    }

    fn get_archive_state(&self, _block: u64) -> BTResult<Box<dyn CarmenState>, Error> {
        unimplemented!()
    }

    fn get_archive_block_height(&self) -> BTResult<Option<u64>, Error> {
        Err(Error::UnsupportedOperation(
            "get_archive_block_height is not supported for in-memory databases".to_string(),
        )
        .into())
    }

    fn get_memory_footprint(&self) -> BTResult<Box<str>, Error> {
        Err(
            Error::UnsupportedOperation("get_memory_footprint is not yet implemented".to_string())
                .into(),
        )
    }
}

/// A file-based `S6` implementation of [`CarmenDb`].
pub struct CarmenS6FileBasedDb<S: Storage, LS: CarmenState> {
    manager: Arc<CachedNodeManager<S>>,
    live_state: Arc<LS>,
}

impl<S: Storage, LS: CarmenState> CarmenS6FileBasedDb<S, LS> {
    /// Creates a new [`CarmenS6FileBasedDb`] with the provided node manager and live state.
    pub fn new(manager: Arc<CachedNodeManager<S>>, live_state: LS) -> Self {
        Self {
            manager,
            live_state: Arc::new(live_state),
        }
    }
}

impl<S, LS> CarmenDb for CarmenS6FileBasedDb<S, LS>
where
    S: Storage<Id = VerkleNodeId, Item = VerkleNode> + RootIdProvider<Id = VerkleNodeId> + 'static,
    LS: CarmenState + IsArchive + 'static,
{
    fn checkpoint(&self) -> BTResult<(), Error> {
        // TODO: Support checkpoints for archive
        Err(
            Error::UnsupportedOperation("cannot create checkpoint for live state".to_owned())
                .into(),
        )
    }

    fn close(self: Box<Self>) -> BTResult<(), Error> {
        // Ensure that we have no dirty commitments before flushing to disk
        self.live_state.get_hash()?;

        // Release live state first, since it holds a reference to the manager
        drop(self.live_state);
        let manager = Arc::into_inner(self.manager).ok_or_else(|| {
            Error::CorruptedState("node manager reference count is not 1 on close".to_owned())
        })?;
        manager.close()?;

        SPECIALIZATION_TRANSACTIONS.lock().unwrap().print(&mut std::io::stdout())?; 

        Ok(())
    }

    fn get_live_state(&self) -> BTResult<Box<dyn CarmenState>, Error> {
        Ok(Box::new(self.live_state.clone()))
    }

    fn get_archive_state(&self, block: u64) -> BTResult<Box<dyn CarmenState>, Error> {
        if !self.live_state.is_archive() {
            return Err(Error::UnsupportedOperation(
                "creating an archive state failed: the database was opened in live only mode"
                    .into(),
            )
            .into());
        }
        Ok(Box::new(Arc::new(VerkleTrieCarmenState::<_>::try_new(
            self.manager.clone(),
            StateMode::Archive(block),
        )?)))
    }

    fn get_archive_block_height(&self) -> BTResult<Option<u64>, Error> {
        if !self.live_state.is_archive() {
            return Err(Error::UnsupportedOperation(
                "get_archive_block_height is not supported for live only databases".to_string(),
            )
            .into());
        }
        Ok(self.manager.highest_block_number()?)
    }

    fn get_memory_footprint(&self) -> BTResult<Box<str>, Error> {
        Err(
            Error::UnsupportedOperation("get_memory_footprint is not yet implemented".to_string())
                .into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use crypto_bigint::U256;
    use zerocopy::transmute;

    use super::*;
    use crate::utils::test_dir::{Permissions, TestDir};

    #[rstest_reuse::template]
    #[rstest::rstest]
    #[case::live(b"none")]
    #[case::archive(b"file")]
    fn archive_impl(#[case] archive_impl: &[u8]) {}

    #[rstest_reuse::apply(archive_impl)]
    fn file_based_verkle_trie_implementation_supports_closing_and_reopening(
        #[case] archive_impl: &[u8],
        #[values(DbOpenMode::ReadOnly, DbOpenMode::ReadWrite)] db_open_mode: DbOpenMode,
    ) {
        // This test writes to 512 leaf nodes. In two leaf nodes only one slot gets set, in two leaf
        // nodes two slots get set and so on.
        // This makes sure that no matter the variants of sparse leaf nodes that are used for
        // storage optimization, there will always be at least two nodes for each variant.

        // Skip the first 256 indices to avoid special casing in embedding, where the first leaf
        // only stores 64 values, and the second second 192.
        let key_indices_offset: u16 = 256;

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let db =
            open_carmen_db(6, b"file", archive_impl, dir.path(), DbOpenMode::ReadWrite).unwrap();

        let mut slot_updates = Vec::new();
        for address_idx in 0..256 * 2 {
            for key_idx in key_indices_offset..=key_indices_offset + address_idx {
                let mut addr = [0; 20];
                addr[..2].copy_from_slice(&address_idx.to_be_bytes());
                let key = U256::from(key_idx);
                slot_updates.push(SlotUpdate {
                    addr,
                    key: key.to_be_bytes(),
                    value: key.to_be_bytes(),
                });
            }
        }
        let update = Update {
            slots: &slot_updates,
            ..Default::default()
        };

        db.get_live_state()
            .unwrap()
            .apply_block_update(0, update)
            .unwrap();

        db.close().unwrap();

        dir.set_permissions(db_open_mode.to_permissions()).unwrap();
        let db = open_carmen_db(6, b"file", archive_impl, &dir, db_open_mode).unwrap();
        let live = db.get_live_state().unwrap();
        for address_idx in 0..2 * 256 {
            for key_idx in key_indices_offset..=key_indices_offset + address_idx {
                let mut addr = [0; 20];
                addr[..2].copy_from_slice(&address_idx.to_be_bytes());
                let key = U256::from(key_idx);
                assert_eq!(
                    live.get_storage_value(&addr, &key.to_be_bytes()).unwrap(),
                    key.to_be_bytes()
                );
            }
        }
    }

    #[test]
    fn file_based_verkle_trie_implementation_supports_archive_state_semantics() {
        let addr = [1; 20];
        let balance1 = transmute!([[0u8; 16], [2; 16]]);
        let balance2 = transmute!([[0u8; 16], [3; 16]]);

        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let db = open_carmen_db(6, b"file", b"file", &dir, DbOpenMode::ReadWrite).unwrap();
        let live_state = db.get_live_state().unwrap();

        live_state
            .apply_block_update(
                0,
                Update {
                    balances: &[BalanceUpdate {
                        addr,
                        balance: balance1,
                    }],
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(live_state.get_balance(&addr).unwrap(), balance1);

        live_state
            .apply_block_update(
                1,
                Update {
                    balances: &[BalanceUpdate {
                        addr,
                        balance: balance2,
                    }],
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(live_state.get_balance(&addr).unwrap(), balance2);

        let archive_state = db.get_archive_state(0).unwrap();
        assert_eq!(archive_state.get_balance(&addr).unwrap(), balance1);
        let archive_state = db.get_archive_state(1).unwrap();
        assert_eq!(archive_state.get_balance(&addr).unwrap(), balance2);
    }

    #[rstest_reuse::apply(archive_impl)]
    fn carmen_s6_file_based_db_checkpoint_returns_error(#[case] archive_impl: &[u8]) {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let db = open_carmen_db(6, b"file", archive_impl, &dir, DbOpenMode::ReadWrite).unwrap();

        let result = db.checkpoint();
        assert_eq!(
            result,
            Err(
                Error::UnsupportedOperation("cannot create checkpoint for live state".to_owned())
                    .into()
            )
        );
    }

    #[rstest_reuse::apply(archive_impl)]
    fn carmen_s6_file_based_db_close_fails_if_node_manager_refcount_not_one(
        #[case] archive_impl: &[u8],
    ) {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let db = open_carmen_db(6, b"file", archive_impl, &dir, DbOpenMode::ReadWrite).unwrap();
        let _live_state = db.get_live_state().unwrap();

        let result = db.close();
        assert_eq!(
            result,
            Err(
                Error::CorruptedState("node manager reference count is not 1 on close".to_owned())
                    .into()
            )
        );
    }

    #[test]
    fn carmen_s6_file_based_db_get_archive_block_height_fails_in_live_only_mode() {
        let dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let db = open_carmen_db(6, b"file", b"none", &dir, DbOpenMode::ReadWrite).unwrap();
        let result = db.get_archive_block_height();
        assert_eq!(
            result,
            Err(Error::UnsupportedOperation(
                "get_archive_block_height is not supported for live only databases".to_string(),
            )
            .into())
        );
    }
}
