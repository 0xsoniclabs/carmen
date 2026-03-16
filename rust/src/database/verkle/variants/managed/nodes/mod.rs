// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{array, ops::Deref};

use derive_deftly::Deftly;
use zerocopy::{FromBytes, Immutable, IntoBytes, Unaligned};

use crate::{
    database::{
        managed_trie::{LookupResult, ManagedTrieNode, StoreAction, UnionManagedTrieNode},
        verkle::{
            KeyedUpdate, KeyedUpdateBatch,
            variants::managed::{
                VerkleNodeId,
                commitment::{
                    VerkleCommitment, VerkleCommitmentInput, VerkleInnerCommitment,
                    VerkleLeafCommitment,
                },
                nodes::{
                    empty::EmptyNode, inner::FullInnerNode, inner_delta::InnerDeltaNode,
                    leaf::FullLeafNode, leaf_delta::LeafDeltaNode, sparse_inner::SparseInnerNode,
                    sparse_leaf::SparseLeafNode,
                },
            },
        },
        visitor::NodeVisitor,
    },
    error::{BTResult, Error},
    node_manager::NodeManager,
    statistics::node_count::NodeCountVisitor,
    storage::file::derive_deftly_template_FileStorageManager,
    types::{HasDeltaVariant, HasEmptyNode, Key, NodeSize, ToNodeKind, Value},
};

pub mod empty;
pub mod id;
pub mod inner;
pub mod inner_delta;
pub mod leaf;
pub mod leaf_delta;
pub mod sparse_inner;
pub mod sparse_leaf;

#[cfg(test)]
pub use tests::{NodeAccess, VerkleManagedTrieNode};

/// A node in a managed Verkle trie.
//
/// Non-empty nodes are stored as boxed to save memory (otherwise the size of the enum would be
/// dictated by the largest variant).
#[derive(Debug, Clone, PartialEq, Eq, Deftly)]
#[derive_deftly(FileStorageManager)]
pub enum VerkleNode {
    Empty(EmptyVerkleNode),
    Inner9(Box<Inner9VerkleNode>),
    Inner15(Box<Inner15VerkleNode>),
    Inner21(Box<Inner21VerkleNode>),
    Inner256(Box<Inner256VerkleNode>),
    InnerDelta(Box<InnerDeltaVerkleNode>),
    Leaf1(Box<Leaf1VerkleNode>),
    Leaf2(Box<Leaf2VerkleNode>),
    Leaf3(Box<Leaf3VerkleNode>),
    Leaf4(Box<Leaf4VerkleNode>),
    Leaf5(Box<Leaf5VerkleNode>),
    Leaf6(Box<Leaf6VerkleNode>),
    Leaf7(Box<Leaf7VerkleNode>),
    Leaf8(Box<Leaf8VerkleNode>),
    Leaf9(Box<Leaf9VerkleNode>),
    Leaf10(Box<Leaf10VerkleNode>),
    Leaf11(Box<Leaf11VerkleNode>),
    Leaf12(Box<Leaf12VerkleNode>),
    Leaf13(Box<Leaf13VerkleNode>),
    Leaf14(Box<Leaf14VerkleNode>),
    Leaf15(Box<Leaf15VerkleNode>),
    Leaf16(Box<Leaf16VerkleNode>),
    Leaf18(Box<Leaf18VerkleNode>),
    Leaf20(Box<Leaf20VerkleNode>),
    Leaf23(Box<Leaf23VerkleNode>),
    Leaf26(Box<Leaf26VerkleNode>),
    Leaf29(Box<Leaf29VerkleNode>),
    Leaf33(Box<Leaf33VerkleNode>),
    Leaf36(Box<Leaf36VerkleNode>),
    Leaf39(Box<Leaf39VerkleNode>),
    Leaf40(Box<Leaf40VerkleNode>),
    Leaf41(Box<Leaf41VerkleNode>),
    Leaf45(Box<Leaf45VerkleNode>),
    Leaf48(Box<Leaf48VerkleNode>),
    Leaf52(Box<Leaf52VerkleNode>),
    Leaf57(Box<Leaf57VerkleNode>),
    Leaf61(Box<Leaf61VerkleNode>),
    Leaf64(Box<Leaf64VerkleNode>),
    Leaf70(Box<Leaf70VerkleNode>),
    Leaf75(Box<Leaf75VerkleNode>),
    Leaf80(Box<Leaf80VerkleNode>),
    Leaf84(Box<Leaf84VerkleNode>),
    Leaf89(Box<Leaf89VerkleNode>),
    Leaf94(Box<Leaf94VerkleNode>),
    Leaf100(Box<Leaf100VerkleNode>),
    Leaf107(Box<Leaf107VerkleNode>),
    Leaf113(Box<Leaf113VerkleNode>),
    Leaf120(Box<Leaf120VerkleNode>),
    Leaf124(Box<Leaf124VerkleNode>),
    Leaf130(Box<Leaf130VerkleNode>),
    Leaf132(Box<Leaf132VerkleNode>),
    Leaf134(Box<Leaf134VerkleNode>),
    Leaf136(Box<Leaf136VerkleNode>),
    Leaf138(Box<Leaf138VerkleNode>),
    Leaf140(Box<Leaf140VerkleNode>),
    Leaf142(Box<Leaf142VerkleNode>),
    Leaf146(Box<Leaf146VerkleNode>),
    Leaf149(Box<Leaf149VerkleNode>),
    Leaf156(Box<Leaf156VerkleNode>),
    Leaf162(Box<Leaf162VerkleNode>),
    Leaf168(Box<Leaf168VerkleNode>),
    Leaf178(Box<Leaf178VerkleNode>),
    Leaf186(Box<Leaf186VerkleNode>),
    Leaf195(Box<Leaf195VerkleNode>),
    Leaf203(Box<Leaf203VerkleNode>),
    Leaf208(Box<Leaf208VerkleNode>),
    Leaf225(Box<Leaf225VerkleNode>),
    Leaf239(Box<Leaf239VerkleNode>),
    Leaf256(Box<Leaf256VerkleNode>),
    LeafDelta(Box<LeafDeltaVerkleNode>),
}

type EmptyVerkleNode = EmptyNode;
type Inner1VerkleNode = SparseInnerNode<1>;
type Inner2VerkleNode = SparseInnerNode<2>;
type Inner3VerkleNode = SparseInnerNode<3>;
type Inner4VerkleNode = SparseInnerNode<4>;
type Inner5VerkleNode = SparseInnerNode<5>;
type Inner6VerkleNode = SparseInnerNode<6>;
type Inner7VerkleNode = SparseInnerNode<7>;
type Inner8VerkleNode = SparseInnerNode<8>;
type Inner9VerkleNode = SparseInnerNode<9>;
type Inner10VerkleNode = SparseInnerNode<10>;
type Inner11VerkleNode = SparseInnerNode<11>;
type Inner12VerkleNode = SparseInnerNode<12>;
type Inner13VerkleNode = SparseInnerNode<13>;
type Inner14VerkleNode = SparseInnerNode<14>;
type Inner15VerkleNode = SparseInnerNode<15>;
type Inner16VerkleNode = SparseInnerNode<16>;
type Inner17VerkleNode = SparseInnerNode<17>;
type Inner18VerkleNode = SparseInnerNode<18>;
type Inner19VerkleNode = SparseInnerNode<19>;
type Inner20VerkleNode = SparseInnerNode<20>;
type Inner21VerkleNode = SparseInnerNode<21>;
type Inner22VerkleNode = SparseInnerNode<22>;
type Inner23VerkleNode = SparseInnerNode<23>;
type Inner24VerkleNode = SparseInnerNode<24>;
type Inner25VerkleNode = SparseInnerNode<25>;
type Inner26VerkleNode = SparseInnerNode<26>;
type Inner27VerkleNode = SparseInnerNode<27>;
type Inner28VerkleNode = SparseInnerNode<28>;
type Inner29VerkleNode = SparseInnerNode<29>;
type Inner30VerkleNode = SparseInnerNode<30>;
type Inner31VerkleNode = SparseInnerNode<31>;
type Inner32VerkleNode = SparseInnerNode<32>;
type Inner33VerkleNode = SparseInnerNode<33>;
type Inner34VerkleNode = SparseInnerNode<34>;
type Inner35VerkleNode = SparseInnerNode<35>;
type Inner36VerkleNode = SparseInnerNode<36>;
type Inner37VerkleNode = SparseInnerNode<37>;
type Inner38VerkleNode = SparseInnerNode<38>;
type Inner39VerkleNode = SparseInnerNode<39>;
type Inner40VerkleNode = SparseInnerNode<40>;
type Inner41VerkleNode = SparseInnerNode<41>;
type Inner42VerkleNode = SparseInnerNode<42>;
type Inner43VerkleNode = SparseInnerNode<43>;
type Inner44VerkleNode = SparseInnerNode<44>;
type Inner45VerkleNode = SparseInnerNode<45>;
type Inner46VerkleNode = SparseInnerNode<46>;
type Inner47VerkleNode = SparseInnerNode<47>;
type Inner48VerkleNode = SparseInnerNode<48>;
type Inner49VerkleNode = SparseInnerNode<49>;
type Inner50VerkleNode = SparseInnerNode<50>;
type Inner51VerkleNode = SparseInnerNode<51>;
type Inner52VerkleNode = SparseInnerNode<52>;
type Inner53VerkleNode = SparseInnerNode<53>;
type Inner54VerkleNode = SparseInnerNode<54>;
type Inner55VerkleNode = SparseInnerNode<55>;
type Inner56VerkleNode = SparseInnerNode<56>;
type Inner57VerkleNode = SparseInnerNode<57>;
type Inner58VerkleNode = SparseInnerNode<58>;
type Inner59VerkleNode = SparseInnerNode<59>;
type Inner60VerkleNode = SparseInnerNode<60>;
type Inner61VerkleNode = SparseInnerNode<61>;
type Inner62VerkleNode = SparseInnerNode<62>;
type Inner63VerkleNode = SparseInnerNode<63>;
type Inner64VerkleNode = SparseInnerNode<64>;
type Inner65VerkleNode = SparseInnerNode<65>;
type Inner66VerkleNode = SparseInnerNode<66>;
type Inner67VerkleNode = SparseInnerNode<67>;
type Inner68VerkleNode = SparseInnerNode<68>;
type Inner69VerkleNode = SparseInnerNode<69>;
type Inner70VerkleNode = SparseInnerNode<70>;
type Inner71VerkleNode = SparseInnerNode<71>;
type Inner72VerkleNode = SparseInnerNode<72>;
type Inner73VerkleNode = SparseInnerNode<73>;
type Inner74VerkleNode = SparseInnerNode<74>;
type Inner75VerkleNode = SparseInnerNode<75>;
type Inner76VerkleNode = SparseInnerNode<76>;
type Inner77VerkleNode = SparseInnerNode<77>;
type Inner78VerkleNode = SparseInnerNode<78>;
type Inner79VerkleNode = SparseInnerNode<79>;
type Inner80VerkleNode = SparseInnerNode<80>;
type Inner81VerkleNode = SparseInnerNode<81>;
type Inner82VerkleNode = SparseInnerNode<82>;
type Inner83VerkleNode = SparseInnerNode<83>;
type Inner84VerkleNode = SparseInnerNode<84>;
type Inner85VerkleNode = SparseInnerNode<85>;
type Inner86VerkleNode = SparseInnerNode<86>;
type Inner87VerkleNode = SparseInnerNode<87>;
type Inner88VerkleNode = SparseInnerNode<88>;
type Inner89VerkleNode = SparseInnerNode<89>;
type Inner90VerkleNode = SparseInnerNode<90>;
type Inner91VerkleNode = SparseInnerNode<91>;
type Inner92VerkleNode = SparseInnerNode<92>;
type Inner93VerkleNode = SparseInnerNode<93>;
type Inner94VerkleNode = SparseInnerNode<94>;
type Inner95VerkleNode = SparseInnerNode<95>;
type Inner96VerkleNode = SparseInnerNode<96>;
type Inner97VerkleNode = SparseInnerNode<97>;
type Inner98VerkleNode = SparseInnerNode<98>;
type Inner99VerkleNode = SparseInnerNode<99>;
type Inner100VerkleNode = SparseInnerNode<100>;
type Inner101VerkleNode = SparseInnerNode<101>;
type Inner102VerkleNode = SparseInnerNode<102>;
type Inner103VerkleNode = SparseInnerNode<103>;
type Inner104VerkleNode = SparseInnerNode<104>;
type Inner105VerkleNode = SparseInnerNode<105>;
type Inner106VerkleNode = SparseInnerNode<106>;
type Inner107VerkleNode = SparseInnerNode<107>;
type Inner108VerkleNode = SparseInnerNode<108>;
type Inner109VerkleNode = SparseInnerNode<109>;
type Inner110VerkleNode = SparseInnerNode<110>;
type Inner111VerkleNode = SparseInnerNode<111>;
type Inner112VerkleNode = SparseInnerNode<112>;
type Inner113VerkleNode = SparseInnerNode<113>;
type Inner114VerkleNode = SparseInnerNode<114>;
type Inner115VerkleNode = SparseInnerNode<115>;
type Inner116VerkleNode = SparseInnerNode<116>;
type Inner117VerkleNode = SparseInnerNode<117>;
type Inner118VerkleNode = SparseInnerNode<118>;
type Inner119VerkleNode = SparseInnerNode<119>;
type Inner120VerkleNode = SparseInnerNode<120>;
type Inner121VerkleNode = SparseInnerNode<121>;
type Inner122VerkleNode = SparseInnerNode<122>;
type Inner123VerkleNode = SparseInnerNode<123>;
type Inner124VerkleNode = SparseInnerNode<124>;
type Inner125VerkleNode = SparseInnerNode<125>;
type Inner126VerkleNode = SparseInnerNode<126>;
type Inner127VerkleNode = SparseInnerNode<127>;
type Inner128VerkleNode = SparseInnerNode<128>;
type Inner129VerkleNode = SparseInnerNode<129>;
type Inner130VerkleNode = SparseInnerNode<130>;
type Inner131VerkleNode = SparseInnerNode<131>;
type Inner132VerkleNode = SparseInnerNode<132>;
type Inner133VerkleNode = SparseInnerNode<133>;
type Inner134VerkleNode = SparseInnerNode<134>;
type Inner135VerkleNode = SparseInnerNode<135>;
type Inner136VerkleNode = SparseInnerNode<136>;
type Inner137VerkleNode = SparseInnerNode<137>;
type Inner138VerkleNode = SparseInnerNode<138>;
type Inner139VerkleNode = SparseInnerNode<139>;
type Inner140VerkleNode = SparseInnerNode<140>;
type Inner141VerkleNode = SparseInnerNode<141>;
type Inner142VerkleNode = SparseInnerNode<142>;
type Inner143VerkleNode = SparseInnerNode<143>;
type Inner144VerkleNode = SparseInnerNode<144>;
type Inner145VerkleNode = SparseInnerNode<145>;
type Inner146VerkleNode = SparseInnerNode<146>;
type Inner147VerkleNode = SparseInnerNode<147>;
type Inner148VerkleNode = SparseInnerNode<148>;
type Inner149VerkleNode = SparseInnerNode<149>;
type Inner150VerkleNode = SparseInnerNode<150>;
type Inner151VerkleNode = SparseInnerNode<151>;
type Inner152VerkleNode = SparseInnerNode<152>;
type Inner153VerkleNode = SparseInnerNode<153>;
type Inner154VerkleNode = SparseInnerNode<154>;
type Inner155VerkleNode = SparseInnerNode<155>;
type Inner156VerkleNode = SparseInnerNode<156>;
type Inner157VerkleNode = SparseInnerNode<157>;
type Inner158VerkleNode = SparseInnerNode<158>;
type Inner159VerkleNode = SparseInnerNode<159>;
type Inner160VerkleNode = SparseInnerNode<160>;
type Inner161VerkleNode = SparseInnerNode<161>;
type Inner162VerkleNode = SparseInnerNode<162>;
type Inner163VerkleNode = SparseInnerNode<163>;
type Inner164VerkleNode = SparseInnerNode<164>;
type Inner165VerkleNode = SparseInnerNode<165>;
type Inner166VerkleNode = SparseInnerNode<166>;
type Inner167VerkleNode = SparseInnerNode<167>;
type Inner168VerkleNode = SparseInnerNode<168>;
type Inner169VerkleNode = SparseInnerNode<169>;
type Inner170VerkleNode = SparseInnerNode<170>;
type Inner171VerkleNode = SparseInnerNode<171>;
type Inner172VerkleNode = SparseInnerNode<172>;
type Inner173VerkleNode = SparseInnerNode<173>;
type Inner174VerkleNode = SparseInnerNode<174>;
type Inner175VerkleNode = SparseInnerNode<175>;
type Inner176VerkleNode = SparseInnerNode<176>;
type Inner177VerkleNode = SparseInnerNode<177>;
type Inner178VerkleNode = SparseInnerNode<178>;
type Inner179VerkleNode = SparseInnerNode<179>;
type Inner180VerkleNode = SparseInnerNode<180>;
type Inner181VerkleNode = SparseInnerNode<181>;
type Inner182VerkleNode = SparseInnerNode<182>;
type Inner183VerkleNode = SparseInnerNode<183>;
type Inner184VerkleNode = SparseInnerNode<184>;
type Inner185VerkleNode = SparseInnerNode<185>;
type Inner186VerkleNode = SparseInnerNode<186>;
type Inner187VerkleNode = SparseInnerNode<187>;
type Inner188VerkleNode = SparseInnerNode<188>;
type Inner189VerkleNode = SparseInnerNode<189>;
type Inner190VerkleNode = SparseInnerNode<190>;
type Inner191VerkleNode = SparseInnerNode<191>;
type Inner192VerkleNode = SparseInnerNode<192>;
type Inner193VerkleNode = SparseInnerNode<193>;
type Inner194VerkleNode = SparseInnerNode<194>;
type Inner195VerkleNode = SparseInnerNode<195>;
type Inner196VerkleNode = SparseInnerNode<196>;
type Inner197VerkleNode = SparseInnerNode<197>;
type Inner198VerkleNode = SparseInnerNode<198>;
type Inner199VerkleNode = SparseInnerNode<199>;
type Inner200VerkleNode = SparseInnerNode<200>;
type Inner201VerkleNode = SparseInnerNode<201>;
type Inner202VerkleNode = SparseInnerNode<202>;
type Inner203VerkleNode = SparseInnerNode<203>;
type Inner204VerkleNode = SparseInnerNode<204>;
type Inner205VerkleNode = SparseInnerNode<205>;
type Inner206VerkleNode = SparseInnerNode<206>;
type Inner207VerkleNode = SparseInnerNode<207>;
type Inner208VerkleNode = SparseInnerNode<208>;
type Inner209VerkleNode = SparseInnerNode<209>;
type Inner210VerkleNode = SparseInnerNode<210>;
type Inner211VerkleNode = SparseInnerNode<211>;
type Inner212VerkleNode = SparseInnerNode<212>;
type Inner213VerkleNode = SparseInnerNode<213>;
type Inner214VerkleNode = SparseInnerNode<214>;
type Inner215VerkleNode = SparseInnerNode<215>;
type Inner216VerkleNode = SparseInnerNode<216>;
type Inner217VerkleNode = SparseInnerNode<217>;
type Inner218VerkleNode = SparseInnerNode<218>;
type Inner219VerkleNode = SparseInnerNode<219>;
type Inner220VerkleNode = SparseInnerNode<220>;
type Inner221VerkleNode = SparseInnerNode<221>;
type Inner222VerkleNode = SparseInnerNode<222>;
type Inner223VerkleNode = SparseInnerNode<223>;
type Inner224VerkleNode = SparseInnerNode<224>;
type Inner225VerkleNode = SparseInnerNode<225>;
type Inner226VerkleNode = SparseInnerNode<226>;
type Inner227VerkleNode = SparseInnerNode<227>;
type Inner228VerkleNode = SparseInnerNode<228>;
type Inner229VerkleNode = SparseInnerNode<229>;
type Inner230VerkleNode = SparseInnerNode<230>;
type Inner231VerkleNode = SparseInnerNode<231>;
type Inner232VerkleNode = SparseInnerNode<232>;
type Inner233VerkleNode = SparseInnerNode<233>;
type Inner234VerkleNode = SparseInnerNode<234>;
type Inner235VerkleNode = SparseInnerNode<235>;
type Inner236VerkleNode = SparseInnerNode<236>;
type Inner237VerkleNode = SparseInnerNode<237>;
type Inner238VerkleNode = SparseInnerNode<238>;
type Inner239VerkleNode = SparseInnerNode<239>;
type Inner240VerkleNode = SparseInnerNode<240>;
type Inner241VerkleNode = SparseInnerNode<241>;
type Inner242VerkleNode = SparseInnerNode<242>;
type Inner243VerkleNode = SparseInnerNode<243>;
type Inner244VerkleNode = SparseInnerNode<244>;
type Inner245VerkleNode = SparseInnerNode<245>;
type Inner246VerkleNode = SparseInnerNode<246>;
type Inner247VerkleNode = SparseInnerNode<247>;
type Inner248VerkleNode = SparseInnerNode<248>;
type Inner249VerkleNode = SparseInnerNode<249>;
type Inner250VerkleNode = SparseInnerNode<250>;
type Inner251VerkleNode = SparseInnerNode<251>;
type Inner252VerkleNode = SparseInnerNode<252>;
type Inner253VerkleNode = SparseInnerNode<253>;
type Inner254VerkleNode = SparseInnerNode<254>;
type Inner255VerkleNode = SparseInnerNode<255>;
type Inner256VerkleNode = FullInnerNode;
type InnerDeltaVerkleNode = InnerDeltaNode;
type Leaf1VerkleNode = SparseLeafNode<1>;
type Leaf2VerkleNode = SparseLeafNode<2>;
type Leaf3VerkleNode = SparseLeafNode<3>;
type Leaf4VerkleNode = SparseLeafNode<4>;
type Leaf5VerkleNode = SparseLeafNode<5>;
type Leaf6VerkleNode = SparseLeafNode<6>;
type Leaf7VerkleNode = SparseLeafNode<7>;
type Leaf8VerkleNode = SparseLeafNode<8>;
type Leaf9VerkleNode = SparseLeafNode<9>;
type Leaf10VerkleNode = SparseLeafNode<10>;
type Leaf11VerkleNode = SparseLeafNode<11>;
type Leaf12VerkleNode = SparseLeafNode<12>;
type Leaf13VerkleNode = SparseLeafNode<13>;
type Leaf14VerkleNode = SparseLeafNode<14>;
type Leaf15VerkleNode = SparseLeafNode<15>;
type Leaf16VerkleNode = SparseLeafNode<16>;
type Leaf17VerkleNode = SparseLeafNode<17>;
type Leaf18VerkleNode = SparseLeafNode<18>;
type Leaf19VerkleNode = SparseLeafNode<19>;
type Leaf20VerkleNode = SparseLeafNode<20>;
type Leaf21VerkleNode = SparseLeafNode<21>;
type Leaf22VerkleNode = SparseLeafNode<22>;
type Leaf23VerkleNode = SparseLeafNode<23>;
type Leaf24VerkleNode = SparseLeafNode<24>;
type Leaf25VerkleNode = SparseLeafNode<25>;
type Leaf26VerkleNode = SparseLeafNode<26>;
type Leaf27VerkleNode = SparseLeafNode<27>;
type Leaf28VerkleNode = SparseLeafNode<28>;
type Leaf29VerkleNode = SparseLeafNode<29>;
type Leaf30VerkleNode = SparseLeafNode<30>;
type Leaf31VerkleNode = SparseLeafNode<31>;
type Leaf32VerkleNode = SparseLeafNode<32>;
type Leaf33VerkleNode = SparseLeafNode<33>;
type Leaf34VerkleNode = SparseLeafNode<34>;
type Leaf35VerkleNode = SparseLeafNode<35>;
type Leaf36VerkleNode = SparseLeafNode<36>;
type Leaf37VerkleNode = SparseLeafNode<37>;
type Leaf38VerkleNode = SparseLeafNode<38>;
type Leaf39VerkleNode = SparseLeafNode<39>;
type Leaf40VerkleNode = SparseLeafNode<40>;
type Leaf41VerkleNode = SparseLeafNode<41>;
type Leaf42VerkleNode = SparseLeafNode<42>;
type Leaf43VerkleNode = SparseLeafNode<43>;
type Leaf44VerkleNode = SparseLeafNode<44>;
type Leaf45VerkleNode = SparseLeafNode<45>;
type Leaf46VerkleNode = SparseLeafNode<46>;
type Leaf47VerkleNode = SparseLeafNode<47>;
type Leaf48VerkleNode = SparseLeafNode<48>;
type Leaf49VerkleNode = SparseLeafNode<49>;
type Leaf50VerkleNode = SparseLeafNode<50>;
type Leaf51VerkleNode = SparseLeafNode<51>;
type Leaf52VerkleNode = SparseLeafNode<52>;
type Leaf53VerkleNode = SparseLeafNode<53>;
type Leaf54VerkleNode = SparseLeafNode<54>;
type Leaf55VerkleNode = SparseLeafNode<55>;
type Leaf56VerkleNode = SparseLeafNode<56>;
type Leaf57VerkleNode = SparseLeafNode<57>;
type Leaf58VerkleNode = SparseLeafNode<58>;
type Leaf59VerkleNode = SparseLeafNode<59>;
type Leaf60VerkleNode = SparseLeafNode<60>;
type Leaf61VerkleNode = SparseLeafNode<61>;
type Leaf62VerkleNode = SparseLeafNode<62>;
type Leaf63VerkleNode = SparseLeafNode<63>;
type Leaf64VerkleNode = SparseLeafNode<64>;
type Leaf65VerkleNode = SparseLeafNode<65>;
type Leaf66VerkleNode = SparseLeafNode<66>;
type Leaf67VerkleNode = SparseLeafNode<67>;
type Leaf68VerkleNode = SparseLeafNode<68>;
type Leaf69VerkleNode = SparseLeafNode<69>;
type Leaf70VerkleNode = SparseLeafNode<70>;
type Leaf71VerkleNode = SparseLeafNode<71>;
type Leaf72VerkleNode = SparseLeafNode<72>;
type Leaf73VerkleNode = SparseLeafNode<73>;
type Leaf74VerkleNode = SparseLeafNode<74>;
type Leaf75VerkleNode = SparseLeafNode<75>;
type Leaf76VerkleNode = SparseLeafNode<76>;
type Leaf77VerkleNode = SparseLeafNode<77>;
type Leaf78VerkleNode = SparseLeafNode<78>;
type Leaf79VerkleNode = SparseLeafNode<79>;
type Leaf80VerkleNode = SparseLeafNode<80>;
type Leaf81VerkleNode = SparseLeafNode<81>;
type Leaf82VerkleNode = SparseLeafNode<82>;
type Leaf83VerkleNode = SparseLeafNode<83>;
type Leaf84VerkleNode = SparseLeafNode<84>;
type Leaf85VerkleNode = SparseLeafNode<85>;
type Leaf86VerkleNode = SparseLeafNode<86>;
type Leaf87VerkleNode = SparseLeafNode<87>;
type Leaf88VerkleNode = SparseLeafNode<88>;
type Leaf89VerkleNode = SparseLeafNode<89>;
type Leaf90VerkleNode = SparseLeafNode<90>;
type Leaf91VerkleNode = SparseLeafNode<91>;
type Leaf92VerkleNode = SparseLeafNode<92>;
type Leaf93VerkleNode = SparseLeafNode<93>;
type Leaf94VerkleNode = SparseLeafNode<94>;
type Leaf95VerkleNode = SparseLeafNode<95>;
type Leaf96VerkleNode = SparseLeafNode<96>;
type Leaf97VerkleNode = SparseLeafNode<97>;
type Leaf98VerkleNode = SparseLeafNode<98>;
type Leaf99VerkleNode = SparseLeafNode<99>;
type Leaf100VerkleNode = SparseLeafNode<100>;
type Leaf101VerkleNode = SparseLeafNode<101>;
type Leaf102VerkleNode = SparseLeafNode<102>;
type Leaf103VerkleNode = SparseLeafNode<103>;
type Leaf104VerkleNode = SparseLeafNode<104>;
type Leaf105VerkleNode = SparseLeafNode<105>;
type Leaf106VerkleNode = SparseLeafNode<106>;
type Leaf107VerkleNode = SparseLeafNode<107>;
type Leaf108VerkleNode = SparseLeafNode<108>;
type Leaf109VerkleNode = SparseLeafNode<109>;
type Leaf110VerkleNode = SparseLeafNode<110>;
type Leaf111VerkleNode = SparseLeafNode<111>;
type Leaf112VerkleNode = SparseLeafNode<112>;
type Leaf113VerkleNode = SparseLeafNode<113>;
type Leaf114VerkleNode = SparseLeafNode<114>;
type Leaf115VerkleNode = SparseLeafNode<115>;
type Leaf116VerkleNode = SparseLeafNode<116>;
type Leaf117VerkleNode = SparseLeafNode<117>;
type Leaf118VerkleNode = SparseLeafNode<118>;
type Leaf119VerkleNode = SparseLeafNode<119>;
type Leaf120VerkleNode = SparseLeafNode<120>;
type Leaf121VerkleNode = SparseLeafNode<121>;
type Leaf122VerkleNode = SparseLeafNode<122>;
type Leaf123VerkleNode = SparseLeafNode<123>;
type Leaf124VerkleNode = SparseLeafNode<124>;
type Leaf125VerkleNode = SparseLeafNode<125>;
type Leaf126VerkleNode = SparseLeafNode<126>;
type Leaf127VerkleNode = SparseLeafNode<127>;
type Leaf128VerkleNode = SparseLeafNode<128>;
type Leaf129VerkleNode = SparseLeafNode<129>;
type Leaf130VerkleNode = SparseLeafNode<130>;
type Leaf131VerkleNode = SparseLeafNode<131>;
type Leaf132VerkleNode = SparseLeafNode<132>;
type Leaf133VerkleNode = SparseLeafNode<133>;
type Leaf134VerkleNode = SparseLeafNode<134>;
type Leaf135VerkleNode = SparseLeafNode<135>;
type Leaf136VerkleNode = SparseLeafNode<136>;
type Leaf137VerkleNode = SparseLeafNode<137>;
type Leaf138VerkleNode = SparseLeafNode<138>;
type Leaf139VerkleNode = SparseLeafNode<139>;
type Leaf140VerkleNode = SparseLeafNode<140>;
type Leaf141VerkleNode = SparseLeafNode<141>;
type Leaf142VerkleNode = SparseLeafNode<142>;
type Leaf143VerkleNode = SparseLeafNode<143>;
type Leaf144VerkleNode = SparseLeafNode<144>;
type Leaf145VerkleNode = SparseLeafNode<145>;
type Leaf146VerkleNode = SparseLeafNode<146>;
type Leaf147VerkleNode = SparseLeafNode<147>;
type Leaf148VerkleNode = SparseLeafNode<148>;
type Leaf149VerkleNode = SparseLeafNode<149>;
type Leaf150VerkleNode = SparseLeafNode<150>;
type Leaf151VerkleNode = SparseLeafNode<151>;
type Leaf152VerkleNode = SparseLeafNode<152>;
type Leaf153VerkleNode = SparseLeafNode<153>;
type Leaf154VerkleNode = SparseLeafNode<154>;
type Leaf155VerkleNode = SparseLeafNode<155>;
type Leaf156VerkleNode = SparseLeafNode<156>;
type Leaf157VerkleNode = SparseLeafNode<157>;
type Leaf158VerkleNode = SparseLeafNode<158>;
type Leaf159VerkleNode = SparseLeafNode<159>;
type Leaf160VerkleNode = SparseLeafNode<160>;
type Leaf161VerkleNode = SparseLeafNode<161>;
type Leaf162VerkleNode = SparseLeafNode<162>;
type Leaf163VerkleNode = SparseLeafNode<163>;
type Leaf164VerkleNode = SparseLeafNode<164>;
type Leaf165VerkleNode = SparseLeafNode<165>;
type Leaf166VerkleNode = SparseLeafNode<166>;
type Leaf167VerkleNode = SparseLeafNode<167>;
type Leaf168VerkleNode = SparseLeafNode<168>;
type Leaf169VerkleNode = SparseLeafNode<169>;
type Leaf170VerkleNode = SparseLeafNode<170>;
type Leaf171VerkleNode = SparseLeafNode<171>;
type Leaf172VerkleNode = SparseLeafNode<172>;
type Leaf173VerkleNode = SparseLeafNode<173>;
type Leaf174VerkleNode = SparseLeafNode<174>;
type Leaf175VerkleNode = SparseLeafNode<175>;
type Leaf176VerkleNode = SparseLeafNode<176>;
type Leaf177VerkleNode = SparseLeafNode<177>;
type Leaf178VerkleNode = SparseLeafNode<178>;
type Leaf179VerkleNode = SparseLeafNode<179>;
type Leaf180VerkleNode = SparseLeafNode<180>;
type Leaf181VerkleNode = SparseLeafNode<181>;
type Leaf182VerkleNode = SparseLeafNode<182>;
type Leaf183VerkleNode = SparseLeafNode<183>;
type Leaf184VerkleNode = SparseLeafNode<184>;
type Leaf185VerkleNode = SparseLeafNode<185>;
type Leaf186VerkleNode = SparseLeafNode<186>;
type Leaf187VerkleNode = SparseLeafNode<187>;
type Leaf188VerkleNode = SparseLeafNode<188>;
type Leaf189VerkleNode = SparseLeafNode<189>;
type Leaf190VerkleNode = SparseLeafNode<190>;
type Leaf191VerkleNode = SparseLeafNode<191>;
type Leaf192VerkleNode = SparseLeafNode<192>;
type Leaf193VerkleNode = SparseLeafNode<193>;
type Leaf194VerkleNode = SparseLeafNode<194>;
type Leaf195VerkleNode = SparseLeafNode<195>;
type Leaf196VerkleNode = SparseLeafNode<196>;
type Leaf197VerkleNode = SparseLeafNode<197>;
type Leaf198VerkleNode = SparseLeafNode<198>;
type Leaf199VerkleNode = SparseLeafNode<199>;
type Leaf200VerkleNode = SparseLeafNode<200>;
type Leaf201VerkleNode = SparseLeafNode<201>;
type Leaf202VerkleNode = SparseLeafNode<202>;
type Leaf203VerkleNode = SparseLeafNode<203>;
type Leaf204VerkleNode = SparseLeafNode<204>;
type Leaf205VerkleNode = SparseLeafNode<205>;
type Leaf206VerkleNode = SparseLeafNode<206>;
type Leaf207VerkleNode = SparseLeafNode<207>;
type Leaf208VerkleNode = SparseLeafNode<208>;
type Leaf209VerkleNode = SparseLeafNode<209>;
type Leaf210VerkleNode = SparseLeafNode<210>;
type Leaf211VerkleNode = SparseLeafNode<211>;
type Leaf212VerkleNode = SparseLeafNode<212>;
type Leaf213VerkleNode = SparseLeafNode<213>;
type Leaf214VerkleNode = SparseLeafNode<214>;
type Leaf215VerkleNode = SparseLeafNode<215>;
type Leaf216VerkleNode = SparseLeafNode<216>;
type Leaf217VerkleNode = SparseLeafNode<217>;
type Leaf218VerkleNode = SparseLeafNode<218>;
type Leaf219VerkleNode = SparseLeafNode<219>;
type Leaf220VerkleNode = SparseLeafNode<220>;
type Leaf221VerkleNode = SparseLeafNode<221>;
type Leaf222VerkleNode = SparseLeafNode<222>;
type Leaf223VerkleNode = SparseLeafNode<223>;
type Leaf224VerkleNode = SparseLeafNode<224>;
type Leaf225VerkleNode = SparseLeafNode<225>;
type Leaf226VerkleNode = SparseLeafNode<226>;
type Leaf227VerkleNode = SparseLeafNode<227>;
type Leaf228VerkleNode = SparseLeafNode<228>;
type Leaf229VerkleNode = SparseLeafNode<229>;
type Leaf230VerkleNode = SparseLeafNode<230>;
type Leaf231VerkleNode = SparseLeafNode<231>;
type Leaf232VerkleNode = SparseLeafNode<232>;
type Leaf233VerkleNode = SparseLeafNode<233>;
type Leaf234VerkleNode = SparseLeafNode<234>;
type Leaf235VerkleNode = SparseLeafNode<235>;
type Leaf236VerkleNode = SparseLeafNode<236>;
type Leaf237VerkleNode = SparseLeafNode<237>;
type Leaf238VerkleNode = SparseLeafNode<238>;
type Leaf239VerkleNode = SparseLeafNode<239>;
type Leaf240VerkleNode = SparseLeafNode<240>;
type Leaf241VerkleNode = SparseLeafNode<241>;
type Leaf242VerkleNode = SparseLeafNode<242>;
type Leaf243VerkleNode = SparseLeafNode<243>;
type Leaf244VerkleNode = SparseLeafNode<244>;
type Leaf245VerkleNode = SparseLeafNode<245>;
type Leaf246VerkleNode = SparseLeafNode<246>;
type Leaf247VerkleNode = SparseLeafNode<247>;
type Leaf248VerkleNode = SparseLeafNode<248>;
type Leaf249VerkleNode = SparseLeafNode<249>;
type Leaf250VerkleNode = SparseLeafNode<250>;
type Leaf251VerkleNode = SparseLeafNode<251>;
type Leaf252VerkleNode = SparseLeafNode<252>;
type Leaf253VerkleNode = SparseLeafNode<253>;
type Leaf254VerkleNode = SparseLeafNode<254>;
type Leaf255VerkleNode = SparseLeafNode<255>;
type Leaf256VerkleNode = FullLeafNode;
type LeafDeltaVerkleNode = LeafDeltaNode;

impl VerkleNode {
    /// Returns the smallest leaf node type capable of storing `n` values.
/// Returns the smallest leaf node type capable of storing `n` values.
pub fn smallest_leaf_type_for(n: usize) -> VerkleNodeKind {
match n {
1..=1 => VerkleNodeKind::Leaf1,
2..=2 => VerkleNodeKind::Leaf2,
3..=3 => VerkleNodeKind::Leaf3,
4..=4 => VerkleNodeKind::Leaf4,
5..=5 => VerkleNodeKind::Leaf5,
6..=6 => VerkleNodeKind::Leaf6,
7..=7 => VerkleNodeKind::Leaf7,
8..=8 => VerkleNodeKind::Leaf8,
9..=9 => VerkleNodeKind::Leaf9,
10..=10 => VerkleNodeKind::Leaf10,
11..=11 => VerkleNodeKind::Leaf11,
12..=12 => VerkleNodeKind::Leaf12,
13..=13 => VerkleNodeKind::Leaf13,
14..=14 => VerkleNodeKind::Leaf14,
15..=15 => VerkleNodeKind::Leaf15,
16..=16 => VerkleNodeKind::Leaf16,
17..=18 => VerkleNodeKind::Leaf18,
19..=20 => VerkleNodeKind::Leaf20,
21..=23 => VerkleNodeKind::Leaf23,
24..=26 => VerkleNodeKind::Leaf26,
27..=29 => VerkleNodeKind::Leaf29,
30..=33 => VerkleNodeKind::Leaf33,
34..=36 => VerkleNodeKind::Leaf36,
37..=39 => VerkleNodeKind::Leaf39,
40..=40 => VerkleNodeKind::Leaf40,
41..=41 => VerkleNodeKind::Leaf41,
42..=45 => VerkleNodeKind::Leaf45,
46..=48 => VerkleNodeKind::Leaf48,
49..=52 => VerkleNodeKind::Leaf52,
53..=57 => VerkleNodeKind::Leaf57,
58..=61 => VerkleNodeKind::Leaf61,
62..=64 => VerkleNodeKind::Leaf64,
65..=70 => VerkleNodeKind::Leaf70,
71..=75 => VerkleNodeKind::Leaf75,
76..=80 => VerkleNodeKind::Leaf80,
81..=84 => VerkleNodeKind::Leaf84,
85..=89 => VerkleNodeKind::Leaf89,
90..=94 => VerkleNodeKind::Leaf94,
95..=100 => VerkleNodeKind::Leaf100,
101..=107 => VerkleNodeKind::Leaf107,
108..=113 => VerkleNodeKind::Leaf113,
114..=120 => VerkleNodeKind::Leaf120,
121..=124 => VerkleNodeKind::Leaf124,
125..=130 => VerkleNodeKind::Leaf130,
131..=132 => VerkleNodeKind::Leaf132,
133..=134 => VerkleNodeKind::Leaf134,
135..=136 => VerkleNodeKind::Leaf136,
137..=138 => VerkleNodeKind::Leaf138,
139..=140 => VerkleNodeKind::Leaf140,
141..=142 => VerkleNodeKind::Leaf142,
143..=146 => VerkleNodeKind::Leaf146,
147..=149 => VerkleNodeKind::Leaf149,
150..=156 => VerkleNodeKind::Leaf156,
157..=162 => VerkleNodeKind::Leaf162,
163..=168 => VerkleNodeKind::Leaf168,
169..=178 => VerkleNodeKind::Leaf178,
179..=186 => VerkleNodeKind::Leaf186,
187..=195 => VerkleNodeKind::Leaf195,
196..=203 => VerkleNodeKind::Leaf203,
204..=208 => VerkleNodeKind::Leaf208,
209..=225 => VerkleNodeKind::Leaf225,
226..=239 => VerkleNodeKind::Leaf239,
240..=256 => VerkleNodeKind::Leaf256,
_ => panic!("No leaf node type can store more than 256 values"),
}
}

    /// Returns the smallest inner node type capable of storing `n` values.
/// Returns the smallest leaf node type capable of storing `n` values.
pub fn smallest_inner_type_for(n: usize) -> VerkleNodeKind {
match n {
1..=9 => VerkleNodeKind::Inner9,
10..=15 => VerkleNodeKind::Inner15,
16..=21 => VerkleNodeKind::Inner21,
22..=256 => VerkleNodeKind::Inner256,
_ => panic!("No leaf node type can store more than 256 values"),
}
}

    /// Returns the commitment input for computing the commitment of this node.
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
        match self {
            VerkleNode::Empty(n) => n.get_commitment_input(),
            VerkleNode::Inner9(n) => n.get_commitment_input(),
            VerkleNode::Inner15(n) => n.get_commitment_input(),
            VerkleNode::Inner21(n) => n.get_commitment_input(),
            VerkleNode::Inner256(n) => n.get_commitment_input(),
            VerkleNode::InnerDelta(n) => n.get_commitment_input(),
            VerkleNode::Leaf1(n) => n.get_commitment_input(),
            VerkleNode::Leaf2(n) => n.get_commitment_input(),
            VerkleNode::Leaf3(n) => n.get_commitment_input(),
            VerkleNode::Leaf4(n) => n.get_commitment_input(),
            VerkleNode::Leaf5(n) => n.get_commitment_input(),
            VerkleNode::Leaf6(n) => n.get_commitment_input(),
            VerkleNode::Leaf7(n) => n.get_commitment_input(),
            VerkleNode::Leaf8(n) => n.get_commitment_input(),
            VerkleNode::Leaf9(n) => n.get_commitment_input(),
            VerkleNode::Leaf10(n) => n.get_commitment_input(),
            VerkleNode::Leaf11(n) => n.get_commitment_input(),
            VerkleNode::Leaf12(n) => n.get_commitment_input(),
            VerkleNode::Leaf13(n) => n.get_commitment_input(),
            VerkleNode::Leaf14(n) => n.get_commitment_input(),
            VerkleNode::Leaf15(n) => n.get_commitment_input(),
            VerkleNode::Leaf16(n) => n.get_commitment_input(),
            VerkleNode::Leaf18(n) => n.get_commitment_input(),
            VerkleNode::Leaf20(n) => n.get_commitment_input(),
            VerkleNode::Leaf23(n) => n.get_commitment_input(),
            VerkleNode::Leaf26(n) => n.get_commitment_input(),
            VerkleNode::Leaf29(n) => n.get_commitment_input(),
            VerkleNode::Leaf33(n) => n.get_commitment_input(),
            VerkleNode::Leaf36(n) => n.get_commitment_input(),
            VerkleNode::Leaf39(n) => n.get_commitment_input(),
            VerkleNode::Leaf40(n) => n.get_commitment_input(),
            VerkleNode::Leaf41(n) => n.get_commitment_input(),
            VerkleNode::Leaf45(n) => n.get_commitment_input(),
            VerkleNode::Leaf48(n) => n.get_commitment_input(),
            VerkleNode::Leaf52(n) => n.get_commitment_input(),
            VerkleNode::Leaf57(n) => n.get_commitment_input(),
            VerkleNode::Leaf61(n) => n.get_commitment_input(),
            VerkleNode::Leaf64(n) => n.get_commitment_input(),
            VerkleNode::Leaf70(n) => n.get_commitment_input(),
            VerkleNode::Leaf75(n) => n.get_commitment_input(),
            VerkleNode::Leaf80(n) => n.get_commitment_input(),
            VerkleNode::Leaf84(n) => n.get_commitment_input(),
            VerkleNode::Leaf89(n) => n.get_commitment_input(),
            VerkleNode::Leaf94(n) => n.get_commitment_input(),
            VerkleNode::Leaf100(n) => n.get_commitment_input(),
            VerkleNode::Leaf107(n) => n.get_commitment_input(),
            VerkleNode::Leaf113(n) => n.get_commitment_input(),
            VerkleNode::Leaf120(n) => n.get_commitment_input(),
            VerkleNode::Leaf124(n) => n.get_commitment_input(),
            VerkleNode::Leaf130(n) => n.get_commitment_input(),
            VerkleNode::Leaf132(n) => n.get_commitment_input(),
            VerkleNode::Leaf134(n) => n.get_commitment_input(),
            VerkleNode::Leaf136(n) => n.get_commitment_input(),
            VerkleNode::Leaf138(n) => n.get_commitment_input(),
            VerkleNode::Leaf140(n) => n.get_commitment_input(),
            VerkleNode::Leaf142(n) => n.get_commitment_input(),
            VerkleNode::Leaf146(n) => n.get_commitment_input(),
            VerkleNode::Leaf149(n) => n.get_commitment_input(),
            VerkleNode::Leaf156(n) => n.get_commitment_input(),
            VerkleNode::Leaf162(n) => n.get_commitment_input(),
            VerkleNode::Leaf168(n) => n.get_commitment_input(),
            VerkleNode::Leaf178(n) => n.get_commitment_input(),
            VerkleNode::Leaf186(n) => n.get_commitment_input(),
            VerkleNode::Leaf195(n) => n.get_commitment_input(),
            VerkleNode::Leaf203(n) => n.get_commitment_input(),
            VerkleNode::Leaf208(n) => n.get_commitment_input(),
            VerkleNode::Leaf225(n) => n.get_commitment_input(),
            VerkleNode::Leaf239(n) => n.get_commitment_input(),
            VerkleNode::Leaf256(n) => n.get_commitment_input(),
            VerkleNode::LeafDelta(n) => n.get_commitment_input(),
        }
    }

    /// Converts this node to an inner node, if it is one.
    pub fn as_inner_node(&self) -> Option<&dyn VerkleManagedInnerNode> {
        match self {
            VerkleNode::Inner9(n) => Some(n.deref()),
            VerkleNode::Inner15(n) => Some(n.deref()),
            VerkleNode::Inner21(n) => Some(n.deref()),
            VerkleNode::Inner256(n) => Some(n.deref()),
            VerkleNode::InnerDelta(n) => Some(n.deref()),
            _ => None,
        }
    }

    /// Accepts a visitor for recursively traversing the node and its children.
    pub fn accept(
        &self,
        visitor: &mut impl NodeVisitor<Self>,
        manager: &impl NodeManager<Id = VerkleNodeId, Node = VerkleNode>,
        level: u64,
    ) -> BTResult<(), Error> {
        visitor.visit(self, level)?;
        match self {
            VerkleNode::Empty(_)
            | VerkleNode::Leaf1(_)
            | VerkleNode::Leaf2(_)
            | VerkleNode::Leaf3(_)
            | VerkleNode::Leaf4(_)
            | VerkleNode::Leaf5(_)
            | VerkleNode::Leaf6(_)
            | VerkleNode::Leaf7(_)
            | VerkleNode::Leaf8(_)
            | VerkleNode::Leaf9(_)
            | VerkleNode::Leaf10(_)
            | VerkleNode::Leaf11(_)
            | VerkleNode::Leaf12(_)
            | VerkleNode::Leaf13(_)
            | VerkleNode::Leaf14(_)
            | VerkleNode::Leaf15(_)
            | VerkleNode::Leaf16(_)
            | VerkleNode::Leaf18(_)
            | VerkleNode::Leaf20(_)
            | VerkleNode::Leaf23(_)
            | VerkleNode::Leaf26(_)
            | VerkleNode::Leaf29(_)
            | VerkleNode::Leaf33(_)
            | VerkleNode::Leaf36(_)
            | VerkleNode::Leaf39(_)
            | VerkleNode::Leaf40(_)
            | VerkleNode::Leaf41(_)
            | VerkleNode::Leaf45(_)
            | VerkleNode::Leaf48(_)
            | VerkleNode::Leaf52(_)
            | VerkleNode::Leaf57(_)
            | VerkleNode::Leaf61(_)
            | VerkleNode::Leaf64(_)
            | VerkleNode::Leaf70(_)
            | VerkleNode::Leaf75(_)
            | VerkleNode::Leaf80(_)
            | VerkleNode::Leaf84(_)
            | VerkleNode::Leaf89(_)
            | VerkleNode::Leaf94(_)
            | VerkleNode::Leaf100(_)
            | VerkleNode::Leaf107(_)
            | VerkleNode::Leaf113(_)
            | VerkleNode::Leaf120(_)
            | VerkleNode::Leaf124(_)
            | VerkleNode::Leaf130(_)
            | VerkleNode::Leaf132(_)
            | VerkleNode::Leaf134(_)
            | VerkleNode::Leaf136(_)
            | VerkleNode::Leaf138(_)
            | VerkleNode::Leaf140(_)
            | VerkleNode::Leaf142(_)
            | VerkleNode::Leaf146(_)
            | VerkleNode::Leaf149(_)
            | VerkleNode::Leaf156(_)
            | VerkleNode::Leaf162(_)
            | VerkleNode::Leaf168(_)
            | VerkleNode::Leaf178(_)
            | VerkleNode::Leaf186(_)
            | VerkleNode::Leaf195(_)
            | VerkleNode::Leaf203(_)
            | VerkleNode::Leaf208(_)
            | VerkleNode::Leaf225(_)
            | VerkleNode::Leaf239(_)
            | VerkleNode::Leaf256(_)
            | VerkleNode::LeafDelta(_) => {}
            | VerkleNode::Inner9(_)
            | VerkleNode::Inner15(_)
            | VerkleNode::Inner21(_)
            | VerkleNode::Inner256(_)
            | VerkleNode::InnerDelta(_) => {
                let inner = self.as_inner_node().ok_or(Error::CorruptedState(
                    "expected inner node in accept method. Maybe you added a new leaf variant and forgot to update the accept method".to_owned(),
                ))?;
                for child_id in inner.iter_children() {
                    let child = manager.get_read_access(child_id.item)?;
                    child.accept(visitor, manager, level + 1)?;
                }
            }
        }
        Ok(())
    }
}

impl HasDeltaVariant for VerkleNode {
    type Id = VerkleNodeId;

    fn needs_delta_base(&self) -> Option<Self::Id> {
        match self {
            VerkleNode::InnerDelta(n) => Some(n.base_node_id),
            VerkleNode::LeafDelta(n) => Some(n.base_node_id),
            _ => None,
        }
    }

    fn copy_from_delta_base(&mut self, base: &Self) -> BTResult<(), Error> {
        // match self {
        //     VerkleNode::LeafDelta(n) => match base {
        //         VerkleNode::Leaf256(f) => n.values = f.values,
        //         VerkleNode::Leaf146(f) => {
        //             n.values = array::from_fn(|i| {
        //                 ValueWithIndex::get_slot_for(&f.values, i as u8)
        //                     .map(|slot| f.values[slot].item)
        //                     .unwrap_or_default()
        //             });
        //         }
        //         _ => {
        //             return Err(Error::Internal(
        //                 "copy_from_delta_base called with unsupported node".to_owned(),
        //             )
        //             .into());
        //         }
        //     },
        //     VerkleNode::InnerDelta(n) => {
        //         if let VerkleNode::Inner256(i) = base {
        //             n.children = i.children;
        //         } else {
        //             return Err(Error::Internal(
        //                 "copy_from_delta_base called with unsupported node".to_owned(),
        //             )
        //             .into());
        //         }
        //     }
        //     _ => (),
        // }
        Ok(())
    }
}

impl NodeVisitor<VerkleNode> for NodeCountVisitor {
    fn visit(&mut self, node: &VerkleNode, level: u64) -> BTResult<(), Error> {
        // match node {
        //     VerkleNode::Empty(n) => self.visit(n, level),
        //     VerkleNode::Inner9(n) => self.visit(n.deref(), level),
        //     VerkleNode::Inner15(n) => self.visit(n.deref(), level),
        //     VerkleNode::Inner21(n) => self.visit(n.deref(), level),
        //     VerkleNode::Inner256(n) => self.visit(n.deref(), level),
        //     VerkleNode::InnerDelta(n) => self.visit(n.deref(), level),
        //     VerkleNode::Leaf1(n) => self.visit(n.deref(), level),
        //     VerkleNode::Leaf2(n) => self.visit(n.deref(), level),
        //     VerkleNode::Leaf5(n) => self.visit(n.deref(), level),
        //     VerkleNode::Leaf18(n) => self.visit(n.deref(), level),
        //     VerkleNode::Leaf146(n) => self.visit(n.deref(), level),
        //     VerkleNode::Leaf256(n) => self.visit(n.deref(), level),
        //     VerkleNode::LeafDelta(n) => self.visit(n.deref(), level),
        // }
        Ok(())
    }
}

impl ToNodeKind for VerkleNode {
    type Target = VerkleNodeKind;

    /// Converts the ID to its corresponding node kind. This conversion will always succeed.
    fn to_node_kind(&self) -> Option<Self::Target> {
        match self {
            VerkleNode::Empty(_) => Some(VerkleNodeKind::Empty),
            VerkleNode::Inner9(_) => Some(VerkleNodeKind::Inner9),
            VerkleNode::Inner15(_) => Some(VerkleNodeKind::Inner15),
            VerkleNode::Inner21(_) => Some(VerkleNodeKind::Inner21),
            VerkleNode::Inner256(_) => Some(VerkleNodeKind::Inner256),
            VerkleNode::InnerDelta(_) => Some(VerkleNodeKind::InnerDelta),
            VerkleNode::Leaf1(_) => Some(VerkleNodeKind::Leaf1),
            VerkleNode::Leaf2(_) => Some(VerkleNodeKind::Leaf2),
            VerkleNode::Leaf3(_) => Some(VerkleNodeKind::Leaf3),
            VerkleNode::Leaf4(_) => Some(VerkleNodeKind::Leaf4),
            VerkleNode::Leaf5(_) => Some(VerkleNodeKind::Leaf5),
            VerkleNode::Leaf6(_) => Some(VerkleNodeKind::Leaf6),
            VerkleNode::Leaf7(_) => Some(VerkleNodeKind::Leaf7),
            VerkleNode::Leaf8(_) => Some(VerkleNodeKind::Leaf8),
            VerkleNode::Leaf9(_) => Some(VerkleNodeKind::Leaf9),
            VerkleNode::Leaf10(_) => Some(VerkleNodeKind::Leaf10),
            VerkleNode::Leaf11(_) => Some(VerkleNodeKind::Leaf11),
            VerkleNode::Leaf12(_) => Some(VerkleNodeKind::Leaf12),
            VerkleNode::Leaf13(_) => Some(VerkleNodeKind::Leaf13),
            VerkleNode::Leaf14(_) => Some(VerkleNodeKind::Leaf14),
            VerkleNode::Leaf15(_) => Some(VerkleNodeKind::Leaf15),
            VerkleNode::Leaf16(_) => Some(VerkleNodeKind::Leaf16),
            VerkleNode::Leaf18(_) => Some(VerkleNodeKind::Leaf18),
            VerkleNode::Leaf20(_) => Some(VerkleNodeKind::Leaf20),
            VerkleNode::Leaf23(_) => Some(VerkleNodeKind::Leaf23),
            VerkleNode::Leaf26(_) => Some(VerkleNodeKind::Leaf26),
            VerkleNode::Leaf29(_) => Some(VerkleNodeKind::Leaf29),
            VerkleNode::Leaf33(_) => Some(VerkleNodeKind::Leaf33),
            VerkleNode::Leaf36(_) => Some(VerkleNodeKind::Leaf36),
            VerkleNode::Leaf39(_) => Some(VerkleNodeKind::Leaf39),
            VerkleNode::Leaf40(_) => Some(VerkleNodeKind::Leaf40),
            VerkleNode::Leaf41(_) => Some(VerkleNodeKind::Leaf41),
            VerkleNode::Leaf45(_) => Some(VerkleNodeKind::Leaf45),
            VerkleNode::Leaf48(_) => Some(VerkleNodeKind::Leaf48),
            VerkleNode::Leaf52(_) => Some(VerkleNodeKind::Leaf52),
            VerkleNode::Leaf57(_) => Some(VerkleNodeKind::Leaf57),
            VerkleNode::Leaf61(_) => Some(VerkleNodeKind::Leaf61),
            VerkleNode::Leaf64(_) => Some(VerkleNodeKind::Leaf64),
            VerkleNode::Leaf70(_) => Some(VerkleNodeKind::Leaf70),
            VerkleNode::Leaf75(_) => Some(VerkleNodeKind::Leaf75),
            VerkleNode::Leaf80(_) => Some(VerkleNodeKind::Leaf80),
            VerkleNode::Leaf84(_) => Some(VerkleNodeKind::Leaf84),
            VerkleNode::Leaf89(_) => Some(VerkleNodeKind::Leaf89),
            VerkleNode::Leaf94(_) => Some(VerkleNodeKind::Leaf94),
            VerkleNode::Leaf100(_) => Some(VerkleNodeKind::Leaf100),
            VerkleNode::Leaf107(_) => Some(VerkleNodeKind::Leaf107),
            VerkleNode::Leaf113(_) => Some(VerkleNodeKind::Leaf113),
            VerkleNode::Leaf120(_) => Some(VerkleNodeKind::Leaf120),
            VerkleNode::Leaf124(_) => Some(VerkleNodeKind::Leaf124),
            VerkleNode::Leaf130(_) => Some(VerkleNodeKind::Leaf130),
            VerkleNode::Leaf132(_) => Some(VerkleNodeKind::Leaf132),
            VerkleNode::Leaf134(_) => Some(VerkleNodeKind::Leaf134),
            VerkleNode::Leaf136(_) => Some(VerkleNodeKind::Leaf136),
            VerkleNode::Leaf138(_) => Some(VerkleNodeKind::Leaf138),
            VerkleNode::Leaf140(_) => Some(VerkleNodeKind::Leaf140),
            VerkleNode::Leaf142(_) => Some(VerkleNodeKind::Leaf142),
            VerkleNode::Leaf146(_) => Some(VerkleNodeKind::Leaf146),
            VerkleNode::Leaf149(_) => Some(VerkleNodeKind::Leaf149),
            VerkleNode::Leaf156(_) => Some(VerkleNodeKind::Leaf156),
            VerkleNode::Leaf162(_) => Some(VerkleNodeKind::Leaf162),
            VerkleNode::Leaf168(_) => Some(VerkleNodeKind::Leaf168),
            VerkleNode::Leaf178(_) => Some(VerkleNodeKind::Leaf178),
            VerkleNode::Leaf186(_) => Some(VerkleNodeKind::Leaf186),
            VerkleNode::Leaf195(_) => Some(VerkleNodeKind::Leaf195),
            VerkleNode::Leaf203(_) => Some(VerkleNodeKind::Leaf203),
            VerkleNode::Leaf208(_) => Some(VerkleNodeKind::Leaf208),
            VerkleNode::Leaf225(_) => Some(VerkleNodeKind::Leaf225),
            VerkleNode::Leaf239(_) => Some(VerkleNodeKind::Leaf239),
            VerkleNode::Leaf256(_) => Some(VerkleNodeKind::Leaf256),
            VerkleNode::LeafDelta(_) => Some(VerkleNodeKind::LeafDelta),
        }
    }
}

impl NodeSize for VerkleNode {
    fn node_byte_size(&self) -> usize {
        self.to_node_kind().unwrap().node_byte_size()
    }

    fn min_non_empty_node_size() -> usize {
        VerkleNodeKind::min_non_empty_node_size()
    }
}

impl HasEmptyNode for VerkleNode {
    fn is_empty_node(&self) -> bool {
        matches!(self, VerkleNode::Empty(_))
    }

    fn empty_node() -> Self {
        VerkleNode::Empty(EmptyNode)
    }
}

impl Default for VerkleNode {
    fn default() -> Self {
        VerkleNode::Empty(EmptyNode)
    }
}

impl UnionManagedTrieNode for VerkleNode {
    fn copy_on_write(&self, id: Self::Id, changed_indices: Vec<u8>) -> BTResult<Self, Error> {
        // // Note: This method is only called in archive mode, so using the delta node is fine.
        // match self {
        //     VerkleNode::Inner256(n) => {
        //         if changed_indices.len() <= InnerDeltaNode::DELTA_SIZE {
        //             Ok(VerkleNode::InnerDelta(Box::new(
        //                 InnerDeltaNode::from_full_inner(n, id),
        //             )))
        //         } else {
        //             Ok(VerkleNode::Inner256(n.clone()))
        //         }
        //     }
        //     VerkleNode::InnerDelta(n) => {
        //         let enough_slots = ItemWithIndex::required_slot_count_for(
        //             &n.children_delta,
        //             changed_indices.into_iter(),
        //         ) <= InnerDeltaNode::DELTA_SIZE;
        //         if enough_slots {
        //             Ok(VerkleNode::InnerDelta(n.clone()))
        //         } else {
        //             Ok(VerkleNode::Inner256(Box::new(FullInnerNode::from(
        //                 (**n).clone(),
        //             ))))
        //         }
        //     }
        //     VerkleNode::Leaf256(n) => {
        //         if changed_indices.len() <= LeafDeltaNode::DELTA_SIZE {
        //             Ok(VerkleNode::LeafDelta(Box::new(
        //                 LeafDeltaNode::from_full_leaf(n, id),
        //             )))
        //         } else {
        //             Ok(VerkleNode::Leaf256(n.clone()))
        //         }
        //     }
        //     VerkleNode::Leaf146(n) => {
        //         if changed_indices.len() <= LeafDeltaNode::DELTA_SIZE {
        //             Ok(VerkleNode::LeafDelta(Box::new(
        //                 LeafDeltaNode::from_sparse_leaf(n, id),
        //             )))
        //         } else {
        //             Ok(VerkleNode::Leaf146(n.clone()))
        //         }
        //     }
        //     VerkleNode::LeafDelta(n) => {
        //         const DELTA_PLUS_ONE: usize = LeafDeltaNode::DELTA_SIZE + 1;
        //         match ItemWithIndex::required_slot_count_for(
        //             &n.values_delta,
        //             changed_indices.into_iter(),
        //         ) {
        //             ..=LeafDeltaNode::DELTA_SIZE => Ok(VerkleNode::LeafDelta(n.clone())),
        //             DELTA_PLUS_ONE..=146 => Ok(VerkleNode::Leaf146(Box::new(
        //                 SparseLeafNode::try_from((**n).clone())?,
        //             ))),
        //             _ => Ok(VerkleNode::Leaf256(Box::new(FullLeafNode::from(
        //                 (**n).clone(),
        //             )))),
        //         }
        //     }
        //     _ => Ok(self.clone()),
        // }
        Ok(self.clone())
    }
}

impl ManagedTrieNode for VerkleNode {
    type Union = VerkleNode;
    type Id = VerkleNodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, key: &Key, depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {
        match self {
            VerkleNode::Empty(n) => n.lookup(key, depth),
            VerkleNode::Inner9(n) => n.lookup(key, depth),
            VerkleNode::Inner15(n) => n.lookup(key, depth),
            VerkleNode::Inner21(n) => n.lookup(key, depth),
            VerkleNode::Inner256(n) => n.lookup(key, depth),
            VerkleNode::InnerDelta(n) => n.lookup(key, depth),
            VerkleNode::Leaf1(n) => n.lookup(key, depth),
            VerkleNode::Leaf2(n) => n.lookup(key, depth),
            VerkleNode::Leaf3(n) => n.lookup(key, depth),
            VerkleNode::Leaf4(n) => n.lookup(key, depth),
            VerkleNode::Leaf5(n) => n.lookup(key, depth),
            VerkleNode::Leaf6(n) => n.lookup(key, depth),
            VerkleNode::Leaf7(n) => n.lookup(key, depth),
            VerkleNode::Leaf8(n) => n.lookup(key, depth),
            VerkleNode::Leaf9(n) => n.lookup(key, depth),
            VerkleNode::Leaf10(n) => n.lookup(key, depth),
            VerkleNode::Leaf11(n) => n.lookup(key, depth),
            VerkleNode::Leaf12(n) => n.lookup(key, depth),
            VerkleNode::Leaf13(n) => n.lookup(key, depth),
            VerkleNode::Leaf14(n) => n.lookup(key, depth),
            VerkleNode::Leaf15(n) => n.lookup(key, depth),
            VerkleNode::Leaf16(n) => n.lookup(key, depth),
            VerkleNode::Leaf18(n) => n.lookup(key, depth),
            VerkleNode::Leaf20(n) => n.lookup(key, depth),
            VerkleNode::Leaf23(n) => n.lookup(key, depth),
            VerkleNode::Leaf26(n) => n.lookup(key, depth),
            VerkleNode::Leaf29(n) => n.lookup(key, depth),
            VerkleNode::Leaf33(n) => n.lookup(key, depth),
            VerkleNode::Leaf36(n) => n.lookup(key, depth),
            VerkleNode::Leaf39(n) => n.lookup(key, depth),
            VerkleNode::Leaf40(n) => n.lookup(key, depth),
            VerkleNode::Leaf41(n) => n.lookup(key, depth),
            VerkleNode::Leaf45(n) => n.lookup(key, depth),
            VerkleNode::Leaf48(n) => n.lookup(key, depth),
            VerkleNode::Leaf52(n) => n.lookup(key, depth),
            VerkleNode::Leaf57(n) => n.lookup(key, depth),
            VerkleNode::Leaf61(n) => n.lookup(key, depth),
            VerkleNode::Leaf64(n) => n.lookup(key, depth),
            VerkleNode::Leaf70(n) => n.lookup(key, depth),
            VerkleNode::Leaf75(n) => n.lookup(key, depth),
            VerkleNode::Leaf80(n) => n.lookup(key, depth),
            VerkleNode::Leaf84(n) => n.lookup(key, depth),
            VerkleNode::Leaf89(n) => n.lookup(key, depth),
            VerkleNode::Leaf94(n) => n.lookup(key, depth),
            VerkleNode::Leaf100(n) => n.lookup(key, depth),
            VerkleNode::Leaf107(n) => n.lookup(key, depth),
            VerkleNode::Leaf113(n) => n.lookup(key, depth),
            VerkleNode::Leaf120(n) => n.lookup(key, depth),
            VerkleNode::Leaf124(n) => n.lookup(key, depth),
            VerkleNode::Leaf130(n) => n.lookup(key, depth),
            VerkleNode::Leaf132(n) => n.lookup(key, depth),
            VerkleNode::Leaf134(n) => n.lookup(key, depth),
            VerkleNode::Leaf136(n) => n.lookup(key, depth),
            VerkleNode::Leaf138(n) => n.lookup(key, depth),
            VerkleNode::Leaf140(n) => n.lookup(key, depth),
            VerkleNode::Leaf142(n) => n.lookup(key, depth),
            VerkleNode::Leaf146(n) => n.lookup(key, depth),
            VerkleNode::Leaf149(n) => n.lookup(key, depth),
            VerkleNode::Leaf156(n) => n.lookup(key, depth),
            VerkleNode::Leaf162(n) => n.lookup(key, depth),
            VerkleNode::Leaf168(n) => n.lookup(key, depth),
            VerkleNode::Leaf178(n) => n.lookup(key, depth),
            VerkleNode::Leaf186(n) => n.lookup(key, depth),
            VerkleNode::Leaf195(n) => n.lookup(key, depth),
            VerkleNode::Leaf203(n) => n.lookup(key, depth),
            VerkleNode::Leaf208(n) => n.lookup(key, depth),
            VerkleNode::Leaf225(n) => n.lookup(key, depth),
            VerkleNode::Leaf239(n) => n.lookup(key, depth),
            VerkleNode::Leaf256(n) => n.lookup(key, depth),
            VerkleNode::LeafDelta(n) => n.lookup(key, depth),
        }
    }

    fn next_store_action<'a>(
        &self,
        updates: KeyedUpdateBatch<'a>,
        depth: u8,
        self_id: Self::Id,
    ) -> BTResult<StoreAction<'a, Self::Id, Self::Union>, Error> {
        match self {
            VerkleNode::Empty(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner9(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner15(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner21(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner256(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::InnerDelta(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf1(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf2(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf3(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf4(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf5(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf6(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf7(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf8(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf9(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf10(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf11(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf12(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf13(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf14(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf15(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf16(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf18(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf20(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf23(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf26(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf29(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf33(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf36(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf39(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf40(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf41(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf45(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf48(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf52(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf57(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf61(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf64(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf70(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf75(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf80(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf84(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf89(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf94(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf100(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf107(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf113(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf120(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf124(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf130(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf132(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf134(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf136(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf138(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf140(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf142(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf146(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf149(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf156(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf162(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf168(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf178(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf186(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf195(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf203(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf208(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf225(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf239(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf256(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::LeafDelta(n) => n.next_store_action(updates, depth, self_id),
        }
    }

    fn replace_child(&mut self, key: &Key, depth: u8, new: VerkleNodeId) -> BTResult<(), Error> {
        match self {
            VerkleNode::Empty(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner9(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner15(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner21(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner256(n) => n.replace_child(key, depth, new),
            VerkleNode::InnerDelta(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf1(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf2(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf3(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf4(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf5(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf6(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf7(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf8(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf9(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf10(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf11(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf12(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf13(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf14(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf15(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf16(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf18(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf20(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf23(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf26(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf29(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf33(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf36(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf39(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf40(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf41(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf45(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf48(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf52(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf57(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf61(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf64(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf70(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf75(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf80(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf84(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf89(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf94(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf100(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf107(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf113(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf120(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf124(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf130(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf132(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf134(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf136(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf138(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf140(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf142(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf146(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf149(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf156(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf162(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf168(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf178(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf186(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf195(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf203(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf208(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf225(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf239(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf256(n) => n.replace_child(key, depth, new),
            VerkleNode::LeafDelta(n) => n.replace_child(key, depth, new),
        }
    }

    fn store(&mut self, update: &KeyedUpdate) -> BTResult<Value, Error> {
        match self {
            VerkleNode::Empty(n) => n.store(update),
            VerkleNode::Inner9(n) => n.store(update),
            VerkleNode::Inner15(n) => n.store(update),
            VerkleNode::Inner21(n) => n.store(update),
            VerkleNode::Inner256(n) => n.store(update),
            VerkleNode::InnerDelta(n) => n.store(update),
            VerkleNode::Leaf1(n) => n.store(update),
            VerkleNode::Leaf2(n) => n.store(update),
            VerkleNode::Leaf3(n) => n.store(update),
            VerkleNode::Leaf4(n) => n.store(update),
            VerkleNode::Leaf5(n) => n.store(update),
            VerkleNode::Leaf6(n) => n.store(update),
            VerkleNode::Leaf7(n) => n.store(update),
            VerkleNode::Leaf8(n) => n.store(update),
            VerkleNode::Leaf9(n) => n.store(update),
            VerkleNode::Leaf10(n) => n.store(update),
            VerkleNode::Leaf11(n) => n.store(update),
            VerkleNode::Leaf12(n) => n.store(update),
            VerkleNode::Leaf13(n) => n.store(update),
            VerkleNode::Leaf14(n) => n.store(update),
            VerkleNode::Leaf15(n) => n.store(update),
            VerkleNode::Leaf16(n) => n.store(update),
            VerkleNode::Leaf18(n) => n.store(update),
            VerkleNode::Leaf20(n) => n.store(update),
            VerkleNode::Leaf23(n) => n.store(update),
            VerkleNode::Leaf26(n) => n.store(update),
            VerkleNode::Leaf29(n) => n.store(update),
            VerkleNode::Leaf33(n) => n.store(update),
            VerkleNode::Leaf36(n) => n.store(update),
            VerkleNode::Leaf39(n) => n.store(update),
            VerkleNode::Leaf40(n) => n.store(update),
            VerkleNode::Leaf41(n) => n.store(update),
            VerkleNode::Leaf45(n) => n.store(update),
            VerkleNode::Leaf48(n) => n.store(update),
            VerkleNode::Leaf52(n) => n.store(update),
            VerkleNode::Leaf57(n) => n.store(update),
            VerkleNode::Leaf61(n) => n.store(update),
            VerkleNode::Leaf64(n) => n.store(update),
            VerkleNode::Leaf70(n) => n.store(update),
            VerkleNode::Leaf75(n) => n.store(update),
            VerkleNode::Leaf80(n) => n.store(update),
            VerkleNode::Leaf84(n) => n.store(update),
            VerkleNode::Leaf89(n) => n.store(update),
            VerkleNode::Leaf94(n) => n.store(update),
            VerkleNode::Leaf100(n) => n.store(update),
            VerkleNode::Leaf107(n) => n.store(update),
            VerkleNode::Leaf113(n) => n.store(update),
            VerkleNode::Leaf120(n) => n.store(update),
            VerkleNode::Leaf124(n) => n.store(update),
            VerkleNode::Leaf130(n) => n.store(update),
            VerkleNode::Leaf132(n) => n.store(update),
            VerkleNode::Leaf134(n) => n.store(update),
            VerkleNode::Leaf136(n) => n.store(update),
            VerkleNode::Leaf138(n) => n.store(update),
            VerkleNode::Leaf140(n) => n.store(update),
            VerkleNode::Leaf142(n) => n.store(update),
            VerkleNode::Leaf146(n) => n.store(update),
            VerkleNode::Leaf149(n) => n.store(update),
            VerkleNode::Leaf156(n) => n.store(update),
            VerkleNode::Leaf162(n) => n.store(update),
            VerkleNode::Leaf168(n) => n.store(update),
            VerkleNode::Leaf178(n) => n.store(update),
            VerkleNode::Leaf186(n) => n.store(update),
            VerkleNode::Leaf195(n) => n.store(update),
            VerkleNode::Leaf203(n) => n.store(update),
            VerkleNode::Leaf208(n) => n.store(update),
            VerkleNode::Leaf225(n) => n.store(update),
            VerkleNode::Leaf239(n) => n.store(update),
            VerkleNode::Leaf256(n) => n.store(update),
            VerkleNode::LeafDelta(n) => n.store(update),
        }
    }

    fn get_commitment(&self) -> Self::Commitment {
        match self {
            VerkleNode::Empty(n) => n.get_commitment(),
            VerkleNode::Inner9(n) => n.get_commitment(),
            VerkleNode::Inner15(n) => n.get_commitment(),
            VerkleNode::Inner21(n) => n.get_commitment(),
            VerkleNode::Inner256(n) => n.get_commitment(),
            VerkleNode::InnerDelta(n) => n.get_commitment(),
            VerkleNode::Leaf1(n) => n.get_commitment(),
            VerkleNode::Leaf2(n) => n.get_commitment(),
            VerkleNode::Leaf3(n) => n.get_commitment(),
            VerkleNode::Leaf4(n) => n.get_commitment(),
            VerkleNode::Leaf5(n) => n.get_commitment(),
            VerkleNode::Leaf6(n) => n.get_commitment(),
            VerkleNode::Leaf7(n) => n.get_commitment(),
            VerkleNode::Leaf8(n) => n.get_commitment(),
            VerkleNode::Leaf9(n) => n.get_commitment(),
            VerkleNode::Leaf10(n) => n.get_commitment(),
            VerkleNode::Leaf11(n) => n.get_commitment(),
            VerkleNode::Leaf12(n) => n.get_commitment(),
            VerkleNode::Leaf13(n) => n.get_commitment(),
            VerkleNode::Leaf14(n) => n.get_commitment(),
            VerkleNode::Leaf15(n) => n.get_commitment(),
            VerkleNode::Leaf16(n) => n.get_commitment(),
            VerkleNode::Leaf18(n) => n.get_commitment(),
            VerkleNode::Leaf20(n) => n.get_commitment(),
            VerkleNode::Leaf23(n) => n.get_commitment(),
            VerkleNode::Leaf26(n) => n.get_commitment(),
            VerkleNode::Leaf29(n) => n.get_commitment(),
            VerkleNode::Leaf33(n) => n.get_commitment(),
            VerkleNode::Leaf36(n) => n.get_commitment(),
            VerkleNode::Leaf39(n) => n.get_commitment(),
            VerkleNode::Leaf40(n) => n.get_commitment(),
            VerkleNode::Leaf41(n) => n.get_commitment(),
            VerkleNode::Leaf45(n) => n.get_commitment(),
            VerkleNode::Leaf48(n) => n.get_commitment(),
            VerkleNode::Leaf52(n) => n.get_commitment(),
            VerkleNode::Leaf57(n) => n.get_commitment(),
            VerkleNode::Leaf61(n) => n.get_commitment(),
            VerkleNode::Leaf64(n) => n.get_commitment(),
            VerkleNode::Leaf70(n) => n.get_commitment(),
            VerkleNode::Leaf75(n) => n.get_commitment(),
            VerkleNode::Leaf80(n) => n.get_commitment(),
            VerkleNode::Leaf84(n) => n.get_commitment(),
            VerkleNode::Leaf89(n) => n.get_commitment(),
            VerkleNode::Leaf94(n) => n.get_commitment(),
            VerkleNode::Leaf100(n) => n.get_commitment(),
            VerkleNode::Leaf107(n) => n.get_commitment(),
            VerkleNode::Leaf113(n) => n.get_commitment(),
            VerkleNode::Leaf120(n) => n.get_commitment(),
            VerkleNode::Leaf124(n) => n.get_commitment(),
            VerkleNode::Leaf130(n) => n.get_commitment(),
            VerkleNode::Leaf132(n) => n.get_commitment(),
            VerkleNode::Leaf134(n) => n.get_commitment(),
            VerkleNode::Leaf136(n) => n.get_commitment(),
            VerkleNode::Leaf138(n) => n.get_commitment(),
            VerkleNode::Leaf140(n) => n.get_commitment(),
            VerkleNode::Leaf142(n) => n.get_commitment(),
            VerkleNode::Leaf146(n) => n.get_commitment(),
            VerkleNode::Leaf149(n) => n.get_commitment(),
            VerkleNode::Leaf156(n) => n.get_commitment(),
            VerkleNode::Leaf162(n) => n.get_commitment(),
            VerkleNode::Leaf168(n) => n.get_commitment(),
            VerkleNode::Leaf178(n) => n.get_commitment(),
            VerkleNode::Leaf186(n) => n.get_commitment(),
            VerkleNode::Leaf195(n) => n.get_commitment(),
            VerkleNode::Leaf203(n) => n.get_commitment(),
            VerkleNode::Leaf208(n) => n.get_commitment(),
            VerkleNode::Leaf225(n) => n.get_commitment(),
            VerkleNode::Leaf239(n) => n.get_commitment(),
            VerkleNode::Leaf256(n) => n.get_commitment(),
            VerkleNode::LeafDelta(n) => n.get_commitment(),
        }
    }

    fn set_commitment(&mut self, cache: Self::Commitment) -> BTResult<(), Error> {
        match self {
            VerkleNode::Empty(n) => n.set_commitment(cache),
            VerkleNode::Inner9(n) => n.set_commitment(cache),
            VerkleNode::Inner15(n) => n.set_commitment(cache),
            VerkleNode::Inner21(n) => n.set_commitment(cache),
            VerkleNode::Inner256(n) => n.set_commitment(cache),
            VerkleNode::InnerDelta(n) => n.set_commitment(cache),
            VerkleNode::Leaf1(n) => n.set_commitment(cache),
            VerkleNode::Leaf2(n) => n.set_commitment(cache),
            VerkleNode::Leaf3(n) => n.set_commitment(cache),
            VerkleNode::Leaf4(n) => n.set_commitment(cache),
            VerkleNode::Leaf5(n) => n.set_commitment(cache),
            VerkleNode::Leaf6(n) => n.set_commitment(cache),
            VerkleNode::Leaf7(n) => n.set_commitment(cache),
            VerkleNode::Leaf8(n) => n.set_commitment(cache),
            VerkleNode::Leaf9(n) => n.set_commitment(cache),
            VerkleNode::Leaf10(n) => n.set_commitment(cache),
            VerkleNode::Leaf11(n) => n.set_commitment(cache),
            VerkleNode::Leaf12(n) => n.set_commitment(cache),
            VerkleNode::Leaf13(n) => n.set_commitment(cache),
            VerkleNode::Leaf14(n) => n.set_commitment(cache),
            VerkleNode::Leaf15(n) => n.set_commitment(cache),
            VerkleNode::Leaf16(n) => n.set_commitment(cache),
            VerkleNode::Leaf18(n) => n.set_commitment(cache),
            VerkleNode::Leaf20(n) => n.set_commitment(cache),
            VerkleNode::Leaf23(n) => n.set_commitment(cache),
            VerkleNode::Leaf26(n) => n.set_commitment(cache),
            VerkleNode::Leaf29(n) => n.set_commitment(cache),
            VerkleNode::Leaf33(n) => n.set_commitment(cache),
            VerkleNode::Leaf36(n) => n.set_commitment(cache),
            VerkleNode::Leaf39(n) => n.set_commitment(cache),
            VerkleNode::Leaf40(n) => n.set_commitment(cache),
            VerkleNode::Leaf41(n) => n.set_commitment(cache),
            VerkleNode::Leaf45(n) => n.set_commitment(cache),
            VerkleNode::Leaf48(n) => n.set_commitment(cache),
            VerkleNode::Leaf52(n) => n.set_commitment(cache),
            VerkleNode::Leaf57(n) => n.set_commitment(cache),
            VerkleNode::Leaf61(n) => n.set_commitment(cache),
            VerkleNode::Leaf64(n) => n.set_commitment(cache),
            VerkleNode::Leaf70(n) => n.set_commitment(cache),
            VerkleNode::Leaf75(n) => n.set_commitment(cache),
            VerkleNode::Leaf80(n) => n.set_commitment(cache),
            VerkleNode::Leaf84(n) => n.set_commitment(cache),
            VerkleNode::Leaf89(n) => n.set_commitment(cache),
            VerkleNode::Leaf94(n) => n.set_commitment(cache),
            VerkleNode::Leaf100(n) => n.set_commitment(cache),
            VerkleNode::Leaf107(n) => n.set_commitment(cache),
            VerkleNode::Leaf113(n) => n.set_commitment(cache),
            VerkleNode::Leaf120(n) => n.set_commitment(cache),
            VerkleNode::Leaf124(n) => n.set_commitment(cache),
            VerkleNode::Leaf130(n) => n.set_commitment(cache),
            VerkleNode::Leaf132(n) => n.set_commitment(cache),
            VerkleNode::Leaf134(n) => n.set_commitment(cache),
            VerkleNode::Leaf136(n) => n.set_commitment(cache),
            VerkleNode::Leaf138(n) => n.set_commitment(cache),
            VerkleNode::Leaf140(n) => n.set_commitment(cache),
            VerkleNode::Leaf142(n) => n.set_commitment(cache),
            VerkleNode::Leaf146(n) => n.set_commitment(cache),
            VerkleNode::Leaf149(n) => n.set_commitment(cache),
            VerkleNode::Leaf156(n) => n.set_commitment(cache),
            VerkleNode::Leaf162(n) => n.set_commitment(cache),
            VerkleNode::Leaf168(n) => n.set_commitment(cache),
            VerkleNode::Leaf178(n) => n.set_commitment(cache),
            VerkleNode::Leaf186(n) => n.set_commitment(cache),
            VerkleNode::Leaf195(n) => n.set_commitment(cache),
            VerkleNode::Leaf203(n) => n.set_commitment(cache),
            VerkleNode::Leaf208(n) => n.set_commitment(cache),
            VerkleNode::Leaf225(n) => n.set_commitment(cache),
            VerkleNode::Leaf239(n) => n.set_commitment(cache),
            VerkleNode::Leaf256(n) => n.set_commitment(cache),
            VerkleNode::LeafDelta(n) => n.set_commitment(cache),
        }
    }
}

/// A node type of a node in a managed Verkle trie.
/// This type is primarily used for conversion between [`VerkleNode`] and indexes in the file
/// storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VerkleNodeKind {
    Empty,
    Inner9,
    Inner15,
    Inner21,
    Inner256,
    InnerDelta,
    Leaf1,
    Leaf2,
    Leaf3,
    Leaf4,
    Leaf5,
    Leaf6,
    Leaf7,
    Leaf8,
    Leaf9,
    Leaf10,
    Leaf11,
    Leaf12,
    Leaf13,
    Leaf14,
    Leaf15,
    Leaf16,
    Leaf18,
    Leaf20,
    Leaf23,
    Leaf26,
    Leaf29,
    Leaf33,
    Leaf36,
    Leaf39,
    Leaf40,
    Leaf41,
    Leaf45,
    Leaf48,
    Leaf52,
    Leaf57,
    Leaf61,
    Leaf64,
    Leaf70,
    Leaf75,
    Leaf80,
    Leaf84,
    Leaf89,
    Leaf94,
    Leaf100,
    Leaf107,
    Leaf113,
    Leaf120,
    Leaf124,
    Leaf130,
    Leaf132,
    Leaf134,
    Leaf136,
    Leaf138,
    Leaf140,
    Leaf142,
    Leaf146,
    Leaf149,
    Leaf156,
    Leaf162,
    Leaf168,
    Leaf178,
    Leaf186,
    Leaf195,
    Leaf203,
    Leaf208,
    Leaf225,
    Leaf239,
    Leaf256,
    LeafDelta,
}

impl NodeSize for VerkleNodeKind {
    fn node_byte_size(&self) -> usize {
        let inner_size = match self {
            VerkleNodeKind::Empty => std::mem::size_of::<VerkleNode>(),
            VerkleNodeKind::Inner9 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<9>>>()
                    + std::mem::size_of::<SparseInnerNode<9>>()
            }
            VerkleNodeKind::Inner15 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<15>>>()
                    + std::mem::size_of::<SparseInnerNode<15>>()
            }
            VerkleNodeKind::Inner21 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<21>>>()
                    + std::mem::size_of::<SparseInnerNode<21>>()
            }
            VerkleNodeKind::Inner256 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<FullInnerNode>>()
                    + std::mem::size_of::<FullInnerNode>()
            }
            VerkleNodeKind::InnerDelta => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<InnerDeltaNode>>()
                    + std::mem::size_of::<InnerDeltaNode>()
            }
            VerkleNodeKind::Leaf1 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<1>>>()
                    + std::mem::size_of::<SparseLeafNode<1>>()
            }
            VerkleNodeKind::Leaf2 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<2>>>()
                    + std::mem::size_of::<SparseLeafNode<2>>()
            }
            VerkleNodeKind::Leaf3 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<3>>>()
                    + std::mem::size_of::<SparseLeafNode<3>>()
            }
            VerkleNodeKind::Leaf4 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<4>>>()
                    + std::mem::size_of::<SparseLeafNode<4>>()
            }
            VerkleNodeKind::Leaf5 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<5>>>()
                    + std::mem::size_of::<SparseLeafNode<5>>()
            }
            VerkleNodeKind::Leaf6 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<6>>>()
                    + std::mem::size_of::<SparseLeafNode<6>>()
            }
            VerkleNodeKind::Leaf7 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<7>>>()
                    + std::mem::size_of::<SparseLeafNode<7>>()
            }
            VerkleNodeKind::Leaf8 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<8>>>()
                    + std::mem::size_of::<SparseLeafNode<8>>()
            }
            VerkleNodeKind::Leaf9 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<9>>>()
                    + std::mem::size_of::<SparseLeafNode<9>>()
            }
            VerkleNodeKind::Leaf10 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<10>>>()
                    + std::mem::size_of::<SparseLeafNode<10>>()
            }
            VerkleNodeKind::Leaf11 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<11>>>()
                    + std::mem::size_of::<SparseLeafNode<11>>()
            }
            VerkleNodeKind::Leaf12 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<12>>>()
                    + std::mem::size_of::<SparseLeafNode<12>>()
            }
            VerkleNodeKind::Leaf13 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<13>>>()
                    + std::mem::size_of::<SparseLeafNode<13>>()
            }
            VerkleNodeKind::Leaf14 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<14>>>()
                    + std::mem::size_of::<SparseLeafNode<14>>()
            }
            VerkleNodeKind::Leaf15 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<15>>>()
                    + std::mem::size_of::<SparseLeafNode<15>>()
            }
            VerkleNodeKind::Leaf16 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<16>>>()
                    + std::mem::size_of::<SparseLeafNode<16>>()
            }
            VerkleNodeKind::Leaf18 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<18>>>()
                    + std::mem::size_of::<SparseLeafNode<18>>()
            }
            VerkleNodeKind::Leaf20 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<20>>>()
                    + std::mem::size_of::<SparseLeafNode<20>>()
            }
            VerkleNodeKind::Leaf23 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<23>>>()
                    + std::mem::size_of::<SparseLeafNode<23>>()
            }
            VerkleNodeKind::Leaf26 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<26>>>()
                    + std::mem::size_of::<SparseLeafNode<26>>()
            }
            VerkleNodeKind::Leaf29 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<29>>>()
                    + std::mem::size_of::<SparseLeafNode<29>>()
            }
            VerkleNodeKind::Leaf33 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<33>>>()
                    + std::mem::size_of::<SparseLeafNode<33>>()
            }
            VerkleNodeKind::Leaf36 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<36>>>()
                    + std::mem::size_of::<SparseLeafNode<36>>()
            }
            VerkleNodeKind::Leaf39 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<39>>>()
                    + std::mem::size_of::<SparseLeafNode<39>>()
            }
            VerkleNodeKind::Leaf40 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<40>>>()
                    + std::mem::size_of::<SparseLeafNode<40>>()
            }
            VerkleNodeKind::Leaf41 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<41>>>()
                    + std::mem::size_of::<SparseLeafNode<41>>()
            }
            VerkleNodeKind::Leaf45 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<45>>>()
                    + std::mem::size_of::<SparseLeafNode<45>>()
            }
            VerkleNodeKind::Leaf48 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<48>>>()
                    + std::mem::size_of::<SparseLeafNode<48>>()
            }
            VerkleNodeKind::Leaf52 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<52>>>()
                    + std::mem::size_of::<SparseLeafNode<52>>()
            }
            VerkleNodeKind::Leaf57 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<57>>>()
                    + std::mem::size_of::<SparseLeafNode<57>>()
            }
            VerkleNodeKind::Leaf61 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<61>>>()
                    + std::mem::size_of::<SparseLeafNode<61>>()
            }
            VerkleNodeKind::Leaf64 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<64>>>()
                    + std::mem::size_of::<SparseLeafNode<64>>()
            }
            VerkleNodeKind::Leaf70 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<70>>>()
                    + std::mem::size_of::<SparseLeafNode<70>>()
            }
            VerkleNodeKind::Leaf75 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<75>>>()
                    + std::mem::size_of::<SparseLeafNode<75>>()
            }
            VerkleNodeKind::Leaf80 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<80>>>()
                    + std::mem::size_of::<SparseLeafNode<80>>()
            }
            VerkleNodeKind::Leaf84 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<84>>>()
                    + std::mem::size_of::<SparseLeafNode<84>>()
            }
            VerkleNodeKind::Leaf89 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<89>>>()
                    + std::mem::size_of::<SparseLeafNode<89>>()
            }
            VerkleNodeKind::Leaf94 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<94>>>()
                    + std::mem::size_of::<SparseLeafNode<94>>()
            }
            VerkleNodeKind::Leaf100 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<100>>>()
                    + std::mem::size_of::<SparseLeafNode<100>>()
            }
            VerkleNodeKind::Leaf107 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<107>>>()
                    + std::mem::size_of::<SparseLeafNode<107>>()
            }
            VerkleNodeKind::Leaf113 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<113>>>()
                    + std::mem::size_of::<SparseLeafNode<113>>()
            }
            VerkleNodeKind::Leaf120 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<120>>>()
                    + std::mem::size_of::<SparseLeafNode<120>>()
            }
            VerkleNodeKind::Leaf124 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<124>>>()
                    + std::mem::size_of::<SparseLeafNode<124>>()
            }
            VerkleNodeKind::Leaf130 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<130>>>()
                    + std::mem::size_of::<SparseLeafNode<130>>()
            }
            VerkleNodeKind::Leaf132 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<132>>>()
                    + std::mem::size_of::<SparseLeafNode<132>>()
            }
            VerkleNodeKind::Leaf134 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<134>>>()
                    + std::mem::size_of::<SparseLeafNode<134>>()
            }
            VerkleNodeKind::Leaf136 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<136>>>()
                    + std::mem::size_of::<SparseLeafNode<136>>()
            }
            VerkleNodeKind::Leaf138 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<138>>>()
                    + std::mem::size_of::<SparseLeafNode<138>>()
            }
            VerkleNodeKind::Leaf140 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<140>>>()
                    + std::mem::size_of::<SparseLeafNode<140>>()
            }
            VerkleNodeKind::Leaf142 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<142>>>()
                    + std::mem::size_of::<SparseLeafNode<142>>()
            }
            VerkleNodeKind::Leaf146 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<146>>>()
                    + std::mem::size_of::<SparseLeafNode<146>>()
            }
            VerkleNodeKind::Leaf149 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<149>>>()
                    + std::mem::size_of::<SparseLeafNode<149>>()
            }
            VerkleNodeKind::Leaf156 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<156>>>()
                    + std::mem::size_of::<SparseLeafNode<156>>()
            }
            VerkleNodeKind::Leaf162 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<162>>>()
                    + std::mem::size_of::<SparseLeafNode<162>>()
            }
            VerkleNodeKind::Leaf168 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<168>>>()
                    + std::mem::size_of::<SparseLeafNode<168>>()
            }
            VerkleNodeKind::Leaf178 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<178>>>()
                    + std::mem::size_of::<SparseLeafNode<178>>()
            }
            VerkleNodeKind::Leaf186 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<186>>>()
                    + std::mem::size_of::<SparseLeafNode<186>>()
            }
            VerkleNodeKind::Leaf195 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<195>>>()
                    + std::mem::size_of::<SparseLeafNode<195>>()
            }
            VerkleNodeKind::Leaf203 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<203>>>()
                    + std::mem::size_of::<SparseLeafNode<203>>()
            }
            VerkleNodeKind::Leaf208 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<208>>>()
                    + std::mem::size_of::<SparseLeafNode<208>>()
            }
            VerkleNodeKind::Leaf225 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<225>>>()
                    + std::mem::size_of::<SparseLeafNode<225>>()
            }
            VerkleNodeKind::Leaf239 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<239>>>()
                    + std::mem::size_of::<SparseLeafNode<239>>()
            }
            VerkleNodeKind::Leaf256 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<FullLeafNode>>()
                    + std::mem::size_of::<FullLeafNode>()
            }
            VerkleNodeKind::LeafDelta => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<LeafDeltaNode>>()
                    + std::mem::size_of::<LeafDeltaNode>()
            }
        };
        inner_size
    }

    fn min_non_empty_node_size() -> usize {
        // Because we don't store empty nodes, the minimum size is the smallest non-empty node.
        VerkleNodeKind::Leaf1.node_byte_size()
    }
}

/// An item (value or child ID) stored in a sparse trie node, together with its index.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, Unaligned,
)]
#[repr(C)]
pub struct ItemWithIndex<T> {
    pub index: u8,
    pub item: T,
}

/// A value of a sparse leaf node in a managed Verkle trie, together with its index.
pub type ValueWithIndex = ItemWithIndex<Value>;
/// An ID in a sparse inner node, together with its index.
pub type VerkleIdWithIndex = ItemWithIndex<VerkleNodeId>;

impl<T> ItemWithIndex<T>
where
    T: Default + PartialEq,
{
    /// Creates an array of `N` default-initialized `ItemWithIndex<T>` items, with indexes from `0`
    /// to `N-1`.
    fn default_array<const N: usize>() -> [ItemWithIndex<T>; N] {
        std::array::from_fn(|i| ItemWithIndex {
            index: i as u8,
            item: T::default(),
        })
    }

    /// Returns a slot in `items` for storing an item with the given index, or `None` if no such
    /// slot exists. A slot is suitable if it either already holds the given index, or if it is
    /// empty (i.e., holds the default item).
    fn get_slot_for<const N: usize>(items: &[ItemWithIndex<T>; N], index: u8) -> Option<usize> {
        let mut empty_slot = None;
        // We always do a linear search over all items to ensure that we never hold the same index
        // twice in different slots. By starting the search at the given index we are very likely
        // to find the matching slot immediately in practice (if index < N).
        for (i, iwi) in items
            .iter()
            .enumerate()
            .cycle()
            .skip(index as usize)
            .take(N)
        {
            if iwi.index == index {
                return Some(i);
            } else if empty_slot.is_none() && iwi.item == T::default() {
                empty_slot = Some(i);
            }
        }
        empty_slot
    }

    /// Returns the number of slots that are required to store the given items.
    fn required_slot_count_for<const N: usize>(
        items: &[ItemWithIndex<T>; N],
        indices: impl Iterator<Item = u8>,
    ) -> usize {
        let empty_slots = items.iter().filter(|iwi| iwi.item == T::default()).count();
        let mut new_slots = 0;
        for index in indices {
            if items
                .iter()
                .any(|iwi| iwi.index == index && iwi.item != T::default())
            {
                continue;
            }
            new_slots += 1;
        }
        N - empty_slots + new_slots
    }
}

/// Creates the smallest leaf node capable of storing `n` values, initialized with the given
/// `stem`, `values` and `commitment`.
pub fn make_smallest_leaf_node_for(
    n: usize,
    stem: [u8; 31],
    values: &[ValueWithIndex],
    commitment: &VerkleLeafCommitment,
) -> BTResult<VerkleNode, Error> {
    match VerkleNode::smallest_leaf_type_for(n) {
        VerkleNodeKind::Empty => Ok(VerkleNode::Empty(EmptyNode)),
        VerkleNodeKind::Leaf1 => Ok(VerkleNode::Leaf1(Box::new(
            SparseLeafNode::<1>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf2 => Ok(VerkleNode::Leaf2(Box::new(
            SparseLeafNode::<2>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf3 => Ok(VerkleNode::Leaf3(Box::new(
            SparseLeafNode::<3>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf4 => Ok(VerkleNode::Leaf4(Box::new(
            SparseLeafNode::<4>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf5 => Ok(VerkleNode::Leaf5(Box::new(
            SparseLeafNode::<5>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf6 => Ok(VerkleNode::Leaf6(Box::new(
            SparseLeafNode::<6>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf7 => Ok(VerkleNode::Leaf7(Box::new(
            SparseLeafNode::<7>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf8 => Ok(VerkleNode::Leaf8(Box::new(
            SparseLeafNode::<8>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf9 => Ok(VerkleNode::Leaf9(Box::new(
            SparseLeafNode::<9>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf10 => Ok(VerkleNode::Leaf10(Box::new(
            SparseLeafNode::<10>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf11 => Ok(VerkleNode::Leaf11(Box::new(
            SparseLeafNode::<11>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf12 => Ok(VerkleNode::Leaf12(Box::new(
            SparseLeafNode::<12>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf13 => Ok(VerkleNode::Leaf13(Box::new(
            SparseLeafNode::<13>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf14 => Ok(VerkleNode::Leaf14(Box::new(
            SparseLeafNode::<14>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf15 => Ok(VerkleNode::Leaf15(Box::new(
            SparseLeafNode::<15>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf16 => Ok(VerkleNode::Leaf16(Box::new(
            SparseLeafNode::<16>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf18 => Ok(VerkleNode::Leaf18(Box::new(
            SparseLeafNode::<18>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf20 => Ok(VerkleNode::Leaf20(Box::new(
            SparseLeafNode::<20>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf23 => Ok(VerkleNode::Leaf23(Box::new(
            SparseLeafNode::<23>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf26 => Ok(VerkleNode::Leaf26(Box::new(
            SparseLeafNode::<26>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf29 => Ok(VerkleNode::Leaf29(Box::new(
            SparseLeafNode::<29>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf33 => Ok(VerkleNode::Leaf33(Box::new(
            SparseLeafNode::<33>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf36 => Ok(VerkleNode::Leaf36(Box::new(
            SparseLeafNode::<36>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf39 => Ok(VerkleNode::Leaf39(Box::new(
            SparseLeafNode::<39>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf40 => Ok(VerkleNode::Leaf40(Box::new(
            SparseLeafNode::<40>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf41 => Ok(VerkleNode::Leaf41(Box::new(
            SparseLeafNode::<41>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf45 => Ok(VerkleNode::Leaf45(Box::new(
            SparseLeafNode::<45>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf48 => Ok(VerkleNode::Leaf48(Box::new(
            SparseLeafNode::<48>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf52 => Ok(VerkleNode::Leaf52(Box::new(
            SparseLeafNode::<52>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf57 => Ok(VerkleNode::Leaf57(Box::new(
            SparseLeafNode::<57>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf61 => Ok(VerkleNode::Leaf61(Box::new(
            SparseLeafNode::<61>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf64 => Ok(VerkleNode::Leaf64(Box::new(
            SparseLeafNode::<64>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf70 => Ok(VerkleNode::Leaf70(Box::new(
            SparseLeafNode::<70>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf75 => Ok(VerkleNode::Leaf75(Box::new(
            SparseLeafNode::<75>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf80 => Ok(VerkleNode::Leaf80(Box::new(
            SparseLeafNode::<80>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf84 => Ok(VerkleNode::Leaf84(Box::new(
            SparseLeafNode::<84>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf89 => Ok(VerkleNode::Leaf89(Box::new(
            SparseLeafNode::<89>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf94 => Ok(VerkleNode::Leaf94(Box::new(
            SparseLeafNode::<94>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf100 => Ok(VerkleNode::Leaf100(Box::new(
            SparseLeafNode::<100>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf107 => Ok(VerkleNode::Leaf107(Box::new(
            SparseLeafNode::<107>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf113 => Ok(VerkleNode::Leaf113(Box::new(
            SparseLeafNode::<113>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf120 => Ok(VerkleNode::Leaf120(Box::new(
            SparseLeafNode::<120>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf124 => Ok(VerkleNode::Leaf124(Box::new(
            SparseLeafNode::<124>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf130 => Ok(VerkleNode::Leaf130(Box::new(
            SparseLeafNode::<130>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf132 => Ok(VerkleNode::Leaf132(Box::new(
            SparseLeafNode::<132>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf134 => Ok(VerkleNode::Leaf134(Box::new(
            SparseLeafNode::<134>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf136 => Ok(VerkleNode::Leaf136(Box::new(
            SparseLeafNode::<136>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf138 => Ok(VerkleNode::Leaf138(Box::new(
            SparseLeafNode::<138>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf140 => Ok(VerkleNode::Leaf140(Box::new(
            SparseLeafNode::<140>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf142 => Ok(VerkleNode::Leaf142(Box::new(
            SparseLeafNode::<142>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf146 => Ok(VerkleNode::Leaf146(Box::new(
            SparseLeafNode::<146>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf149 => Ok(VerkleNode::Leaf149(Box::new(
            SparseLeafNode::<149>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf156 => Ok(VerkleNode::Leaf156(Box::new(
            SparseLeafNode::<156>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf162 => Ok(VerkleNode::Leaf162(Box::new(
            SparseLeafNode::<162>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf168 => Ok(VerkleNode::Leaf168(Box::new(
            SparseLeafNode::<168>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf178 => Ok(VerkleNode::Leaf178(Box::new(
            SparseLeafNode::<178>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf186 => Ok(VerkleNode::Leaf186(Box::new(
            SparseLeafNode::<186>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf195 => Ok(VerkleNode::Leaf195(Box::new(
            SparseLeafNode::<195>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf203 => Ok(VerkleNode::Leaf203(Box::new(
            SparseLeafNode::<203>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf208 => Ok(VerkleNode::Leaf208(Box::new(
            SparseLeafNode::<208>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf225 => Ok(VerkleNode::Leaf225(Box::new(
            SparseLeafNode::<225>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf239 => Ok(VerkleNode::Leaf239(Box::new(
            SparseLeafNode::<239>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf256 => {
            let mut new_leaf = FullLeafNode {
                stem,
                commitment: *commitment,
                ..Default::default()
            };
            for v in values {
                new_leaf.values[v.index as usize] = v.item;
            }
            Ok(VerkleNode::Leaf256(Box::new(new_leaf)))
        }
        VerkleNodeKind::LeafDelta => Err(Error::CorruptedState(
            "LeafDelta is not a valid choice for make_smallest_leaf_node_for".to_owned(),
        )
        .into()),
        _ => Err(Error::CorruptedState(
            "received non-leaf type in make_smallest_leaf_node_for".to_owned(),
        )
        .into()),
    }
}

/// Creates the smallest inner node capable of storing `n` children, initialized with the given
/// `children` and `commitment`.
pub fn make_smallest_inner_node_for(
    n: usize,
    children: &[VerkleIdWithIndex],
    commitment: &VerkleInnerCommitment,
) -> BTResult<VerkleNode, Error> {
    match VerkleNode::smallest_inner_type_for(n) {
        VerkleNodeKind::Empty => Ok(VerkleNode::Empty(EmptyNode)),
        VerkleNodeKind::Inner9 => Ok(VerkleNode::Inner9(Box::new(
            SparseInnerNode::<9>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner15 => Ok(VerkleNode::Inner15(Box::new(
            SparseInnerNode::<15>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner21 => Ok(VerkleNode::Inner21(Box::new(
            SparseInnerNode::<21>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner256 => {
            let mut new_inner = FullInnerNode {
                commitment: *commitment,
                ..Default::default()
            };
            for c in children {
                new_inner.children[c.index as usize] = c.item;
            }
            Ok(VerkleNode::Inner256(Box::new(new_inner)))
        }
        VerkleNodeKind::InnerDelta => Err(Error::CorruptedState(
            "InnerDelta is not a valid choice for make_smallest_inner_node_for".to_owned(),
        )
        .into()),
        _ => Err(Error::CorruptedState(
            "received non-inner type in make_smallest_inner_node_for".to_owned(),
        )
        .into()),
    }
}

/// A trait to link together full and sparse inner nodes.
/// It provides a set of operations common to all inner node types.
pub trait VerkleManagedInnerNode {
    /// Returns an iterator over all children in the inner node, together with their indexes.
    fn iter_children(&self) -> Box<dyn Iterator<Item = VerkleIdWithIndex> + '_>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{error::BTError, types::TreeId};

    // NOTE: Tests for the accept method are in managed/mod.rs

    #[test]
    fn node_type_byte_size_returns_correct_size() {
        let empty_node = VerkleNodeKind::Empty;
        let inner9_node = VerkleNodeKind::Inner9;
        let inner15_node = VerkleNodeKind::Inner15;
        let inner21_node = VerkleNodeKind::Inner21;
        let inner256_node = VerkleNodeKind::Inner256;
        let leaf1_node = VerkleNodeKind::Leaf1;
        let leaf2_node = VerkleNodeKind::Leaf2;
        let leaf5_node = VerkleNodeKind::Leaf5;
        let leaf18_node = VerkleNodeKind::Leaf18;
        let leaf146_node = VerkleNodeKind::Leaf146;
        let leaf256_node = VerkleNodeKind::Leaf256;

        assert_eq!(
            empty_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
        );
        assert_eq!(
            inner9_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseInnerNode<9>>>()
                + std::mem::size_of::<SparseInnerNode<9>>()
        );
        assert_eq!(
            inner15_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseInnerNode<15>>>()
                + std::mem::size_of::<SparseInnerNode<15>>()
        );
        assert_eq!(
            inner21_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseInnerNode<21>>>()
                + std::mem::size_of::<SparseInnerNode<21>>()
        );
        assert_eq!(
            inner256_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<FullInnerNode>>()
                + std::mem::size_of::<FullInnerNode>()
        );
        assert_eq!(
            leaf1_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseLeafNode<1>>>()
                + std::mem::size_of::<SparseLeafNode<1>>()
        );
        assert_eq!(
            leaf2_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseLeafNode<2>>>()
                + std::mem::size_of::<SparseLeafNode<2>>()
        );
        assert_eq!(
            leaf5_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseLeafNode<5>>>()
                + std::mem::size_of::<SparseLeafNode<5>>()
        );
        assert_eq!(
            leaf18_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseLeafNode<18>>>()
                + std::mem::size_of::<SparseLeafNode<18>>()
        );
        assert_eq!(
            leaf146_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<SparseLeafNode<146>>>()
                + std::mem::size_of::<SparseLeafNode<146>>()
        );
        assert_eq!(
            leaf256_node.node_byte_size(),
            std::mem::size_of::<VerkleNode>()
                + std::mem::size_of::<Box<FullLeafNode>>()
                + std::mem::size_of::<FullLeafNode>()
        );
    }

    #[test]
    fn node_type_min_non_empty_node_size_returns_size_of_smallest_non_empty_node() {
        assert_eq!(
            VerkleNodeKind::min_non_empty_node_size(),
            VerkleNode::Leaf1(Box::default()).node_byte_size()
        );
    }

    #[test]
    fn node_byte_size_returns_node_type_byte_size() {
        let empty_node = VerkleNode::Empty(EmptyNode);
        let inner9_node = VerkleNode::Inner9(Box::default());
        let inner15_node = VerkleNode::Inner15(Box::default());
        let inner21_node = VerkleNode::Inner21(Box::default());
        let inner256_node = VerkleNode::Inner256(Box::default());
        let leaf1_node = VerkleNode::Leaf1(Box::default());
        let leaf2_node = VerkleNode::Leaf2(Box::default());
        let leaf5_node = VerkleNode::Leaf5(Box::default());
        let leaf18_node = VerkleNode::Leaf18(Box::default());
        let leaf146_node = VerkleNode::Leaf146(Box::default());
        let leaf256_node = VerkleNode::Leaf256(Box::default());

        assert_eq!(
            VerkleNodeKind::Empty.node_byte_size(),
            empty_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Inner9.node_byte_size(),
            inner9_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Inner15.node_byte_size(),
            inner15_node.node_byte_size()
        );

        assert_eq!(
            VerkleNodeKind::Inner21.node_byte_size(),
            inner21_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Inner256.node_byte_size(),
            inner256_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Leaf1.node_byte_size(),
            leaf1_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Leaf2.node_byte_size(),
            leaf2_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Leaf5.node_byte_size(),
            leaf5_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Leaf18.node_byte_size(),
            leaf18_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Leaf146.node_byte_size(),
            leaf146_node.node_byte_size()
        );
        assert_eq!(
            VerkleNodeKind::Leaf256.node_byte_size(),
            leaf256_node.node_byte_size()
        );
    }

    #[test]
    fn node_min_non_empty_node_size_returns_node_type_min_size() {
        assert_eq!(
            VerkleNodeKind::min_non_empty_node_size(),
            VerkleNode::min_non_empty_node_size()
        );
    }

    #[test]
    fn node_count_visitor_visit_visit_nodes() {
        let mut visitor = NodeCountVisitor::default();
        let level = 0;

        let node = VerkleNode::Empty(EmptyNode);
        assert!(visitor.visit(&node, level).is_ok());

        let mut node = FullInnerNode::default();
        for i in 0..256 {
            node.children[i] = VerkleNodeId::from_idx_and_node_kind(1, VerkleNodeKind::Inner256);
        }
        assert!(visitor.visit(&node, level + 1).is_ok());

        let mut node = Leaf2VerkleNode::default();
        for i in 0..2 {
            node.values[i] = ValueWithIndex {
                index: i as u8,
                item: [1; 32],
            };
        }
        let node = VerkleNode::Leaf2(Box::new(node));
        assert!(visitor.visit(&node, level + 2).is_ok());

        let mut node = Leaf256VerkleNode::default();
        for i in 0..256 {
            node.values[i] = [1; 32];
        }
        let node = VerkleNode::Leaf256(Box::new(node));
        assert!(visitor.visit(&node, level + 3).is_ok());

        assert_eq!(visitor.node_count.levels_count.len(), 4);
        assert_eq!(
            visitor.node_count.levels_count[0]
                .get("Empty")
                .unwrap()
                .size_count
                .get(&0),
            Some(&1)
        );
        assert_eq!(
            visitor.node_count.levels_count[1]
                .get("Inner")
                .unwrap()
                .size_count
                .get(&256),
            Some(&1)
        );
        assert_eq!(
            visitor.node_count.levels_count[2]
                .get("Leaf")
                .unwrap()
                .size_count
                .get(&2),
            Some(&1)
        );
        assert_eq!(
            visitor.node_count.levels_count[3]
                .get("Leaf")
                .unwrap()
                .size_count
                .get(&256),
            Some(&1)
        );
    }

    #[test]
    fn item_with_index_default_array_creates_array_of_default_initialized_items_and_unique_ids() {
        let items: [ItemWithIndex<u8>; 5] = ItemWithIndex::default_array();
        for (i, item) in items.iter().enumerate() {
            assert_eq!(item.index, i as u8);
            assert_eq!(item.item, u8::default());
        }
    }

    #[test]
    fn item_with_index_get_slot_returns_slot_with_matching_index_or_empty_slot() {
        type TestItemWithIndex = ItemWithIndex<u8>;
        let mut values = [TestItemWithIndex::default(); 4];
        values[0] = TestItemWithIndex { index: 0, item: 10 };
        values[3] = TestItemWithIndex { index: 5, item: 20 };

        // Matching index
        let slot = TestItemWithIndex::get_slot_for(&values, 0);
        assert_eq!(slot, Some(0));

        // Matching index has precedence over empty slot
        let slot = TestItemWithIndex::get_slot_for(&values, 5);
        assert_eq!(slot, Some(3));

        // No matching index, so we return first empty slot
        let slot = TestItemWithIndex::get_slot_for(&values, 8); // 8 % 4 = 0, so start start search at 0
        assert_eq!(slot, Some(1));

        // No matching index and no empty slot
        values[1] = TestItemWithIndex { index: 1, item: 30 };
        values[2] = TestItemWithIndex { index: 2, item: 40 };
        let slot = TestItemWithIndex::get_slot_for(&values, 250);
        assert_eq!(slot, None);
    }

    #[test]
    fn item_with_index_required_slot_count_for_returns_number_of_required_slots() {
        let mut items = [ItemWithIndex::default(); 5];
        items[1] = ItemWithIndex {
            index: 1,
            item: [1; 32],
        };
        items[2] = ItemWithIndex {
            index: 10,
            item: [2; 32],
        };
        items[3] = ItemWithIndex {
            index: 100,
            item: Value::default(),
        };
        // `items` now has 2 occupied slots (for indices 1 and 10) and 3 empty slots

        // Enough empty slots for all new indices
        let slots = ItemWithIndex::required_slot_count_for(&items, [100, 101, 102].into_iter());
        assert_eq!(slots, 5);

        // Enough empty slots and slots which get overwritten
        let slots =
            ItemWithIndex::required_slot_count_for(&items, [100, 101, 102, 10, 1].into_iter());
        assert_eq!(slots, 5);

        // Not enough empty slots
        let slots =
            ItemWithIndex::required_slot_count_for(&items, [100, 101, 102, 103].into_iter());
        assert_eq!(slots, 6); // 2 existing + 1 reused + 3 new
    }

    #[test]
    fn needs_full_returns_id_of_full_node_if_is_delta_node() {
        let full_inner_node_id = VerkleNodeId::from_idx_and_node_kind(42, VerkleNodeKind::Inner256);
        let inner_delta = InnerDeltaNode {
            children: [VerkleNodeId::default(); 256],
            children_delta: [ItemWithIndex::default(); InnerDeltaNode::DELTA_SIZE],
            base_node_id: full_inner_node_id,
            commitment: VerkleInnerCommitment::default(),
        };
        let verkle_node = VerkleNode::InnerDelta(Box::new(inner_delta));
        assert_eq!(verkle_node.needs_delta_base(), Some(full_inner_node_id));
    }

    #[test]
    fn copy_from_full_copies_children_from_full_node_into_delta_node() {
        let inner_children =
            [VerkleNodeId::from_idx_and_node_kind(1, VerkleNodeKind::Inner256); 256];
        let full_node = VerkleNode::Inner256(Box::new(FullInnerNode {
            children: inner_children,
            commitment: VerkleInnerCommitment::default(),
        }));
        let mut delta_node = VerkleNode::InnerDelta(Box::new(InnerDeltaNode {
            children: [VerkleNodeId::default(); 256],
            children_delta: [ItemWithIndex::default(); InnerDeltaNode::DELTA_SIZE],
            base_node_id: VerkleNodeId::default(),
            commitment: VerkleInnerCommitment::default(),
        }));

        delta_node.copy_from_delta_base(&full_node).unwrap();
        assert!(matches!(delta_node, VerkleNode::InnerDelta(n) if n.children == inner_children));
    }

    #[test]
    fn copy_from_full_returns_error_if_provided_node_is_not_a_full_node() {
        let full_node = VerkleNode::Inner15(Box::new(Inner15VerkleNode {
            children: [ItemWithIndex::default(); 15],
            commitment: VerkleInnerCommitment::default(),
        }));
        let mut delta_node = VerkleNode::InnerDelta(Box::new(InnerDeltaNode {
            children: [VerkleNodeId::default(); 256],
            children_delta: [ItemWithIndex::default(); InnerDeltaNode::DELTA_SIZE],
            base_node_id: VerkleNodeId::default(),
            commitment: VerkleInnerCommitment::default(),
        }));

        assert_eq!(
            delta_node
                .copy_from_delta_base(&full_node)
                .map_err(BTError::into_inner),
            Err(Error::Internal(
                "copy_from_delta_base called with unsupported node".into()
            ))
        );
    }

    #[test]
    fn copy_on_write_transforms_between_full_inner_and_inner_delta_depending_on_available_slots() {
        const DELTA_SIZE: usize = InnerDeltaNode::DELTA_SIZE;

        // full inner to full inner if more changes than delta size
        for num_changed in [DELTA_SIZE + 1, DELTA_SIZE + 2] {
            let full_inner = FullInnerNode::default();
            let verkle_node = VerkleNode::Inner256(Box::new(full_inner));
            let changed_children: Vec<u8> = (0..num_changed as u8).collect();
            let cow_node = verkle_node
                .copy_on_write(
                    VerkleNodeId::from_idx_and_node_kind(1, VerkleNodeKind::Inner256),
                    changed_children,
                )
                .unwrap();
            assert!(matches!(cow_node, VerkleNode::Inner256(_)));
        }

        // full inner to inner delta if less or equal changes than delta size
        for num_changed in [0, 1, DELTA_SIZE - 1, DELTA_SIZE] {
            let full_inner = FullInnerNode::default();
            let verkle_node = VerkleNode::Inner256(Box::new(full_inner));
            let changed_children: Vec<u8> = (0..num_changed as u8).collect();
            let cow_node = verkle_node
                .copy_on_write(
                    VerkleNodeId::from_idx_and_node_kind(1, VerkleNodeKind::Inner256),
                    changed_children,
                )
                .unwrap();
            assert!(matches!(cow_node, VerkleNode::InnerDelta(_)));
        }

        // inner delta to inner delta if enough slots
        for (used_slots, new_slots) in [
            (0, 0),
            (1, DELTA_SIZE - 1),
            (DELTA_SIZE - 1, 1),
            (0, DELTA_SIZE),
            (DELTA_SIZE, 0),
            (DELTA_SIZE / 2, DELTA_SIZE / 2),
        ] {
            let mut children_delta = [ItemWithIndex::default(); InnerDeltaNode::DELTA_SIZE];
            #[allow(clippy::needless_range_loop)]
            for i in 0..used_slots {
                children_delta[i] = ItemWithIndex {
                    index: i as u8,
                    item: VerkleNodeId::from_idx_and_node_kind(i as u64, VerkleNodeKind::Inner256),
                };
            }
            let inner_delta = InnerDeltaNode {
                children: [VerkleNodeId::default(); 256],
                children_delta,
                base_node_id: VerkleNodeId::default(),
                commitment: VerkleInnerCommitment::default(),
            };
            let verkle_node = VerkleNode::InnerDelta(Box::new(inner_delta));
            let changed_children: Vec<u8> =
                (used_slots as u8..(used_slots + new_slots) as u8).collect();
            let cow_node = verkle_node
                .copy_on_write(
                    VerkleNodeId::from_idx_and_node_kind(1, VerkleNodeKind::InnerDelta),
                    changed_children,
                )
                .unwrap();
            assert!(matches!(cow_node, VerkleNode::InnerDelta(_)));
        }

        // inner delta to full inner if not enough slots
        for (used_slots, new_slots) in [
            (0, DELTA_SIZE + 1),
            (1, DELTA_SIZE),
            (DELTA_SIZE, 1),
            (DELTA_SIZE / 2 + 1, DELTA_SIZE / 2 + 1),
        ] {
            let mut children_delta = [ItemWithIndex::default(); InnerDeltaNode::DELTA_SIZE];
            #[allow(clippy::needless_range_loop)]
            for i in 0..used_slots {
                children_delta[i] = ItemWithIndex {
                    index: i as u8,
                    item: VerkleNodeId::from_idx_and_node_kind(i as u64, VerkleNodeKind::Inner256),
                };
            }
            let inner_delta = InnerDeltaNode {
                children: [VerkleNodeId::default(); 256],
                children_delta,
                base_node_id: VerkleNodeId::default(),
                commitment: VerkleInnerCommitment::default(),
            };
            let verkle_node = VerkleNode::InnerDelta(Box::new(inner_delta));
            let changed_children: Vec<u8> =
                (used_slots as u8..(used_slots + new_slots) as u8).collect();
            let cow_node = verkle_node
                .copy_on_write(
                    VerkleNodeId::from_idx_and_node_kind(1, VerkleNodeKind::InnerDelta),
                    changed_children,
                )
                .unwrap();
            assert!(matches!(cow_node, VerkleNode::Inner256(_)));
        }
    }

    /// A supertrait combining [`ManagedTrieNode`] and [`NodeAccess`] for use in rstest tests.
    pub trait VerkleManagedTrieNode<T>:
        ManagedTrieNode<Union = VerkleNode, Id = VerkleNodeId, Commitment = VerkleCommitment>
        + NodeAccess<T>
    where
        T: Clone + Copy + Default + PartialEq + Eq + FromBytes + IntoBytes + Immutable + Unaligned,
    {
    }

    impl<const N: usize> VerkleManagedTrieNode<VerkleNodeId> for SparseInnerNode<N> {}

    /// Helper trait to interact with generic node types in rstest tests.
    pub trait NodeAccess<T>
    where
        T: Clone + Copy + Default + PartialEq + Eq + FromBytes + IntoBytes + Immutable + Unaligned,
    {
        fn access_slot(&mut self, slot: usize) -> &mut ItemWithIndex<T>;

        fn access_stem(&mut self) -> Option<&mut [u8; 31]>;

        fn get_commitment_input(&self) -> VerkleCommitmentInput;
    }
}
