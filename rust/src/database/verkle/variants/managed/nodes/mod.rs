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
    Inner1(Box<Inner1VerkleNode>),
    Inner2(Box<Inner2VerkleNode>),
    Inner3(Box<Inner3VerkleNode>),
    Inner4(Box<Inner4VerkleNode>),
    Inner5(Box<Inner5VerkleNode>),
    Inner6(Box<Inner6VerkleNode>),
    Inner7(Box<Inner7VerkleNode>),
    Inner8(Box<Inner8VerkleNode>),
    Inner9(Box<Inner9VerkleNode>),
    Inner10(Box<Inner10VerkleNode>),
    Inner11(Box<Inner11VerkleNode>),
    Inner12(Box<Inner12VerkleNode>),
    Inner13(Box<Inner13VerkleNode>),
    Inner14(Box<Inner14VerkleNode>),
    Inner15(Box<Inner15VerkleNode>),
    Inner16(Box<Inner16VerkleNode>),
    Inner17(Box<Inner17VerkleNode>),
    Inner18(Box<Inner18VerkleNode>),
    Inner19(Box<Inner19VerkleNode>),
    Inner20(Box<Inner20VerkleNode>),
    Inner21(Box<Inner21VerkleNode>),
    Inner22(Box<Inner22VerkleNode>),
    Inner23(Box<Inner23VerkleNode>),
    Inner24(Box<Inner24VerkleNode>),
    Inner25(Box<Inner25VerkleNode>),
    Inner26(Box<Inner26VerkleNode>),
    Inner27(Box<Inner27VerkleNode>),
    Inner28(Box<Inner28VerkleNode>),
    Inner29(Box<Inner29VerkleNode>),
    Inner30(Box<Inner30VerkleNode>),
    Inner31(Box<Inner31VerkleNode>),
    Inner32(Box<Inner32VerkleNode>),
    Inner33(Box<Inner33VerkleNode>),
    Inner34(Box<Inner34VerkleNode>),
    Inner35(Box<Inner35VerkleNode>),
    Inner36(Box<Inner36VerkleNode>),
    Inner37(Box<Inner37VerkleNode>),
    Inner38(Box<Inner38VerkleNode>),
    Inner39(Box<Inner39VerkleNode>),
    Inner40(Box<Inner40VerkleNode>),
    Inner41(Box<Inner41VerkleNode>),
    Inner42(Box<Inner42VerkleNode>),
    Inner43(Box<Inner43VerkleNode>),
    Inner44(Box<Inner44VerkleNode>),
    Inner45(Box<Inner45VerkleNode>),
    Inner46(Box<Inner46VerkleNode>),
    Inner47(Box<Inner47VerkleNode>),
    Inner48(Box<Inner48VerkleNode>),
    Inner49(Box<Inner49VerkleNode>),
    Inner50(Box<Inner50VerkleNode>),
    Inner51(Box<Inner51VerkleNode>),
    Inner52(Box<Inner52VerkleNode>),
    Inner53(Box<Inner53VerkleNode>),
    Inner54(Box<Inner54VerkleNode>),
    Inner55(Box<Inner55VerkleNode>),
    Inner56(Box<Inner56VerkleNode>),
    Inner57(Box<Inner57VerkleNode>),
    Inner58(Box<Inner58VerkleNode>),
    Inner59(Box<Inner59VerkleNode>),
    Inner60(Box<Inner60VerkleNode>),
    Inner61(Box<Inner61VerkleNode>),
    Inner62(Box<Inner62VerkleNode>),
    Inner63(Box<Inner63VerkleNode>),
    Inner64(Box<Inner64VerkleNode>),
    Inner65(Box<Inner65VerkleNode>),
    Inner66(Box<Inner66VerkleNode>),
    Inner67(Box<Inner67VerkleNode>),
    Inner68(Box<Inner68VerkleNode>),
    Inner69(Box<Inner69VerkleNode>),
    Inner70(Box<Inner70VerkleNode>),
    Inner71(Box<Inner71VerkleNode>),
    Inner72(Box<Inner72VerkleNode>),
    Inner73(Box<Inner73VerkleNode>),
    Inner74(Box<Inner74VerkleNode>),
    Inner75(Box<Inner75VerkleNode>),
    Inner76(Box<Inner76VerkleNode>),
    Inner77(Box<Inner77VerkleNode>),
    Inner78(Box<Inner78VerkleNode>),
    Inner79(Box<Inner79VerkleNode>),
    Inner80(Box<Inner80VerkleNode>),
    Inner81(Box<Inner81VerkleNode>),
    Inner82(Box<Inner82VerkleNode>),
    Inner83(Box<Inner83VerkleNode>),
    Inner84(Box<Inner84VerkleNode>),
    Inner85(Box<Inner85VerkleNode>),
    Inner86(Box<Inner86VerkleNode>),
    Inner87(Box<Inner87VerkleNode>),
    Inner88(Box<Inner88VerkleNode>),
    Inner89(Box<Inner89VerkleNode>),
    Inner90(Box<Inner90VerkleNode>),
    Inner91(Box<Inner91VerkleNode>),
    Inner92(Box<Inner92VerkleNode>),
    Inner93(Box<Inner93VerkleNode>),
    Inner94(Box<Inner94VerkleNode>),
    Inner95(Box<Inner95VerkleNode>),
    Inner96(Box<Inner96VerkleNode>),
    Inner97(Box<Inner97VerkleNode>),
    Inner98(Box<Inner98VerkleNode>),
    Inner99(Box<Inner99VerkleNode>),
    Inner100(Box<Inner100VerkleNode>),
    Inner101(Box<Inner101VerkleNode>),
    Inner102(Box<Inner102VerkleNode>),
    Inner103(Box<Inner103VerkleNode>),
    Inner104(Box<Inner104VerkleNode>),
    Inner105(Box<Inner105VerkleNode>),
    Inner106(Box<Inner106VerkleNode>),
    Inner107(Box<Inner107VerkleNode>),
    Inner108(Box<Inner108VerkleNode>),
    Inner109(Box<Inner109VerkleNode>),
    Inner110(Box<Inner110VerkleNode>),
    Inner111(Box<Inner111VerkleNode>),
    Inner112(Box<Inner112VerkleNode>),
    Inner113(Box<Inner113VerkleNode>),
    Inner114(Box<Inner114VerkleNode>),
    Inner115(Box<Inner115VerkleNode>),
    Inner116(Box<Inner116VerkleNode>),
    Inner117(Box<Inner117VerkleNode>),
    Inner118(Box<Inner118VerkleNode>),
    Inner119(Box<Inner119VerkleNode>),
    Inner120(Box<Inner120VerkleNode>),
    Inner121(Box<Inner121VerkleNode>),
    Inner122(Box<Inner122VerkleNode>),
    Inner123(Box<Inner123VerkleNode>),
    Inner124(Box<Inner124VerkleNode>),
    Inner125(Box<Inner125VerkleNode>),
    Inner126(Box<Inner126VerkleNode>),
    Inner127(Box<Inner127VerkleNode>),
    Inner128(Box<Inner128VerkleNode>),
    Inner129(Box<Inner129VerkleNode>),
    Inner130(Box<Inner130VerkleNode>),
    Inner131(Box<Inner131VerkleNode>),
    Inner132(Box<Inner132VerkleNode>),
    Inner133(Box<Inner133VerkleNode>),
    Inner134(Box<Inner134VerkleNode>),
    Inner135(Box<Inner135VerkleNode>),
    Inner136(Box<Inner136VerkleNode>),
    Inner137(Box<Inner137VerkleNode>),
    Inner138(Box<Inner138VerkleNode>),
    Inner139(Box<Inner139VerkleNode>),
    Inner140(Box<Inner140VerkleNode>),
    Inner141(Box<Inner141VerkleNode>),
    Inner142(Box<Inner142VerkleNode>),
    Inner143(Box<Inner143VerkleNode>),
    Inner144(Box<Inner144VerkleNode>),
    Inner145(Box<Inner145VerkleNode>),
    Inner146(Box<Inner146VerkleNode>),
    Inner147(Box<Inner147VerkleNode>),
    Inner148(Box<Inner148VerkleNode>),
    Inner149(Box<Inner149VerkleNode>),
    Inner150(Box<Inner150VerkleNode>),
    Inner151(Box<Inner151VerkleNode>),
    Inner152(Box<Inner152VerkleNode>),
    Inner153(Box<Inner153VerkleNode>),
    Inner154(Box<Inner154VerkleNode>),
    Inner155(Box<Inner155VerkleNode>),
    Inner156(Box<Inner156VerkleNode>),
    Inner157(Box<Inner157VerkleNode>),
    Inner158(Box<Inner158VerkleNode>),
    Inner159(Box<Inner159VerkleNode>),
    Inner160(Box<Inner160VerkleNode>),
    Inner161(Box<Inner161VerkleNode>),
    Inner162(Box<Inner162VerkleNode>),
    Inner163(Box<Inner163VerkleNode>),
    Inner164(Box<Inner164VerkleNode>),
    Inner165(Box<Inner165VerkleNode>),
    Inner166(Box<Inner166VerkleNode>),
    Inner167(Box<Inner167VerkleNode>),
    Inner168(Box<Inner168VerkleNode>),
    Inner169(Box<Inner169VerkleNode>),
    Inner170(Box<Inner170VerkleNode>),
    Inner171(Box<Inner171VerkleNode>),
    Inner172(Box<Inner172VerkleNode>),
    Inner173(Box<Inner173VerkleNode>),
    Inner174(Box<Inner174VerkleNode>),
    Inner175(Box<Inner175VerkleNode>),
    Inner176(Box<Inner176VerkleNode>),
    Inner177(Box<Inner177VerkleNode>),
    Inner178(Box<Inner178VerkleNode>),
    Inner179(Box<Inner179VerkleNode>),
    Inner180(Box<Inner180VerkleNode>),
    Inner181(Box<Inner181VerkleNode>),
    Inner182(Box<Inner182VerkleNode>),
    Inner183(Box<Inner183VerkleNode>),
    Inner184(Box<Inner184VerkleNode>),
    Inner185(Box<Inner185VerkleNode>),
    Inner186(Box<Inner186VerkleNode>),
    Inner187(Box<Inner187VerkleNode>),
    Inner188(Box<Inner188VerkleNode>),
    Inner189(Box<Inner189VerkleNode>),
    Inner190(Box<Inner190VerkleNode>),
    Inner191(Box<Inner191VerkleNode>),
    Inner192(Box<Inner192VerkleNode>),
    Inner193(Box<Inner193VerkleNode>),
    Inner194(Box<Inner194VerkleNode>),
    Inner195(Box<Inner195VerkleNode>),
    Inner196(Box<Inner196VerkleNode>),
    Inner197(Box<Inner197VerkleNode>),
    Inner198(Box<Inner198VerkleNode>),
    Inner199(Box<Inner199VerkleNode>),
    Inner200(Box<Inner200VerkleNode>),
    Inner201(Box<Inner201VerkleNode>),
    Inner202(Box<Inner202VerkleNode>),
    Inner203(Box<Inner203VerkleNode>),
    Inner204(Box<Inner204VerkleNode>),
    Inner205(Box<Inner205VerkleNode>),
    Inner206(Box<Inner206VerkleNode>),
    Inner207(Box<Inner207VerkleNode>),
    Inner208(Box<Inner208VerkleNode>),
    Inner209(Box<Inner209VerkleNode>),
    Inner210(Box<Inner210VerkleNode>),
    Inner211(Box<Inner211VerkleNode>),
    Inner212(Box<Inner212VerkleNode>),
    Inner213(Box<Inner213VerkleNode>),
    Inner214(Box<Inner214VerkleNode>),
    Inner215(Box<Inner215VerkleNode>),
    Inner216(Box<Inner216VerkleNode>),
    Inner217(Box<Inner217VerkleNode>),
    Inner218(Box<Inner218VerkleNode>),
    Inner219(Box<Inner219VerkleNode>),
    Inner220(Box<Inner220VerkleNode>),
    Inner221(Box<Inner221VerkleNode>),
    Inner222(Box<Inner222VerkleNode>),
    Inner223(Box<Inner223VerkleNode>),
    Inner224(Box<Inner224VerkleNode>),
    Inner225(Box<Inner225VerkleNode>),
    Inner226(Box<Inner226VerkleNode>),
    Inner227(Box<Inner227VerkleNode>),
    Inner228(Box<Inner228VerkleNode>),
    Inner229(Box<Inner229VerkleNode>),
    Inner230(Box<Inner230VerkleNode>),
    Inner231(Box<Inner231VerkleNode>),
    Inner232(Box<Inner232VerkleNode>),
    Inner233(Box<Inner233VerkleNode>),
    Inner234(Box<Inner234VerkleNode>),
    Inner235(Box<Inner235VerkleNode>),
    Inner236(Box<Inner236VerkleNode>),
    Inner237(Box<Inner237VerkleNode>),
    Inner238(Box<Inner238VerkleNode>),
    Inner239(Box<Inner239VerkleNode>),
    Inner240(Box<Inner240VerkleNode>),
    Inner241(Box<Inner241VerkleNode>),
    Inner242(Box<Inner242VerkleNode>),
    Inner243(Box<Inner243VerkleNode>),
    Inner244(Box<Inner244VerkleNode>),
    Inner245(Box<Inner245VerkleNode>),
    Inner246(Box<Inner246VerkleNode>),
    Inner247(Box<Inner247VerkleNode>),
    Inner248(Box<Inner248VerkleNode>),
    Inner249(Box<Inner249VerkleNode>),
    Inner250(Box<Inner250VerkleNode>),
    Inner251(Box<Inner251VerkleNode>),
    Inner252(Box<Inner252VerkleNode>),
    Inner253(Box<Inner253VerkleNode>),
    Inner254(Box<Inner254VerkleNode>),
    Inner255(Box<Inner255VerkleNode>),
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
    Leaf17(Box<Leaf17VerkleNode>),
    Leaf18(Box<Leaf18VerkleNode>),
    Leaf19(Box<Leaf19VerkleNode>),
    Leaf20(Box<Leaf20VerkleNode>),
    Leaf21(Box<Leaf21VerkleNode>),
    Leaf22(Box<Leaf22VerkleNode>),
    Leaf23(Box<Leaf23VerkleNode>),
    Leaf24(Box<Leaf24VerkleNode>),
    Leaf25(Box<Leaf25VerkleNode>),
    Leaf26(Box<Leaf26VerkleNode>),
    Leaf27(Box<Leaf27VerkleNode>),
    Leaf28(Box<Leaf28VerkleNode>),
    Leaf29(Box<Leaf29VerkleNode>),
    Leaf30(Box<Leaf30VerkleNode>),
    Leaf31(Box<Leaf31VerkleNode>),
    Leaf32(Box<Leaf32VerkleNode>),
    Leaf33(Box<Leaf33VerkleNode>),
    Leaf34(Box<Leaf34VerkleNode>),
    Leaf35(Box<Leaf35VerkleNode>),
    Leaf36(Box<Leaf36VerkleNode>),
    Leaf37(Box<Leaf37VerkleNode>),
    Leaf38(Box<Leaf38VerkleNode>),
    Leaf39(Box<Leaf39VerkleNode>),
    Leaf40(Box<Leaf40VerkleNode>),
    Leaf41(Box<Leaf41VerkleNode>),
    Leaf42(Box<Leaf42VerkleNode>),
    Leaf43(Box<Leaf43VerkleNode>),
    Leaf44(Box<Leaf44VerkleNode>),
    Leaf45(Box<Leaf45VerkleNode>),
    Leaf46(Box<Leaf46VerkleNode>),
    Leaf47(Box<Leaf47VerkleNode>),
    Leaf48(Box<Leaf48VerkleNode>),
    Leaf49(Box<Leaf49VerkleNode>),
    Leaf50(Box<Leaf50VerkleNode>),
    Leaf51(Box<Leaf51VerkleNode>),
    Leaf52(Box<Leaf52VerkleNode>),
    Leaf53(Box<Leaf53VerkleNode>),
    Leaf54(Box<Leaf54VerkleNode>),
    Leaf55(Box<Leaf55VerkleNode>),
    Leaf56(Box<Leaf56VerkleNode>),
    Leaf57(Box<Leaf57VerkleNode>),
    Leaf58(Box<Leaf58VerkleNode>),
    Leaf59(Box<Leaf59VerkleNode>),
    Leaf60(Box<Leaf60VerkleNode>),
    Leaf61(Box<Leaf61VerkleNode>),
    Leaf62(Box<Leaf62VerkleNode>),
    Leaf63(Box<Leaf63VerkleNode>),
    Leaf64(Box<Leaf64VerkleNode>),
    Leaf65(Box<Leaf65VerkleNode>),
    Leaf66(Box<Leaf66VerkleNode>),
    Leaf67(Box<Leaf67VerkleNode>),
    Leaf68(Box<Leaf68VerkleNode>),
    Leaf69(Box<Leaf69VerkleNode>),
    Leaf70(Box<Leaf70VerkleNode>),
    Leaf71(Box<Leaf71VerkleNode>),
    Leaf72(Box<Leaf72VerkleNode>),
    Leaf73(Box<Leaf73VerkleNode>),
    Leaf74(Box<Leaf74VerkleNode>),
    Leaf75(Box<Leaf75VerkleNode>),
    Leaf76(Box<Leaf76VerkleNode>),
    Leaf77(Box<Leaf77VerkleNode>),
    Leaf78(Box<Leaf78VerkleNode>),
    Leaf79(Box<Leaf79VerkleNode>),
    Leaf80(Box<Leaf80VerkleNode>),
    Leaf81(Box<Leaf81VerkleNode>),
    Leaf82(Box<Leaf82VerkleNode>),
    Leaf83(Box<Leaf83VerkleNode>),
    Leaf84(Box<Leaf84VerkleNode>),
    Leaf85(Box<Leaf85VerkleNode>),
    Leaf86(Box<Leaf86VerkleNode>),
    Leaf87(Box<Leaf87VerkleNode>),
    Leaf88(Box<Leaf88VerkleNode>),
    Leaf89(Box<Leaf89VerkleNode>),
    Leaf90(Box<Leaf90VerkleNode>),
    Leaf91(Box<Leaf91VerkleNode>),
    Leaf92(Box<Leaf92VerkleNode>),
    Leaf93(Box<Leaf93VerkleNode>),
    Leaf94(Box<Leaf94VerkleNode>),
    Leaf95(Box<Leaf95VerkleNode>),
    Leaf96(Box<Leaf96VerkleNode>),
    Leaf97(Box<Leaf97VerkleNode>),
    Leaf98(Box<Leaf98VerkleNode>),
    Leaf99(Box<Leaf99VerkleNode>),
    Leaf100(Box<Leaf100VerkleNode>),
    Leaf101(Box<Leaf101VerkleNode>),
    Leaf102(Box<Leaf102VerkleNode>),
    Leaf103(Box<Leaf103VerkleNode>),
    Leaf104(Box<Leaf104VerkleNode>),
    Leaf105(Box<Leaf105VerkleNode>),
    Leaf106(Box<Leaf106VerkleNode>),
    Leaf107(Box<Leaf107VerkleNode>),
    Leaf108(Box<Leaf108VerkleNode>),
    Leaf109(Box<Leaf109VerkleNode>),
    Leaf110(Box<Leaf110VerkleNode>),
    Leaf111(Box<Leaf111VerkleNode>),
    Leaf112(Box<Leaf112VerkleNode>),
    Leaf113(Box<Leaf113VerkleNode>),
    Leaf114(Box<Leaf114VerkleNode>),
    Leaf115(Box<Leaf115VerkleNode>),
    Leaf116(Box<Leaf116VerkleNode>),
    Leaf117(Box<Leaf117VerkleNode>),
    Leaf118(Box<Leaf118VerkleNode>),
    Leaf119(Box<Leaf119VerkleNode>),
    Leaf120(Box<Leaf120VerkleNode>),
    Leaf121(Box<Leaf121VerkleNode>),
    Leaf122(Box<Leaf122VerkleNode>),
    Leaf123(Box<Leaf123VerkleNode>),
    Leaf124(Box<Leaf124VerkleNode>),
    Leaf125(Box<Leaf125VerkleNode>),
    Leaf126(Box<Leaf126VerkleNode>),
    Leaf127(Box<Leaf127VerkleNode>),
    Leaf128(Box<Leaf128VerkleNode>),
    Leaf129(Box<Leaf129VerkleNode>),
    Leaf130(Box<Leaf130VerkleNode>),
    Leaf131(Box<Leaf131VerkleNode>),
    Leaf132(Box<Leaf132VerkleNode>),
    Leaf133(Box<Leaf133VerkleNode>),
    Leaf134(Box<Leaf134VerkleNode>),
    Leaf135(Box<Leaf135VerkleNode>),
    Leaf136(Box<Leaf136VerkleNode>),
    Leaf137(Box<Leaf137VerkleNode>),
    Leaf138(Box<Leaf138VerkleNode>),
    Leaf139(Box<Leaf139VerkleNode>),
    Leaf140(Box<Leaf140VerkleNode>),
    Leaf141(Box<Leaf141VerkleNode>),
    Leaf142(Box<Leaf142VerkleNode>),
    Leaf143(Box<Leaf143VerkleNode>),
    Leaf144(Box<Leaf144VerkleNode>),
    Leaf145(Box<Leaf145VerkleNode>),
    Leaf146(Box<Leaf146VerkleNode>),
    Leaf147(Box<Leaf147VerkleNode>),
    Leaf148(Box<Leaf148VerkleNode>),
    Leaf149(Box<Leaf149VerkleNode>),
    Leaf150(Box<Leaf150VerkleNode>),
    Leaf151(Box<Leaf151VerkleNode>),
    Leaf152(Box<Leaf152VerkleNode>),
    Leaf153(Box<Leaf153VerkleNode>),
    Leaf154(Box<Leaf154VerkleNode>),
    Leaf155(Box<Leaf155VerkleNode>),
    Leaf156(Box<Leaf156VerkleNode>),
    Leaf157(Box<Leaf157VerkleNode>),
    Leaf158(Box<Leaf158VerkleNode>),
    Leaf159(Box<Leaf159VerkleNode>),
    Leaf160(Box<Leaf160VerkleNode>),
    Leaf161(Box<Leaf161VerkleNode>),
    Leaf162(Box<Leaf162VerkleNode>),
    Leaf163(Box<Leaf163VerkleNode>),
    Leaf164(Box<Leaf164VerkleNode>),
    Leaf165(Box<Leaf165VerkleNode>),
    Leaf166(Box<Leaf166VerkleNode>),
    Leaf167(Box<Leaf167VerkleNode>),
    Leaf168(Box<Leaf168VerkleNode>),
    Leaf169(Box<Leaf169VerkleNode>),
    Leaf170(Box<Leaf170VerkleNode>),
    Leaf171(Box<Leaf171VerkleNode>),
    Leaf172(Box<Leaf172VerkleNode>),
    Leaf173(Box<Leaf173VerkleNode>),
    Leaf174(Box<Leaf174VerkleNode>),
    Leaf175(Box<Leaf175VerkleNode>),
    Leaf176(Box<Leaf176VerkleNode>),
    Leaf177(Box<Leaf177VerkleNode>),
    Leaf178(Box<Leaf178VerkleNode>),
    Leaf179(Box<Leaf179VerkleNode>),
    Leaf180(Box<Leaf180VerkleNode>),
    Leaf181(Box<Leaf181VerkleNode>),
    Leaf182(Box<Leaf182VerkleNode>),
    Leaf183(Box<Leaf183VerkleNode>),
    Leaf184(Box<Leaf184VerkleNode>),
    Leaf185(Box<Leaf185VerkleNode>),
    Leaf186(Box<Leaf186VerkleNode>),
    Leaf187(Box<Leaf187VerkleNode>),
    Leaf188(Box<Leaf188VerkleNode>),
    Leaf189(Box<Leaf189VerkleNode>),
    Leaf190(Box<Leaf190VerkleNode>),
    Leaf191(Box<Leaf191VerkleNode>),
    Leaf192(Box<Leaf192VerkleNode>),
    Leaf193(Box<Leaf193VerkleNode>),
    Leaf194(Box<Leaf194VerkleNode>),
    Leaf195(Box<Leaf195VerkleNode>),
    Leaf196(Box<Leaf196VerkleNode>),
    Leaf197(Box<Leaf197VerkleNode>),
    Leaf198(Box<Leaf198VerkleNode>),
    Leaf199(Box<Leaf199VerkleNode>),
    Leaf200(Box<Leaf200VerkleNode>),
    Leaf201(Box<Leaf201VerkleNode>),
    Leaf202(Box<Leaf202VerkleNode>),
    Leaf203(Box<Leaf203VerkleNode>),
    Leaf204(Box<Leaf204VerkleNode>),
    Leaf205(Box<Leaf205VerkleNode>),
    Leaf206(Box<Leaf206VerkleNode>),
    Leaf207(Box<Leaf207VerkleNode>),
    Leaf208(Box<Leaf208VerkleNode>),
    Leaf209(Box<Leaf209VerkleNode>),
    Leaf210(Box<Leaf210VerkleNode>),
    Leaf211(Box<Leaf211VerkleNode>),
    Leaf212(Box<Leaf212VerkleNode>),
    Leaf213(Box<Leaf213VerkleNode>),
    Leaf214(Box<Leaf214VerkleNode>),
    Leaf215(Box<Leaf215VerkleNode>),
    Leaf216(Box<Leaf216VerkleNode>),
    Leaf217(Box<Leaf217VerkleNode>),
    Leaf218(Box<Leaf218VerkleNode>),
    Leaf219(Box<Leaf219VerkleNode>),
    Leaf220(Box<Leaf220VerkleNode>),
    Leaf221(Box<Leaf221VerkleNode>),
    Leaf222(Box<Leaf222VerkleNode>),
    Leaf223(Box<Leaf223VerkleNode>),
    Leaf224(Box<Leaf224VerkleNode>),
    Leaf225(Box<Leaf225VerkleNode>),
    Leaf226(Box<Leaf226VerkleNode>),
    Leaf227(Box<Leaf227VerkleNode>),
    Leaf228(Box<Leaf228VerkleNode>),
    Leaf229(Box<Leaf229VerkleNode>),
    Leaf230(Box<Leaf230VerkleNode>),
    Leaf231(Box<Leaf231VerkleNode>),
    Leaf232(Box<Leaf232VerkleNode>),
    Leaf233(Box<Leaf233VerkleNode>),
    Leaf234(Box<Leaf234VerkleNode>),
    Leaf235(Box<Leaf235VerkleNode>),
    Leaf236(Box<Leaf236VerkleNode>),
    Leaf237(Box<Leaf237VerkleNode>),
    Leaf238(Box<Leaf238VerkleNode>),
    Leaf239(Box<Leaf239VerkleNode>),
    Leaf240(Box<Leaf240VerkleNode>),
    Leaf241(Box<Leaf241VerkleNode>),
    Leaf242(Box<Leaf242VerkleNode>),
    Leaf243(Box<Leaf243VerkleNode>),
    Leaf244(Box<Leaf244VerkleNode>),
    Leaf245(Box<Leaf245VerkleNode>),
    Leaf246(Box<Leaf246VerkleNode>),
    Leaf247(Box<Leaf247VerkleNode>),
    Leaf248(Box<Leaf248VerkleNode>),
    Leaf249(Box<Leaf249VerkleNode>),
    Leaf250(Box<Leaf250VerkleNode>),
    Leaf251(Box<Leaf251VerkleNode>),
    Leaf252(Box<Leaf252VerkleNode>),
    Leaf253(Box<Leaf253VerkleNode>),
    Leaf254(Box<Leaf254VerkleNode>),
    Leaf255(Box<Leaf255VerkleNode>),
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
    pub fn smallest_leaf_type_for(n: usize) -> VerkleNodeKind {
        match n {
            0 => VerkleNodeKind::Empty,
            1 => VerkleNodeKind::Leaf1,
            2 => VerkleNodeKind::Leaf2,
            3 => VerkleNodeKind::Leaf3,
            4 => VerkleNodeKind::Leaf4,
            5 => VerkleNodeKind::Leaf5,
            6 => VerkleNodeKind::Leaf6,
            7 => VerkleNodeKind::Leaf7,
            8 => VerkleNodeKind::Leaf8,
            9 => VerkleNodeKind::Leaf9,
            10 => VerkleNodeKind::Leaf10,
            11 => VerkleNodeKind::Leaf11,
            12 => VerkleNodeKind::Leaf12,
            13 => VerkleNodeKind::Leaf13,
            14 => VerkleNodeKind::Leaf14,
            15 => VerkleNodeKind::Leaf15,
            16 => VerkleNodeKind::Leaf16,
            17 => VerkleNodeKind::Leaf17,
            18 => VerkleNodeKind::Leaf18,
            19 => VerkleNodeKind::Leaf19,
            20 => VerkleNodeKind::Leaf20,
            21 => VerkleNodeKind::Leaf21,
            22 => VerkleNodeKind::Leaf22,
            23 => VerkleNodeKind::Leaf23,
            24 => VerkleNodeKind::Leaf24,
            25 => VerkleNodeKind::Leaf25,
            26 => VerkleNodeKind::Leaf26,
            27 => VerkleNodeKind::Leaf27,
            28 => VerkleNodeKind::Leaf28,
            29 => VerkleNodeKind::Leaf29,
            30 => VerkleNodeKind::Leaf30,
            31 => VerkleNodeKind::Leaf31,
            32 => VerkleNodeKind::Leaf32,
            33 => VerkleNodeKind::Leaf33,
            34 => VerkleNodeKind::Leaf34,
            35 => VerkleNodeKind::Leaf35,
            36 => VerkleNodeKind::Leaf36,
            37 => VerkleNodeKind::Leaf37,
            38 => VerkleNodeKind::Leaf38,
            39 => VerkleNodeKind::Leaf39,
            40 => VerkleNodeKind::Leaf40,
            41 => VerkleNodeKind::Leaf41,
            42 => VerkleNodeKind::Leaf42,
            43 => VerkleNodeKind::Leaf43,
            44 => VerkleNodeKind::Leaf44,
            45 => VerkleNodeKind::Leaf45,
            46 => VerkleNodeKind::Leaf46,
            47 => VerkleNodeKind::Leaf47,
            48 => VerkleNodeKind::Leaf48,
            49 => VerkleNodeKind::Leaf49,
            50 => VerkleNodeKind::Leaf50,
            51 => VerkleNodeKind::Leaf51,
            52 => VerkleNodeKind::Leaf52,
            53 => VerkleNodeKind::Leaf53,
            54 => VerkleNodeKind::Leaf54,
            55 => VerkleNodeKind::Leaf55,
            56 => VerkleNodeKind::Leaf56,
            57 => VerkleNodeKind::Leaf57,
            58 => VerkleNodeKind::Leaf58,
            59 => VerkleNodeKind::Leaf59,
            60 => VerkleNodeKind::Leaf60,
            61 => VerkleNodeKind::Leaf61,
            62 => VerkleNodeKind::Leaf62,
            63 => VerkleNodeKind::Leaf63,
            64 => VerkleNodeKind::Leaf64,
            65 => VerkleNodeKind::Leaf65,
            66 => VerkleNodeKind::Leaf66,
            67 => VerkleNodeKind::Leaf67,
            68 => VerkleNodeKind::Leaf68,
            69 => VerkleNodeKind::Leaf69,
            70 => VerkleNodeKind::Leaf70,
            71 => VerkleNodeKind::Leaf71,
            72 => VerkleNodeKind::Leaf72,
            73 => VerkleNodeKind::Leaf73,
            74 => VerkleNodeKind::Leaf74,
            75 => VerkleNodeKind::Leaf75,
            76 => VerkleNodeKind::Leaf76,
            77 => VerkleNodeKind::Leaf77,
            78 => VerkleNodeKind::Leaf78,
            79 => VerkleNodeKind::Leaf79,
            80 => VerkleNodeKind::Leaf80,
            81 => VerkleNodeKind::Leaf81,
            82 => VerkleNodeKind::Leaf82,
            83 => VerkleNodeKind::Leaf83,
            84 => VerkleNodeKind::Leaf84,
            85 => VerkleNodeKind::Leaf85,
            86 => VerkleNodeKind::Leaf86,
            87 => VerkleNodeKind::Leaf87,
            88 => VerkleNodeKind::Leaf88,
            89 => VerkleNodeKind::Leaf89,
            90 => VerkleNodeKind::Leaf90,
            91 => VerkleNodeKind::Leaf91,
            92 => VerkleNodeKind::Leaf92,
            93 => VerkleNodeKind::Leaf93,
            94 => VerkleNodeKind::Leaf94,
            95 => VerkleNodeKind::Leaf95,
            96 => VerkleNodeKind::Leaf96,
            97 => VerkleNodeKind::Leaf97,
            98 => VerkleNodeKind::Leaf98,
            99 => VerkleNodeKind::Leaf99,
            100 => VerkleNodeKind::Leaf100,
            101 => VerkleNodeKind::Leaf101,
            102 => VerkleNodeKind::Leaf102,
            103 => VerkleNodeKind::Leaf103,
            104 => VerkleNodeKind::Leaf104,
            105 => VerkleNodeKind::Leaf105,
            106 => VerkleNodeKind::Leaf106,
            107 => VerkleNodeKind::Leaf107,
            108 => VerkleNodeKind::Leaf108,
            109 => VerkleNodeKind::Leaf109,
            110 => VerkleNodeKind::Leaf110,
            111 => VerkleNodeKind::Leaf111,
            112 => VerkleNodeKind::Leaf112,
            113 => VerkleNodeKind::Leaf113,
            114 => VerkleNodeKind::Leaf114,
            115 => VerkleNodeKind::Leaf115,
            116 => VerkleNodeKind::Leaf116,
            117 => VerkleNodeKind::Leaf117,
            118 => VerkleNodeKind::Leaf118,
            119 => VerkleNodeKind::Leaf119,
            120 => VerkleNodeKind::Leaf120,
            121 => VerkleNodeKind::Leaf121,
            122 => VerkleNodeKind::Leaf122,
            123 => VerkleNodeKind::Leaf123,
            124 => VerkleNodeKind::Leaf124,
            125 => VerkleNodeKind::Leaf125,
            126 => VerkleNodeKind::Leaf126,
            127 => VerkleNodeKind::Leaf127,
            128 => VerkleNodeKind::Leaf128,
            129 => VerkleNodeKind::Leaf129,
            130 => VerkleNodeKind::Leaf130,
            131 => VerkleNodeKind::Leaf131,
            132 => VerkleNodeKind::Leaf132,
            133 => VerkleNodeKind::Leaf133,
            134 => VerkleNodeKind::Leaf134,
            135 => VerkleNodeKind::Leaf135,
            136 => VerkleNodeKind::Leaf136,
            137 => VerkleNodeKind::Leaf137,
            138 => VerkleNodeKind::Leaf138,
            139 => VerkleNodeKind::Leaf139,
            140 => VerkleNodeKind::Leaf140,
            141 => VerkleNodeKind::Leaf141,
            142 => VerkleNodeKind::Leaf142,
            143 => VerkleNodeKind::Leaf143,
            144 => VerkleNodeKind::Leaf144,
            145 => VerkleNodeKind::Leaf145,
            146 => VerkleNodeKind::Leaf146,
            147 => VerkleNodeKind::Leaf147,
            148 => VerkleNodeKind::Leaf148,
            149 => VerkleNodeKind::Leaf149,
            150 => VerkleNodeKind::Leaf150,
            151 => VerkleNodeKind::Leaf151,
            152 => VerkleNodeKind::Leaf152,
            153 => VerkleNodeKind::Leaf153,
            154 => VerkleNodeKind::Leaf154,
            155 => VerkleNodeKind::Leaf155,
            156 => VerkleNodeKind::Leaf156,
            157 => VerkleNodeKind::Leaf157,
            158 => VerkleNodeKind::Leaf158,
            159 => VerkleNodeKind::Leaf159,
            160 => VerkleNodeKind::Leaf160,
            161 => VerkleNodeKind::Leaf161,
            162 => VerkleNodeKind::Leaf162,
            163 => VerkleNodeKind::Leaf163,
            164 => VerkleNodeKind::Leaf164,
            165 => VerkleNodeKind::Leaf165,
            166 => VerkleNodeKind::Leaf166,
            167 => VerkleNodeKind::Leaf167,
            168 => VerkleNodeKind::Leaf168,
            169 => VerkleNodeKind::Leaf169,
            170 => VerkleNodeKind::Leaf170,
            171 => VerkleNodeKind::Leaf171,
            172 => VerkleNodeKind::Leaf172,
            173 => VerkleNodeKind::Leaf173,
            174 => VerkleNodeKind::Leaf174,
            175 => VerkleNodeKind::Leaf175,
            176 => VerkleNodeKind::Leaf176,
            177 => VerkleNodeKind::Leaf177,
            178 => VerkleNodeKind::Leaf178,
            179 => VerkleNodeKind::Leaf179,
            180 => VerkleNodeKind::Leaf180,
            181 => VerkleNodeKind::Leaf181,
            182 => VerkleNodeKind::Leaf182,
            183 => VerkleNodeKind::Leaf183,
            184 => VerkleNodeKind::Leaf184,
            185 => VerkleNodeKind::Leaf185,
            186 => VerkleNodeKind::Leaf186,
            187 => VerkleNodeKind::Leaf187,
            188 => VerkleNodeKind::Leaf188,
            189 => VerkleNodeKind::Leaf189,
            190 => VerkleNodeKind::Leaf190,
            191 => VerkleNodeKind::Leaf191,
            192 => VerkleNodeKind::Leaf192,
            193 => VerkleNodeKind::Leaf193,
            194 => VerkleNodeKind::Leaf194,
            195 => VerkleNodeKind::Leaf195,
            196 => VerkleNodeKind::Leaf196,
            197 => VerkleNodeKind::Leaf197,
            198 => VerkleNodeKind::Leaf198,
            199 => VerkleNodeKind::Leaf199,
            200 => VerkleNodeKind::Leaf200,
            201 => VerkleNodeKind::Leaf201,
            202 => VerkleNodeKind::Leaf202,
            203 => VerkleNodeKind::Leaf203,
            204 => VerkleNodeKind::Leaf204,
            205 => VerkleNodeKind::Leaf205,
            206 => VerkleNodeKind::Leaf206,
            207 => VerkleNodeKind::Leaf207,
            208 => VerkleNodeKind::Leaf208,
            209 => VerkleNodeKind::Leaf209,
            210 => VerkleNodeKind::Leaf210,
            211 => VerkleNodeKind::Leaf211,
            212 => VerkleNodeKind::Leaf212,
            213 => VerkleNodeKind::Leaf213,
            214 => VerkleNodeKind::Leaf214,
            215 => VerkleNodeKind::Leaf215,
            216 => VerkleNodeKind::Leaf216,
            217 => VerkleNodeKind::Leaf217,
            218 => VerkleNodeKind::Leaf218,
            219 => VerkleNodeKind::Leaf219,
            220 => VerkleNodeKind::Leaf220,
            221 => VerkleNodeKind::Leaf221,
            222 => VerkleNodeKind::Leaf222,
            223 => VerkleNodeKind::Leaf223,
            224 => VerkleNodeKind::Leaf224,
            225 => VerkleNodeKind::Leaf225,
            226 => VerkleNodeKind::Leaf226,
            227 => VerkleNodeKind::Leaf227,
            228 => VerkleNodeKind::Leaf228,
            229 => VerkleNodeKind::Leaf229,
            230 => VerkleNodeKind::Leaf230,
            231 => VerkleNodeKind::Leaf231,
            232 => VerkleNodeKind::Leaf232,
            233 => VerkleNodeKind::Leaf233,
            234 => VerkleNodeKind::Leaf234,
            235 => VerkleNodeKind::Leaf235,
            236 => VerkleNodeKind::Leaf236,
            237 => VerkleNodeKind::Leaf237,
            238 => VerkleNodeKind::Leaf238,
            239 => VerkleNodeKind::Leaf239,
            240 => VerkleNodeKind::Leaf240,
            241 => VerkleNodeKind::Leaf241,
            242 => VerkleNodeKind::Leaf242,
            243 => VerkleNodeKind::Leaf243,
            244 => VerkleNodeKind::Leaf244,
            245 => VerkleNodeKind::Leaf245,
            246 => VerkleNodeKind::Leaf246,
            247 => VerkleNodeKind::Leaf247,
            248 => VerkleNodeKind::Leaf248,
            249 => VerkleNodeKind::Leaf249,
            250 => VerkleNodeKind::Leaf250,
            251 => VerkleNodeKind::Leaf251,
            252 => VerkleNodeKind::Leaf252,
            253 => VerkleNodeKind::Leaf253,
            254 => VerkleNodeKind::Leaf254,
            255 => VerkleNodeKind::Leaf255,
            256 => VerkleNodeKind::Leaf256,
            _ => panic!("no leaf type for more than 256 values"),
        }
    }

    /// Returns the smallest inner node type capable of storing `n` values.
    pub fn smallest_inner_type_for(n: usize) -> VerkleNodeKind {
        match n {
            0 => VerkleNodeKind::Empty,
            1 => VerkleNodeKind::Inner1,
            2 => VerkleNodeKind::Inner2,
            3 => VerkleNodeKind::Inner3,
            4 => VerkleNodeKind::Inner4,
            5 => VerkleNodeKind::Inner5,
            6 => VerkleNodeKind::Inner6,
            7 => VerkleNodeKind::Inner7,
            8 => VerkleNodeKind::Inner8,
            9 => VerkleNodeKind::Inner9,
            10 => VerkleNodeKind::Inner10,
            11 => VerkleNodeKind::Inner11,
            12 => VerkleNodeKind::Inner12,
            13 => VerkleNodeKind::Inner13,
            14 => VerkleNodeKind::Inner14,
            15 => VerkleNodeKind::Inner15,
            16 => VerkleNodeKind::Inner16,
            17 => VerkleNodeKind::Inner17,
            18 => VerkleNodeKind::Inner18,
            19 => VerkleNodeKind::Inner19,
            20 => VerkleNodeKind::Inner20,
            21 => VerkleNodeKind::Inner21,
            22 => VerkleNodeKind::Inner22,
            23 => VerkleNodeKind::Inner23,
            24 => VerkleNodeKind::Inner24,
            25 => VerkleNodeKind::Inner25,
            26 => VerkleNodeKind::Inner26,
            27 => VerkleNodeKind::Inner27,
            28 => VerkleNodeKind::Inner28,
            29 => VerkleNodeKind::Inner29,
            30 => VerkleNodeKind::Inner30,
            31 => VerkleNodeKind::Inner31,
            32 => VerkleNodeKind::Inner32,
            33 => VerkleNodeKind::Inner33,
            34 => VerkleNodeKind::Inner34,
            35 => VerkleNodeKind::Inner35,
            36 => VerkleNodeKind::Inner36,
            37 => VerkleNodeKind::Inner37,
            38 => VerkleNodeKind::Inner38,
            39 => VerkleNodeKind::Inner39,
            40 => VerkleNodeKind::Inner40,
            41 => VerkleNodeKind::Inner41,
            42 => VerkleNodeKind::Inner42,
            43 => VerkleNodeKind::Inner43,
            44 => VerkleNodeKind::Inner44,
            45 => VerkleNodeKind::Inner45,
            46 => VerkleNodeKind::Inner46,
            47 => VerkleNodeKind::Inner47,
            48 => VerkleNodeKind::Inner48,
            49 => VerkleNodeKind::Inner49,
            50 => VerkleNodeKind::Inner50,
            51 => VerkleNodeKind::Inner51,
            52 => VerkleNodeKind::Inner52,
            53 => VerkleNodeKind::Inner53,
            54 => VerkleNodeKind::Inner54,
            55 => VerkleNodeKind::Inner55,
            56 => VerkleNodeKind::Inner56,
            57 => VerkleNodeKind::Inner57,
            58 => VerkleNodeKind::Inner58,
            59 => VerkleNodeKind::Inner59,
            60 => VerkleNodeKind::Inner60,
            61 => VerkleNodeKind::Inner61,
            62 => VerkleNodeKind::Inner62,
            63 => VerkleNodeKind::Inner63,
            64 => VerkleNodeKind::Inner64,
            65 => VerkleNodeKind::Inner65,
            66 => VerkleNodeKind::Inner66,
            67 => VerkleNodeKind::Inner67,
            68 => VerkleNodeKind::Inner68,
            69 => VerkleNodeKind::Inner69,
            70 => VerkleNodeKind::Inner70,
            71 => VerkleNodeKind::Inner71,
            72 => VerkleNodeKind::Inner72,
            73 => VerkleNodeKind::Inner73,
            74 => VerkleNodeKind::Inner74,
            75 => VerkleNodeKind::Inner75,
            76 => VerkleNodeKind::Inner76,
            77 => VerkleNodeKind::Inner77,
            78 => VerkleNodeKind::Inner78,
            79 => VerkleNodeKind::Inner79,
            80 => VerkleNodeKind::Inner80,
            81 => VerkleNodeKind::Inner81,
            82 => VerkleNodeKind::Inner82,
            83 => VerkleNodeKind::Inner83,
            84 => VerkleNodeKind::Inner84,
            85 => VerkleNodeKind::Inner85,
            86 => VerkleNodeKind::Inner86,
            87 => VerkleNodeKind::Inner87,
            88 => VerkleNodeKind::Inner88,
            89 => VerkleNodeKind::Inner89,
            90 => VerkleNodeKind::Inner90,
            91 => VerkleNodeKind::Inner91,
            92 => VerkleNodeKind::Inner92,
            93 => VerkleNodeKind::Inner93,
            94 => VerkleNodeKind::Inner94,
            95 => VerkleNodeKind::Inner95,
            96 => VerkleNodeKind::Inner96,
            97 => VerkleNodeKind::Inner97,
            98 => VerkleNodeKind::Inner98,
            99 => VerkleNodeKind::Inner99,
            100 => VerkleNodeKind::Inner100,
            101 => VerkleNodeKind::Inner101,
            102 => VerkleNodeKind::Inner102,
            103 => VerkleNodeKind::Inner103,
            104 => VerkleNodeKind::Inner104,
            105 => VerkleNodeKind::Inner105,
            106 => VerkleNodeKind::Inner106,
            107 => VerkleNodeKind::Inner107,
            108 => VerkleNodeKind::Inner108,
            109 => VerkleNodeKind::Inner109,
            110 => VerkleNodeKind::Inner110,
            111 => VerkleNodeKind::Inner111,
            112 => VerkleNodeKind::Inner112,
            113 => VerkleNodeKind::Inner113,
            114 => VerkleNodeKind::Inner114,
            115 => VerkleNodeKind::Inner115,
            116 => VerkleNodeKind::Inner116,
            117 => VerkleNodeKind::Inner117,
            118 => VerkleNodeKind::Inner118,
            119 => VerkleNodeKind::Inner119,
            120 => VerkleNodeKind::Inner120,
            121 => VerkleNodeKind::Inner121,
            122 => VerkleNodeKind::Inner122,
            123 => VerkleNodeKind::Inner123,
            124 => VerkleNodeKind::Inner124,
            125 => VerkleNodeKind::Inner125,
            126 => VerkleNodeKind::Inner126,
            127 => VerkleNodeKind::Inner127,
            128 => VerkleNodeKind::Inner128,
            129 => VerkleNodeKind::Inner129,
            130 => VerkleNodeKind::Inner130,
            131 => VerkleNodeKind::Inner131,
            132 => VerkleNodeKind::Inner132,
            133 => VerkleNodeKind::Inner133,
            134 => VerkleNodeKind::Inner134,
            135 => VerkleNodeKind::Inner135,
            136 => VerkleNodeKind::Inner136,
            137 => VerkleNodeKind::Inner137,
            138 => VerkleNodeKind::Inner138,
            139 => VerkleNodeKind::Inner139,
            140 => VerkleNodeKind::Inner140,
            141 => VerkleNodeKind::Inner141,
            142 => VerkleNodeKind::Inner142,
            143 => VerkleNodeKind::Inner143,
            144 => VerkleNodeKind::Inner144,
            145 => VerkleNodeKind::Inner145,
            146 => VerkleNodeKind::Inner146,
            147 => VerkleNodeKind::Inner147,
            148 => VerkleNodeKind::Inner148,
            149 => VerkleNodeKind::Inner149,
            150 => VerkleNodeKind::Inner150,
            151 => VerkleNodeKind::Inner151,
            152 => VerkleNodeKind::Inner152,
            153 => VerkleNodeKind::Inner153,
            154 => VerkleNodeKind::Inner154,
            155 => VerkleNodeKind::Inner155,
            156 => VerkleNodeKind::Inner156,
            157 => VerkleNodeKind::Inner157,
            158 => VerkleNodeKind::Inner158,
            159 => VerkleNodeKind::Inner159,
            160 => VerkleNodeKind::Inner160,
            161 => VerkleNodeKind::Inner161,
            162 => VerkleNodeKind::Inner162,
            163 => VerkleNodeKind::Inner163,
            164 => VerkleNodeKind::Inner164,
            165 => VerkleNodeKind::Inner165,
            166 => VerkleNodeKind::Inner166,
            167 => VerkleNodeKind::Inner167,
            168 => VerkleNodeKind::Inner168,
            169 => VerkleNodeKind::Inner169,
            170 => VerkleNodeKind::Inner170,
            171 => VerkleNodeKind::Inner171,
            172 => VerkleNodeKind::Inner172,
            173 => VerkleNodeKind::Inner173,
            174 => VerkleNodeKind::Inner174,
            175 => VerkleNodeKind::Inner175,
            176 => VerkleNodeKind::Inner176,
            177 => VerkleNodeKind::Inner177,
            178 => VerkleNodeKind::Inner178,
            179 => VerkleNodeKind::Inner179,
            180 => VerkleNodeKind::Inner180,
            181 => VerkleNodeKind::Inner181,
            182 => VerkleNodeKind::Inner182,
            183 => VerkleNodeKind::Inner183,
            184 => VerkleNodeKind::Inner184,
            185 => VerkleNodeKind::Inner185,
            186 => VerkleNodeKind::Inner186,
            187 => VerkleNodeKind::Inner187,
            188 => VerkleNodeKind::Inner188,
            189 => VerkleNodeKind::Inner189,
            190 => VerkleNodeKind::Inner190,
            191 => VerkleNodeKind::Inner191,
            192 => VerkleNodeKind::Inner192,
            193 => VerkleNodeKind::Inner193,
            194 => VerkleNodeKind::Inner194,
            195 => VerkleNodeKind::Inner195,
            196 => VerkleNodeKind::Inner196,
            197 => VerkleNodeKind::Inner197,
            198 => VerkleNodeKind::Inner198,
            199 => VerkleNodeKind::Inner199,
            200 => VerkleNodeKind::Inner200,
            201 => VerkleNodeKind::Inner201,
            202 => VerkleNodeKind::Inner202,
            203 => VerkleNodeKind::Inner203,
            204 => VerkleNodeKind::Inner204,
            205 => VerkleNodeKind::Inner205,
            206 => VerkleNodeKind::Inner206,
            207 => VerkleNodeKind::Inner207,
            208 => VerkleNodeKind::Inner208,
            209 => VerkleNodeKind::Inner209,
            210 => VerkleNodeKind::Inner210,
            211 => VerkleNodeKind::Inner211,
            212 => VerkleNodeKind::Inner212,
            213 => VerkleNodeKind::Inner213,
            214 => VerkleNodeKind::Inner214,
            215 => VerkleNodeKind::Inner215,
            216 => VerkleNodeKind::Inner216,
            217 => VerkleNodeKind::Inner217,
            218 => VerkleNodeKind::Inner218,
            219 => VerkleNodeKind::Inner219,
            220 => VerkleNodeKind::Inner220,
            221 => VerkleNodeKind::Inner221,
            222 => VerkleNodeKind::Inner222,
            223 => VerkleNodeKind::Inner223,
            224 => VerkleNodeKind::Inner224,
            225 => VerkleNodeKind::Inner225,
            226 => VerkleNodeKind::Inner226,
            227 => VerkleNodeKind::Inner227,
            228 => VerkleNodeKind::Inner228,
            229 => VerkleNodeKind::Inner229,
            230 => VerkleNodeKind::Inner230,
            231 => VerkleNodeKind::Inner231,
            232 => VerkleNodeKind::Inner232,
            233 => VerkleNodeKind::Inner233,
            234 => VerkleNodeKind::Inner234,
            235 => VerkleNodeKind::Inner235,
            236 => VerkleNodeKind::Inner236,
            237 => VerkleNodeKind::Inner237,
            238 => VerkleNodeKind::Inner238,
            239 => VerkleNodeKind::Inner239,
            240 => VerkleNodeKind::Inner240,
            241 => VerkleNodeKind::Inner241,
            242 => VerkleNodeKind::Inner242,
            243 => VerkleNodeKind::Inner243,
            244 => VerkleNodeKind::Inner244,
            245 => VerkleNodeKind::Inner245,
            246 => VerkleNodeKind::Inner246,
            247 => VerkleNodeKind::Inner247,
            248 => VerkleNodeKind::Inner248,
            249 => VerkleNodeKind::Inner249,
            250 => VerkleNodeKind::Inner250,
            251 => VerkleNodeKind::Inner251,
            252 => VerkleNodeKind::Inner252,
            253 => VerkleNodeKind::Inner253,
            254 => VerkleNodeKind::Inner254,
            255 => VerkleNodeKind::Inner255,
            256 => VerkleNodeKind::Inner256,
            _ => panic!("no inner type for more than 256 children"),
        }
    }

    /// Returns the commitment input for computing the commitment of this node.
    pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {
        match self {
            VerkleNode::Empty(n) => n.get_commitment_input(),
            VerkleNode::Inner1(n) => n.get_commitment_input(),
            VerkleNode::Inner2(n) => n.get_commitment_input(),
            VerkleNode::Inner3(n) => n.get_commitment_input(),
            VerkleNode::Inner4(n) => n.get_commitment_input(),
            VerkleNode::Inner5(n) => n.get_commitment_input(),
            VerkleNode::Inner6(n) => n.get_commitment_input(),
            VerkleNode::Inner7(n) => n.get_commitment_input(),
            VerkleNode::Inner8(n) => n.get_commitment_input(),
            VerkleNode::Inner9(n) => n.get_commitment_input(),
            VerkleNode::Inner10(n) => n.get_commitment_input(),
            VerkleNode::Inner11(n) => n.get_commitment_input(),
            VerkleNode::Inner12(n) => n.get_commitment_input(),
            VerkleNode::Inner13(n) => n.get_commitment_input(),
            VerkleNode::Inner14(n) => n.get_commitment_input(),
            VerkleNode::Inner15(n) => n.get_commitment_input(),
            VerkleNode::Inner16(n) => n.get_commitment_input(),
            VerkleNode::Inner17(n) => n.get_commitment_input(),
            VerkleNode::Inner18(n) => n.get_commitment_input(),
            VerkleNode::Inner19(n) => n.get_commitment_input(),
            VerkleNode::Inner20(n) => n.get_commitment_input(),
            VerkleNode::Inner21(n) => n.get_commitment_input(),
            VerkleNode::Inner22(n) => n.get_commitment_input(),
            VerkleNode::Inner23(n) => n.get_commitment_input(),
            VerkleNode::Inner24(n) => n.get_commitment_input(),
            VerkleNode::Inner25(n) => n.get_commitment_input(),
            VerkleNode::Inner26(n) => n.get_commitment_input(),
            VerkleNode::Inner27(n) => n.get_commitment_input(),
            VerkleNode::Inner28(n) => n.get_commitment_input(),
            VerkleNode::Inner29(n) => n.get_commitment_input(),
            VerkleNode::Inner30(n) => n.get_commitment_input(),
            VerkleNode::Inner31(n) => n.get_commitment_input(),
            VerkleNode::Inner32(n) => n.get_commitment_input(),
            VerkleNode::Inner33(n) => n.get_commitment_input(),
            VerkleNode::Inner34(n) => n.get_commitment_input(),
            VerkleNode::Inner35(n) => n.get_commitment_input(),
            VerkleNode::Inner36(n) => n.get_commitment_input(),
            VerkleNode::Inner37(n) => n.get_commitment_input(),
            VerkleNode::Inner38(n) => n.get_commitment_input(),
            VerkleNode::Inner39(n) => n.get_commitment_input(),
            VerkleNode::Inner40(n) => n.get_commitment_input(),
            VerkleNode::Inner41(n) => n.get_commitment_input(),
            VerkleNode::Inner42(n) => n.get_commitment_input(),
            VerkleNode::Inner43(n) => n.get_commitment_input(),
            VerkleNode::Inner44(n) => n.get_commitment_input(),
            VerkleNode::Inner45(n) => n.get_commitment_input(),
            VerkleNode::Inner46(n) => n.get_commitment_input(),
            VerkleNode::Inner47(n) => n.get_commitment_input(),
            VerkleNode::Inner48(n) => n.get_commitment_input(),
            VerkleNode::Inner49(n) => n.get_commitment_input(),
            VerkleNode::Inner50(n) => n.get_commitment_input(),
            VerkleNode::Inner51(n) => n.get_commitment_input(),
            VerkleNode::Inner52(n) => n.get_commitment_input(),
            VerkleNode::Inner53(n) => n.get_commitment_input(),
            VerkleNode::Inner54(n) => n.get_commitment_input(),
            VerkleNode::Inner55(n) => n.get_commitment_input(),
            VerkleNode::Inner56(n) => n.get_commitment_input(),
            VerkleNode::Inner57(n) => n.get_commitment_input(),
            VerkleNode::Inner58(n) => n.get_commitment_input(),
            VerkleNode::Inner59(n) => n.get_commitment_input(),
            VerkleNode::Inner60(n) => n.get_commitment_input(),
            VerkleNode::Inner61(n) => n.get_commitment_input(),
            VerkleNode::Inner62(n) => n.get_commitment_input(),
            VerkleNode::Inner63(n) => n.get_commitment_input(),
            VerkleNode::Inner64(n) => n.get_commitment_input(),
            VerkleNode::Inner65(n) => n.get_commitment_input(),
            VerkleNode::Inner66(n) => n.get_commitment_input(),
            VerkleNode::Inner67(n) => n.get_commitment_input(),
            VerkleNode::Inner68(n) => n.get_commitment_input(),
            VerkleNode::Inner69(n) => n.get_commitment_input(),
            VerkleNode::Inner70(n) => n.get_commitment_input(),
            VerkleNode::Inner71(n) => n.get_commitment_input(),
            VerkleNode::Inner72(n) => n.get_commitment_input(),
            VerkleNode::Inner73(n) => n.get_commitment_input(),
            VerkleNode::Inner74(n) => n.get_commitment_input(),
            VerkleNode::Inner75(n) => n.get_commitment_input(),
            VerkleNode::Inner76(n) => n.get_commitment_input(),
            VerkleNode::Inner77(n) => n.get_commitment_input(),
            VerkleNode::Inner78(n) => n.get_commitment_input(),
            VerkleNode::Inner79(n) => n.get_commitment_input(),
            VerkleNode::Inner80(n) => n.get_commitment_input(),
            VerkleNode::Inner81(n) => n.get_commitment_input(),
            VerkleNode::Inner82(n) => n.get_commitment_input(),
            VerkleNode::Inner83(n) => n.get_commitment_input(),
            VerkleNode::Inner84(n) => n.get_commitment_input(),
            VerkleNode::Inner85(n) => n.get_commitment_input(),
            VerkleNode::Inner86(n) => n.get_commitment_input(),
            VerkleNode::Inner87(n) => n.get_commitment_input(),
            VerkleNode::Inner88(n) => n.get_commitment_input(),
            VerkleNode::Inner89(n) => n.get_commitment_input(),
            VerkleNode::Inner90(n) => n.get_commitment_input(),
            VerkleNode::Inner91(n) => n.get_commitment_input(),
            VerkleNode::Inner92(n) => n.get_commitment_input(),
            VerkleNode::Inner93(n) => n.get_commitment_input(),
            VerkleNode::Inner94(n) => n.get_commitment_input(),
            VerkleNode::Inner95(n) => n.get_commitment_input(),
            VerkleNode::Inner96(n) => n.get_commitment_input(),
            VerkleNode::Inner97(n) => n.get_commitment_input(),
            VerkleNode::Inner98(n) => n.get_commitment_input(),
            VerkleNode::Inner99(n) => n.get_commitment_input(),
            VerkleNode::Inner100(n) => n.get_commitment_input(),
            VerkleNode::Inner101(n) => n.get_commitment_input(),
            VerkleNode::Inner102(n) => n.get_commitment_input(),
            VerkleNode::Inner103(n) => n.get_commitment_input(),
            VerkleNode::Inner104(n) => n.get_commitment_input(),
            VerkleNode::Inner105(n) => n.get_commitment_input(),
            VerkleNode::Inner106(n) => n.get_commitment_input(),
            VerkleNode::Inner107(n) => n.get_commitment_input(),
            VerkleNode::Inner108(n) => n.get_commitment_input(),
            VerkleNode::Inner109(n) => n.get_commitment_input(),
            VerkleNode::Inner110(n) => n.get_commitment_input(),
            VerkleNode::Inner111(n) => n.get_commitment_input(),
            VerkleNode::Inner112(n) => n.get_commitment_input(),
            VerkleNode::Inner113(n) => n.get_commitment_input(),
            VerkleNode::Inner114(n) => n.get_commitment_input(),
            VerkleNode::Inner115(n) => n.get_commitment_input(),
            VerkleNode::Inner116(n) => n.get_commitment_input(),
            VerkleNode::Inner117(n) => n.get_commitment_input(),
            VerkleNode::Inner118(n) => n.get_commitment_input(),
            VerkleNode::Inner119(n) => n.get_commitment_input(),
            VerkleNode::Inner120(n) => n.get_commitment_input(),
            VerkleNode::Inner121(n) => n.get_commitment_input(),
            VerkleNode::Inner122(n) => n.get_commitment_input(),
            VerkleNode::Inner123(n) => n.get_commitment_input(),
            VerkleNode::Inner124(n) => n.get_commitment_input(),
            VerkleNode::Inner125(n) => n.get_commitment_input(),
            VerkleNode::Inner126(n) => n.get_commitment_input(),
            VerkleNode::Inner127(n) => n.get_commitment_input(),
            VerkleNode::Inner128(n) => n.get_commitment_input(),
            VerkleNode::Inner129(n) => n.get_commitment_input(),
            VerkleNode::Inner130(n) => n.get_commitment_input(),
            VerkleNode::Inner131(n) => n.get_commitment_input(),
            VerkleNode::Inner132(n) => n.get_commitment_input(),
            VerkleNode::Inner133(n) => n.get_commitment_input(),
            VerkleNode::Inner134(n) => n.get_commitment_input(),
            VerkleNode::Inner135(n) => n.get_commitment_input(),
            VerkleNode::Inner136(n) => n.get_commitment_input(),
            VerkleNode::Inner137(n) => n.get_commitment_input(),
            VerkleNode::Inner138(n) => n.get_commitment_input(),
            VerkleNode::Inner139(n) => n.get_commitment_input(),
            VerkleNode::Inner140(n) => n.get_commitment_input(),
            VerkleNode::Inner141(n) => n.get_commitment_input(),
            VerkleNode::Inner142(n) => n.get_commitment_input(),
            VerkleNode::Inner143(n) => n.get_commitment_input(),
            VerkleNode::Inner144(n) => n.get_commitment_input(),
            VerkleNode::Inner145(n) => n.get_commitment_input(),
            VerkleNode::Inner146(n) => n.get_commitment_input(),
            VerkleNode::Inner147(n) => n.get_commitment_input(),
            VerkleNode::Inner148(n) => n.get_commitment_input(),
            VerkleNode::Inner149(n) => n.get_commitment_input(),
            VerkleNode::Inner150(n) => n.get_commitment_input(),
            VerkleNode::Inner151(n) => n.get_commitment_input(),
            VerkleNode::Inner152(n) => n.get_commitment_input(),
            VerkleNode::Inner153(n) => n.get_commitment_input(),
            VerkleNode::Inner154(n) => n.get_commitment_input(),
            VerkleNode::Inner155(n) => n.get_commitment_input(),
            VerkleNode::Inner156(n) => n.get_commitment_input(),
            VerkleNode::Inner157(n) => n.get_commitment_input(),
            VerkleNode::Inner158(n) => n.get_commitment_input(),
            VerkleNode::Inner159(n) => n.get_commitment_input(),
            VerkleNode::Inner160(n) => n.get_commitment_input(),
            VerkleNode::Inner161(n) => n.get_commitment_input(),
            VerkleNode::Inner162(n) => n.get_commitment_input(),
            VerkleNode::Inner163(n) => n.get_commitment_input(),
            VerkleNode::Inner164(n) => n.get_commitment_input(),
            VerkleNode::Inner165(n) => n.get_commitment_input(),
            VerkleNode::Inner166(n) => n.get_commitment_input(),
            VerkleNode::Inner167(n) => n.get_commitment_input(),
            VerkleNode::Inner168(n) => n.get_commitment_input(),
            VerkleNode::Inner169(n) => n.get_commitment_input(),
            VerkleNode::Inner170(n) => n.get_commitment_input(),
            VerkleNode::Inner171(n) => n.get_commitment_input(),
            VerkleNode::Inner172(n) => n.get_commitment_input(),
            VerkleNode::Inner173(n) => n.get_commitment_input(),
            VerkleNode::Inner174(n) => n.get_commitment_input(),
            VerkleNode::Inner175(n) => n.get_commitment_input(),
            VerkleNode::Inner176(n) => n.get_commitment_input(),
            VerkleNode::Inner177(n) => n.get_commitment_input(),
            VerkleNode::Inner178(n) => n.get_commitment_input(),
            VerkleNode::Inner179(n) => n.get_commitment_input(),
            VerkleNode::Inner180(n) => n.get_commitment_input(),
            VerkleNode::Inner181(n) => n.get_commitment_input(),
            VerkleNode::Inner182(n) => n.get_commitment_input(),
            VerkleNode::Inner183(n) => n.get_commitment_input(),
            VerkleNode::Inner184(n) => n.get_commitment_input(),
            VerkleNode::Inner185(n) => n.get_commitment_input(),
            VerkleNode::Inner186(n) => n.get_commitment_input(),
            VerkleNode::Inner187(n) => n.get_commitment_input(),
            VerkleNode::Inner188(n) => n.get_commitment_input(),
            VerkleNode::Inner189(n) => n.get_commitment_input(),
            VerkleNode::Inner190(n) => n.get_commitment_input(),
            VerkleNode::Inner191(n) => n.get_commitment_input(),
            VerkleNode::Inner192(n) => n.get_commitment_input(),
            VerkleNode::Inner193(n) => n.get_commitment_input(),
            VerkleNode::Inner194(n) => n.get_commitment_input(),
            VerkleNode::Inner195(n) => n.get_commitment_input(),
            VerkleNode::Inner196(n) => n.get_commitment_input(),
            VerkleNode::Inner197(n) => n.get_commitment_input(),
            VerkleNode::Inner198(n) => n.get_commitment_input(),
            VerkleNode::Inner199(n) => n.get_commitment_input(),
            VerkleNode::Inner200(n) => n.get_commitment_input(),
            VerkleNode::Inner201(n) => n.get_commitment_input(),
            VerkleNode::Inner202(n) => n.get_commitment_input(),
            VerkleNode::Inner203(n) => n.get_commitment_input(),
            VerkleNode::Inner204(n) => n.get_commitment_input(),
            VerkleNode::Inner205(n) => n.get_commitment_input(),
            VerkleNode::Inner206(n) => n.get_commitment_input(),
            VerkleNode::Inner207(n) => n.get_commitment_input(),
            VerkleNode::Inner208(n) => n.get_commitment_input(),
            VerkleNode::Inner209(n) => n.get_commitment_input(),
            VerkleNode::Inner210(n) => n.get_commitment_input(),
            VerkleNode::Inner211(n) => n.get_commitment_input(),
            VerkleNode::Inner212(n) => n.get_commitment_input(),
            VerkleNode::Inner213(n) => n.get_commitment_input(),
            VerkleNode::Inner214(n) => n.get_commitment_input(),
            VerkleNode::Inner215(n) => n.get_commitment_input(),
            VerkleNode::Inner216(n) => n.get_commitment_input(),
            VerkleNode::Inner217(n) => n.get_commitment_input(),
            VerkleNode::Inner218(n) => n.get_commitment_input(),
            VerkleNode::Inner219(n) => n.get_commitment_input(),
            VerkleNode::Inner220(n) => n.get_commitment_input(),
            VerkleNode::Inner221(n) => n.get_commitment_input(),
            VerkleNode::Inner222(n) => n.get_commitment_input(),
            VerkleNode::Inner223(n) => n.get_commitment_input(),
            VerkleNode::Inner224(n) => n.get_commitment_input(),
            VerkleNode::Inner225(n) => n.get_commitment_input(),
            VerkleNode::Inner226(n) => n.get_commitment_input(),
            VerkleNode::Inner227(n) => n.get_commitment_input(),
            VerkleNode::Inner228(n) => n.get_commitment_input(),
            VerkleNode::Inner229(n) => n.get_commitment_input(),
            VerkleNode::Inner230(n) => n.get_commitment_input(),
            VerkleNode::Inner231(n) => n.get_commitment_input(),
            VerkleNode::Inner232(n) => n.get_commitment_input(),
            VerkleNode::Inner233(n) => n.get_commitment_input(),
            VerkleNode::Inner234(n) => n.get_commitment_input(),
            VerkleNode::Inner235(n) => n.get_commitment_input(),
            VerkleNode::Inner236(n) => n.get_commitment_input(),
            VerkleNode::Inner237(n) => n.get_commitment_input(),
            VerkleNode::Inner238(n) => n.get_commitment_input(),
            VerkleNode::Inner239(n) => n.get_commitment_input(),
            VerkleNode::Inner240(n) => n.get_commitment_input(),
            VerkleNode::Inner241(n) => n.get_commitment_input(),
            VerkleNode::Inner242(n) => n.get_commitment_input(),
            VerkleNode::Inner243(n) => n.get_commitment_input(),
            VerkleNode::Inner244(n) => n.get_commitment_input(),
            VerkleNode::Inner245(n) => n.get_commitment_input(),
            VerkleNode::Inner246(n) => n.get_commitment_input(),
            VerkleNode::Inner247(n) => n.get_commitment_input(),
            VerkleNode::Inner248(n) => n.get_commitment_input(),
            VerkleNode::Inner249(n) => n.get_commitment_input(),
            VerkleNode::Inner250(n) => n.get_commitment_input(),
            VerkleNode::Inner251(n) => n.get_commitment_input(),
            VerkleNode::Inner252(n) => n.get_commitment_input(),
            VerkleNode::Inner253(n) => n.get_commitment_input(),
            VerkleNode::Inner254(n) => n.get_commitment_input(),
            VerkleNode::Inner255(n) => n.get_commitment_input(),
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
            VerkleNode::Leaf17(n) => n.get_commitment_input(),
            VerkleNode::Leaf18(n) => n.get_commitment_input(),
            VerkleNode::Leaf19(n) => n.get_commitment_input(),
            VerkleNode::Leaf20(n) => n.get_commitment_input(),
            VerkleNode::Leaf21(n) => n.get_commitment_input(),
            VerkleNode::Leaf22(n) => n.get_commitment_input(),
            VerkleNode::Leaf23(n) => n.get_commitment_input(),
            VerkleNode::Leaf24(n) => n.get_commitment_input(),
            VerkleNode::Leaf25(n) => n.get_commitment_input(),
            VerkleNode::Leaf26(n) => n.get_commitment_input(),
            VerkleNode::Leaf27(n) => n.get_commitment_input(),
            VerkleNode::Leaf28(n) => n.get_commitment_input(),
            VerkleNode::Leaf29(n) => n.get_commitment_input(),
            VerkleNode::Leaf30(n) => n.get_commitment_input(),
            VerkleNode::Leaf31(n) => n.get_commitment_input(),
            VerkleNode::Leaf32(n) => n.get_commitment_input(),
            VerkleNode::Leaf33(n) => n.get_commitment_input(),
            VerkleNode::Leaf34(n) => n.get_commitment_input(),
            VerkleNode::Leaf35(n) => n.get_commitment_input(),
            VerkleNode::Leaf36(n) => n.get_commitment_input(),
            VerkleNode::Leaf37(n) => n.get_commitment_input(),
            VerkleNode::Leaf38(n) => n.get_commitment_input(),
            VerkleNode::Leaf39(n) => n.get_commitment_input(),
            VerkleNode::Leaf40(n) => n.get_commitment_input(),
            VerkleNode::Leaf41(n) => n.get_commitment_input(),
            VerkleNode::Leaf42(n) => n.get_commitment_input(),
            VerkleNode::Leaf43(n) => n.get_commitment_input(),
            VerkleNode::Leaf44(n) => n.get_commitment_input(),
            VerkleNode::Leaf45(n) => n.get_commitment_input(),
            VerkleNode::Leaf46(n) => n.get_commitment_input(),
            VerkleNode::Leaf47(n) => n.get_commitment_input(),
            VerkleNode::Leaf48(n) => n.get_commitment_input(),
            VerkleNode::Leaf49(n) => n.get_commitment_input(),
            VerkleNode::Leaf50(n) => n.get_commitment_input(),
            VerkleNode::Leaf51(n) => n.get_commitment_input(),
            VerkleNode::Leaf52(n) => n.get_commitment_input(),
            VerkleNode::Leaf53(n) => n.get_commitment_input(),
            VerkleNode::Leaf54(n) => n.get_commitment_input(),
            VerkleNode::Leaf55(n) => n.get_commitment_input(),
            VerkleNode::Leaf56(n) => n.get_commitment_input(),
            VerkleNode::Leaf57(n) => n.get_commitment_input(),
            VerkleNode::Leaf58(n) => n.get_commitment_input(),
            VerkleNode::Leaf59(n) => n.get_commitment_input(),
            VerkleNode::Leaf60(n) => n.get_commitment_input(),
            VerkleNode::Leaf61(n) => n.get_commitment_input(),
            VerkleNode::Leaf62(n) => n.get_commitment_input(),
            VerkleNode::Leaf63(n) => n.get_commitment_input(),
            VerkleNode::Leaf64(n) => n.get_commitment_input(),
            VerkleNode::Leaf65(n) => n.get_commitment_input(),
            VerkleNode::Leaf66(n) => n.get_commitment_input(),
            VerkleNode::Leaf67(n) => n.get_commitment_input(),
            VerkleNode::Leaf68(n) => n.get_commitment_input(),
            VerkleNode::Leaf69(n) => n.get_commitment_input(),
            VerkleNode::Leaf70(n) => n.get_commitment_input(),
            VerkleNode::Leaf71(n) => n.get_commitment_input(),
            VerkleNode::Leaf72(n) => n.get_commitment_input(),
            VerkleNode::Leaf73(n) => n.get_commitment_input(),
            VerkleNode::Leaf74(n) => n.get_commitment_input(),
            VerkleNode::Leaf75(n) => n.get_commitment_input(),
            VerkleNode::Leaf76(n) => n.get_commitment_input(),
            VerkleNode::Leaf77(n) => n.get_commitment_input(),
            VerkleNode::Leaf78(n) => n.get_commitment_input(),
            VerkleNode::Leaf79(n) => n.get_commitment_input(),
            VerkleNode::Leaf80(n) => n.get_commitment_input(),
            VerkleNode::Leaf81(n) => n.get_commitment_input(),
            VerkleNode::Leaf82(n) => n.get_commitment_input(),
            VerkleNode::Leaf83(n) => n.get_commitment_input(),
            VerkleNode::Leaf84(n) => n.get_commitment_input(),
            VerkleNode::Leaf85(n) => n.get_commitment_input(),
            VerkleNode::Leaf86(n) => n.get_commitment_input(),
            VerkleNode::Leaf87(n) => n.get_commitment_input(),
            VerkleNode::Leaf88(n) => n.get_commitment_input(),
            VerkleNode::Leaf89(n) => n.get_commitment_input(),
            VerkleNode::Leaf90(n) => n.get_commitment_input(),
            VerkleNode::Leaf91(n) => n.get_commitment_input(),
            VerkleNode::Leaf92(n) => n.get_commitment_input(),
            VerkleNode::Leaf93(n) => n.get_commitment_input(),
            VerkleNode::Leaf94(n) => n.get_commitment_input(),
            VerkleNode::Leaf95(n) => n.get_commitment_input(),
            VerkleNode::Leaf96(n) => n.get_commitment_input(),
            VerkleNode::Leaf97(n) => n.get_commitment_input(),
            VerkleNode::Leaf98(n) => n.get_commitment_input(),
            VerkleNode::Leaf99(n) => n.get_commitment_input(),
            VerkleNode::Leaf100(n) => n.get_commitment_input(),
            VerkleNode::Leaf101(n) => n.get_commitment_input(),
            VerkleNode::Leaf102(n) => n.get_commitment_input(),
            VerkleNode::Leaf103(n) => n.get_commitment_input(),
            VerkleNode::Leaf104(n) => n.get_commitment_input(),
            VerkleNode::Leaf105(n) => n.get_commitment_input(),
            VerkleNode::Leaf106(n) => n.get_commitment_input(),
            VerkleNode::Leaf107(n) => n.get_commitment_input(),
            VerkleNode::Leaf108(n) => n.get_commitment_input(),
            VerkleNode::Leaf109(n) => n.get_commitment_input(),
            VerkleNode::Leaf110(n) => n.get_commitment_input(),
            VerkleNode::Leaf111(n) => n.get_commitment_input(),
            VerkleNode::Leaf112(n) => n.get_commitment_input(),
            VerkleNode::Leaf113(n) => n.get_commitment_input(),
            VerkleNode::Leaf114(n) => n.get_commitment_input(),
            VerkleNode::Leaf115(n) => n.get_commitment_input(),
            VerkleNode::Leaf116(n) => n.get_commitment_input(),
            VerkleNode::Leaf117(n) => n.get_commitment_input(),
            VerkleNode::Leaf118(n) => n.get_commitment_input(),
            VerkleNode::Leaf119(n) => n.get_commitment_input(),
            VerkleNode::Leaf120(n) => n.get_commitment_input(),
            VerkleNode::Leaf121(n) => n.get_commitment_input(),
            VerkleNode::Leaf122(n) => n.get_commitment_input(),
            VerkleNode::Leaf123(n) => n.get_commitment_input(),
            VerkleNode::Leaf124(n) => n.get_commitment_input(),
            VerkleNode::Leaf125(n) => n.get_commitment_input(),
            VerkleNode::Leaf126(n) => n.get_commitment_input(),
            VerkleNode::Leaf127(n) => n.get_commitment_input(),
            VerkleNode::Leaf128(n) => n.get_commitment_input(),
            VerkleNode::Leaf129(n) => n.get_commitment_input(),
            VerkleNode::Leaf130(n) => n.get_commitment_input(),
            VerkleNode::Leaf131(n) => n.get_commitment_input(),
            VerkleNode::Leaf132(n) => n.get_commitment_input(),
            VerkleNode::Leaf133(n) => n.get_commitment_input(),
            VerkleNode::Leaf134(n) => n.get_commitment_input(),
            VerkleNode::Leaf135(n) => n.get_commitment_input(),
            VerkleNode::Leaf136(n) => n.get_commitment_input(),
            VerkleNode::Leaf137(n) => n.get_commitment_input(),
            VerkleNode::Leaf138(n) => n.get_commitment_input(),
            VerkleNode::Leaf139(n) => n.get_commitment_input(),
            VerkleNode::Leaf140(n) => n.get_commitment_input(),
            VerkleNode::Leaf141(n) => n.get_commitment_input(),
            VerkleNode::Leaf142(n) => n.get_commitment_input(),
            VerkleNode::Leaf143(n) => n.get_commitment_input(),
            VerkleNode::Leaf144(n) => n.get_commitment_input(),
            VerkleNode::Leaf145(n) => n.get_commitment_input(),
            VerkleNode::Leaf146(n) => n.get_commitment_input(),
            VerkleNode::Leaf147(n) => n.get_commitment_input(),
            VerkleNode::Leaf148(n) => n.get_commitment_input(),
            VerkleNode::Leaf149(n) => n.get_commitment_input(),
            VerkleNode::Leaf150(n) => n.get_commitment_input(),
            VerkleNode::Leaf151(n) => n.get_commitment_input(),
            VerkleNode::Leaf152(n) => n.get_commitment_input(),
            VerkleNode::Leaf153(n) => n.get_commitment_input(),
            VerkleNode::Leaf154(n) => n.get_commitment_input(),
            VerkleNode::Leaf155(n) => n.get_commitment_input(),
            VerkleNode::Leaf156(n) => n.get_commitment_input(),
            VerkleNode::Leaf157(n) => n.get_commitment_input(),
            VerkleNode::Leaf158(n) => n.get_commitment_input(),
            VerkleNode::Leaf159(n) => n.get_commitment_input(),
            VerkleNode::Leaf160(n) => n.get_commitment_input(),
            VerkleNode::Leaf161(n) => n.get_commitment_input(),
            VerkleNode::Leaf162(n) => n.get_commitment_input(),
            VerkleNode::Leaf163(n) => n.get_commitment_input(),
            VerkleNode::Leaf164(n) => n.get_commitment_input(),
            VerkleNode::Leaf165(n) => n.get_commitment_input(),
            VerkleNode::Leaf166(n) => n.get_commitment_input(),
            VerkleNode::Leaf167(n) => n.get_commitment_input(),
            VerkleNode::Leaf168(n) => n.get_commitment_input(),
            VerkleNode::Leaf169(n) => n.get_commitment_input(),
            VerkleNode::Leaf170(n) => n.get_commitment_input(),
            VerkleNode::Leaf171(n) => n.get_commitment_input(),
            VerkleNode::Leaf172(n) => n.get_commitment_input(),
            VerkleNode::Leaf173(n) => n.get_commitment_input(),
            VerkleNode::Leaf174(n) => n.get_commitment_input(),
            VerkleNode::Leaf175(n) => n.get_commitment_input(),
            VerkleNode::Leaf176(n) => n.get_commitment_input(),
            VerkleNode::Leaf177(n) => n.get_commitment_input(),
            VerkleNode::Leaf178(n) => n.get_commitment_input(),
            VerkleNode::Leaf179(n) => n.get_commitment_input(),
            VerkleNode::Leaf180(n) => n.get_commitment_input(),
            VerkleNode::Leaf181(n) => n.get_commitment_input(),
            VerkleNode::Leaf182(n) => n.get_commitment_input(),
            VerkleNode::Leaf183(n) => n.get_commitment_input(),
            VerkleNode::Leaf184(n) => n.get_commitment_input(),
            VerkleNode::Leaf185(n) => n.get_commitment_input(),
            VerkleNode::Leaf186(n) => n.get_commitment_input(),
            VerkleNode::Leaf187(n) => n.get_commitment_input(),
            VerkleNode::Leaf188(n) => n.get_commitment_input(),
            VerkleNode::Leaf189(n) => n.get_commitment_input(),
            VerkleNode::Leaf190(n) => n.get_commitment_input(),
            VerkleNode::Leaf191(n) => n.get_commitment_input(),
            VerkleNode::Leaf192(n) => n.get_commitment_input(),
            VerkleNode::Leaf193(n) => n.get_commitment_input(),
            VerkleNode::Leaf194(n) => n.get_commitment_input(),
            VerkleNode::Leaf195(n) => n.get_commitment_input(),
            VerkleNode::Leaf196(n) => n.get_commitment_input(),
            VerkleNode::Leaf197(n) => n.get_commitment_input(),
            VerkleNode::Leaf198(n) => n.get_commitment_input(),
            VerkleNode::Leaf199(n) => n.get_commitment_input(),
            VerkleNode::Leaf200(n) => n.get_commitment_input(),
            VerkleNode::Leaf201(n) => n.get_commitment_input(),
            VerkleNode::Leaf202(n) => n.get_commitment_input(),
            VerkleNode::Leaf203(n) => n.get_commitment_input(),
            VerkleNode::Leaf204(n) => n.get_commitment_input(),
            VerkleNode::Leaf205(n) => n.get_commitment_input(),
            VerkleNode::Leaf206(n) => n.get_commitment_input(),
            VerkleNode::Leaf207(n) => n.get_commitment_input(),
            VerkleNode::Leaf208(n) => n.get_commitment_input(),
            VerkleNode::Leaf209(n) => n.get_commitment_input(),
            VerkleNode::Leaf210(n) => n.get_commitment_input(),
            VerkleNode::Leaf211(n) => n.get_commitment_input(),
            VerkleNode::Leaf212(n) => n.get_commitment_input(),
            VerkleNode::Leaf213(n) => n.get_commitment_input(),
            VerkleNode::Leaf214(n) => n.get_commitment_input(),
            VerkleNode::Leaf215(n) => n.get_commitment_input(),
            VerkleNode::Leaf216(n) => n.get_commitment_input(),
            VerkleNode::Leaf217(n) => n.get_commitment_input(),
            VerkleNode::Leaf218(n) => n.get_commitment_input(),
            VerkleNode::Leaf219(n) => n.get_commitment_input(),
            VerkleNode::Leaf220(n) => n.get_commitment_input(),
            VerkleNode::Leaf221(n) => n.get_commitment_input(),
            VerkleNode::Leaf222(n) => n.get_commitment_input(),
            VerkleNode::Leaf223(n) => n.get_commitment_input(),
            VerkleNode::Leaf224(n) => n.get_commitment_input(),
            VerkleNode::Leaf225(n) => n.get_commitment_input(),
            VerkleNode::Leaf226(n) => n.get_commitment_input(),
            VerkleNode::Leaf227(n) => n.get_commitment_input(),
            VerkleNode::Leaf228(n) => n.get_commitment_input(),
            VerkleNode::Leaf229(n) => n.get_commitment_input(),
            VerkleNode::Leaf230(n) => n.get_commitment_input(),
            VerkleNode::Leaf231(n) => n.get_commitment_input(),
            VerkleNode::Leaf232(n) => n.get_commitment_input(),
            VerkleNode::Leaf233(n) => n.get_commitment_input(),
            VerkleNode::Leaf234(n) => n.get_commitment_input(),
            VerkleNode::Leaf235(n) => n.get_commitment_input(),
            VerkleNode::Leaf236(n) => n.get_commitment_input(),
            VerkleNode::Leaf237(n) => n.get_commitment_input(),
            VerkleNode::Leaf238(n) => n.get_commitment_input(),
            VerkleNode::Leaf239(n) => n.get_commitment_input(),
            VerkleNode::Leaf240(n) => n.get_commitment_input(),
            VerkleNode::Leaf241(n) => n.get_commitment_input(),
            VerkleNode::Leaf242(n) => n.get_commitment_input(),
            VerkleNode::Leaf243(n) => n.get_commitment_input(),
            VerkleNode::Leaf244(n) => n.get_commitment_input(),
            VerkleNode::Leaf245(n) => n.get_commitment_input(),
            VerkleNode::Leaf246(n) => n.get_commitment_input(),
            VerkleNode::Leaf247(n) => n.get_commitment_input(),
            VerkleNode::Leaf248(n) => n.get_commitment_input(),
            VerkleNode::Leaf249(n) => n.get_commitment_input(),
            VerkleNode::Leaf250(n) => n.get_commitment_input(),
            VerkleNode::Leaf251(n) => n.get_commitment_input(),
            VerkleNode::Leaf252(n) => n.get_commitment_input(),
            VerkleNode::Leaf253(n) => n.get_commitment_input(),
            VerkleNode::Leaf254(n) => n.get_commitment_input(),
            VerkleNode::Leaf255(n) => n.get_commitment_input(),
            VerkleNode::Leaf256(n) => n.get_commitment_input(),
            VerkleNode::LeafDelta(n) => n.get_commitment_input(),
        }
    }

    /// Converts this node to an inner node, if it is one.
    pub fn as_inner_node(&self) -> Option<&dyn VerkleManagedInnerNode> {
        match self {
            VerkleNode::Inner1(n) => Some(n.deref()),
            VerkleNode::Inner2(n) => Some(n.deref()),
            VerkleNode::Inner3(n) => Some(n.deref()),
            VerkleNode::Inner4(n) => Some(n.deref()),
            VerkleNode::Inner5(n) => Some(n.deref()),
            VerkleNode::Inner6(n) => Some(n.deref()),
            VerkleNode::Inner7(n) => Some(n.deref()),
            VerkleNode::Inner8(n) => Some(n.deref()),
            VerkleNode::Inner9(n) => Some(n.deref()),
            VerkleNode::Inner10(n) => Some(n.deref()),
            VerkleNode::Inner11(n) => Some(n.deref()),
            VerkleNode::Inner12(n) => Some(n.deref()),
            VerkleNode::Inner13(n) => Some(n.deref()),
            VerkleNode::Inner14(n) => Some(n.deref()),
            VerkleNode::Inner15(n) => Some(n.deref()),
            VerkleNode::Inner16(n) => Some(n.deref()),
            VerkleNode::Inner17(n) => Some(n.deref()),
            VerkleNode::Inner18(n) => Some(n.deref()),
            VerkleNode::Inner19(n) => Some(n.deref()),
            VerkleNode::Inner20(n) => Some(n.deref()),
            VerkleNode::Inner21(n) => Some(n.deref()),
            VerkleNode::Inner22(n) => Some(n.deref()),
            VerkleNode::Inner23(n) => Some(n.deref()),
            VerkleNode::Inner24(n) => Some(n.deref()),
            VerkleNode::Inner25(n) => Some(n.deref()),
            VerkleNode::Inner26(n) => Some(n.deref()),
            VerkleNode::Inner27(n) => Some(n.deref()),
            VerkleNode::Inner28(n) => Some(n.deref()),
            VerkleNode::Inner29(n) => Some(n.deref()),
            VerkleNode::Inner30(n) => Some(n.deref()),
            VerkleNode::Inner31(n) => Some(n.deref()),
            VerkleNode::Inner32(n) => Some(n.deref()),
            VerkleNode::Inner33(n) => Some(n.deref()),
            VerkleNode::Inner34(n) => Some(n.deref()),
            VerkleNode::Inner35(n) => Some(n.deref()),
            VerkleNode::Inner36(n) => Some(n.deref()),
            VerkleNode::Inner37(n) => Some(n.deref()),
            VerkleNode::Inner38(n) => Some(n.deref()),
            VerkleNode::Inner39(n) => Some(n.deref()),
            VerkleNode::Inner40(n) => Some(n.deref()),
            VerkleNode::Inner41(n) => Some(n.deref()),
            VerkleNode::Inner42(n) => Some(n.deref()),
            VerkleNode::Inner43(n) => Some(n.deref()),
            VerkleNode::Inner44(n) => Some(n.deref()),
            VerkleNode::Inner45(n) => Some(n.deref()),
            VerkleNode::Inner46(n) => Some(n.deref()),
            VerkleNode::Inner47(n) => Some(n.deref()),
            VerkleNode::Inner48(n) => Some(n.deref()),
            VerkleNode::Inner49(n) => Some(n.deref()),
            VerkleNode::Inner50(n) => Some(n.deref()),
            VerkleNode::Inner51(n) => Some(n.deref()),
            VerkleNode::Inner52(n) => Some(n.deref()),
            VerkleNode::Inner53(n) => Some(n.deref()),
            VerkleNode::Inner54(n) => Some(n.deref()),
            VerkleNode::Inner55(n) => Some(n.deref()),
            VerkleNode::Inner56(n) => Some(n.deref()),
            VerkleNode::Inner57(n) => Some(n.deref()),
            VerkleNode::Inner58(n) => Some(n.deref()),
            VerkleNode::Inner59(n) => Some(n.deref()),
            VerkleNode::Inner60(n) => Some(n.deref()),
            VerkleNode::Inner61(n) => Some(n.deref()),
            VerkleNode::Inner62(n) => Some(n.deref()),
            VerkleNode::Inner63(n) => Some(n.deref()),
            VerkleNode::Inner64(n) => Some(n.deref()),
            VerkleNode::Inner65(n) => Some(n.deref()),
            VerkleNode::Inner66(n) => Some(n.deref()),
            VerkleNode::Inner67(n) => Some(n.deref()),
            VerkleNode::Inner68(n) => Some(n.deref()),
            VerkleNode::Inner69(n) => Some(n.deref()),
            VerkleNode::Inner70(n) => Some(n.deref()),
            VerkleNode::Inner71(n) => Some(n.deref()),
            VerkleNode::Inner72(n) => Some(n.deref()),
            VerkleNode::Inner73(n) => Some(n.deref()),
            VerkleNode::Inner74(n) => Some(n.deref()),
            VerkleNode::Inner75(n) => Some(n.deref()),
            VerkleNode::Inner76(n) => Some(n.deref()),
            VerkleNode::Inner77(n) => Some(n.deref()),
            VerkleNode::Inner78(n) => Some(n.deref()),
            VerkleNode::Inner79(n) => Some(n.deref()),
            VerkleNode::Inner80(n) => Some(n.deref()),
            VerkleNode::Inner81(n) => Some(n.deref()),
            VerkleNode::Inner82(n) => Some(n.deref()),
            VerkleNode::Inner83(n) => Some(n.deref()),
            VerkleNode::Inner84(n) => Some(n.deref()),
            VerkleNode::Inner85(n) => Some(n.deref()),
            VerkleNode::Inner86(n) => Some(n.deref()),
            VerkleNode::Inner87(n) => Some(n.deref()),
            VerkleNode::Inner88(n) => Some(n.deref()),
            VerkleNode::Inner89(n) => Some(n.deref()),
            VerkleNode::Inner90(n) => Some(n.deref()),
            VerkleNode::Inner91(n) => Some(n.deref()),
            VerkleNode::Inner92(n) => Some(n.deref()),
            VerkleNode::Inner93(n) => Some(n.deref()),
            VerkleNode::Inner94(n) => Some(n.deref()),
            VerkleNode::Inner95(n) => Some(n.deref()),
            VerkleNode::Inner96(n) => Some(n.deref()),
            VerkleNode::Inner97(n) => Some(n.deref()),
            VerkleNode::Inner98(n) => Some(n.deref()),
            VerkleNode::Inner99(n) => Some(n.deref()),
            VerkleNode::Inner100(n) => Some(n.deref()),
            VerkleNode::Inner101(n) => Some(n.deref()),
            VerkleNode::Inner102(n) => Some(n.deref()),
            VerkleNode::Inner103(n) => Some(n.deref()),
            VerkleNode::Inner104(n) => Some(n.deref()),
            VerkleNode::Inner105(n) => Some(n.deref()),
            VerkleNode::Inner106(n) => Some(n.deref()),
            VerkleNode::Inner107(n) => Some(n.deref()),
            VerkleNode::Inner108(n) => Some(n.deref()),
            VerkleNode::Inner109(n) => Some(n.deref()),
            VerkleNode::Inner110(n) => Some(n.deref()),
            VerkleNode::Inner111(n) => Some(n.deref()),
            VerkleNode::Inner112(n) => Some(n.deref()),
            VerkleNode::Inner113(n) => Some(n.deref()),
            VerkleNode::Inner114(n) => Some(n.deref()),
            VerkleNode::Inner115(n) => Some(n.deref()),
            VerkleNode::Inner116(n) => Some(n.deref()),
            VerkleNode::Inner117(n) => Some(n.deref()),
            VerkleNode::Inner118(n) => Some(n.deref()),
            VerkleNode::Inner119(n) => Some(n.deref()),
            VerkleNode::Inner120(n) => Some(n.deref()),
            VerkleNode::Inner121(n) => Some(n.deref()),
            VerkleNode::Inner122(n) => Some(n.deref()),
            VerkleNode::Inner123(n) => Some(n.deref()),
            VerkleNode::Inner124(n) => Some(n.deref()),
            VerkleNode::Inner125(n) => Some(n.deref()),
            VerkleNode::Inner126(n) => Some(n.deref()),
            VerkleNode::Inner127(n) => Some(n.deref()),
            VerkleNode::Inner128(n) => Some(n.deref()),
            VerkleNode::Inner129(n) => Some(n.deref()),
            VerkleNode::Inner130(n) => Some(n.deref()),
            VerkleNode::Inner131(n) => Some(n.deref()),
            VerkleNode::Inner132(n) => Some(n.deref()),
            VerkleNode::Inner133(n) => Some(n.deref()),
            VerkleNode::Inner134(n) => Some(n.deref()),
            VerkleNode::Inner135(n) => Some(n.deref()),
            VerkleNode::Inner136(n) => Some(n.deref()),
            VerkleNode::Inner137(n) => Some(n.deref()),
            VerkleNode::Inner138(n) => Some(n.deref()),
            VerkleNode::Inner139(n) => Some(n.deref()),
            VerkleNode::Inner140(n) => Some(n.deref()),
            VerkleNode::Inner141(n) => Some(n.deref()),
            VerkleNode::Inner142(n) => Some(n.deref()),
            VerkleNode::Inner143(n) => Some(n.deref()),
            VerkleNode::Inner144(n) => Some(n.deref()),
            VerkleNode::Inner145(n) => Some(n.deref()),
            VerkleNode::Inner146(n) => Some(n.deref()),
            VerkleNode::Inner147(n) => Some(n.deref()),
            VerkleNode::Inner148(n) => Some(n.deref()),
            VerkleNode::Inner149(n) => Some(n.deref()),
            VerkleNode::Inner150(n) => Some(n.deref()),
            VerkleNode::Inner151(n) => Some(n.deref()),
            VerkleNode::Inner152(n) => Some(n.deref()),
            VerkleNode::Inner153(n) => Some(n.deref()),
            VerkleNode::Inner154(n) => Some(n.deref()),
            VerkleNode::Inner155(n) => Some(n.deref()),
            VerkleNode::Inner156(n) => Some(n.deref()),
            VerkleNode::Inner157(n) => Some(n.deref()),
            VerkleNode::Inner158(n) => Some(n.deref()),
            VerkleNode::Inner159(n) => Some(n.deref()),
            VerkleNode::Inner160(n) => Some(n.deref()),
            VerkleNode::Inner161(n) => Some(n.deref()),
            VerkleNode::Inner162(n) => Some(n.deref()),
            VerkleNode::Inner163(n) => Some(n.deref()),
            VerkleNode::Inner164(n) => Some(n.deref()),
            VerkleNode::Inner165(n) => Some(n.deref()),
            VerkleNode::Inner166(n) => Some(n.deref()),
            VerkleNode::Inner167(n) => Some(n.deref()),
            VerkleNode::Inner168(n) => Some(n.deref()),
            VerkleNode::Inner169(n) => Some(n.deref()),
            VerkleNode::Inner170(n) => Some(n.deref()),
            VerkleNode::Inner171(n) => Some(n.deref()),
            VerkleNode::Inner172(n) => Some(n.deref()),
            VerkleNode::Inner173(n) => Some(n.deref()),
            VerkleNode::Inner174(n) => Some(n.deref()),
            VerkleNode::Inner175(n) => Some(n.deref()),
            VerkleNode::Inner176(n) => Some(n.deref()),
            VerkleNode::Inner177(n) => Some(n.deref()),
            VerkleNode::Inner178(n) => Some(n.deref()),
            VerkleNode::Inner179(n) => Some(n.deref()),
            VerkleNode::Inner180(n) => Some(n.deref()),
            VerkleNode::Inner181(n) => Some(n.deref()),
            VerkleNode::Inner182(n) => Some(n.deref()),
            VerkleNode::Inner183(n) => Some(n.deref()),
            VerkleNode::Inner184(n) => Some(n.deref()),
            VerkleNode::Inner185(n) => Some(n.deref()),
            VerkleNode::Inner186(n) => Some(n.deref()),
            VerkleNode::Inner187(n) => Some(n.deref()),
            VerkleNode::Inner188(n) => Some(n.deref()),
            VerkleNode::Inner189(n) => Some(n.deref()),
            VerkleNode::Inner190(n) => Some(n.deref()),
            VerkleNode::Inner191(n) => Some(n.deref()),
            VerkleNode::Inner192(n) => Some(n.deref()),
            VerkleNode::Inner193(n) => Some(n.deref()),
            VerkleNode::Inner194(n) => Some(n.deref()),
            VerkleNode::Inner195(n) => Some(n.deref()),
            VerkleNode::Inner196(n) => Some(n.deref()),
            VerkleNode::Inner197(n) => Some(n.deref()),
            VerkleNode::Inner198(n) => Some(n.deref()),
            VerkleNode::Inner199(n) => Some(n.deref()),
            VerkleNode::Inner200(n) => Some(n.deref()),
            VerkleNode::Inner201(n) => Some(n.deref()),
            VerkleNode::Inner202(n) => Some(n.deref()),
            VerkleNode::Inner203(n) => Some(n.deref()),
            VerkleNode::Inner204(n) => Some(n.deref()),
            VerkleNode::Inner205(n) => Some(n.deref()),
            VerkleNode::Inner206(n) => Some(n.deref()),
            VerkleNode::Inner207(n) => Some(n.deref()),
            VerkleNode::Inner208(n) => Some(n.deref()),
            VerkleNode::Inner209(n) => Some(n.deref()),
            VerkleNode::Inner210(n) => Some(n.deref()),
            VerkleNode::Inner211(n) => Some(n.deref()),
            VerkleNode::Inner212(n) => Some(n.deref()),
            VerkleNode::Inner213(n) => Some(n.deref()),
            VerkleNode::Inner214(n) => Some(n.deref()),
            VerkleNode::Inner215(n) => Some(n.deref()),
            VerkleNode::Inner216(n) => Some(n.deref()),
            VerkleNode::Inner217(n) => Some(n.deref()),
            VerkleNode::Inner218(n) => Some(n.deref()),
            VerkleNode::Inner219(n) => Some(n.deref()),
            VerkleNode::Inner220(n) => Some(n.deref()),
            VerkleNode::Inner221(n) => Some(n.deref()),
            VerkleNode::Inner222(n) => Some(n.deref()),
            VerkleNode::Inner223(n) => Some(n.deref()),
            VerkleNode::Inner224(n) => Some(n.deref()),
            VerkleNode::Inner225(n) => Some(n.deref()),
            VerkleNode::Inner226(n) => Some(n.deref()),
            VerkleNode::Inner227(n) => Some(n.deref()),
            VerkleNode::Inner228(n) => Some(n.deref()),
            VerkleNode::Inner229(n) => Some(n.deref()),
            VerkleNode::Inner230(n) => Some(n.deref()),
            VerkleNode::Inner231(n) => Some(n.deref()),
            VerkleNode::Inner232(n) => Some(n.deref()),
            VerkleNode::Inner233(n) => Some(n.deref()),
            VerkleNode::Inner234(n) => Some(n.deref()),
            VerkleNode::Inner235(n) => Some(n.deref()),
            VerkleNode::Inner236(n) => Some(n.deref()),
            VerkleNode::Inner237(n) => Some(n.deref()),
            VerkleNode::Inner238(n) => Some(n.deref()),
            VerkleNode::Inner239(n) => Some(n.deref()),
            VerkleNode::Inner240(n) => Some(n.deref()),
            VerkleNode::Inner241(n) => Some(n.deref()),
            VerkleNode::Inner242(n) => Some(n.deref()),
            VerkleNode::Inner243(n) => Some(n.deref()),
            VerkleNode::Inner244(n) => Some(n.deref()),
            VerkleNode::Inner245(n) => Some(n.deref()),
            VerkleNode::Inner246(n) => Some(n.deref()),
            VerkleNode::Inner247(n) => Some(n.deref()),
            VerkleNode::Inner248(n) => Some(n.deref()),
            VerkleNode::Inner249(n) => Some(n.deref()),
            VerkleNode::Inner250(n) => Some(n.deref()),
            VerkleNode::Inner251(n) => Some(n.deref()),
            VerkleNode::Inner252(n) => Some(n.deref()),
            VerkleNode::Inner253(n) => Some(n.deref()),
            VerkleNode::Inner254(n) => Some(n.deref()),
            VerkleNode::Inner255(n) => Some(n.deref()),
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
            | VerkleNode::Leaf17(_)
            | VerkleNode::Leaf18(_)
            | VerkleNode::Leaf19(_)
            | VerkleNode::Leaf20(_)
            | VerkleNode::Leaf21(_)
            | VerkleNode::Leaf22(_)
            | VerkleNode::Leaf23(_)
            | VerkleNode::Leaf24(_)
            | VerkleNode::Leaf25(_)
            | VerkleNode::Leaf26(_)
            | VerkleNode::Leaf27(_)
            | VerkleNode::Leaf28(_)
            | VerkleNode::Leaf29(_)
            | VerkleNode::Leaf30(_)
            | VerkleNode::Leaf31(_)
            | VerkleNode::Leaf32(_)
            | VerkleNode::Leaf33(_)
            | VerkleNode::Leaf34(_)
            | VerkleNode::Leaf35(_)
            | VerkleNode::Leaf36(_)
            | VerkleNode::Leaf37(_)
            | VerkleNode::Leaf38(_)
            | VerkleNode::Leaf39(_)
            | VerkleNode::Leaf40(_)
            | VerkleNode::Leaf41(_)
            | VerkleNode::Leaf42(_)
            | VerkleNode::Leaf43(_)
            | VerkleNode::Leaf44(_)
            | VerkleNode::Leaf45(_)
            | VerkleNode::Leaf46(_)
            | VerkleNode::Leaf47(_)
            | VerkleNode::Leaf48(_)
            | VerkleNode::Leaf49(_)
            | VerkleNode::Leaf50(_)
            | VerkleNode::Leaf51(_)
            | VerkleNode::Leaf52(_)
            | VerkleNode::Leaf53(_)
            | VerkleNode::Leaf54(_)
            | VerkleNode::Leaf55(_)
            | VerkleNode::Leaf56(_)
            | VerkleNode::Leaf57(_)
            | VerkleNode::Leaf58(_)
            | VerkleNode::Leaf59(_)
            | VerkleNode::Leaf60(_)
            | VerkleNode::Leaf61(_)
            | VerkleNode::Leaf62(_)
            | VerkleNode::Leaf63(_)
            | VerkleNode::Leaf64(_)
            | VerkleNode::Leaf65(_)
            | VerkleNode::Leaf66(_)
            | VerkleNode::Leaf67(_)
            | VerkleNode::Leaf68(_)
            | VerkleNode::Leaf69(_)
            | VerkleNode::Leaf70(_)
            | VerkleNode::Leaf71(_)
            | VerkleNode::Leaf72(_)
            | VerkleNode::Leaf73(_)
            | VerkleNode::Leaf74(_)
            | VerkleNode::Leaf75(_)
            | VerkleNode::Leaf76(_)
            | VerkleNode::Leaf77(_)
            | VerkleNode::Leaf78(_)
            | VerkleNode::Leaf79(_)
            | VerkleNode::Leaf80(_)
            | VerkleNode::Leaf81(_)
            | VerkleNode::Leaf82(_)
            | VerkleNode::Leaf83(_)
            | VerkleNode::Leaf84(_)
            | VerkleNode::Leaf85(_)
            | VerkleNode::Leaf86(_)
            | VerkleNode::Leaf87(_)
            | VerkleNode::Leaf88(_)
            | VerkleNode::Leaf89(_)
            | VerkleNode::Leaf90(_)
            | VerkleNode::Leaf91(_)
            | VerkleNode::Leaf92(_)
            | VerkleNode::Leaf93(_)
            | VerkleNode::Leaf94(_)
            | VerkleNode::Leaf95(_)
            | VerkleNode::Leaf96(_)
            | VerkleNode::Leaf97(_)
            | VerkleNode::Leaf98(_)
            | VerkleNode::Leaf99(_)
            | VerkleNode::Leaf100(_)
            | VerkleNode::Leaf101(_)
            | VerkleNode::Leaf102(_)
            | VerkleNode::Leaf103(_)
            | VerkleNode::Leaf104(_)
            | VerkleNode::Leaf105(_)
            | VerkleNode::Leaf106(_)
            | VerkleNode::Leaf107(_)
            | VerkleNode::Leaf108(_)
            | VerkleNode::Leaf109(_)
            | VerkleNode::Leaf110(_)
            | VerkleNode::Leaf111(_)
            | VerkleNode::Leaf112(_)
            | VerkleNode::Leaf113(_)
            | VerkleNode::Leaf114(_)
            | VerkleNode::Leaf115(_)
            | VerkleNode::Leaf116(_)
            | VerkleNode::Leaf117(_)
            | VerkleNode::Leaf118(_)
            | VerkleNode::Leaf119(_)
            | VerkleNode::Leaf120(_)
            | VerkleNode::Leaf121(_)
            | VerkleNode::Leaf122(_)
            | VerkleNode::Leaf123(_)
            | VerkleNode::Leaf124(_)
            | VerkleNode::Leaf125(_)
            | VerkleNode::Leaf126(_)
            | VerkleNode::Leaf127(_)
            | VerkleNode::Leaf128(_)
            | VerkleNode::Leaf129(_)
            | VerkleNode::Leaf130(_)
            | VerkleNode::Leaf131(_)
            | VerkleNode::Leaf132(_)
            | VerkleNode::Leaf133(_)
            | VerkleNode::Leaf134(_)
            | VerkleNode::Leaf135(_)
            | VerkleNode::Leaf136(_)
            | VerkleNode::Leaf137(_)
            | VerkleNode::Leaf138(_)
            | VerkleNode::Leaf139(_)
            | VerkleNode::Leaf140(_)
            | VerkleNode::Leaf141(_)
            | VerkleNode::Leaf142(_)
            | VerkleNode::Leaf143(_)
            | VerkleNode::Leaf144(_)
            | VerkleNode::Leaf145(_)
            | VerkleNode::Leaf146(_)
            | VerkleNode::Leaf147(_)
            | VerkleNode::Leaf148(_)
            | VerkleNode::Leaf149(_)
            | VerkleNode::Leaf150(_)
            | VerkleNode::Leaf151(_)
            | VerkleNode::Leaf152(_)
            | VerkleNode::Leaf153(_)
            | VerkleNode::Leaf154(_)
            | VerkleNode::Leaf155(_)
            | VerkleNode::Leaf156(_)
            | VerkleNode::Leaf157(_)
            | VerkleNode::Leaf158(_)
            | VerkleNode::Leaf159(_)
            | VerkleNode::Leaf160(_)
            | VerkleNode::Leaf161(_)
            | VerkleNode::Leaf162(_)
            | VerkleNode::Leaf163(_)
            | VerkleNode::Leaf164(_)
            | VerkleNode::Leaf165(_)
            | VerkleNode::Leaf166(_)
            | VerkleNode::Leaf167(_)
            | VerkleNode::Leaf168(_)
            | VerkleNode::Leaf169(_)
            | VerkleNode::Leaf170(_)
            | VerkleNode::Leaf171(_)
            | VerkleNode::Leaf172(_)
            | VerkleNode::Leaf173(_)
            | VerkleNode::Leaf174(_)
            | VerkleNode::Leaf175(_)
            | VerkleNode::Leaf176(_)
            | VerkleNode::Leaf177(_)
            | VerkleNode::Leaf178(_)
            | VerkleNode::Leaf179(_)
            | VerkleNode::Leaf180(_)
            | VerkleNode::Leaf181(_)
            | VerkleNode::Leaf182(_)
            | VerkleNode::Leaf183(_)
            | VerkleNode::Leaf184(_)
            | VerkleNode::Leaf185(_)
            | VerkleNode::Leaf186(_)
            | VerkleNode::Leaf187(_)
            | VerkleNode::Leaf188(_)
            | VerkleNode::Leaf189(_)
            | VerkleNode::Leaf190(_)
            | VerkleNode::Leaf191(_)
            | VerkleNode::Leaf192(_)
            | VerkleNode::Leaf193(_)
            | VerkleNode::Leaf194(_)
            | VerkleNode::Leaf195(_)
            | VerkleNode::Leaf196(_)
            | VerkleNode::Leaf197(_)
            | VerkleNode::Leaf198(_)
            | VerkleNode::Leaf199(_)
            | VerkleNode::Leaf200(_)
            | VerkleNode::Leaf201(_)
            | VerkleNode::Leaf202(_)
            | VerkleNode::Leaf203(_)
            | VerkleNode::Leaf204(_)
            | VerkleNode::Leaf205(_)
            | VerkleNode::Leaf206(_)
            | VerkleNode::Leaf207(_)
            | VerkleNode::Leaf208(_)
            | VerkleNode::Leaf209(_)
            | VerkleNode::Leaf210(_)
            | VerkleNode::Leaf211(_)
            | VerkleNode::Leaf212(_)
            | VerkleNode::Leaf213(_)
            | VerkleNode::Leaf214(_)
            | VerkleNode::Leaf215(_)
            | VerkleNode::Leaf216(_)
            | VerkleNode::Leaf217(_)
            | VerkleNode::Leaf218(_)
            | VerkleNode::Leaf219(_)
            | VerkleNode::Leaf220(_)
            | VerkleNode::Leaf221(_)
            | VerkleNode::Leaf222(_)
            | VerkleNode::Leaf223(_)
            | VerkleNode::Leaf224(_)
            | VerkleNode::Leaf225(_)
            | VerkleNode::Leaf226(_)
            | VerkleNode::Leaf227(_)
            | VerkleNode::Leaf228(_)
            | VerkleNode::Leaf229(_)
            | VerkleNode::Leaf230(_)
            | VerkleNode::Leaf231(_)
            | VerkleNode::Leaf232(_)
            | VerkleNode::Leaf233(_)
            | VerkleNode::Leaf234(_)
            | VerkleNode::Leaf235(_)
            | VerkleNode::Leaf236(_)
            | VerkleNode::Leaf237(_)
            | VerkleNode::Leaf238(_)
            | VerkleNode::Leaf239(_)
            | VerkleNode::Leaf240(_)
            | VerkleNode::Leaf241(_)
            | VerkleNode::Leaf242(_)
            | VerkleNode::Leaf243(_)
            | VerkleNode::Leaf244(_)
            | VerkleNode::Leaf245(_)
            | VerkleNode::Leaf246(_)
            | VerkleNode::Leaf247(_)
            | VerkleNode::Leaf248(_)
            | VerkleNode::Leaf249(_)
            | VerkleNode::Leaf250(_)
            | VerkleNode::Leaf251(_)
            | VerkleNode::Leaf252(_)
            | VerkleNode::Leaf253(_)
            | VerkleNode::Leaf254(_)
            | VerkleNode::Leaf255(_)
            | VerkleNode::Leaf256(_)
            | VerkleNode::LeafDelta(_) => {}
            VerkleNode::Inner1(_)
            | VerkleNode::Inner2(_)
            | VerkleNode::Inner3(_)
            | VerkleNode::Inner4(_)
            | VerkleNode::Inner5(_)
            | VerkleNode::Inner6(_)
            | VerkleNode::Inner7(_)
            | VerkleNode::Inner8(_)
            | VerkleNode::Inner9(_)
            | VerkleNode::Inner10(_)
            | VerkleNode::Inner11(_)
            | VerkleNode::Inner12(_)
            | VerkleNode::Inner13(_)
            | VerkleNode::Inner14(_)
            | VerkleNode::Inner15(_)
            | VerkleNode::Inner16(_)
            | VerkleNode::Inner17(_)
            | VerkleNode::Inner18(_)
            | VerkleNode::Inner19(_)
            | VerkleNode::Inner20(_)
            | VerkleNode::Inner21(_)
            | VerkleNode::Inner22(_)
            | VerkleNode::Inner23(_)
            | VerkleNode::Inner24(_)
            | VerkleNode::Inner25(_)
            | VerkleNode::Inner26(_)
            | VerkleNode::Inner27(_)
            | VerkleNode::Inner28(_)
            | VerkleNode::Inner29(_)
            | VerkleNode::Inner30(_)
            | VerkleNode::Inner31(_)
            | VerkleNode::Inner32(_)
            | VerkleNode::Inner33(_)
            | VerkleNode::Inner34(_)
            | VerkleNode::Inner35(_)
            | VerkleNode::Inner36(_)
            | VerkleNode::Inner37(_)
            | VerkleNode::Inner38(_)
            | VerkleNode::Inner39(_)
            | VerkleNode::Inner40(_)
            | VerkleNode::Inner41(_)
            | VerkleNode::Inner42(_)
            | VerkleNode::Inner43(_)
            | VerkleNode::Inner44(_)
            | VerkleNode::Inner45(_)
            | VerkleNode::Inner46(_)
            | VerkleNode::Inner47(_)
            | VerkleNode::Inner48(_)
            | VerkleNode::Inner49(_)
            | VerkleNode::Inner50(_)
            | VerkleNode::Inner51(_)
            | VerkleNode::Inner52(_)
            | VerkleNode::Inner53(_)
            | VerkleNode::Inner54(_)
            | VerkleNode::Inner55(_)
            | VerkleNode::Inner56(_)
            | VerkleNode::Inner57(_)
            | VerkleNode::Inner58(_)
            | VerkleNode::Inner59(_)
            | VerkleNode::Inner60(_)
            | VerkleNode::Inner61(_)
            | VerkleNode::Inner62(_)
            | VerkleNode::Inner63(_)
            | VerkleNode::Inner64(_)
            | VerkleNode::Inner65(_)
            | VerkleNode::Inner66(_)
            | VerkleNode::Inner67(_)
            | VerkleNode::Inner68(_)
            | VerkleNode::Inner69(_)
            | VerkleNode::Inner70(_)
            | VerkleNode::Inner71(_)
            | VerkleNode::Inner72(_)
            | VerkleNode::Inner73(_)
            | VerkleNode::Inner74(_)
            | VerkleNode::Inner75(_)
            | VerkleNode::Inner76(_)
            | VerkleNode::Inner77(_)
            | VerkleNode::Inner78(_)
            | VerkleNode::Inner79(_)
            | VerkleNode::Inner80(_)
            | VerkleNode::Inner81(_)
            | VerkleNode::Inner82(_)
            | VerkleNode::Inner83(_)
            | VerkleNode::Inner84(_)
            | VerkleNode::Inner85(_)
            | VerkleNode::Inner86(_)
            | VerkleNode::Inner87(_)
            | VerkleNode::Inner88(_)
            | VerkleNode::Inner89(_)
            | VerkleNode::Inner90(_)
            | VerkleNode::Inner91(_)
            | VerkleNode::Inner92(_)
            | VerkleNode::Inner93(_)
            | VerkleNode::Inner94(_)
            | VerkleNode::Inner95(_)
            | VerkleNode::Inner96(_)
            | VerkleNode::Inner97(_)
            | VerkleNode::Inner98(_)
            | VerkleNode::Inner99(_)
            | VerkleNode::Inner100(_)
            | VerkleNode::Inner101(_)
            | VerkleNode::Inner102(_)
            | VerkleNode::Inner103(_)
            | VerkleNode::Inner104(_)
            | VerkleNode::Inner105(_)
            | VerkleNode::Inner106(_)
            | VerkleNode::Inner107(_)
            | VerkleNode::Inner108(_)
            | VerkleNode::Inner109(_)
            | VerkleNode::Inner110(_)
            | VerkleNode::Inner111(_)
            | VerkleNode::Inner112(_)
            | VerkleNode::Inner113(_)
            | VerkleNode::Inner114(_)
            | VerkleNode::Inner115(_)
            | VerkleNode::Inner116(_)
            | VerkleNode::Inner117(_)
            | VerkleNode::Inner118(_)
            | VerkleNode::Inner119(_)
            | VerkleNode::Inner120(_)
            | VerkleNode::Inner121(_)
            | VerkleNode::Inner122(_)
            | VerkleNode::Inner123(_)
            | VerkleNode::Inner124(_)
            | VerkleNode::Inner125(_)
            | VerkleNode::Inner126(_)
            | VerkleNode::Inner127(_)
            | VerkleNode::Inner128(_)
            | VerkleNode::Inner129(_)
            | VerkleNode::Inner130(_)
            | VerkleNode::Inner131(_)
            | VerkleNode::Inner132(_)
            | VerkleNode::Inner133(_)
            | VerkleNode::Inner134(_)
            | VerkleNode::Inner135(_)
            | VerkleNode::Inner136(_)
            | VerkleNode::Inner137(_)
            | VerkleNode::Inner138(_)
            | VerkleNode::Inner139(_)
            | VerkleNode::Inner140(_)
            | VerkleNode::Inner141(_)
            | VerkleNode::Inner142(_)
            | VerkleNode::Inner143(_)
            | VerkleNode::Inner144(_)
            | VerkleNode::Inner145(_)
            | VerkleNode::Inner146(_)
            | VerkleNode::Inner147(_)
            | VerkleNode::Inner148(_)
            | VerkleNode::Inner149(_)
            | VerkleNode::Inner150(_)
            | VerkleNode::Inner151(_)
            | VerkleNode::Inner152(_)
            | VerkleNode::Inner153(_)
            | VerkleNode::Inner154(_)
            | VerkleNode::Inner155(_)
            | VerkleNode::Inner156(_)
            | VerkleNode::Inner157(_)
            | VerkleNode::Inner158(_)
            | VerkleNode::Inner159(_)
            | VerkleNode::Inner160(_)
            | VerkleNode::Inner161(_)
            | VerkleNode::Inner162(_)
            | VerkleNode::Inner163(_)
            | VerkleNode::Inner164(_)
            | VerkleNode::Inner165(_)
            | VerkleNode::Inner166(_)
            | VerkleNode::Inner167(_)
            | VerkleNode::Inner168(_)
            | VerkleNode::Inner169(_)
            | VerkleNode::Inner170(_)
            | VerkleNode::Inner171(_)
            | VerkleNode::Inner172(_)
            | VerkleNode::Inner173(_)
            | VerkleNode::Inner174(_)
            | VerkleNode::Inner175(_)
            | VerkleNode::Inner176(_)
            | VerkleNode::Inner177(_)
            | VerkleNode::Inner178(_)
            | VerkleNode::Inner179(_)
            | VerkleNode::Inner180(_)
            | VerkleNode::Inner181(_)
            | VerkleNode::Inner182(_)
            | VerkleNode::Inner183(_)
            | VerkleNode::Inner184(_)
            | VerkleNode::Inner185(_)
            | VerkleNode::Inner186(_)
            | VerkleNode::Inner187(_)
            | VerkleNode::Inner188(_)
            | VerkleNode::Inner189(_)
            | VerkleNode::Inner190(_)
            | VerkleNode::Inner191(_)
            | VerkleNode::Inner192(_)
            | VerkleNode::Inner193(_)
            | VerkleNode::Inner194(_)
            | VerkleNode::Inner195(_)
            | VerkleNode::Inner196(_)
            | VerkleNode::Inner197(_)
            | VerkleNode::Inner198(_)
            | VerkleNode::Inner199(_)
            | VerkleNode::Inner200(_)
            | VerkleNode::Inner201(_)
            | VerkleNode::Inner202(_)
            | VerkleNode::Inner203(_)
            | VerkleNode::Inner204(_)
            | VerkleNode::Inner205(_)
            | VerkleNode::Inner206(_)
            | VerkleNode::Inner207(_)
            | VerkleNode::Inner208(_)
            | VerkleNode::Inner209(_)
            | VerkleNode::Inner210(_)
            | VerkleNode::Inner211(_)
            | VerkleNode::Inner212(_)
            | VerkleNode::Inner213(_)
            | VerkleNode::Inner214(_)
            | VerkleNode::Inner215(_)
            | VerkleNode::Inner216(_)
            | VerkleNode::Inner217(_)
            | VerkleNode::Inner218(_)
            | VerkleNode::Inner219(_)
            | VerkleNode::Inner220(_)
            | VerkleNode::Inner221(_)
            | VerkleNode::Inner222(_)
            | VerkleNode::Inner223(_)
            | VerkleNode::Inner224(_)
            | VerkleNode::Inner225(_)
            | VerkleNode::Inner226(_)
            | VerkleNode::Inner227(_)
            | VerkleNode::Inner228(_)
            | VerkleNode::Inner229(_)
            | VerkleNode::Inner230(_)
            | VerkleNode::Inner231(_)
            | VerkleNode::Inner232(_)
            | VerkleNode::Inner233(_)
            | VerkleNode::Inner234(_)
            | VerkleNode::Inner235(_)
            | VerkleNode::Inner236(_)
            | VerkleNode::Inner237(_)
            | VerkleNode::Inner238(_)
            | VerkleNode::Inner239(_)
            | VerkleNode::Inner240(_)
            | VerkleNode::Inner241(_)
            | VerkleNode::Inner242(_)
            | VerkleNode::Inner243(_)
            | VerkleNode::Inner244(_)
            | VerkleNode::Inner245(_)
            | VerkleNode::Inner246(_)
            | VerkleNode::Inner247(_)
            | VerkleNode::Inner248(_)
            | VerkleNode::Inner249(_)
            | VerkleNode::Inner250(_)
            | VerkleNode::Inner251(_)
            | VerkleNode::Inner252(_)
            | VerkleNode::Inner253(_)
            | VerkleNode::Inner254(_)
            | VerkleNode::Inner255(_)
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
        match self {
            VerkleNode::LeafDelta(n) => match base {
                VerkleNode::Leaf256(f) => n.values = f.values,
                VerkleNode::Leaf146(f) => {
                    n.values = array::from_fn(|i| {
                        ValueWithIndex::get_slot_for(&f.values, i as u8)
                            .map(|slot| f.values[slot].item)
                            .unwrap_or_default()
                    });
                }
                _ => {
                    return Err(Error::Internal(
                        "copy_from_delta_base called with unsupported node".to_owned(),
                    )
                    .into());
                }
            },
            VerkleNode::InnerDelta(n) => {
                if let VerkleNode::Inner256(i) = base {
                    n.children = i.children;
                } else {
                    return Err(Error::Internal(
                        "copy_from_delta_base called with unsupported node".to_owned(),
                    )
                    .into());
                }
            }
            _ => (),
        }
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
            VerkleNode::Inner1(_) => Some(VerkleNodeKind::Inner1),
            VerkleNode::Inner2(_) => Some(VerkleNodeKind::Inner2),
            VerkleNode::Inner3(_) => Some(VerkleNodeKind::Inner3),
            VerkleNode::Inner4(_) => Some(VerkleNodeKind::Inner4),
            VerkleNode::Inner5(_) => Some(VerkleNodeKind::Inner5),
            VerkleNode::Inner6(_) => Some(VerkleNodeKind::Inner6),
            VerkleNode::Inner7(_) => Some(VerkleNodeKind::Inner7),
            VerkleNode::Inner8(_) => Some(VerkleNodeKind::Inner8),
            VerkleNode::Inner9(_) => Some(VerkleNodeKind::Inner9),
            VerkleNode::Inner10(_) => Some(VerkleNodeKind::Inner10),
            VerkleNode::Inner11(_) => Some(VerkleNodeKind::Inner11),
            VerkleNode::Inner12(_) => Some(VerkleNodeKind::Inner12),
            VerkleNode::Inner13(_) => Some(VerkleNodeKind::Inner13),
            VerkleNode::Inner14(_) => Some(VerkleNodeKind::Inner14),
            VerkleNode::Inner15(_) => Some(VerkleNodeKind::Inner15),
            VerkleNode::Inner16(_) => Some(VerkleNodeKind::Inner16),
            VerkleNode::Inner17(_) => Some(VerkleNodeKind::Inner17),
            VerkleNode::Inner18(_) => Some(VerkleNodeKind::Inner18),
            VerkleNode::Inner19(_) => Some(VerkleNodeKind::Inner19),
            VerkleNode::Inner20(_) => Some(VerkleNodeKind::Inner20),
            VerkleNode::Inner21(_) => Some(VerkleNodeKind::Inner21),
            VerkleNode::Inner22(_) => Some(VerkleNodeKind::Inner22),
            VerkleNode::Inner23(_) => Some(VerkleNodeKind::Inner23),
            VerkleNode::Inner24(_) => Some(VerkleNodeKind::Inner24),
            VerkleNode::Inner25(_) => Some(VerkleNodeKind::Inner25),
            VerkleNode::Inner26(_) => Some(VerkleNodeKind::Inner26),
            VerkleNode::Inner27(_) => Some(VerkleNodeKind::Inner27),
            VerkleNode::Inner28(_) => Some(VerkleNodeKind::Inner28),
            VerkleNode::Inner29(_) => Some(VerkleNodeKind::Inner29),
            VerkleNode::Inner30(_) => Some(VerkleNodeKind::Inner30),
            VerkleNode::Inner31(_) => Some(VerkleNodeKind::Inner31),
            VerkleNode::Inner32(_) => Some(VerkleNodeKind::Inner32),
            VerkleNode::Inner33(_) => Some(VerkleNodeKind::Inner33),
            VerkleNode::Inner34(_) => Some(VerkleNodeKind::Inner34),
            VerkleNode::Inner35(_) => Some(VerkleNodeKind::Inner35),
            VerkleNode::Inner36(_) => Some(VerkleNodeKind::Inner36),
            VerkleNode::Inner37(_) => Some(VerkleNodeKind::Inner37),
            VerkleNode::Inner38(_) => Some(VerkleNodeKind::Inner38),
            VerkleNode::Inner39(_) => Some(VerkleNodeKind::Inner39),
            VerkleNode::Inner40(_) => Some(VerkleNodeKind::Inner40),
            VerkleNode::Inner41(_) => Some(VerkleNodeKind::Inner41),
            VerkleNode::Inner42(_) => Some(VerkleNodeKind::Inner42),
            VerkleNode::Inner43(_) => Some(VerkleNodeKind::Inner43),
            VerkleNode::Inner44(_) => Some(VerkleNodeKind::Inner44),
            VerkleNode::Inner45(_) => Some(VerkleNodeKind::Inner45),
            VerkleNode::Inner46(_) => Some(VerkleNodeKind::Inner46),
            VerkleNode::Inner47(_) => Some(VerkleNodeKind::Inner47),
            VerkleNode::Inner48(_) => Some(VerkleNodeKind::Inner48),
            VerkleNode::Inner49(_) => Some(VerkleNodeKind::Inner49),
            VerkleNode::Inner50(_) => Some(VerkleNodeKind::Inner50),
            VerkleNode::Inner51(_) => Some(VerkleNodeKind::Inner51),
            VerkleNode::Inner52(_) => Some(VerkleNodeKind::Inner52),
            VerkleNode::Inner53(_) => Some(VerkleNodeKind::Inner53),
            VerkleNode::Inner54(_) => Some(VerkleNodeKind::Inner54),
            VerkleNode::Inner55(_) => Some(VerkleNodeKind::Inner55),
            VerkleNode::Inner56(_) => Some(VerkleNodeKind::Inner56),
            VerkleNode::Inner57(_) => Some(VerkleNodeKind::Inner57),
            VerkleNode::Inner58(_) => Some(VerkleNodeKind::Inner58),
            VerkleNode::Inner59(_) => Some(VerkleNodeKind::Inner59),
            VerkleNode::Inner60(_) => Some(VerkleNodeKind::Inner60),
            VerkleNode::Inner61(_) => Some(VerkleNodeKind::Inner61),
            VerkleNode::Inner62(_) => Some(VerkleNodeKind::Inner62),
            VerkleNode::Inner63(_) => Some(VerkleNodeKind::Inner63),
            VerkleNode::Inner64(_) => Some(VerkleNodeKind::Inner64),
            VerkleNode::Inner65(_) => Some(VerkleNodeKind::Inner65),
            VerkleNode::Inner66(_) => Some(VerkleNodeKind::Inner66),
            VerkleNode::Inner67(_) => Some(VerkleNodeKind::Inner67),
            VerkleNode::Inner68(_) => Some(VerkleNodeKind::Inner68),
            VerkleNode::Inner69(_) => Some(VerkleNodeKind::Inner69),
            VerkleNode::Inner70(_) => Some(VerkleNodeKind::Inner70),
            VerkleNode::Inner71(_) => Some(VerkleNodeKind::Inner71),
            VerkleNode::Inner72(_) => Some(VerkleNodeKind::Inner72),
            VerkleNode::Inner73(_) => Some(VerkleNodeKind::Inner73),
            VerkleNode::Inner74(_) => Some(VerkleNodeKind::Inner74),
            VerkleNode::Inner75(_) => Some(VerkleNodeKind::Inner75),
            VerkleNode::Inner76(_) => Some(VerkleNodeKind::Inner76),
            VerkleNode::Inner77(_) => Some(VerkleNodeKind::Inner77),
            VerkleNode::Inner78(_) => Some(VerkleNodeKind::Inner78),
            VerkleNode::Inner79(_) => Some(VerkleNodeKind::Inner79),
            VerkleNode::Inner80(_) => Some(VerkleNodeKind::Inner80),
            VerkleNode::Inner81(_) => Some(VerkleNodeKind::Inner81),
            VerkleNode::Inner82(_) => Some(VerkleNodeKind::Inner82),
            VerkleNode::Inner83(_) => Some(VerkleNodeKind::Inner83),
            VerkleNode::Inner84(_) => Some(VerkleNodeKind::Inner84),
            VerkleNode::Inner85(_) => Some(VerkleNodeKind::Inner85),
            VerkleNode::Inner86(_) => Some(VerkleNodeKind::Inner86),
            VerkleNode::Inner87(_) => Some(VerkleNodeKind::Inner87),
            VerkleNode::Inner88(_) => Some(VerkleNodeKind::Inner88),
            VerkleNode::Inner89(_) => Some(VerkleNodeKind::Inner89),
            VerkleNode::Inner90(_) => Some(VerkleNodeKind::Inner90),
            VerkleNode::Inner91(_) => Some(VerkleNodeKind::Inner91),
            VerkleNode::Inner92(_) => Some(VerkleNodeKind::Inner92),
            VerkleNode::Inner93(_) => Some(VerkleNodeKind::Inner93),
            VerkleNode::Inner94(_) => Some(VerkleNodeKind::Inner94),
            VerkleNode::Inner95(_) => Some(VerkleNodeKind::Inner95),
            VerkleNode::Inner96(_) => Some(VerkleNodeKind::Inner96),
            VerkleNode::Inner97(_) => Some(VerkleNodeKind::Inner97),
            VerkleNode::Inner98(_) => Some(VerkleNodeKind::Inner98),
            VerkleNode::Inner99(_) => Some(VerkleNodeKind::Inner99),
            VerkleNode::Inner100(_) => Some(VerkleNodeKind::Inner100),
            VerkleNode::Inner101(_) => Some(VerkleNodeKind::Inner101),
            VerkleNode::Inner102(_) => Some(VerkleNodeKind::Inner102),
            VerkleNode::Inner103(_) => Some(VerkleNodeKind::Inner103),
            VerkleNode::Inner104(_) => Some(VerkleNodeKind::Inner104),
            VerkleNode::Inner105(_) => Some(VerkleNodeKind::Inner105),
            VerkleNode::Inner106(_) => Some(VerkleNodeKind::Inner106),
            VerkleNode::Inner107(_) => Some(VerkleNodeKind::Inner107),
            VerkleNode::Inner108(_) => Some(VerkleNodeKind::Inner108),
            VerkleNode::Inner109(_) => Some(VerkleNodeKind::Inner109),
            VerkleNode::Inner110(_) => Some(VerkleNodeKind::Inner110),
            VerkleNode::Inner111(_) => Some(VerkleNodeKind::Inner111),
            VerkleNode::Inner112(_) => Some(VerkleNodeKind::Inner112),
            VerkleNode::Inner113(_) => Some(VerkleNodeKind::Inner113),
            VerkleNode::Inner114(_) => Some(VerkleNodeKind::Inner114),
            VerkleNode::Inner115(_) => Some(VerkleNodeKind::Inner115),
            VerkleNode::Inner116(_) => Some(VerkleNodeKind::Inner116),
            VerkleNode::Inner117(_) => Some(VerkleNodeKind::Inner117),
            VerkleNode::Inner118(_) => Some(VerkleNodeKind::Inner118),
            VerkleNode::Inner119(_) => Some(VerkleNodeKind::Inner119),
            VerkleNode::Inner120(_) => Some(VerkleNodeKind::Inner120),
            VerkleNode::Inner121(_) => Some(VerkleNodeKind::Inner121),
            VerkleNode::Inner122(_) => Some(VerkleNodeKind::Inner122),
            VerkleNode::Inner123(_) => Some(VerkleNodeKind::Inner123),
            VerkleNode::Inner124(_) => Some(VerkleNodeKind::Inner124),
            VerkleNode::Inner125(_) => Some(VerkleNodeKind::Inner125),
            VerkleNode::Inner126(_) => Some(VerkleNodeKind::Inner126),
            VerkleNode::Inner127(_) => Some(VerkleNodeKind::Inner127),
            VerkleNode::Inner128(_) => Some(VerkleNodeKind::Inner128),
            VerkleNode::Inner129(_) => Some(VerkleNodeKind::Inner129),
            VerkleNode::Inner130(_) => Some(VerkleNodeKind::Inner130),
            VerkleNode::Inner131(_) => Some(VerkleNodeKind::Inner131),
            VerkleNode::Inner132(_) => Some(VerkleNodeKind::Inner132),
            VerkleNode::Inner133(_) => Some(VerkleNodeKind::Inner133),
            VerkleNode::Inner134(_) => Some(VerkleNodeKind::Inner134),
            VerkleNode::Inner135(_) => Some(VerkleNodeKind::Inner135),
            VerkleNode::Inner136(_) => Some(VerkleNodeKind::Inner136),
            VerkleNode::Inner137(_) => Some(VerkleNodeKind::Inner137),
            VerkleNode::Inner138(_) => Some(VerkleNodeKind::Inner138),
            VerkleNode::Inner139(_) => Some(VerkleNodeKind::Inner139),
            VerkleNode::Inner140(_) => Some(VerkleNodeKind::Inner140),
            VerkleNode::Inner141(_) => Some(VerkleNodeKind::Inner141),
            VerkleNode::Inner142(_) => Some(VerkleNodeKind::Inner142),
            VerkleNode::Inner143(_) => Some(VerkleNodeKind::Inner143),
            VerkleNode::Inner144(_) => Some(VerkleNodeKind::Inner144),
            VerkleNode::Inner145(_) => Some(VerkleNodeKind::Inner145),
            VerkleNode::Inner146(_) => Some(VerkleNodeKind::Inner146),
            VerkleNode::Inner147(_) => Some(VerkleNodeKind::Inner147),
            VerkleNode::Inner148(_) => Some(VerkleNodeKind::Inner148),
            VerkleNode::Inner149(_) => Some(VerkleNodeKind::Inner149),
            VerkleNode::Inner150(_) => Some(VerkleNodeKind::Inner150),
            VerkleNode::Inner151(_) => Some(VerkleNodeKind::Inner151),
            VerkleNode::Inner152(_) => Some(VerkleNodeKind::Inner152),
            VerkleNode::Inner153(_) => Some(VerkleNodeKind::Inner153),
            VerkleNode::Inner154(_) => Some(VerkleNodeKind::Inner154),
            VerkleNode::Inner155(_) => Some(VerkleNodeKind::Inner155),
            VerkleNode::Inner156(_) => Some(VerkleNodeKind::Inner156),
            VerkleNode::Inner157(_) => Some(VerkleNodeKind::Inner157),
            VerkleNode::Inner158(_) => Some(VerkleNodeKind::Inner158),
            VerkleNode::Inner159(_) => Some(VerkleNodeKind::Inner159),
            VerkleNode::Inner160(_) => Some(VerkleNodeKind::Inner160),
            VerkleNode::Inner161(_) => Some(VerkleNodeKind::Inner161),
            VerkleNode::Inner162(_) => Some(VerkleNodeKind::Inner162),
            VerkleNode::Inner163(_) => Some(VerkleNodeKind::Inner163),
            VerkleNode::Inner164(_) => Some(VerkleNodeKind::Inner164),
            VerkleNode::Inner165(_) => Some(VerkleNodeKind::Inner165),
            VerkleNode::Inner166(_) => Some(VerkleNodeKind::Inner166),
            VerkleNode::Inner167(_) => Some(VerkleNodeKind::Inner167),
            VerkleNode::Inner168(_) => Some(VerkleNodeKind::Inner168),
            VerkleNode::Inner169(_) => Some(VerkleNodeKind::Inner169),
            VerkleNode::Inner170(_) => Some(VerkleNodeKind::Inner170),
            VerkleNode::Inner171(_) => Some(VerkleNodeKind::Inner171),
            VerkleNode::Inner172(_) => Some(VerkleNodeKind::Inner172),
            VerkleNode::Inner173(_) => Some(VerkleNodeKind::Inner173),
            VerkleNode::Inner174(_) => Some(VerkleNodeKind::Inner174),
            VerkleNode::Inner175(_) => Some(VerkleNodeKind::Inner175),
            VerkleNode::Inner176(_) => Some(VerkleNodeKind::Inner176),
            VerkleNode::Inner177(_) => Some(VerkleNodeKind::Inner177),
            VerkleNode::Inner178(_) => Some(VerkleNodeKind::Inner178),
            VerkleNode::Inner179(_) => Some(VerkleNodeKind::Inner179),
            VerkleNode::Inner180(_) => Some(VerkleNodeKind::Inner180),
            VerkleNode::Inner181(_) => Some(VerkleNodeKind::Inner181),
            VerkleNode::Inner182(_) => Some(VerkleNodeKind::Inner182),
            VerkleNode::Inner183(_) => Some(VerkleNodeKind::Inner183),
            VerkleNode::Inner184(_) => Some(VerkleNodeKind::Inner184),
            VerkleNode::Inner185(_) => Some(VerkleNodeKind::Inner185),
            VerkleNode::Inner186(_) => Some(VerkleNodeKind::Inner186),
            VerkleNode::Inner187(_) => Some(VerkleNodeKind::Inner187),
            VerkleNode::Inner188(_) => Some(VerkleNodeKind::Inner188),
            VerkleNode::Inner189(_) => Some(VerkleNodeKind::Inner189),
            VerkleNode::Inner190(_) => Some(VerkleNodeKind::Inner190),
            VerkleNode::Inner191(_) => Some(VerkleNodeKind::Inner191),
            VerkleNode::Inner192(_) => Some(VerkleNodeKind::Inner192),
            VerkleNode::Inner193(_) => Some(VerkleNodeKind::Inner193),
            VerkleNode::Inner194(_) => Some(VerkleNodeKind::Inner194),
            VerkleNode::Inner195(_) => Some(VerkleNodeKind::Inner195),
            VerkleNode::Inner196(_) => Some(VerkleNodeKind::Inner196),
            VerkleNode::Inner197(_) => Some(VerkleNodeKind::Inner197),
            VerkleNode::Inner198(_) => Some(VerkleNodeKind::Inner198),
            VerkleNode::Inner199(_) => Some(VerkleNodeKind::Inner199),
            VerkleNode::Inner200(_) => Some(VerkleNodeKind::Inner200),
            VerkleNode::Inner201(_) => Some(VerkleNodeKind::Inner201),
            VerkleNode::Inner202(_) => Some(VerkleNodeKind::Inner202),
            VerkleNode::Inner203(_) => Some(VerkleNodeKind::Inner203),
            VerkleNode::Inner204(_) => Some(VerkleNodeKind::Inner204),
            VerkleNode::Inner205(_) => Some(VerkleNodeKind::Inner205),
            VerkleNode::Inner206(_) => Some(VerkleNodeKind::Inner206),
            VerkleNode::Inner207(_) => Some(VerkleNodeKind::Inner207),
            VerkleNode::Inner208(_) => Some(VerkleNodeKind::Inner208),
            VerkleNode::Inner209(_) => Some(VerkleNodeKind::Inner209),
            VerkleNode::Inner210(_) => Some(VerkleNodeKind::Inner210),
            VerkleNode::Inner211(_) => Some(VerkleNodeKind::Inner211),
            VerkleNode::Inner212(_) => Some(VerkleNodeKind::Inner212),
            VerkleNode::Inner213(_) => Some(VerkleNodeKind::Inner213),
            VerkleNode::Inner214(_) => Some(VerkleNodeKind::Inner214),
            VerkleNode::Inner215(_) => Some(VerkleNodeKind::Inner215),
            VerkleNode::Inner216(_) => Some(VerkleNodeKind::Inner216),
            VerkleNode::Inner217(_) => Some(VerkleNodeKind::Inner217),
            VerkleNode::Inner218(_) => Some(VerkleNodeKind::Inner218),
            VerkleNode::Inner219(_) => Some(VerkleNodeKind::Inner219),
            VerkleNode::Inner220(_) => Some(VerkleNodeKind::Inner220),
            VerkleNode::Inner221(_) => Some(VerkleNodeKind::Inner221),
            VerkleNode::Inner222(_) => Some(VerkleNodeKind::Inner222),
            VerkleNode::Inner223(_) => Some(VerkleNodeKind::Inner223),
            VerkleNode::Inner224(_) => Some(VerkleNodeKind::Inner224),
            VerkleNode::Inner225(_) => Some(VerkleNodeKind::Inner225),
            VerkleNode::Inner226(_) => Some(VerkleNodeKind::Inner226),
            VerkleNode::Inner227(_) => Some(VerkleNodeKind::Inner227),
            VerkleNode::Inner228(_) => Some(VerkleNodeKind::Inner228),
            VerkleNode::Inner229(_) => Some(VerkleNodeKind::Inner229),
            VerkleNode::Inner230(_) => Some(VerkleNodeKind::Inner230),
            VerkleNode::Inner231(_) => Some(VerkleNodeKind::Inner231),
            VerkleNode::Inner232(_) => Some(VerkleNodeKind::Inner232),
            VerkleNode::Inner233(_) => Some(VerkleNodeKind::Inner233),
            VerkleNode::Inner234(_) => Some(VerkleNodeKind::Inner234),
            VerkleNode::Inner235(_) => Some(VerkleNodeKind::Inner235),
            VerkleNode::Inner236(_) => Some(VerkleNodeKind::Inner236),
            VerkleNode::Inner237(_) => Some(VerkleNodeKind::Inner237),
            VerkleNode::Inner238(_) => Some(VerkleNodeKind::Inner238),
            VerkleNode::Inner239(_) => Some(VerkleNodeKind::Inner239),
            VerkleNode::Inner240(_) => Some(VerkleNodeKind::Inner240),
            VerkleNode::Inner241(_) => Some(VerkleNodeKind::Inner241),
            VerkleNode::Inner242(_) => Some(VerkleNodeKind::Inner242),
            VerkleNode::Inner243(_) => Some(VerkleNodeKind::Inner243),
            VerkleNode::Inner244(_) => Some(VerkleNodeKind::Inner244),
            VerkleNode::Inner245(_) => Some(VerkleNodeKind::Inner245),
            VerkleNode::Inner246(_) => Some(VerkleNodeKind::Inner246),
            VerkleNode::Inner247(_) => Some(VerkleNodeKind::Inner247),
            VerkleNode::Inner248(_) => Some(VerkleNodeKind::Inner248),
            VerkleNode::Inner249(_) => Some(VerkleNodeKind::Inner249),
            VerkleNode::Inner250(_) => Some(VerkleNodeKind::Inner250),
            VerkleNode::Inner251(_) => Some(VerkleNodeKind::Inner251),
            VerkleNode::Inner252(_) => Some(VerkleNodeKind::Inner252),
            VerkleNode::Inner253(_) => Some(VerkleNodeKind::Inner253),
            VerkleNode::Inner254(_) => Some(VerkleNodeKind::Inner254),
            VerkleNode::Inner255(_) => Some(VerkleNodeKind::Inner255),
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
            VerkleNode::Leaf17(_) => Some(VerkleNodeKind::Leaf17),
            VerkleNode::Leaf18(_) => Some(VerkleNodeKind::Leaf18),
            VerkleNode::Leaf19(_) => Some(VerkleNodeKind::Leaf19),
            VerkleNode::Leaf20(_) => Some(VerkleNodeKind::Leaf20),
            VerkleNode::Leaf21(_) => Some(VerkleNodeKind::Leaf21),
            VerkleNode::Leaf22(_) => Some(VerkleNodeKind::Leaf22),
            VerkleNode::Leaf23(_) => Some(VerkleNodeKind::Leaf23),
            VerkleNode::Leaf24(_) => Some(VerkleNodeKind::Leaf24),
            VerkleNode::Leaf25(_) => Some(VerkleNodeKind::Leaf25),
            VerkleNode::Leaf26(_) => Some(VerkleNodeKind::Leaf26),
            VerkleNode::Leaf27(_) => Some(VerkleNodeKind::Leaf27),
            VerkleNode::Leaf28(_) => Some(VerkleNodeKind::Leaf28),
            VerkleNode::Leaf29(_) => Some(VerkleNodeKind::Leaf29),
            VerkleNode::Leaf30(_) => Some(VerkleNodeKind::Leaf30),
            VerkleNode::Leaf31(_) => Some(VerkleNodeKind::Leaf31),
            VerkleNode::Leaf32(_) => Some(VerkleNodeKind::Leaf32),
            VerkleNode::Leaf33(_) => Some(VerkleNodeKind::Leaf33),
            VerkleNode::Leaf34(_) => Some(VerkleNodeKind::Leaf34),
            VerkleNode::Leaf35(_) => Some(VerkleNodeKind::Leaf35),
            VerkleNode::Leaf36(_) => Some(VerkleNodeKind::Leaf36),
            VerkleNode::Leaf37(_) => Some(VerkleNodeKind::Leaf37),
            VerkleNode::Leaf38(_) => Some(VerkleNodeKind::Leaf38),
            VerkleNode::Leaf39(_) => Some(VerkleNodeKind::Leaf39),
            VerkleNode::Leaf40(_) => Some(VerkleNodeKind::Leaf40),
            VerkleNode::Leaf41(_) => Some(VerkleNodeKind::Leaf41),
            VerkleNode::Leaf42(_) => Some(VerkleNodeKind::Leaf42),
            VerkleNode::Leaf43(_) => Some(VerkleNodeKind::Leaf43),
            VerkleNode::Leaf44(_) => Some(VerkleNodeKind::Leaf44),
            VerkleNode::Leaf45(_) => Some(VerkleNodeKind::Leaf45),
            VerkleNode::Leaf46(_) => Some(VerkleNodeKind::Leaf46),
            VerkleNode::Leaf47(_) => Some(VerkleNodeKind::Leaf47),
            VerkleNode::Leaf48(_) => Some(VerkleNodeKind::Leaf48),
            VerkleNode::Leaf49(_) => Some(VerkleNodeKind::Leaf49),
            VerkleNode::Leaf50(_) => Some(VerkleNodeKind::Leaf50),
            VerkleNode::Leaf51(_) => Some(VerkleNodeKind::Leaf51),
            VerkleNode::Leaf52(_) => Some(VerkleNodeKind::Leaf52),
            VerkleNode::Leaf53(_) => Some(VerkleNodeKind::Leaf53),
            VerkleNode::Leaf54(_) => Some(VerkleNodeKind::Leaf54),
            VerkleNode::Leaf55(_) => Some(VerkleNodeKind::Leaf55),
            VerkleNode::Leaf56(_) => Some(VerkleNodeKind::Leaf56),
            VerkleNode::Leaf57(_) => Some(VerkleNodeKind::Leaf57),
            VerkleNode::Leaf58(_) => Some(VerkleNodeKind::Leaf58),
            VerkleNode::Leaf59(_) => Some(VerkleNodeKind::Leaf59),
            VerkleNode::Leaf60(_) => Some(VerkleNodeKind::Leaf60),
            VerkleNode::Leaf61(_) => Some(VerkleNodeKind::Leaf61),
            VerkleNode::Leaf62(_) => Some(VerkleNodeKind::Leaf62),
            VerkleNode::Leaf63(_) => Some(VerkleNodeKind::Leaf63),
            VerkleNode::Leaf64(_) => Some(VerkleNodeKind::Leaf64),
            VerkleNode::Leaf65(_) => Some(VerkleNodeKind::Leaf65),
            VerkleNode::Leaf66(_) => Some(VerkleNodeKind::Leaf66),
            VerkleNode::Leaf67(_) => Some(VerkleNodeKind::Leaf67),
            VerkleNode::Leaf68(_) => Some(VerkleNodeKind::Leaf68),
            VerkleNode::Leaf69(_) => Some(VerkleNodeKind::Leaf69),
            VerkleNode::Leaf70(_) => Some(VerkleNodeKind::Leaf70),
            VerkleNode::Leaf71(_) => Some(VerkleNodeKind::Leaf71),
            VerkleNode::Leaf72(_) => Some(VerkleNodeKind::Leaf72),
            VerkleNode::Leaf73(_) => Some(VerkleNodeKind::Leaf73),
            VerkleNode::Leaf74(_) => Some(VerkleNodeKind::Leaf74),
            VerkleNode::Leaf75(_) => Some(VerkleNodeKind::Leaf75),
            VerkleNode::Leaf76(_) => Some(VerkleNodeKind::Leaf76),
            VerkleNode::Leaf77(_) => Some(VerkleNodeKind::Leaf77),
            VerkleNode::Leaf78(_) => Some(VerkleNodeKind::Leaf78),
            VerkleNode::Leaf79(_) => Some(VerkleNodeKind::Leaf79),
            VerkleNode::Leaf80(_) => Some(VerkleNodeKind::Leaf80),
            VerkleNode::Leaf81(_) => Some(VerkleNodeKind::Leaf81),
            VerkleNode::Leaf82(_) => Some(VerkleNodeKind::Leaf82),
            VerkleNode::Leaf83(_) => Some(VerkleNodeKind::Leaf83),
            VerkleNode::Leaf84(_) => Some(VerkleNodeKind::Leaf84),
            VerkleNode::Leaf85(_) => Some(VerkleNodeKind::Leaf85),
            VerkleNode::Leaf86(_) => Some(VerkleNodeKind::Leaf86),
            VerkleNode::Leaf87(_) => Some(VerkleNodeKind::Leaf87),
            VerkleNode::Leaf88(_) => Some(VerkleNodeKind::Leaf88),
            VerkleNode::Leaf89(_) => Some(VerkleNodeKind::Leaf89),
            VerkleNode::Leaf90(_) => Some(VerkleNodeKind::Leaf90),
            VerkleNode::Leaf91(_) => Some(VerkleNodeKind::Leaf91),
            VerkleNode::Leaf92(_) => Some(VerkleNodeKind::Leaf92),
            VerkleNode::Leaf93(_) => Some(VerkleNodeKind::Leaf93),
            VerkleNode::Leaf94(_) => Some(VerkleNodeKind::Leaf94),
            VerkleNode::Leaf95(_) => Some(VerkleNodeKind::Leaf95),
            VerkleNode::Leaf96(_) => Some(VerkleNodeKind::Leaf96),
            VerkleNode::Leaf97(_) => Some(VerkleNodeKind::Leaf97),
            VerkleNode::Leaf98(_) => Some(VerkleNodeKind::Leaf98),
            VerkleNode::Leaf99(_) => Some(VerkleNodeKind::Leaf99),
            VerkleNode::Leaf100(_) => Some(VerkleNodeKind::Leaf100),
            VerkleNode::Leaf101(_) => Some(VerkleNodeKind::Leaf101),
            VerkleNode::Leaf102(_) => Some(VerkleNodeKind::Leaf102),
            VerkleNode::Leaf103(_) => Some(VerkleNodeKind::Leaf103),
            VerkleNode::Leaf104(_) => Some(VerkleNodeKind::Leaf104),
            VerkleNode::Leaf105(_) => Some(VerkleNodeKind::Leaf105),
            VerkleNode::Leaf106(_) => Some(VerkleNodeKind::Leaf106),
            VerkleNode::Leaf107(_) => Some(VerkleNodeKind::Leaf107),
            VerkleNode::Leaf108(_) => Some(VerkleNodeKind::Leaf108),
            VerkleNode::Leaf109(_) => Some(VerkleNodeKind::Leaf109),
            VerkleNode::Leaf110(_) => Some(VerkleNodeKind::Leaf110),
            VerkleNode::Leaf111(_) => Some(VerkleNodeKind::Leaf111),
            VerkleNode::Leaf112(_) => Some(VerkleNodeKind::Leaf112),
            VerkleNode::Leaf113(_) => Some(VerkleNodeKind::Leaf113),
            VerkleNode::Leaf114(_) => Some(VerkleNodeKind::Leaf114),
            VerkleNode::Leaf115(_) => Some(VerkleNodeKind::Leaf115),
            VerkleNode::Leaf116(_) => Some(VerkleNodeKind::Leaf116),
            VerkleNode::Leaf117(_) => Some(VerkleNodeKind::Leaf117),
            VerkleNode::Leaf118(_) => Some(VerkleNodeKind::Leaf118),
            VerkleNode::Leaf119(_) => Some(VerkleNodeKind::Leaf119),
            VerkleNode::Leaf120(_) => Some(VerkleNodeKind::Leaf120),
            VerkleNode::Leaf121(_) => Some(VerkleNodeKind::Leaf121),
            VerkleNode::Leaf122(_) => Some(VerkleNodeKind::Leaf122),
            VerkleNode::Leaf123(_) => Some(VerkleNodeKind::Leaf123),
            VerkleNode::Leaf124(_) => Some(VerkleNodeKind::Leaf124),
            VerkleNode::Leaf125(_) => Some(VerkleNodeKind::Leaf125),
            VerkleNode::Leaf126(_) => Some(VerkleNodeKind::Leaf126),
            VerkleNode::Leaf127(_) => Some(VerkleNodeKind::Leaf127),
            VerkleNode::Leaf128(_) => Some(VerkleNodeKind::Leaf128),
            VerkleNode::Leaf129(_) => Some(VerkleNodeKind::Leaf129),
            VerkleNode::Leaf130(_) => Some(VerkleNodeKind::Leaf130),
            VerkleNode::Leaf131(_) => Some(VerkleNodeKind::Leaf131),
            VerkleNode::Leaf132(_) => Some(VerkleNodeKind::Leaf132),
            VerkleNode::Leaf133(_) => Some(VerkleNodeKind::Leaf133),
            VerkleNode::Leaf134(_) => Some(VerkleNodeKind::Leaf134),
            VerkleNode::Leaf135(_) => Some(VerkleNodeKind::Leaf135),
            VerkleNode::Leaf136(_) => Some(VerkleNodeKind::Leaf136),
            VerkleNode::Leaf137(_) => Some(VerkleNodeKind::Leaf137),
            VerkleNode::Leaf138(_) => Some(VerkleNodeKind::Leaf138),
            VerkleNode::Leaf139(_) => Some(VerkleNodeKind::Leaf139),
            VerkleNode::Leaf140(_) => Some(VerkleNodeKind::Leaf140),
            VerkleNode::Leaf141(_) => Some(VerkleNodeKind::Leaf141),
            VerkleNode::Leaf142(_) => Some(VerkleNodeKind::Leaf142),
            VerkleNode::Leaf143(_) => Some(VerkleNodeKind::Leaf143),
            VerkleNode::Leaf144(_) => Some(VerkleNodeKind::Leaf144),
            VerkleNode::Leaf145(_) => Some(VerkleNodeKind::Leaf145),
            VerkleNode::Leaf146(_) => Some(VerkleNodeKind::Leaf146),
            VerkleNode::Leaf147(_) => Some(VerkleNodeKind::Leaf147),
            VerkleNode::Leaf148(_) => Some(VerkleNodeKind::Leaf148),
            VerkleNode::Leaf149(_) => Some(VerkleNodeKind::Leaf149),
            VerkleNode::Leaf150(_) => Some(VerkleNodeKind::Leaf150),
            VerkleNode::Leaf151(_) => Some(VerkleNodeKind::Leaf151),
            VerkleNode::Leaf152(_) => Some(VerkleNodeKind::Leaf152),
            VerkleNode::Leaf153(_) => Some(VerkleNodeKind::Leaf153),
            VerkleNode::Leaf154(_) => Some(VerkleNodeKind::Leaf154),
            VerkleNode::Leaf155(_) => Some(VerkleNodeKind::Leaf155),
            VerkleNode::Leaf156(_) => Some(VerkleNodeKind::Leaf156),
            VerkleNode::Leaf157(_) => Some(VerkleNodeKind::Leaf157),
            VerkleNode::Leaf158(_) => Some(VerkleNodeKind::Leaf158),
            VerkleNode::Leaf159(_) => Some(VerkleNodeKind::Leaf159),
            VerkleNode::Leaf160(_) => Some(VerkleNodeKind::Leaf160),
            VerkleNode::Leaf161(_) => Some(VerkleNodeKind::Leaf161),
            VerkleNode::Leaf162(_) => Some(VerkleNodeKind::Leaf162),
            VerkleNode::Leaf163(_) => Some(VerkleNodeKind::Leaf163),
            VerkleNode::Leaf164(_) => Some(VerkleNodeKind::Leaf164),
            VerkleNode::Leaf165(_) => Some(VerkleNodeKind::Leaf165),
            VerkleNode::Leaf166(_) => Some(VerkleNodeKind::Leaf166),
            VerkleNode::Leaf167(_) => Some(VerkleNodeKind::Leaf167),
            VerkleNode::Leaf168(_) => Some(VerkleNodeKind::Leaf168),
            VerkleNode::Leaf169(_) => Some(VerkleNodeKind::Leaf169),
            VerkleNode::Leaf170(_) => Some(VerkleNodeKind::Leaf170),
            VerkleNode::Leaf171(_) => Some(VerkleNodeKind::Leaf171),
            VerkleNode::Leaf172(_) => Some(VerkleNodeKind::Leaf172),
            VerkleNode::Leaf173(_) => Some(VerkleNodeKind::Leaf173),
            VerkleNode::Leaf174(_) => Some(VerkleNodeKind::Leaf174),
            VerkleNode::Leaf175(_) => Some(VerkleNodeKind::Leaf175),
            VerkleNode::Leaf176(_) => Some(VerkleNodeKind::Leaf176),
            VerkleNode::Leaf177(_) => Some(VerkleNodeKind::Leaf177),
            VerkleNode::Leaf178(_) => Some(VerkleNodeKind::Leaf178),
            VerkleNode::Leaf179(_) => Some(VerkleNodeKind::Leaf179),
            VerkleNode::Leaf180(_) => Some(VerkleNodeKind::Leaf180),
            VerkleNode::Leaf181(_) => Some(VerkleNodeKind::Leaf181),
            VerkleNode::Leaf182(_) => Some(VerkleNodeKind::Leaf182),
            VerkleNode::Leaf183(_) => Some(VerkleNodeKind::Leaf183),
            VerkleNode::Leaf184(_) => Some(VerkleNodeKind::Leaf184),
            VerkleNode::Leaf185(_) => Some(VerkleNodeKind::Leaf185),
            VerkleNode::Leaf186(_) => Some(VerkleNodeKind::Leaf186),
            VerkleNode::Leaf187(_) => Some(VerkleNodeKind::Leaf187),
            VerkleNode::Leaf188(_) => Some(VerkleNodeKind::Leaf188),
            VerkleNode::Leaf189(_) => Some(VerkleNodeKind::Leaf189),
            VerkleNode::Leaf190(_) => Some(VerkleNodeKind::Leaf190),
            VerkleNode::Leaf191(_) => Some(VerkleNodeKind::Leaf191),
            VerkleNode::Leaf192(_) => Some(VerkleNodeKind::Leaf192),
            VerkleNode::Leaf193(_) => Some(VerkleNodeKind::Leaf193),
            VerkleNode::Leaf194(_) => Some(VerkleNodeKind::Leaf194),
            VerkleNode::Leaf195(_) => Some(VerkleNodeKind::Leaf195),
            VerkleNode::Leaf196(_) => Some(VerkleNodeKind::Leaf196),
            VerkleNode::Leaf197(_) => Some(VerkleNodeKind::Leaf197),
            VerkleNode::Leaf198(_) => Some(VerkleNodeKind::Leaf198),
            VerkleNode::Leaf199(_) => Some(VerkleNodeKind::Leaf199),
            VerkleNode::Leaf200(_) => Some(VerkleNodeKind::Leaf200),
            VerkleNode::Leaf201(_) => Some(VerkleNodeKind::Leaf201),
            VerkleNode::Leaf202(_) => Some(VerkleNodeKind::Leaf202),
            VerkleNode::Leaf203(_) => Some(VerkleNodeKind::Leaf203),
            VerkleNode::Leaf204(_) => Some(VerkleNodeKind::Leaf204),
            VerkleNode::Leaf205(_) => Some(VerkleNodeKind::Leaf205),
            VerkleNode::Leaf206(_) => Some(VerkleNodeKind::Leaf206),
            VerkleNode::Leaf207(_) => Some(VerkleNodeKind::Leaf207),
            VerkleNode::Leaf208(_) => Some(VerkleNodeKind::Leaf208),
            VerkleNode::Leaf209(_) => Some(VerkleNodeKind::Leaf209),
            VerkleNode::Leaf210(_) => Some(VerkleNodeKind::Leaf210),
            VerkleNode::Leaf211(_) => Some(VerkleNodeKind::Leaf211),
            VerkleNode::Leaf212(_) => Some(VerkleNodeKind::Leaf212),
            VerkleNode::Leaf213(_) => Some(VerkleNodeKind::Leaf213),
            VerkleNode::Leaf214(_) => Some(VerkleNodeKind::Leaf214),
            VerkleNode::Leaf215(_) => Some(VerkleNodeKind::Leaf215),
            VerkleNode::Leaf216(_) => Some(VerkleNodeKind::Leaf216),
            VerkleNode::Leaf217(_) => Some(VerkleNodeKind::Leaf217),
            VerkleNode::Leaf218(_) => Some(VerkleNodeKind::Leaf218),
            VerkleNode::Leaf219(_) => Some(VerkleNodeKind::Leaf219),
            VerkleNode::Leaf220(_) => Some(VerkleNodeKind::Leaf220),
            VerkleNode::Leaf221(_) => Some(VerkleNodeKind::Leaf221),
            VerkleNode::Leaf222(_) => Some(VerkleNodeKind::Leaf222),
            VerkleNode::Leaf223(_) => Some(VerkleNodeKind::Leaf223),
            VerkleNode::Leaf224(_) => Some(VerkleNodeKind::Leaf224),
            VerkleNode::Leaf225(_) => Some(VerkleNodeKind::Leaf225),
            VerkleNode::Leaf226(_) => Some(VerkleNodeKind::Leaf226),
            VerkleNode::Leaf227(_) => Some(VerkleNodeKind::Leaf227),
            VerkleNode::Leaf228(_) => Some(VerkleNodeKind::Leaf228),
            VerkleNode::Leaf229(_) => Some(VerkleNodeKind::Leaf229),
            VerkleNode::Leaf230(_) => Some(VerkleNodeKind::Leaf230),
            VerkleNode::Leaf231(_) => Some(VerkleNodeKind::Leaf231),
            VerkleNode::Leaf232(_) => Some(VerkleNodeKind::Leaf232),
            VerkleNode::Leaf233(_) => Some(VerkleNodeKind::Leaf233),
            VerkleNode::Leaf234(_) => Some(VerkleNodeKind::Leaf234),
            VerkleNode::Leaf235(_) => Some(VerkleNodeKind::Leaf235),
            VerkleNode::Leaf236(_) => Some(VerkleNodeKind::Leaf236),
            VerkleNode::Leaf237(_) => Some(VerkleNodeKind::Leaf237),
            VerkleNode::Leaf238(_) => Some(VerkleNodeKind::Leaf238),
            VerkleNode::Leaf239(_) => Some(VerkleNodeKind::Leaf239),
            VerkleNode::Leaf240(_) => Some(VerkleNodeKind::Leaf240),
            VerkleNode::Leaf241(_) => Some(VerkleNodeKind::Leaf241),
            VerkleNode::Leaf242(_) => Some(VerkleNodeKind::Leaf242),
            VerkleNode::Leaf243(_) => Some(VerkleNodeKind::Leaf243),
            VerkleNode::Leaf244(_) => Some(VerkleNodeKind::Leaf244),
            VerkleNode::Leaf245(_) => Some(VerkleNodeKind::Leaf245),
            VerkleNode::Leaf246(_) => Some(VerkleNodeKind::Leaf246),
            VerkleNode::Leaf247(_) => Some(VerkleNodeKind::Leaf247),
            VerkleNode::Leaf248(_) => Some(VerkleNodeKind::Leaf248),
            VerkleNode::Leaf249(_) => Some(VerkleNodeKind::Leaf249),
            VerkleNode::Leaf250(_) => Some(VerkleNodeKind::Leaf250),
            VerkleNode::Leaf251(_) => Some(VerkleNodeKind::Leaf251),
            VerkleNode::Leaf252(_) => Some(VerkleNodeKind::Leaf252),
            VerkleNode::Leaf253(_) => Some(VerkleNodeKind::Leaf253),
            VerkleNode::Leaf254(_) => Some(VerkleNodeKind::Leaf254),
            VerkleNode::Leaf255(_) => Some(VerkleNodeKind::Leaf255),
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
        // Note: This method is only called in archive mode, so using the delta node is fine.
        match self {
            VerkleNode::Inner256(n) => {
                if changed_indices.len() <= InnerDeltaNode::DELTA_SIZE {
                    Ok(VerkleNode::InnerDelta(Box::new(
                        InnerDeltaNode::from_full_inner(n, id),
                    )))
                } else {
                    Ok(VerkleNode::Inner256(n.clone()))
                }
            }
            VerkleNode::InnerDelta(n) => {
                let enough_slots = ItemWithIndex::required_slot_count_for(
                    &n.children_delta,
                    changed_indices.into_iter(),
                ) <= InnerDeltaNode::DELTA_SIZE;
                if enough_slots {
                    Ok(VerkleNode::InnerDelta(n.clone()))
                } else {
                    Ok(VerkleNode::Inner256(Box::new(FullInnerNode::from(
                        (**n).clone(),
                    ))))
                }
            }
            VerkleNode::Leaf256(n) => {
                if changed_indices.len() <= LeafDeltaNode::DELTA_SIZE {
                    Ok(VerkleNode::LeafDelta(Box::new(
                        LeafDeltaNode::from_full_leaf(n, id),
                    )))
                } else {
                    Ok(VerkleNode::Leaf256(n.clone()))
                }
            }
            VerkleNode::Leaf146(n) => {
                if changed_indices.len() <= LeafDeltaNode::DELTA_SIZE {
                    Ok(VerkleNode::LeafDelta(Box::new(
                        LeafDeltaNode::from_sparse_leaf(n, id),
                    )))
                } else {
                    Ok(VerkleNode::Leaf146(n.clone()))
                }
            }
            VerkleNode::LeafDelta(n) => {
                const DELTA_PLUS_ONE: usize = LeafDeltaNode::DELTA_SIZE + 1;
                match ItemWithIndex::required_slot_count_for(
                    &n.values_delta,
                    changed_indices.into_iter(),
                ) {
                    ..=LeafDeltaNode::DELTA_SIZE => Ok(VerkleNode::LeafDelta(n.clone())),
                    DELTA_PLUS_ONE..=146 => Ok(VerkleNode::Leaf146(Box::new(
                        SparseLeafNode::try_from((**n).clone())?,
                    ))),
                    _ => Ok(VerkleNode::Leaf256(Box::new(FullLeafNode::from(
                        (**n).clone(),
                    )))),
                }
            }
            _ => Ok(self.clone()),
        }
    }
}

impl ManagedTrieNode for VerkleNode {
    type Union = VerkleNode;
    type Id = VerkleNodeId;
    type Commitment = VerkleCommitment;

    fn lookup(&self, key: &Key, depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {
        match self {
            VerkleNode::Empty(n) => n.lookup(key, depth),
            VerkleNode::Inner1(n) => n.lookup(key, depth),
            VerkleNode::Inner2(n) => n.lookup(key, depth),
            VerkleNode::Inner3(n) => n.lookup(key, depth),
            VerkleNode::Inner4(n) => n.lookup(key, depth),
            VerkleNode::Inner5(n) => n.lookup(key, depth),
            VerkleNode::Inner6(n) => n.lookup(key, depth),
            VerkleNode::Inner7(n) => n.lookup(key, depth),
            VerkleNode::Inner8(n) => n.lookup(key, depth),
            VerkleNode::Inner9(n) => n.lookup(key, depth),
            VerkleNode::Inner10(n) => n.lookup(key, depth),
            VerkleNode::Inner11(n) => n.lookup(key, depth),
            VerkleNode::Inner12(n) => n.lookup(key, depth),
            VerkleNode::Inner13(n) => n.lookup(key, depth),
            VerkleNode::Inner14(n) => n.lookup(key, depth),
            VerkleNode::Inner15(n) => n.lookup(key, depth),
            VerkleNode::Inner16(n) => n.lookup(key, depth),
            VerkleNode::Inner17(n) => n.lookup(key, depth),
            VerkleNode::Inner18(n) => n.lookup(key, depth),
            VerkleNode::Inner19(n) => n.lookup(key, depth),
            VerkleNode::Inner20(n) => n.lookup(key, depth),
            VerkleNode::Inner21(n) => n.lookup(key, depth),
            VerkleNode::Inner22(n) => n.lookup(key, depth),
            VerkleNode::Inner23(n) => n.lookup(key, depth),
            VerkleNode::Inner24(n) => n.lookup(key, depth),
            VerkleNode::Inner25(n) => n.lookup(key, depth),
            VerkleNode::Inner26(n) => n.lookup(key, depth),
            VerkleNode::Inner27(n) => n.lookup(key, depth),
            VerkleNode::Inner28(n) => n.lookup(key, depth),
            VerkleNode::Inner29(n) => n.lookup(key, depth),
            VerkleNode::Inner30(n) => n.lookup(key, depth),
            VerkleNode::Inner31(n) => n.lookup(key, depth),
            VerkleNode::Inner32(n) => n.lookup(key, depth),
            VerkleNode::Inner33(n) => n.lookup(key, depth),
            VerkleNode::Inner34(n) => n.lookup(key, depth),
            VerkleNode::Inner35(n) => n.lookup(key, depth),
            VerkleNode::Inner36(n) => n.lookup(key, depth),
            VerkleNode::Inner37(n) => n.lookup(key, depth),
            VerkleNode::Inner38(n) => n.lookup(key, depth),
            VerkleNode::Inner39(n) => n.lookup(key, depth),
            VerkleNode::Inner40(n) => n.lookup(key, depth),
            VerkleNode::Inner41(n) => n.lookup(key, depth),
            VerkleNode::Inner42(n) => n.lookup(key, depth),
            VerkleNode::Inner43(n) => n.lookup(key, depth),
            VerkleNode::Inner44(n) => n.lookup(key, depth),
            VerkleNode::Inner45(n) => n.lookup(key, depth),
            VerkleNode::Inner46(n) => n.lookup(key, depth),
            VerkleNode::Inner47(n) => n.lookup(key, depth),
            VerkleNode::Inner48(n) => n.lookup(key, depth),
            VerkleNode::Inner49(n) => n.lookup(key, depth),
            VerkleNode::Inner50(n) => n.lookup(key, depth),
            VerkleNode::Inner51(n) => n.lookup(key, depth),
            VerkleNode::Inner52(n) => n.lookup(key, depth),
            VerkleNode::Inner53(n) => n.lookup(key, depth),
            VerkleNode::Inner54(n) => n.lookup(key, depth),
            VerkleNode::Inner55(n) => n.lookup(key, depth),
            VerkleNode::Inner56(n) => n.lookup(key, depth),
            VerkleNode::Inner57(n) => n.lookup(key, depth),
            VerkleNode::Inner58(n) => n.lookup(key, depth),
            VerkleNode::Inner59(n) => n.lookup(key, depth),
            VerkleNode::Inner60(n) => n.lookup(key, depth),
            VerkleNode::Inner61(n) => n.lookup(key, depth),
            VerkleNode::Inner62(n) => n.lookup(key, depth),
            VerkleNode::Inner63(n) => n.lookup(key, depth),
            VerkleNode::Inner64(n) => n.lookup(key, depth),
            VerkleNode::Inner65(n) => n.lookup(key, depth),
            VerkleNode::Inner66(n) => n.lookup(key, depth),
            VerkleNode::Inner67(n) => n.lookup(key, depth),
            VerkleNode::Inner68(n) => n.lookup(key, depth),
            VerkleNode::Inner69(n) => n.lookup(key, depth),
            VerkleNode::Inner70(n) => n.lookup(key, depth),
            VerkleNode::Inner71(n) => n.lookup(key, depth),
            VerkleNode::Inner72(n) => n.lookup(key, depth),
            VerkleNode::Inner73(n) => n.lookup(key, depth),
            VerkleNode::Inner74(n) => n.lookup(key, depth),
            VerkleNode::Inner75(n) => n.lookup(key, depth),
            VerkleNode::Inner76(n) => n.lookup(key, depth),
            VerkleNode::Inner77(n) => n.lookup(key, depth),
            VerkleNode::Inner78(n) => n.lookup(key, depth),
            VerkleNode::Inner79(n) => n.lookup(key, depth),
            VerkleNode::Inner80(n) => n.lookup(key, depth),
            VerkleNode::Inner81(n) => n.lookup(key, depth),
            VerkleNode::Inner82(n) => n.lookup(key, depth),
            VerkleNode::Inner83(n) => n.lookup(key, depth),
            VerkleNode::Inner84(n) => n.lookup(key, depth),
            VerkleNode::Inner85(n) => n.lookup(key, depth),
            VerkleNode::Inner86(n) => n.lookup(key, depth),
            VerkleNode::Inner87(n) => n.lookup(key, depth),
            VerkleNode::Inner88(n) => n.lookup(key, depth),
            VerkleNode::Inner89(n) => n.lookup(key, depth),
            VerkleNode::Inner90(n) => n.lookup(key, depth),
            VerkleNode::Inner91(n) => n.lookup(key, depth),
            VerkleNode::Inner92(n) => n.lookup(key, depth),
            VerkleNode::Inner93(n) => n.lookup(key, depth),
            VerkleNode::Inner94(n) => n.lookup(key, depth),
            VerkleNode::Inner95(n) => n.lookup(key, depth),
            VerkleNode::Inner96(n) => n.lookup(key, depth),
            VerkleNode::Inner97(n) => n.lookup(key, depth),
            VerkleNode::Inner98(n) => n.lookup(key, depth),
            VerkleNode::Inner99(n) => n.lookup(key, depth),
            VerkleNode::Inner100(n) => n.lookup(key, depth),
            VerkleNode::Inner101(n) => n.lookup(key, depth),
            VerkleNode::Inner102(n) => n.lookup(key, depth),
            VerkleNode::Inner103(n) => n.lookup(key, depth),
            VerkleNode::Inner104(n) => n.lookup(key, depth),
            VerkleNode::Inner105(n) => n.lookup(key, depth),
            VerkleNode::Inner106(n) => n.lookup(key, depth),
            VerkleNode::Inner107(n) => n.lookup(key, depth),
            VerkleNode::Inner108(n) => n.lookup(key, depth),
            VerkleNode::Inner109(n) => n.lookup(key, depth),
            VerkleNode::Inner110(n) => n.lookup(key, depth),
            VerkleNode::Inner111(n) => n.lookup(key, depth),
            VerkleNode::Inner112(n) => n.lookup(key, depth),
            VerkleNode::Inner113(n) => n.lookup(key, depth),
            VerkleNode::Inner114(n) => n.lookup(key, depth),
            VerkleNode::Inner115(n) => n.lookup(key, depth),
            VerkleNode::Inner116(n) => n.lookup(key, depth),
            VerkleNode::Inner117(n) => n.lookup(key, depth),
            VerkleNode::Inner118(n) => n.lookup(key, depth),
            VerkleNode::Inner119(n) => n.lookup(key, depth),
            VerkleNode::Inner120(n) => n.lookup(key, depth),
            VerkleNode::Inner121(n) => n.lookup(key, depth),
            VerkleNode::Inner122(n) => n.lookup(key, depth),
            VerkleNode::Inner123(n) => n.lookup(key, depth),
            VerkleNode::Inner124(n) => n.lookup(key, depth),
            VerkleNode::Inner125(n) => n.lookup(key, depth),
            VerkleNode::Inner126(n) => n.lookup(key, depth),
            VerkleNode::Inner127(n) => n.lookup(key, depth),
            VerkleNode::Inner128(n) => n.lookup(key, depth),
            VerkleNode::Inner129(n) => n.lookup(key, depth),
            VerkleNode::Inner130(n) => n.lookup(key, depth),
            VerkleNode::Inner131(n) => n.lookup(key, depth),
            VerkleNode::Inner132(n) => n.lookup(key, depth),
            VerkleNode::Inner133(n) => n.lookup(key, depth),
            VerkleNode::Inner134(n) => n.lookup(key, depth),
            VerkleNode::Inner135(n) => n.lookup(key, depth),
            VerkleNode::Inner136(n) => n.lookup(key, depth),
            VerkleNode::Inner137(n) => n.lookup(key, depth),
            VerkleNode::Inner138(n) => n.lookup(key, depth),
            VerkleNode::Inner139(n) => n.lookup(key, depth),
            VerkleNode::Inner140(n) => n.lookup(key, depth),
            VerkleNode::Inner141(n) => n.lookup(key, depth),
            VerkleNode::Inner142(n) => n.lookup(key, depth),
            VerkleNode::Inner143(n) => n.lookup(key, depth),
            VerkleNode::Inner144(n) => n.lookup(key, depth),
            VerkleNode::Inner145(n) => n.lookup(key, depth),
            VerkleNode::Inner146(n) => n.lookup(key, depth),
            VerkleNode::Inner147(n) => n.lookup(key, depth),
            VerkleNode::Inner148(n) => n.lookup(key, depth),
            VerkleNode::Inner149(n) => n.lookup(key, depth),
            VerkleNode::Inner150(n) => n.lookup(key, depth),
            VerkleNode::Inner151(n) => n.lookup(key, depth),
            VerkleNode::Inner152(n) => n.lookup(key, depth),
            VerkleNode::Inner153(n) => n.lookup(key, depth),
            VerkleNode::Inner154(n) => n.lookup(key, depth),
            VerkleNode::Inner155(n) => n.lookup(key, depth),
            VerkleNode::Inner156(n) => n.lookup(key, depth),
            VerkleNode::Inner157(n) => n.lookup(key, depth),
            VerkleNode::Inner158(n) => n.lookup(key, depth),
            VerkleNode::Inner159(n) => n.lookup(key, depth),
            VerkleNode::Inner160(n) => n.lookup(key, depth),
            VerkleNode::Inner161(n) => n.lookup(key, depth),
            VerkleNode::Inner162(n) => n.lookup(key, depth),
            VerkleNode::Inner163(n) => n.lookup(key, depth),
            VerkleNode::Inner164(n) => n.lookup(key, depth),
            VerkleNode::Inner165(n) => n.lookup(key, depth),
            VerkleNode::Inner166(n) => n.lookup(key, depth),
            VerkleNode::Inner167(n) => n.lookup(key, depth),
            VerkleNode::Inner168(n) => n.lookup(key, depth),
            VerkleNode::Inner169(n) => n.lookup(key, depth),
            VerkleNode::Inner170(n) => n.lookup(key, depth),
            VerkleNode::Inner171(n) => n.lookup(key, depth),
            VerkleNode::Inner172(n) => n.lookup(key, depth),
            VerkleNode::Inner173(n) => n.lookup(key, depth),
            VerkleNode::Inner174(n) => n.lookup(key, depth),
            VerkleNode::Inner175(n) => n.lookup(key, depth),
            VerkleNode::Inner176(n) => n.lookup(key, depth),
            VerkleNode::Inner177(n) => n.lookup(key, depth),
            VerkleNode::Inner178(n) => n.lookup(key, depth),
            VerkleNode::Inner179(n) => n.lookup(key, depth),
            VerkleNode::Inner180(n) => n.lookup(key, depth),
            VerkleNode::Inner181(n) => n.lookup(key, depth),
            VerkleNode::Inner182(n) => n.lookup(key, depth),
            VerkleNode::Inner183(n) => n.lookup(key, depth),
            VerkleNode::Inner184(n) => n.lookup(key, depth),
            VerkleNode::Inner185(n) => n.lookup(key, depth),
            VerkleNode::Inner186(n) => n.lookup(key, depth),
            VerkleNode::Inner187(n) => n.lookup(key, depth),
            VerkleNode::Inner188(n) => n.lookup(key, depth),
            VerkleNode::Inner189(n) => n.lookup(key, depth),
            VerkleNode::Inner190(n) => n.lookup(key, depth),
            VerkleNode::Inner191(n) => n.lookup(key, depth),
            VerkleNode::Inner192(n) => n.lookup(key, depth),
            VerkleNode::Inner193(n) => n.lookup(key, depth),
            VerkleNode::Inner194(n) => n.lookup(key, depth),
            VerkleNode::Inner195(n) => n.lookup(key, depth),
            VerkleNode::Inner196(n) => n.lookup(key, depth),
            VerkleNode::Inner197(n) => n.lookup(key, depth),
            VerkleNode::Inner198(n) => n.lookup(key, depth),
            VerkleNode::Inner199(n) => n.lookup(key, depth),
            VerkleNode::Inner200(n) => n.lookup(key, depth),
            VerkleNode::Inner201(n) => n.lookup(key, depth),
            VerkleNode::Inner202(n) => n.lookup(key, depth),
            VerkleNode::Inner203(n) => n.lookup(key, depth),
            VerkleNode::Inner204(n) => n.lookup(key, depth),
            VerkleNode::Inner205(n) => n.lookup(key, depth),
            VerkleNode::Inner206(n) => n.lookup(key, depth),
            VerkleNode::Inner207(n) => n.lookup(key, depth),
            VerkleNode::Inner208(n) => n.lookup(key, depth),
            VerkleNode::Inner209(n) => n.lookup(key, depth),
            VerkleNode::Inner210(n) => n.lookup(key, depth),
            VerkleNode::Inner211(n) => n.lookup(key, depth),
            VerkleNode::Inner212(n) => n.lookup(key, depth),
            VerkleNode::Inner213(n) => n.lookup(key, depth),
            VerkleNode::Inner214(n) => n.lookup(key, depth),
            VerkleNode::Inner215(n) => n.lookup(key, depth),
            VerkleNode::Inner216(n) => n.lookup(key, depth),
            VerkleNode::Inner217(n) => n.lookup(key, depth),
            VerkleNode::Inner218(n) => n.lookup(key, depth),
            VerkleNode::Inner219(n) => n.lookup(key, depth),
            VerkleNode::Inner220(n) => n.lookup(key, depth),
            VerkleNode::Inner221(n) => n.lookup(key, depth),
            VerkleNode::Inner222(n) => n.lookup(key, depth),
            VerkleNode::Inner223(n) => n.lookup(key, depth),
            VerkleNode::Inner224(n) => n.lookup(key, depth),
            VerkleNode::Inner225(n) => n.lookup(key, depth),
            VerkleNode::Inner226(n) => n.lookup(key, depth),
            VerkleNode::Inner227(n) => n.lookup(key, depth),
            VerkleNode::Inner228(n) => n.lookup(key, depth),
            VerkleNode::Inner229(n) => n.lookup(key, depth),
            VerkleNode::Inner230(n) => n.lookup(key, depth),
            VerkleNode::Inner231(n) => n.lookup(key, depth),
            VerkleNode::Inner232(n) => n.lookup(key, depth),
            VerkleNode::Inner233(n) => n.lookup(key, depth),
            VerkleNode::Inner234(n) => n.lookup(key, depth),
            VerkleNode::Inner235(n) => n.lookup(key, depth),
            VerkleNode::Inner236(n) => n.lookup(key, depth),
            VerkleNode::Inner237(n) => n.lookup(key, depth),
            VerkleNode::Inner238(n) => n.lookup(key, depth),
            VerkleNode::Inner239(n) => n.lookup(key, depth),
            VerkleNode::Inner240(n) => n.lookup(key, depth),
            VerkleNode::Inner241(n) => n.lookup(key, depth),
            VerkleNode::Inner242(n) => n.lookup(key, depth),
            VerkleNode::Inner243(n) => n.lookup(key, depth),
            VerkleNode::Inner244(n) => n.lookup(key, depth),
            VerkleNode::Inner245(n) => n.lookup(key, depth),
            VerkleNode::Inner246(n) => n.lookup(key, depth),
            VerkleNode::Inner247(n) => n.lookup(key, depth),
            VerkleNode::Inner248(n) => n.lookup(key, depth),
            VerkleNode::Inner249(n) => n.lookup(key, depth),
            VerkleNode::Inner250(n) => n.lookup(key, depth),
            VerkleNode::Inner251(n) => n.lookup(key, depth),
            VerkleNode::Inner252(n) => n.lookup(key, depth),
            VerkleNode::Inner253(n) => n.lookup(key, depth),
            VerkleNode::Inner254(n) => n.lookup(key, depth),
            VerkleNode::Inner255(n) => n.lookup(key, depth),
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
            VerkleNode::Leaf17(n) => n.lookup(key, depth),
            VerkleNode::Leaf18(n) => n.lookup(key, depth),
            VerkleNode::Leaf19(n) => n.lookup(key, depth),
            VerkleNode::Leaf20(n) => n.lookup(key, depth),
            VerkleNode::Leaf21(n) => n.lookup(key, depth),
            VerkleNode::Leaf22(n) => n.lookup(key, depth),
            VerkleNode::Leaf23(n) => n.lookup(key, depth),
            VerkleNode::Leaf24(n) => n.lookup(key, depth),
            VerkleNode::Leaf25(n) => n.lookup(key, depth),
            VerkleNode::Leaf26(n) => n.lookup(key, depth),
            VerkleNode::Leaf27(n) => n.lookup(key, depth),
            VerkleNode::Leaf28(n) => n.lookup(key, depth),
            VerkleNode::Leaf29(n) => n.lookup(key, depth),
            VerkleNode::Leaf30(n) => n.lookup(key, depth),
            VerkleNode::Leaf31(n) => n.lookup(key, depth),
            VerkleNode::Leaf32(n) => n.lookup(key, depth),
            VerkleNode::Leaf33(n) => n.lookup(key, depth),
            VerkleNode::Leaf34(n) => n.lookup(key, depth),
            VerkleNode::Leaf35(n) => n.lookup(key, depth),
            VerkleNode::Leaf36(n) => n.lookup(key, depth),
            VerkleNode::Leaf37(n) => n.lookup(key, depth),
            VerkleNode::Leaf38(n) => n.lookup(key, depth),
            VerkleNode::Leaf39(n) => n.lookup(key, depth),
            VerkleNode::Leaf40(n) => n.lookup(key, depth),
            VerkleNode::Leaf41(n) => n.lookup(key, depth),
            VerkleNode::Leaf42(n) => n.lookup(key, depth),
            VerkleNode::Leaf43(n) => n.lookup(key, depth),
            VerkleNode::Leaf44(n) => n.lookup(key, depth),
            VerkleNode::Leaf45(n) => n.lookup(key, depth),
            VerkleNode::Leaf46(n) => n.lookup(key, depth),
            VerkleNode::Leaf47(n) => n.lookup(key, depth),
            VerkleNode::Leaf48(n) => n.lookup(key, depth),
            VerkleNode::Leaf49(n) => n.lookup(key, depth),
            VerkleNode::Leaf50(n) => n.lookup(key, depth),
            VerkleNode::Leaf51(n) => n.lookup(key, depth),
            VerkleNode::Leaf52(n) => n.lookup(key, depth),
            VerkleNode::Leaf53(n) => n.lookup(key, depth),
            VerkleNode::Leaf54(n) => n.lookup(key, depth),
            VerkleNode::Leaf55(n) => n.lookup(key, depth),
            VerkleNode::Leaf56(n) => n.lookup(key, depth),
            VerkleNode::Leaf57(n) => n.lookup(key, depth),
            VerkleNode::Leaf58(n) => n.lookup(key, depth),
            VerkleNode::Leaf59(n) => n.lookup(key, depth),
            VerkleNode::Leaf60(n) => n.lookup(key, depth),
            VerkleNode::Leaf61(n) => n.lookup(key, depth),
            VerkleNode::Leaf62(n) => n.lookup(key, depth),
            VerkleNode::Leaf63(n) => n.lookup(key, depth),
            VerkleNode::Leaf64(n) => n.lookup(key, depth),
            VerkleNode::Leaf65(n) => n.lookup(key, depth),
            VerkleNode::Leaf66(n) => n.lookup(key, depth),
            VerkleNode::Leaf67(n) => n.lookup(key, depth),
            VerkleNode::Leaf68(n) => n.lookup(key, depth),
            VerkleNode::Leaf69(n) => n.lookup(key, depth),
            VerkleNode::Leaf70(n) => n.lookup(key, depth),
            VerkleNode::Leaf71(n) => n.lookup(key, depth),
            VerkleNode::Leaf72(n) => n.lookup(key, depth),
            VerkleNode::Leaf73(n) => n.lookup(key, depth),
            VerkleNode::Leaf74(n) => n.lookup(key, depth),
            VerkleNode::Leaf75(n) => n.lookup(key, depth),
            VerkleNode::Leaf76(n) => n.lookup(key, depth),
            VerkleNode::Leaf77(n) => n.lookup(key, depth),
            VerkleNode::Leaf78(n) => n.lookup(key, depth),
            VerkleNode::Leaf79(n) => n.lookup(key, depth),
            VerkleNode::Leaf80(n) => n.lookup(key, depth),
            VerkleNode::Leaf81(n) => n.lookup(key, depth),
            VerkleNode::Leaf82(n) => n.lookup(key, depth),
            VerkleNode::Leaf83(n) => n.lookup(key, depth),
            VerkleNode::Leaf84(n) => n.lookup(key, depth),
            VerkleNode::Leaf85(n) => n.lookup(key, depth),
            VerkleNode::Leaf86(n) => n.lookup(key, depth),
            VerkleNode::Leaf87(n) => n.lookup(key, depth),
            VerkleNode::Leaf88(n) => n.lookup(key, depth),
            VerkleNode::Leaf89(n) => n.lookup(key, depth),
            VerkleNode::Leaf90(n) => n.lookup(key, depth),
            VerkleNode::Leaf91(n) => n.lookup(key, depth),
            VerkleNode::Leaf92(n) => n.lookup(key, depth),
            VerkleNode::Leaf93(n) => n.lookup(key, depth),
            VerkleNode::Leaf94(n) => n.lookup(key, depth),
            VerkleNode::Leaf95(n) => n.lookup(key, depth),
            VerkleNode::Leaf96(n) => n.lookup(key, depth),
            VerkleNode::Leaf97(n) => n.lookup(key, depth),
            VerkleNode::Leaf98(n) => n.lookup(key, depth),
            VerkleNode::Leaf99(n) => n.lookup(key, depth),
            VerkleNode::Leaf100(n) => n.lookup(key, depth),
            VerkleNode::Leaf101(n) => n.lookup(key, depth),
            VerkleNode::Leaf102(n) => n.lookup(key, depth),
            VerkleNode::Leaf103(n) => n.lookup(key, depth),
            VerkleNode::Leaf104(n) => n.lookup(key, depth),
            VerkleNode::Leaf105(n) => n.lookup(key, depth),
            VerkleNode::Leaf106(n) => n.lookup(key, depth),
            VerkleNode::Leaf107(n) => n.lookup(key, depth),
            VerkleNode::Leaf108(n) => n.lookup(key, depth),
            VerkleNode::Leaf109(n) => n.lookup(key, depth),
            VerkleNode::Leaf110(n) => n.lookup(key, depth),
            VerkleNode::Leaf111(n) => n.lookup(key, depth),
            VerkleNode::Leaf112(n) => n.lookup(key, depth),
            VerkleNode::Leaf113(n) => n.lookup(key, depth),
            VerkleNode::Leaf114(n) => n.lookup(key, depth),
            VerkleNode::Leaf115(n) => n.lookup(key, depth),
            VerkleNode::Leaf116(n) => n.lookup(key, depth),
            VerkleNode::Leaf117(n) => n.lookup(key, depth),
            VerkleNode::Leaf118(n) => n.lookup(key, depth),
            VerkleNode::Leaf119(n) => n.lookup(key, depth),
            VerkleNode::Leaf120(n) => n.lookup(key, depth),
            VerkleNode::Leaf121(n) => n.lookup(key, depth),
            VerkleNode::Leaf122(n) => n.lookup(key, depth),
            VerkleNode::Leaf123(n) => n.lookup(key, depth),
            VerkleNode::Leaf124(n) => n.lookup(key, depth),
            VerkleNode::Leaf125(n) => n.lookup(key, depth),
            VerkleNode::Leaf126(n) => n.lookup(key, depth),
            VerkleNode::Leaf127(n) => n.lookup(key, depth),
            VerkleNode::Leaf128(n) => n.lookup(key, depth),
            VerkleNode::Leaf129(n) => n.lookup(key, depth),
            VerkleNode::Leaf130(n) => n.lookup(key, depth),
            VerkleNode::Leaf131(n) => n.lookup(key, depth),
            VerkleNode::Leaf132(n) => n.lookup(key, depth),
            VerkleNode::Leaf133(n) => n.lookup(key, depth),
            VerkleNode::Leaf134(n) => n.lookup(key, depth),
            VerkleNode::Leaf135(n) => n.lookup(key, depth),
            VerkleNode::Leaf136(n) => n.lookup(key, depth),
            VerkleNode::Leaf137(n) => n.lookup(key, depth),
            VerkleNode::Leaf138(n) => n.lookup(key, depth),
            VerkleNode::Leaf139(n) => n.lookup(key, depth),
            VerkleNode::Leaf140(n) => n.lookup(key, depth),
            VerkleNode::Leaf141(n) => n.lookup(key, depth),
            VerkleNode::Leaf142(n) => n.lookup(key, depth),
            VerkleNode::Leaf143(n) => n.lookup(key, depth),
            VerkleNode::Leaf144(n) => n.lookup(key, depth),
            VerkleNode::Leaf145(n) => n.lookup(key, depth),
            VerkleNode::Leaf146(n) => n.lookup(key, depth),
            VerkleNode::Leaf147(n) => n.lookup(key, depth),
            VerkleNode::Leaf148(n) => n.lookup(key, depth),
            VerkleNode::Leaf149(n) => n.lookup(key, depth),
            VerkleNode::Leaf150(n) => n.lookup(key, depth),
            VerkleNode::Leaf151(n) => n.lookup(key, depth),
            VerkleNode::Leaf152(n) => n.lookup(key, depth),
            VerkleNode::Leaf153(n) => n.lookup(key, depth),
            VerkleNode::Leaf154(n) => n.lookup(key, depth),
            VerkleNode::Leaf155(n) => n.lookup(key, depth),
            VerkleNode::Leaf156(n) => n.lookup(key, depth),
            VerkleNode::Leaf157(n) => n.lookup(key, depth),
            VerkleNode::Leaf158(n) => n.lookup(key, depth),
            VerkleNode::Leaf159(n) => n.lookup(key, depth),
            VerkleNode::Leaf160(n) => n.lookup(key, depth),
            VerkleNode::Leaf161(n) => n.lookup(key, depth),
            VerkleNode::Leaf162(n) => n.lookup(key, depth),
            VerkleNode::Leaf163(n) => n.lookup(key, depth),
            VerkleNode::Leaf164(n) => n.lookup(key, depth),
            VerkleNode::Leaf165(n) => n.lookup(key, depth),
            VerkleNode::Leaf166(n) => n.lookup(key, depth),
            VerkleNode::Leaf167(n) => n.lookup(key, depth),
            VerkleNode::Leaf168(n) => n.lookup(key, depth),
            VerkleNode::Leaf169(n) => n.lookup(key, depth),
            VerkleNode::Leaf170(n) => n.lookup(key, depth),
            VerkleNode::Leaf171(n) => n.lookup(key, depth),
            VerkleNode::Leaf172(n) => n.lookup(key, depth),
            VerkleNode::Leaf173(n) => n.lookup(key, depth),
            VerkleNode::Leaf174(n) => n.lookup(key, depth),
            VerkleNode::Leaf175(n) => n.lookup(key, depth),
            VerkleNode::Leaf176(n) => n.lookup(key, depth),
            VerkleNode::Leaf177(n) => n.lookup(key, depth),
            VerkleNode::Leaf178(n) => n.lookup(key, depth),
            VerkleNode::Leaf179(n) => n.lookup(key, depth),
            VerkleNode::Leaf180(n) => n.lookup(key, depth),
            VerkleNode::Leaf181(n) => n.lookup(key, depth),
            VerkleNode::Leaf182(n) => n.lookup(key, depth),
            VerkleNode::Leaf183(n) => n.lookup(key, depth),
            VerkleNode::Leaf184(n) => n.lookup(key, depth),
            VerkleNode::Leaf185(n) => n.lookup(key, depth),
            VerkleNode::Leaf186(n) => n.lookup(key, depth),
            VerkleNode::Leaf187(n) => n.lookup(key, depth),
            VerkleNode::Leaf188(n) => n.lookup(key, depth),
            VerkleNode::Leaf189(n) => n.lookup(key, depth),
            VerkleNode::Leaf190(n) => n.lookup(key, depth),
            VerkleNode::Leaf191(n) => n.lookup(key, depth),
            VerkleNode::Leaf192(n) => n.lookup(key, depth),
            VerkleNode::Leaf193(n) => n.lookup(key, depth),
            VerkleNode::Leaf194(n) => n.lookup(key, depth),
            VerkleNode::Leaf195(n) => n.lookup(key, depth),
            VerkleNode::Leaf196(n) => n.lookup(key, depth),
            VerkleNode::Leaf197(n) => n.lookup(key, depth),
            VerkleNode::Leaf198(n) => n.lookup(key, depth),
            VerkleNode::Leaf199(n) => n.lookup(key, depth),
            VerkleNode::Leaf200(n) => n.lookup(key, depth),
            VerkleNode::Leaf201(n) => n.lookup(key, depth),
            VerkleNode::Leaf202(n) => n.lookup(key, depth),
            VerkleNode::Leaf203(n) => n.lookup(key, depth),
            VerkleNode::Leaf204(n) => n.lookup(key, depth),
            VerkleNode::Leaf205(n) => n.lookup(key, depth),
            VerkleNode::Leaf206(n) => n.lookup(key, depth),
            VerkleNode::Leaf207(n) => n.lookup(key, depth),
            VerkleNode::Leaf208(n) => n.lookup(key, depth),
            VerkleNode::Leaf209(n) => n.lookup(key, depth),
            VerkleNode::Leaf210(n) => n.lookup(key, depth),
            VerkleNode::Leaf211(n) => n.lookup(key, depth),
            VerkleNode::Leaf212(n) => n.lookup(key, depth),
            VerkleNode::Leaf213(n) => n.lookup(key, depth),
            VerkleNode::Leaf214(n) => n.lookup(key, depth),
            VerkleNode::Leaf215(n) => n.lookup(key, depth),
            VerkleNode::Leaf216(n) => n.lookup(key, depth),
            VerkleNode::Leaf217(n) => n.lookup(key, depth),
            VerkleNode::Leaf218(n) => n.lookup(key, depth),
            VerkleNode::Leaf219(n) => n.lookup(key, depth),
            VerkleNode::Leaf220(n) => n.lookup(key, depth),
            VerkleNode::Leaf221(n) => n.lookup(key, depth),
            VerkleNode::Leaf222(n) => n.lookup(key, depth),
            VerkleNode::Leaf223(n) => n.lookup(key, depth),
            VerkleNode::Leaf224(n) => n.lookup(key, depth),
            VerkleNode::Leaf225(n) => n.lookup(key, depth),
            VerkleNode::Leaf226(n) => n.lookup(key, depth),
            VerkleNode::Leaf227(n) => n.lookup(key, depth),
            VerkleNode::Leaf228(n) => n.lookup(key, depth),
            VerkleNode::Leaf229(n) => n.lookup(key, depth),
            VerkleNode::Leaf230(n) => n.lookup(key, depth),
            VerkleNode::Leaf231(n) => n.lookup(key, depth),
            VerkleNode::Leaf232(n) => n.lookup(key, depth),
            VerkleNode::Leaf233(n) => n.lookup(key, depth),
            VerkleNode::Leaf234(n) => n.lookup(key, depth),
            VerkleNode::Leaf235(n) => n.lookup(key, depth),
            VerkleNode::Leaf236(n) => n.lookup(key, depth),
            VerkleNode::Leaf237(n) => n.lookup(key, depth),
            VerkleNode::Leaf238(n) => n.lookup(key, depth),
            VerkleNode::Leaf239(n) => n.lookup(key, depth),
            VerkleNode::Leaf240(n) => n.lookup(key, depth),
            VerkleNode::Leaf241(n) => n.lookup(key, depth),
            VerkleNode::Leaf242(n) => n.lookup(key, depth),
            VerkleNode::Leaf243(n) => n.lookup(key, depth),
            VerkleNode::Leaf244(n) => n.lookup(key, depth),
            VerkleNode::Leaf245(n) => n.lookup(key, depth),
            VerkleNode::Leaf246(n) => n.lookup(key, depth),
            VerkleNode::Leaf247(n) => n.lookup(key, depth),
            VerkleNode::Leaf248(n) => n.lookup(key, depth),
            VerkleNode::Leaf249(n) => n.lookup(key, depth),
            VerkleNode::Leaf250(n) => n.lookup(key, depth),
            VerkleNode::Leaf251(n) => n.lookup(key, depth),
            VerkleNode::Leaf252(n) => n.lookup(key, depth),
            VerkleNode::Leaf253(n) => n.lookup(key, depth),
            VerkleNode::Leaf254(n) => n.lookup(key, depth),
            VerkleNode::Leaf255(n) => n.lookup(key, depth),
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
            VerkleNode::Inner1(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner2(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner3(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner4(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner5(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner6(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner7(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner8(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner9(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner10(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner11(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner12(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner13(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner14(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner15(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner16(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner17(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner18(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner19(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner20(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner21(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner22(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner23(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner24(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner25(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner26(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner27(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner28(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner29(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner30(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner31(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner32(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner33(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner34(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner35(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner36(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner37(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner38(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner39(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner40(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner41(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner42(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner43(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner44(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner45(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner46(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner47(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner48(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner49(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner50(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner51(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner52(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner53(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner54(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner55(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner56(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner57(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner58(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner59(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner60(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner61(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner62(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner63(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner64(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner65(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner66(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner67(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner68(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner69(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner70(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner71(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner72(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner73(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner74(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner75(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner76(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner77(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner78(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner79(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner80(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner81(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner82(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner83(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner84(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner85(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner86(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner87(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner88(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner89(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner90(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner91(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner92(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner93(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner94(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner95(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner96(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner97(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner98(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner99(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner100(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner101(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner102(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner103(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner104(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner105(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner106(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner107(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner108(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner109(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner110(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner111(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner112(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner113(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner114(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner115(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner116(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner117(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner118(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner119(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner120(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner121(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner122(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner123(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner124(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner125(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner126(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner127(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner128(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner129(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner130(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner131(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner132(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner133(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner134(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner135(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner136(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner137(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner138(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner139(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner140(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner141(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner142(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner143(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner144(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner145(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner146(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner147(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner148(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner149(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner150(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner151(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner152(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner153(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner154(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner155(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner156(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner157(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner158(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner159(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner160(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner161(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner162(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner163(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner164(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner165(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner166(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner167(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner168(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner169(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner170(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner171(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner172(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner173(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner174(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner175(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner176(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner177(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner178(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner179(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner180(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner181(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner182(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner183(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner184(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner185(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner186(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner187(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner188(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner189(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner190(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner191(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner192(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner193(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner194(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner195(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner196(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner197(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner198(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner199(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner200(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner201(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner202(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner203(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner204(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner205(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner206(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner207(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner208(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner209(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner210(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner211(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner212(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner213(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner214(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner215(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner216(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner217(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner218(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner219(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner220(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner221(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner222(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner223(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner224(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner225(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner226(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner227(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner228(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner229(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner230(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner231(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner232(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner233(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner234(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner235(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner236(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner237(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner238(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner239(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner240(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner241(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner242(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner243(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner244(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner245(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner246(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner247(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner248(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner249(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner250(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner251(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner252(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner253(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner254(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Inner255(n) => n.next_store_action(updates, depth, self_id),
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
            VerkleNode::Leaf17(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf18(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf19(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf20(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf21(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf22(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf23(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf24(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf25(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf26(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf27(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf28(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf29(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf30(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf31(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf32(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf33(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf34(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf35(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf36(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf37(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf38(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf39(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf40(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf41(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf42(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf43(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf44(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf45(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf46(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf47(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf48(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf49(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf50(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf51(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf52(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf53(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf54(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf55(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf56(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf57(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf58(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf59(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf60(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf61(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf62(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf63(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf64(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf65(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf66(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf67(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf68(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf69(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf70(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf71(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf72(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf73(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf74(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf75(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf76(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf77(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf78(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf79(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf80(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf81(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf82(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf83(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf84(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf85(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf86(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf87(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf88(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf89(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf90(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf91(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf92(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf93(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf94(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf95(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf96(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf97(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf98(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf99(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf100(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf101(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf102(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf103(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf104(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf105(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf106(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf107(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf108(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf109(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf110(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf111(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf112(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf113(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf114(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf115(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf116(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf117(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf118(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf119(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf120(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf121(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf122(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf123(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf124(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf125(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf126(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf127(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf128(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf129(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf130(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf131(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf132(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf133(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf134(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf135(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf136(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf137(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf138(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf139(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf140(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf141(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf142(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf143(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf144(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf145(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf146(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf147(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf148(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf149(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf150(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf151(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf152(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf153(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf154(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf155(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf156(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf157(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf158(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf159(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf160(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf161(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf162(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf163(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf164(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf165(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf166(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf167(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf168(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf169(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf170(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf171(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf172(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf173(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf174(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf175(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf176(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf177(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf178(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf179(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf180(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf181(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf182(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf183(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf184(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf185(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf186(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf187(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf188(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf189(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf190(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf191(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf192(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf193(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf194(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf195(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf196(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf197(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf198(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf199(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf200(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf201(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf202(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf203(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf204(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf205(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf206(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf207(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf208(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf209(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf210(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf211(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf212(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf213(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf214(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf215(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf216(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf217(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf218(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf219(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf220(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf221(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf222(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf223(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf224(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf225(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf226(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf227(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf228(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf229(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf230(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf231(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf232(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf233(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf234(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf235(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf236(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf237(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf238(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf239(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf240(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf241(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf242(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf243(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf244(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf245(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf246(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf247(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf248(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf249(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf250(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf251(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf252(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf253(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf254(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf255(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::Leaf256(n) => n.next_store_action(updates, depth, self_id),
            VerkleNode::LeafDelta(n) => n.next_store_action(updates, depth, self_id),
        }
    }

    fn replace_child(&mut self, key: &Key, depth: u8, new: VerkleNodeId) -> BTResult<(), Error> {
        match self {
            VerkleNode::Empty(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner1(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner2(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner3(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner4(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner5(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner6(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner7(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner8(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner9(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner10(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner11(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner12(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner13(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner14(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner15(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner16(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner17(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner18(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner19(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner20(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner21(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner22(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner23(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner24(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner25(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner26(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner27(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner28(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner29(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner30(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner31(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner32(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner33(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner34(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner35(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner36(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner37(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner38(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner39(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner40(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner41(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner42(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner43(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner44(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner45(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner46(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner47(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner48(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner49(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner50(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner51(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner52(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner53(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner54(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner55(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner56(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner57(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner58(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner59(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner60(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner61(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner62(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner63(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner64(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner65(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner66(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner67(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner68(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner69(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner70(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner71(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner72(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner73(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner74(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner75(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner76(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner77(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner78(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner79(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner80(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner81(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner82(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner83(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner84(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner85(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner86(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner87(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner88(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner89(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner90(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner91(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner92(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner93(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner94(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner95(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner96(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner97(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner98(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner99(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner100(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner101(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner102(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner103(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner104(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner105(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner106(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner107(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner108(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner109(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner110(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner111(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner112(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner113(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner114(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner115(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner116(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner117(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner118(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner119(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner120(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner121(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner122(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner123(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner124(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner125(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner126(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner127(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner128(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner129(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner130(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner131(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner132(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner133(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner134(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner135(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner136(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner137(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner138(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner139(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner140(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner141(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner142(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner143(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner144(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner145(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner146(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner147(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner148(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner149(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner150(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner151(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner152(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner153(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner154(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner155(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner156(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner157(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner158(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner159(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner160(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner161(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner162(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner163(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner164(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner165(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner166(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner167(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner168(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner169(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner170(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner171(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner172(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner173(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner174(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner175(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner176(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner177(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner178(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner179(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner180(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner181(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner182(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner183(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner184(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner185(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner186(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner187(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner188(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner189(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner190(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner191(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner192(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner193(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner194(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner195(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner196(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner197(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner198(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner199(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner200(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner201(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner202(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner203(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner204(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner205(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner206(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner207(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner208(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner209(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner210(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner211(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner212(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner213(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner214(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner215(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner216(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner217(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner218(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner219(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner220(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner221(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner222(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner223(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner224(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner225(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner226(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner227(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner228(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner229(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner230(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner231(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner232(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner233(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner234(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner235(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner236(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner237(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner238(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner239(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner240(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner241(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner242(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner243(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner244(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner245(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner246(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner247(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner248(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner249(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner250(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner251(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner252(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner253(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner254(n) => n.replace_child(key, depth, new),
            VerkleNode::Inner255(n) => n.replace_child(key, depth, new),
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
            VerkleNode::Leaf17(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf18(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf19(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf20(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf21(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf22(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf23(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf24(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf25(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf26(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf27(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf28(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf29(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf30(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf31(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf32(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf33(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf34(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf35(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf36(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf37(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf38(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf39(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf40(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf41(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf42(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf43(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf44(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf45(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf46(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf47(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf48(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf49(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf50(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf51(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf52(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf53(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf54(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf55(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf56(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf57(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf58(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf59(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf60(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf61(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf62(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf63(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf64(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf65(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf66(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf67(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf68(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf69(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf70(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf71(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf72(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf73(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf74(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf75(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf76(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf77(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf78(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf79(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf80(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf81(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf82(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf83(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf84(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf85(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf86(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf87(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf88(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf89(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf90(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf91(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf92(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf93(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf94(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf95(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf96(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf97(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf98(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf99(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf100(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf101(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf102(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf103(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf104(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf105(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf106(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf107(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf108(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf109(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf110(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf111(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf112(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf113(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf114(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf115(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf116(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf117(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf118(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf119(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf120(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf121(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf122(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf123(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf124(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf125(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf126(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf127(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf128(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf129(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf130(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf131(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf132(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf133(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf134(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf135(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf136(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf137(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf138(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf139(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf140(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf141(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf142(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf143(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf144(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf145(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf146(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf147(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf148(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf149(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf150(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf151(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf152(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf153(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf154(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf155(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf156(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf157(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf158(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf159(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf160(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf161(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf162(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf163(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf164(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf165(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf166(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf167(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf168(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf169(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf170(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf171(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf172(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf173(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf174(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf175(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf176(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf177(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf178(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf179(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf180(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf181(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf182(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf183(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf184(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf185(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf186(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf187(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf188(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf189(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf190(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf191(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf192(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf193(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf194(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf195(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf196(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf197(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf198(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf199(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf200(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf201(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf202(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf203(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf204(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf205(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf206(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf207(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf208(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf209(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf210(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf211(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf212(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf213(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf214(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf215(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf216(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf217(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf218(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf219(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf220(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf221(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf222(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf223(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf224(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf225(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf226(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf227(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf228(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf229(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf230(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf231(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf232(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf233(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf234(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf235(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf236(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf237(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf238(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf239(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf240(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf241(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf242(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf243(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf244(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf245(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf246(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf247(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf248(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf249(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf250(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf251(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf252(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf253(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf254(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf255(n) => n.replace_child(key, depth, new),
            VerkleNode::Leaf256(n) => n.replace_child(key, depth, new),
            VerkleNode::LeafDelta(n) => n.replace_child(key, depth, new),
        }
    }

    fn store(&mut self, update: &KeyedUpdate) -> BTResult<Value, Error> {
        match self {
            VerkleNode::Empty(n) => n.store(update),
            VerkleNode::Inner1(n) => n.store(update),
            VerkleNode::Inner2(n) => n.store(update),
            VerkleNode::Inner3(n) => n.store(update),
            VerkleNode::Inner4(n) => n.store(update),
            VerkleNode::Inner5(n) => n.store(update),
            VerkleNode::Inner6(n) => n.store(update),
            VerkleNode::Inner7(n) => n.store(update),
            VerkleNode::Inner8(n) => n.store(update),
            VerkleNode::Inner9(n) => n.store(update),
            VerkleNode::Inner10(n) => n.store(update),
            VerkleNode::Inner11(n) => n.store(update),
            VerkleNode::Inner12(n) => n.store(update),
            VerkleNode::Inner13(n) => n.store(update),
            VerkleNode::Inner14(n) => n.store(update),
            VerkleNode::Inner15(n) => n.store(update),
            VerkleNode::Inner16(n) => n.store(update),
            VerkleNode::Inner17(n) => n.store(update),
            VerkleNode::Inner18(n) => n.store(update),
            VerkleNode::Inner19(n) => n.store(update),
            VerkleNode::Inner20(n) => n.store(update),
            VerkleNode::Inner21(n) => n.store(update),
            VerkleNode::Inner22(n) => n.store(update),
            VerkleNode::Inner23(n) => n.store(update),
            VerkleNode::Inner24(n) => n.store(update),
            VerkleNode::Inner25(n) => n.store(update),
            VerkleNode::Inner26(n) => n.store(update),
            VerkleNode::Inner27(n) => n.store(update),
            VerkleNode::Inner28(n) => n.store(update),
            VerkleNode::Inner29(n) => n.store(update),
            VerkleNode::Inner30(n) => n.store(update),
            VerkleNode::Inner31(n) => n.store(update),
            VerkleNode::Inner32(n) => n.store(update),
            VerkleNode::Inner33(n) => n.store(update),
            VerkleNode::Inner34(n) => n.store(update),
            VerkleNode::Inner35(n) => n.store(update),
            VerkleNode::Inner36(n) => n.store(update),
            VerkleNode::Inner37(n) => n.store(update),
            VerkleNode::Inner38(n) => n.store(update),
            VerkleNode::Inner39(n) => n.store(update),
            VerkleNode::Inner40(n) => n.store(update),
            VerkleNode::Inner41(n) => n.store(update),
            VerkleNode::Inner42(n) => n.store(update),
            VerkleNode::Inner43(n) => n.store(update),
            VerkleNode::Inner44(n) => n.store(update),
            VerkleNode::Inner45(n) => n.store(update),
            VerkleNode::Inner46(n) => n.store(update),
            VerkleNode::Inner47(n) => n.store(update),
            VerkleNode::Inner48(n) => n.store(update),
            VerkleNode::Inner49(n) => n.store(update),
            VerkleNode::Inner50(n) => n.store(update),
            VerkleNode::Inner51(n) => n.store(update),
            VerkleNode::Inner52(n) => n.store(update),
            VerkleNode::Inner53(n) => n.store(update),
            VerkleNode::Inner54(n) => n.store(update),
            VerkleNode::Inner55(n) => n.store(update),
            VerkleNode::Inner56(n) => n.store(update),
            VerkleNode::Inner57(n) => n.store(update),
            VerkleNode::Inner58(n) => n.store(update),
            VerkleNode::Inner59(n) => n.store(update),
            VerkleNode::Inner60(n) => n.store(update),
            VerkleNode::Inner61(n) => n.store(update),
            VerkleNode::Inner62(n) => n.store(update),
            VerkleNode::Inner63(n) => n.store(update),
            VerkleNode::Inner64(n) => n.store(update),
            VerkleNode::Inner65(n) => n.store(update),
            VerkleNode::Inner66(n) => n.store(update),
            VerkleNode::Inner67(n) => n.store(update),
            VerkleNode::Inner68(n) => n.store(update),
            VerkleNode::Inner69(n) => n.store(update),
            VerkleNode::Inner70(n) => n.store(update),
            VerkleNode::Inner71(n) => n.store(update),
            VerkleNode::Inner72(n) => n.store(update),
            VerkleNode::Inner73(n) => n.store(update),
            VerkleNode::Inner74(n) => n.store(update),
            VerkleNode::Inner75(n) => n.store(update),
            VerkleNode::Inner76(n) => n.store(update),
            VerkleNode::Inner77(n) => n.store(update),
            VerkleNode::Inner78(n) => n.store(update),
            VerkleNode::Inner79(n) => n.store(update),
            VerkleNode::Inner80(n) => n.store(update),
            VerkleNode::Inner81(n) => n.store(update),
            VerkleNode::Inner82(n) => n.store(update),
            VerkleNode::Inner83(n) => n.store(update),
            VerkleNode::Inner84(n) => n.store(update),
            VerkleNode::Inner85(n) => n.store(update),
            VerkleNode::Inner86(n) => n.store(update),
            VerkleNode::Inner87(n) => n.store(update),
            VerkleNode::Inner88(n) => n.store(update),
            VerkleNode::Inner89(n) => n.store(update),
            VerkleNode::Inner90(n) => n.store(update),
            VerkleNode::Inner91(n) => n.store(update),
            VerkleNode::Inner92(n) => n.store(update),
            VerkleNode::Inner93(n) => n.store(update),
            VerkleNode::Inner94(n) => n.store(update),
            VerkleNode::Inner95(n) => n.store(update),
            VerkleNode::Inner96(n) => n.store(update),
            VerkleNode::Inner97(n) => n.store(update),
            VerkleNode::Inner98(n) => n.store(update),
            VerkleNode::Inner99(n) => n.store(update),
            VerkleNode::Inner100(n) => n.store(update),
            VerkleNode::Inner101(n) => n.store(update),
            VerkleNode::Inner102(n) => n.store(update),
            VerkleNode::Inner103(n) => n.store(update),
            VerkleNode::Inner104(n) => n.store(update),
            VerkleNode::Inner105(n) => n.store(update),
            VerkleNode::Inner106(n) => n.store(update),
            VerkleNode::Inner107(n) => n.store(update),
            VerkleNode::Inner108(n) => n.store(update),
            VerkleNode::Inner109(n) => n.store(update),
            VerkleNode::Inner110(n) => n.store(update),
            VerkleNode::Inner111(n) => n.store(update),
            VerkleNode::Inner112(n) => n.store(update),
            VerkleNode::Inner113(n) => n.store(update),
            VerkleNode::Inner114(n) => n.store(update),
            VerkleNode::Inner115(n) => n.store(update),
            VerkleNode::Inner116(n) => n.store(update),
            VerkleNode::Inner117(n) => n.store(update),
            VerkleNode::Inner118(n) => n.store(update),
            VerkleNode::Inner119(n) => n.store(update),
            VerkleNode::Inner120(n) => n.store(update),
            VerkleNode::Inner121(n) => n.store(update),
            VerkleNode::Inner122(n) => n.store(update),
            VerkleNode::Inner123(n) => n.store(update),
            VerkleNode::Inner124(n) => n.store(update),
            VerkleNode::Inner125(n) => n.store(update),
            VerkleNode::Inner126(n) => n.store(update),
            VerkleNode::Inner127(n) => n.store(update),
            VerkleNode::Inner128(n) => n.store(update),
            VerkleNode::Inner129(n) => n.store(update),
            VerkleNode::Inner130(n) => n.store(update),
            VerkleNode::Inner131(n) => n.store(update),
            VerkleNode::Inner132(n) => n.store(update),
            VerkleNode::Inner133(n) => n.store(update),
            VerkleNode::Inner134(n) => n.store(update),
            VerkleNode::Inner135(n) => n.store(update),
            VerkleNode::Inner136(n) => n.store(update),
            VerkleNode::Inner137(n) => n.store(update),
            VerkleNode::Inner138(n) => n.store(update),
            VerkleNode::Inner139(n) => n.store(update),
            VerkleNode::Inner140(n) => n.store(update),
            VerkleNode::Inner141(n) => n.store(update),
            VerkleNode::Inner142(n) => n.store(update),
            VerkleNode::Inner143(n) => n.store(update),
            VerkleNode::Inner144(n) => n.store(update),
            VerkleNode::Inner145(n) => n.store(update),
            VerkleNode::Inner146(n) => n.store(update),
            VerkleNode::Inner147(n) => n.store(update),
            VerkleNode::Inner148(n) => n.store(update),
            VerkleNode::Inner149(n) => n.store(update),
            VerkleNode::Inner150(n) => n.store(update),
            VerkleNode::Inner151(n) => n.store(update),
            VerkleNode::Inner152(n) => n.store(update),
            VerkleNode::Inner153(n) => n.store(update),
            VerkleNode::Inner154(n) => n.store(update),
            VerkleNode::Inner155(n) => n.store(update),
            VerkleNode::Inner156(n) => n.store(update),
            VerkleNode::Inner157(n) => n.store(update),
            VerkleNode::Inner158(n) => n.store(update),
            VerkleNode::Inner159(n) => n.store(update),
            VerkleNode::Inner160(n) => n.store(update),
            VerkleNode::Inner161(n) => n.store(update),
            VerkleNode::Inner162(n) => n.store(update),
            VerkleNode::Inner163(n) => n.store(update),
            VerkleNode::Inner164(n) => n.store(update),
            VerkleNode::Inner165(n) => n.store(update),
            VerkleNode::Inner166(n) => n.store(update),
            VerkleNode::Inner167(n) => n.store(update),
            VerkleNode::Inner168(n) => n.store(update),
            VerkleNode::Inner169(n) => n.store(update),
            VerkleNode::Inner170(n) => n.store(update),
            VerkleNode::Inner171(n) => n.store(update),
            VerkleNode::Inner172(n) => n.store(update),
            VerkleNode::Inner173(n) => n.store(update),
            VerkleNode::Inner174(n) => n.store(update),
            VerkleNode::Inner175(n) => n.store(update),
            VerkleNode::Inner176(n) => n.store(update),
            VerkleNode::Inner177(n) => n.store(update),
            VerkleNode::Inner178(n) => n.store(update),
            VerkleNode::Inner179(n) => n.store(update),
            VerkleNode::Inner180(n) => n.store(update),
            VerkleNode::Inner181(n) => n.store(update),
            VerkleNode::Inner182(n) => n.store(update),
            VerkleNode::Inner183(n) => n.store(update),
            VerkleNode::Inner184(n) => n.store(update),
            VerkleNode::Inner185(n) => n.store(update),
            VerkleNode::Inner186(n) => n.store(update),
            VerkleNode::Inner187(n) => n.store(update),
            VerkleNode::Inner188(n) => n.store(update),
            VerkleNode::Inner189(n) => n.store(update),
            VerkleNode::Inner190(n) => n.store(update),
            VerkleNode::Inner191(n) => n.store(update),
            VerkleNode::Inner192(n) => n.store(update),
            VerkleNode::Inner193(n) => n.store(update),
            VerkleNode::Inner194(n) => n.store(update),
            VerkleNode::Inner195(n) => n.store(update),
            VerkleNode::Inner196(n) => n.store(update),
            VerkleNode::Inner197(n) => n.store(update),
            VerkleNode::Inner198(n) => n.store(update),
            VerkleNode::Inner199(n) => n.store(update),
            VerkleNode::Inner200(n) => n.store(update),
            VerkleNode::Inner201(n) => n.store(update),
            VerkleNode::Inner202(n) => n.store(update),
            VerkleNode::Inner203(n) => n.store(update),
            VerkleNode::Inner204(n) => n.store(update),
            VerkleNode::Inner205(n) => n.store(update),
            VerkleNode::Inner206(n) => n.store(update),
            VerkleNode::Inner207(n) => n.store(update),
            VerkleNode::Inner208(n) => n.store(update),
            VerkleNode::Inner209(n) => n.store(update),
            VerkleNode::Inner210(n) => n.store(update),
            VerkleNode::Inner211(n) => n.store(update),
            VerkleNode::Inner212(n) => n.store(update),
            VerkleNode::Inner213(n) => n.store(update),
            VerkleNode::Inner214(n) => n.store(update),
            VerkleNode::Inner215(n) => n.store(update),
            VerkleNode::Inner216(n) => n.store(update),
            VerkleNode::Inner217(n) => n.store(update),
            VerkleNode::Inner218(n) => n.store(update),
            VerkleNode::Inner219(n) => n.store(update),
            VerkleNode::Inner220(n) => n.store(update),
            VerkleNode::Inner221(n) => n.store(update),
            VerkleNode::Inner222(n) => n.store(update),
            VerkleNode::Inner223(n) => n.store(update),
            VerkleNode::Inner224(n) => n.store(update),
            VerkleNode::Inner225(n) => n.store(update),
            VerkleNode::Inner226(n) => n.store(update),
            VerkleNode::Inner227(n) => n.store(update),
            VerkleNode::Inner228(n) => n.store(update),
            VerkleNode::Inner229(n) => n.store(update),
            VerkleNode::Inner230(n) => n.store(update),
            VerkleNode::Inner231(n) => n.store(update),
            VerkleNode::Inner232(n) => n.store(update),
            VerkleNode::Inner233(n) => n.store(update),
            VerkleNode::Inner234(n) => n.store(update),
            VerkleNode::Inner235(n) => n.store(update),
            VerkleNode::Inner236(n) => n.store(update),
            VerkleNode::Inner237(n) => n.store(update),
            VerkleNode::Inner238(n) => n.store(update),
            VerkleNode::Inner239(n) => n.store(update),
            VerkleNode::Inner240(n) => n.store(update),
            VerkleNode::Inner241(n) => n.store(update),
            VerkleNode::Inner242(n) => n.store(update),
            VerkleNode::Inner243(n) => n.store(update),
            VerkleNode::Inner244(n) => n.store(update),
            VerkleNode::Inner245(n) => n.store(update),
            VerkleNode::Inner246(n) => n.store(update),
            VerkleNode::Inner247(n) => n.store(update),
            VerkleNode::Inner248(n) => n.store(update),
            VerkleNode::Inner249(n) => n.store(update),
            VerkleNode::Inner250(n) => n.store(update),
            VerkleNode::Inner251(n) => n.store(update),
            VerkleNode::Inner252(n) => n.store(update),
            VerkleNode::Inner253(n) => n.store(update),
            VerkleNode::Inner254(n) => n.store(update),
            VerkleNode::Inner255(n) => n.store(update),
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
            VerkleNode::Leaf17(n) => n.store(update),
            VerkleNode::Leaf18(n) => n.store(update),
            VerkleNode::Leaf19(n) => n.store(update),
            VerkleNode::Leaf20(n) => n.store(update),
            VerkleNode::Leaf21(n) => n.store(update),
            VerkleNode::Leaf22(n) => n.store(update),
            VerkleNode::Leaf23(n) => n.store(update),
            VerkleNode::Leaf24(n) => n.store(update),
            VerkleNode::Leaf25(n) => n.store(update),
            VerkleNode::Leaf26(n) => n.store(update),
            VerkleNode::Leaf27(n) => n.store(update),
            VerkleNode::Leaf28(n) => n.store(update),
            VerkleNode::Leaf29(n) => n.store(update),
            VerkleNode::Leaf30(n) => n.store(update),
            VerkleNode::Leaf31(n) => n.store(update),
            VerkleNode::Leaf32(n) => n.store(update),
            VerkleNode::Leaf33(n) => n.store(update),
            VerkleNode::Leaf34(n) => n.store(update),
            VerkleNode::Leaf35(n) => n.store(update),
            VerkleNode::Leaf36(n) => n.store(update),
            VerkleNode::Leaf37(n) => n.store(update),
            VerkleNode::Leaf38(n) => n.store(update),
            VerkleNode::Leaf39(n) => n.store(update),
            VerkleNode::Leaf40(n) => n.store(update),
            VerkleNode::Leaf41(n) => n.store(update),
            VerkleNode::Leaf42(n) => n.store(update),
            VerkleNode::Leaf43(n) => n.store(update),
            VerkleNode::Leaf44(n) => n.store(update),
            VerkleNode::Leaf45(n) => n.store(update),
            VerkleNode::Leaf46(n) => n.store(update),
            VerkleNode::Leaf47(n) => n.store(update),
            VerkleNode::Leaf48(n) => n.store(update),
            VerkleNode::Leaf49(n) => n.store(update),
            VerkleNode::Leaf50(n) => n.store(update),
            VerkleNode::Leaf51(n) => n.store(update),
            VerkleNode::Leaf52(n) => n.store(update),
            VerkleNode::Leaf53(n) => n.store(update),
            VerkleNode::Leaf54(n) => n.store(update),
            VerkleNode::Leaf55(n) => n.store(update),
            VerkleNode::Leaf56(n) => n.store(update),
            VerkleNode::Leaf57(n) => n.store(update),
            VerkleNode::Leaf58(n) => n.store(update),
            VerkleNode::Leaf59(n) => n.store(update),
            VerkleNode::Leaf60(n) => n.store(update),
            VerkleNode::Leaf61(n) => n.store(update),
            VerkleNode::Leaf62(n) => n.store(update),
            VerkleNode::Leaf63(n) => n.store(update),
            VerkleNode::Leaf64(n) => n.store(update),
            VerkleNode::Leaf65(n) => n.store(update),
            VerkleNode::Leaf66(n) => n.store(update),
            VerkleNode::Leaf67(n) => n.store(update),
            VerkleNode::Leaf68(n) => n.store(update),
            VerkleNode::Leaf69(n) => n.store(update),
            VerkleNode::Leaf70(n) => n.store(update),
            VerkleNode::Leaf71(n) => n.store(update),
            VerkleNode::Leaf72(n) => n.store(update),
            VerkleNode::Leaf73(n) => n.store(update),
            VerkleNode::Leaf74(n) => n.store(update),
            VerkleNode::Leaf75(n) => n.store(update),
            VerkleNode::Leaf76(n) => n.store(update),
            VerkleNode::Leaf77(n) => n.store(update),
            VerkleNode::Leaf78(n) => n.store(update),
            VerkleNode::Leaf79(n) => n.store(update),
            VerkleNode::Leaf80(n) => n.store(update),
            VerkleNode::Leaf81(n) => n.store(update),
            VerkleNode::Leaf82(n) => n.store(update),
            VerkleNode::Leaf83(n) => n.store(update),
            VerkleNode::Leaf84(n) => n.store(update),
            VerkleNode::Leaf85(n) => n.store(update),
            VerkleNode::Leaf86(n) => n.store(update),
            VerkleNode::Leaf87(n) => n.store(update),
            VerkleNode::Leaf88(n) => n.store(update),
            VerkleNode::Leaf89(n) => n.store(update),
            VerkleNode::Leaf90(n) => n.store(update),
            VerkleNode::Leaf91(n) => n.store(update),
            VerkleNode::Leaf92(n) => n.store(update),
            VerkleNode::Leaf93(n) => n.store(update),
            VerkleNode::Leaf94(n) => n.store(update),
            VerkleNode::Leaf95(n) => n.store(update),
            VerkleNode::Leaf96(n) => n.store(update),
            VerkleNode::Leaf97(n) => n.store(update),
            VerkleNode::Leaf98(n) => n.store(update),
            VerkleNode::Leaf99(n) => n.store(update),
            VerkleNode::Leaf100(n) => n.store(update),
            VerkleNode::Leaf101(n) => n.store(update),
            VerkleNode::Leaf102(n) => n.store(update),
            VerkleNode::Leaf103(n) => n.store(update),
            VerkleNode::Leaf104(n) => n.store(update),
            VerkleNode::Leaf105(n) => n.store(update),
            VerkleNode::Leaf106(n) => n.store(update),
            VerkleNode::Leaf107(n) => n.store(update),
            VerkleNode::Leaf108(n) => n.store(update),
            VerkleNode::Leaf109(n) => n.store(update),
            VerkleNode::Leaf110(n) => n.store(update),
            VerkleNode::Leaf111(n) => n.store(update),
            VerkleNode::Leaf112(n) => n.store(update),
            VerkleNode::Leaf113(n) => n.store(update),
            VerkleNode::Leaf114(n) => n.store(update),
            VerkleNode::Leaf115(n) => n.store(update),
            VerkleNode::Leaf116(n) => n.store(update),
            VerkleNode::Leaf117(n) => n.store(update),
            VerkleNode::Leaf118(n) => n.store(update),
            VerkleNode::Leaf119(n) => n.store(update),
            VerkleNode::Leaf120(n) => n.store(update),
            VerkleNode::Leaf121(n) => n.store(update),
            VerkleNode::Leaf122(n) => n.store(update),
            VerkleNode::Leaf123(n) => n.store(update),
            VerkleNode::Leaf124(n) => n.store(update),
            VerkleNode::Leaf125(n) => n.store(update),
            VerkleNode::Leaf126(n) => n.store(update),
            VerkleNode::Leaf127(n) => n.store(update),
            VerkleNode::Leaf128(n) => n.store(update),
            VerkleNode::Leaf129(n) => n.store(update),
            VerkleNode::Leaf130(n) => n.store(update),
            VerkleNode::Leaf131(n) => n.store(update),
            VerkleNode::Leaf132(n) => n.store(update),
            VerkleNode::Leaf133(n) => n.store(update),
            VerkleNode::Leaf134(n) => n.store(update),
            VerkleNode::Leaf135(n) => n.store(update),
            VerkleNode::Leaf136(n) => n.store(update),
            VerkleNode::Leaf137(n) => n.store(update),
            VerkleNode::Leaf138(n) => n.store(update),
            VerkleNode::Leaf139(n) => n.store(update),
            VerkleNode::Leaf140(n) => n.store(update),
            VerkleNode::Leaf141(n) => n.store(update),
            VerkleNode::Leaf142(n) => n.store(update),
            VerkleNode::Leaf143(n) => n.store(update),
            VerkleNode::Leaf144(n) => n.store(update),
            VerkleNode::Leaf145(n) => n.store(update),
            VerkleNode::Leaf146(n) => n.store(update),
            VerkleNode::Leaf147(n) => n.store(update),
            VerkleNode::Leaf148(n) => n.store(update),
            VerkleNode::Leaf149(n) => n.store(update),
            VerkleNode::Leaf150(n) => n.store(update),
            VerkleNode::Leaf151(n) => n.store(update),
            VerkleNode::Leaf152(n) => n.store(update),
            VerkleNode::Leaf153(n) => n.store(update),
            VerkleNode::Leaf154(n) => n.store(update),
            VerkleNode::Leaf155(n) => n.store(update),
            VerkleNode::Leaf156(n) => n.store(update),
            VerkleNode::Leaf157(n) => n.store(update),
            VerkleNode::Leaf158(n) => n.store(update),
            VerkleNode::Leaf159(n) => n.store(update),
            VerkleNode::Leaf160(n) => n.store(update),
            VerkleNode::Leaf161(n) => n.store(update),
            VerkleNode::Leaf162(n) => n.store(update),
            VerkleNode::Leaf163(n) => n.store(update),
            VerkleNode::Leaf164(n) => n.store(update),
            VerkleNode::Leaf165(n) => n.store(update),
            VerkleNode::Leaf166(n) => n.store(update),
            VerkleNode::Leaf167(n) => n.store(update),
            VerkleNode::Leaf168(n) => n.store(update),
            VerkleNode::Leaf169(n) => n.store(update),
            VerkleNode::Leaf170(n) => n.store(update),
            VerkleNode::Leaf171(n) => n.store(update),
            VerkleNode::Leaf172(n) => n.store(update),
            VerkleNode::Leaf173(n) => n.store(update),
            VerkleNode::Leaf174(n) => n.store(update),
            VerkleNode::Leaf175(n) => n.store(update),
            VerkleNode::Leaf176(n) => n.store(update),
            VerkleNode::Leaf177(n) => n.store(update),
            VerkleNode::Leaf178(n) => n.store(update),
            VerkleNode::Leaf179(n) => n.store(update),
            VerkleNode::Leaf180(n) => n.store(update),
            VerkleNode::Leaf181(n) => n.store(update),
            VerkleNode::Leaf182(n) => n.store(update),
            VerkleNode::Leaf183(n) => n.store(update),
            VerkleNode::Leaf184(n) => n.store(update),
            VerkleNode::Leaf185(n) => n.store(update),
            VerkleNode::Leaf186(n) => n.store(update),
            VerkleNode::Leaf187(n) => n.store(update),
            VerkleNode::Leaf188(n) => n.store(update),
            VerkleNode::Leaf189(n) => n.store(update),
            VerkleNode::Leaf190(n) => n.store(update),
            VerkleNode::Leaf191(n) => n.store(update),
            VerkleNode::Leaf192(n) => n.store(update),
            VerkleNode::Leaf193(n) => n.store(update),
            VerkleNode::Leaf194(n) => n.store(update),
            VerkleNode::Leaf195(n) => n.store(update),
            VerkleNode::Leaf196(n) => n.store(update),
            VerkleNode::Leaf197(n) => n.store(update),
            VerkleNode::Leaf198(n) => n.store(update),
            VerkleNode::Leaf199(n) => n.store(update),
            VerkleNode::Leaf200(n) => n.store(update),
            VerkleNode::Leaf201(n) => n.store(update),
            VerkleNode::Leaf202(n) => n.store(update),
            VerkleNode::Leaf203(n) => n.store(update),
            VerkleNode::Leaf204(n) => n.store(update),
            VerkleNode::Leaf205(n) => n.store(update),
            VerkleNode::Leaf206(n) => n.store(update),
            VerkleNode::Leaf207(n) => n.store(update),
            VerkleNode::Leaf208(n) => n.store(update),
            VerkleNode::Leaf209(n) => n.store(update),
            VerkleNode::Leaf210(n) => n.store(update),
            VerkleNode::Leaf211(n) => n.store(update),
            VerkleNode::Leaf212(n) => n.store(update),
            VerkleNode::Leaf213(n) => n.store(update),
            VerkleNode::Leaf214(n) => n.store(update),
            VerkleNode::Leaf215(n) => n.store(update),
            VerkleNode::Leaf216(n) => n.store(update),
            VerkleNode::Leaf217(n) => n.store(update),
            VerkleNode::Leaf218(n) => n.store(update),
            VerkleNode::Leaf219(n) => n.store(update),
            VerkleNode::Leaf220(n) => n.store(update),
            VerkleNode::Leaf221(n) => n.store(update),
            VerkleNode::Leaf222(n) => n.store(update),
            VerkleNode::Leaf223(n) => n.store(update),
            VerkleNode::Leaf224(n) => n.store(update),
            VerkleNode::Leaf225(n) => n.store(update),
            VerkleNode::Leaf226(n) => n.store(update),
            VerkleNode::Leaf227(n) => n.store(update),
            VerkleNode::Leaf228(n) => n.store(update),
            VerkleNode::Leaf229(n) => n.store(update),
            VerkleNode::Leaf230(n) => n.store(update),
            VerkleNode::Leaf231(n) => n.store(update),
            VerkleNode::Leaf232(n) => n.store(update),
            VerkleNode::Leaf233(n) => n.store(update),
            VerkleNode::Leaf234(n) => n.store(update),
            VerkleNode::Leaf235(n) => n.store(update),
            VerkleNode::Leaf236(n) => n.store(update),
            VerkleNode::Leaf237(n) => n.store(update),
            VerkleNode::Leaf238(n) => n.store(update),
            VerkleNode::Leaf239(n) => n.store(update),
            VerkleNode::Leaf240(n) => n.store(update),
            VerkleNode::Leaf241(n) => n.store(update),
            VerkleNode::Leaf242(n) => n.store(update),
            VerkleNode::Leaf243(n) => n.store(update),
            VerkleNode::Leaf244(n) => n.store(update),
            VerkleNode::Leaf245(n) => n.store(update),
            VerkleNode::Leaf246(n) => n.store(update),
            VerkleNode::Leaf247(n) => n.store(update),
            VerkleNode::Leaf248(n) => n.store(update),
            VerkleNode::Leaf249(n) => n.store(update),
            VerkleNode::Leaf250(n) => n.store(update),
            VerkleNode::Leaf251(n) => n.store(update),
            VerkleNode::Leaf252(n) => n.store(update),
            VerkleNode::Leaf253(n) => n.store(update),
            VerkleNode::Leaf254(n) => n.store(update),
            VerkleNode::Leaf255(n) => n.store(update),
            VerkleNode::Leaf256(n) => n.store(update),
            VerkleNode::LeafDelta(n) => n.store(update),
        }
    }

    fn get_commitment(&self) -> Self::Commitment {
        match self {
            VerkleNode::Empty(n) => n.get_commitment(),
            VerkleNode::Inner1(n) => n.get_commitment(),
            VerkleNode::Inner2(n) => n.get_commitment(),
            VerkleNode::Inner3(n) => n.get_commitment(),
            VerkleNode::Inner4(n) => n.get_commitment(),
            VerkleNode::Inner5(n) => n.get_commitment(),
            VerkleNode::Inner6(n) => n.get_commitment(),
            VerkleNode::Inner7(n) => n.get_commitment(),
            VerkleNode::Inner8(n) => n.get_commitment(),
            VerkleNode::Inner9(n) => n.get_commitment(),
            VerkleNode::Inner10(n) => n.get_commitment(),
            VerkleNode::Inner11(n) => n.get_commitment(),
            VerkleNode::Inner12(n) => n.get_commitment(),
            VerkleNode::Inner13(n) => n.get_commitment(),
            VerkleNode::Inner14(n) => n.get_commitment(),
            VerkleNode::Inner15(n) => n.get_commitment(),
            VerkleNode::Inner16(n) => n.get_commitment(),
            VerkleNode::Inner17(n) => n.get_commitment(),
            VerkleNode::Inner18(n) => n.get_commitment(),
            VerkleNode::Inner19(n) => n.get_commitment(),
            VerkleNode::Inner20(n) => n.get_commitment(),
            VerkleNode::Inner21(n) => n.get_commitment(),
            VerkleNode::Inner22(n) => n.get_commitment(),
            VerkleNode::Inner23(n) => n.get_commitment(),
            VerkleNode::Inner24(n) => n.get_commitment(),
            VerkleNode::Inner25(n) => n.get_commitment(),
            VerkleNode::Inner26(n) => n.get_commitment(),
            VerkleNode::Inner27(n) => n.get_commitment(),
            VerkleNode::Inner28(n) => n.get_commitment(),
            VerkleNode::Inner29(n) => n.get_commitment(),
            VerkleNode::Inner30(n) => n.get_commitment(),
            VerkleNode::Inner31(n) => n.get_commitment(),
            VerkleNode::Inner32(n) => n.get_commitment(),
            VerkleNode::Inner33(n) => n.get_commitment(),
            VerkleNode::Inner34(n) => n.get_commitment(),
            VerkleNode::Inner35(n) => n.get_commitment(),
            VerkleNode::Inner36(n) => n.get_commitment(),
            VerkleNode::Inner37(n) => n.get_commitment(),
            VerkleNode::Inner38(n) => n.get_commitment(),
            VerkleNode::Inner39(n) => n.get_commitment(),
            VerkleNode::Inner40(n) => n.get_commitment(),
            VerkleNode::Inner41(n) => n.get_commitment(),
            VerkleNode::Inner42(n) => n.get_commitment(),
            VerkleNode::Inner43(n) => n.get_commitment(),
            VerkleNode::Inner44(n) => n.get_commitment(),
            VerkleNode::Inner45(n) => n.get_commitment(),
            VerkleNode::Inner46(n) => n.get_commitment(),
            VerkleNode::Inner47(n) => n.get_commitment(),
            VerkleNode::Inner48(n) => n.get_commitment(),
            VerkleNode::Inner49(n) => n.get_commitment(),
            VerkleNode::Inner50(n) => n.get_commitment(),
            VerkleNode::Inner51(n) => n.get_commitment(),
            VerkleNode::Inner52(n) => n.get_commitment(),
            VerkleNode::Inner53(n) => n.get_commitment(),
            VerkleNode::Inner54(n) => n.get_commitment(),
            VerkleNode::Inner55(n) => n.get_commitment(),
            VerkleNode::Inner56(n) => n.get_commitment(),
            VerkleNode::Inner57(n) => n.get_commitment(),
            VerkleNode::Inner58(n) => n.get_commitment(),
            VerkleNode::Inner59(n) => n.get_commitment(),
            VerkleNode::Inner60(n) => n.get_commitment(),
            VerkleNode::Inner61(n) => n.get_commitment(),
            VerkleNode::Inner62(n) => n.get_commitment(),
            VerkleNode::Inner63(n) => n.get_commitment(),
            VerkleNode::Inner64(n) => n.get_commitment(),
            VerkleNode::Inner65(n) => n.get_commitment(),
            VerkleNode::Inner66(n) => n.get_commitment(),
            VerkleNode::Inner67(n) => n.get_commitment(),
            VerkleNode::Inner68(n) => n.get_commitment(),
            VerkleNode::Inner69(n) => n.get_commitment(),
            VerkleNode::Inner70(n) => n.get_commitment(),
            VerkleNode::Inner71(n) => n.get_commitment(),
            VerkleNode::Inner72(n) => n.get_commitment(),
            VerkleNode::Inner73(n) => n.get_commitment(),
            VerkleNode::Inner74(n) => n.get_commitment(),
            VerkleNode::Inner75(n) => n.get_commitment(),
            VerkleNode::Inner76(n) => n.get_commitment(),
            VerkleNode::Inner77(n) => n.get_commitment(),
            VerkleNode::Inner78(n) => n.get_commitment(),
            VerkleNode::Inner79(n) => n.get_commitment(),
            VerkleNode::Inner80(n) => n.get_commitment(),
            VerkleNode::Inner81(n) => n.get_commitment(),
            VerkleNode::Inner82(n) => n.get_commitment(),
            VerkleNode::Inner83(n) => n.get_commitment(),
            VerkleNode::Inner84(n) => n.get_commitment(),
            VerkleNode::Inner85(n) => n.get_commitment(),
            VerkleNode::Inner86(n) => n.get_commitment(),
            VerkleNode::Inner87(n) => n.get_commitment(),
            VerkleNode::Inner88(n) => n.get_commitment(),
            VerkleNode::Inner89(n) => n.get_commitment(),
            VerkleNode::Inner90(n) => n.get_commitment(),
            VerkleNode::Inner91(n) => n.get_commitment(),
            VerkleNode::Inner92(n) => n.get_commitment(),
            VerkleNode::Inner93(n) => n.get_commitment(),
            VerkleNode::Inner94(n) => n.get_commitment(),
            VerkleNode::Inner95(n) => n.get_commitment(),
            VerkleNode::Inner96(n) => n.get_commitment(),
            VerkleNode::Inner97(n) => n.get_commitment(),
            VerkleNode::Inner98(n) => n.get_commitment(),
            VerkleNode::Inner99(n) => n.get_commitment(),
            VerkleNode::Inner100(n) => n.get_commitment(),
            VerkleNode::Inner101(n) => n.get_commitment(),
            VerkleNode::Inner102(n) => n.get_commitment(),
            VerkleNode::Inner103(n) => n.get_commitment(),
            VerkleNode::Inner104(n) => n.get_commitment(),
            VerkleNode::Inner105(n) => n.get_commitment(),
            VerkleNode::Inner106(n) => n.get_commitment(),
            VerkleNode::Inner107(n) => n.get_commitment(),
            VerkleNode::Inner108(n) => n.get_commitment(),
            VerkleNode::Inner109(n) => n.get_commitment(),
            VerkleNode::Inner110(n) => n.get_commitment(),
            VerkleNode::Inner111(n) => n.get_commitment(),
            VerkleNode::Inner112(n) => n.get_commitment(),
            VerkleNode::Inner113(n) => n.get_commitment(),
            VerkleNode::Inner114(n) => n.get_commitment(),
            VerkleNode::Inner115(n) => n.get_commitment(),
            VerkleNode::Inner116(n) => n.get_commitment(),
            VerkleNode::Inner117(n) => n.get_commitment(),
            VerkleNode::Inner118(n) => n.get_commitment(),
            VerkleNode::Inner119(n) => n.get_commitment(),
            VerkleNode::Inner120(n) => n.get_commitment(),
            VerkleNode::Inner121(n) => n.get_commitment(),
            VerkleNode::Inner122(n) => n.get_commitment(),
            VerkleNode::Inner123(n) => n.get_commitment(),
            VerkleNode::Inner124(n) => n.get_commitment(),
            VerkleNode::Inner125(n) => n.get_commitment(),
            VerkleNode::Inner126(n) => n.get_commitment(),
            VerkleNode::Inner127(n) => n.get_commitment(),
            VerkleNode::Inner128(n) => n.get_commitment(),
            VerkleNode::Inner129(n) => n.get_commitment(),
            VerkleNode::Inner130(n) => n.get_commitment(),
            VerkleNode::Inner131(n) => n.get_commitment(),
            VerkleNode::Inner132(n) => n.get_commitment(),
            VerkleNode::Inner133(n) => n.get_commitment(),
            VerkleNode::Inner134(n) => n.get_commitment(),
            VerkleNode::Inner135(n) => n.get_commitment(),
            VerkleNode::Inner136(n) => n.get_commitment(),
            VerkleNode::Inner137(n) => n.get_commitment(),
            VerkleNode::Inner138(n) => n.get_commitment(),
            VerkleNode::Inner139(n) => n.get_commitment(),
            VerkleNode::Inner140(n) => n.get_commitment(),
            VerkleNode::Inner141(n) => n.get_commitment(),
            VerkleNode::Inner142(n) => n.get_commitment(),
            VerkleNode::Inner143(n) => n.get_commitment(),
            VerkleNode::Inner144(n) => n.get_commitment(),
            VerkleNode::Inner145(n) => n.get_commitment(),
            VerkleNode::Inner146(n) => n.get_commitment(),
            VerkleNode::Inner147(n) => n.get_commitment(),
            VerkleNode::Inner148(n) => n.get_commitment(),
            VerkleNode::Inner149(n) => n.get_commitment(),
            VerkleNode::Inner150(n) => n.get_commitment(),
            VerkleNode::Inner151(n) => n.get_commitment(),
            VerkleNode::Inner152(n) => n.get_commitment(),
            VerkleNode::Inner153(n) => n.get_commitment(),
            VerkleNode::Inner154(n) => n.get_commitment(),
            VerkleNode::Inner155(n) => n.get_commitment(),
            VerkleNode::Inner156(n) => n.get_commitment(),
            VerkleNode::Inner157(n) => n.get_commitment(),
            VerkleNode::Inner158(n) => n.get_commitment(),
            VerkleNode::Inner159(n) => n.get_commitment(),
            VerkleNode::Inner160(n) => n.get_commitment(),
            VerkleNode::Inner161(n) => n.get_commitment(),
            VerkleNode::Inner162(n) => n.get_commitment(),
            VerkleNode::Inner163(n) => n.get_commitment(),
            VerkleNode::Inner164(n) => n.get_commitment(),
            VerkleNode::Inner165(n) => n.get_commitment(),
            VerkleNode::Inner166(n) => n.get_commitment(),
            VerkleNode::Inner167(n) => n.get_commitment(),
            VerkleNode::Inner168(n) => n.get_commitment(),
            VerkleNode::Inner169(n) => n.get_commitment(),
            VerkleNode::Inner170(n) => n.get_commitment(),
            VerkleNode::Inner171(n) => n.get_commitment(),
            VerkleNode::Inner172(n) => n.get_commitment(),
            VerkleNode::Inner173(n) => n.get_commitment(),
            VerkleNode::Inner174(n) => n.get_commitment(),
            VerkleNode::Inner175(n) => n.get_commitment(),
            VerkleNode::Inner176(n) => n.get_commitment(),
            VerkleNode::Inner177(n) => n.get_commitment(),
            VerkleNode::Inner178(n) => n.get_commitment(),
            VerkleNode::Inner179(n) => n.get_commitment(),
            VerkleNode::Inner180(n) => n.get_commitment(),
            VerkleNode::Inner181(n) => n.get_commitment(),
            VerkleNode::Inner182(n) => n.get_commitment(),
            VerkleNode::Inner183(n) => n.get_commitment(),
            VerkleNode::Inner184(n) => n.get_commitment(),
            VerkleNode::Inner185(n) => n.get_commitment(),
            VerkleNode::Inner186(n) => n.get_commitment(),
            VerkleNode::Inner187(n) => n.get_commitment(),
            VerkleNode::Inner188(n) => n.get_commitment(),
            VerkleNode::Inner189(n) => n.get_commitment(),
            VerkleNode::Inner190(n) => n.get_commitment(),
            VerkleNode::Inner191(n) => n.get_commitment(),
            VerkleNode::Inner192(n) => n.get_commitment(),
            VerkleNode::Inner193(n) => n.get_commitment(),
            VerkleNode::Inner194(n) => n.get_commitment(),
            VerkleNode::Inner195(n) => n.get_commitment(),
            VerkleNode::Inner196(n) => n.get_commitment(),
            VerkleNode::Inner197(n) => n.get_commitment(),
            VerkleNode::Inner198(n) => n.get_commitment(),
            VerkleNode::Inner199(n) => n.get_commitment(),
            VerkleNode::Inner200(n) => n.get_commitment(),
            VerkleNode::Inner201(n) => n.get_commitment(),
            VerkleNode::Inner202(n) => n.get_commitment(),
            VerkleNode::Inner203(n) => n.get_commitment(),
            VerkleNode::Inner204(n) => n.get_commitment(),
            VerkleNode::Inner205(n) => n.get_commitment(),
            VerkleNode::Inner206(n) => n.get_commitment(),
            VerkleNode::Inner207(n) => n.get_commitment(),
            VerkleNode::Inner208(n) => n.get_commitment(),
            VerkleNode::Inner209(n) => n.get_commitment(),
            VerkleNode::Inner210(n) => n.get_commitment(),
            VerkleNode::Inner211(n) => n.get_commitment(),
            VerkleNode::Inner212(n) => n.get_commitment(),
            VerkleNode::Inner213(n) => n.get_commitment(),
            VerkleNode::Inner214(n) => n.get_commitment(),
            VerkleNode::Inner215(n) => n.get_commitment(),
            VerkleNode::Inner216(n) => n.get_commitment(),
            VerkleNode::Inner217(n) => n.get_commitment(),
            VerkleNode::Inner218(n) => n.get_commitment(),
            VerkleNode::Inner219(n) => n.get_commitment(),
            VerkleNode::Inner220(n) => n.get_commitment(),
            VerkleNode::Inner221(n) => n.get_commitment(),
            VerkleNode::Inner222(n) => n.get_commitment(),
            VerkleNode::Inner223(n) => n.get_commitment(),
            VerkleNode::Inner224(n) => n.get_commitment(),
            VerkleNode::Inner225(n) => n.get_commitment(),
            VerkleNode::Inner226(n) => n.get_commitment(),
            VerkleNode::Inner227(n) => n.get_commitment(),
            VerkleNode::Inner228(n) => n.get_commitment(),
            VerkleNode::Inner229(n) => n.get_commitment(),
            VerkleNode::Inner230(n) => n.get_commitment(),
            VerkleNode::Inner231(n) => n.get_commitment(),
            VerkleNode::Inner232(n) => n.get_commitment(),
            VerkleNode::Inner233(n) => n.get_commitment(),
            VerkleNode::Inner234(n) => n.get_commitment(),
            VerkleNode::Inner235(n) => n.get_commitment(),
            VerkleNode::Inner236(n) => n.get_commitment(),
            VerkleNode::Inner237(n) => n.get_commitment(),
            VerkleNode::Inner238(n) => n.get_commitment(),
            VerkleNode::Inner239(n) => n.get_commitment(),
            VerkleNode::Inner240(n) => n.get_commitment(),
            VerkleNode::Inner241(n) => n.get_commitment(),
            VerkleNode::Inner242(n) => n.get_commitment(),
            VerkleNode::Inner243(n) => n.get_commitment(),
            VerkleNode::Inner244(n) => n.get_commitment(),
            VerkleNode::Inner245(n) => n.get_commitment(),
            VerkleNode::Inner246(n) => n.get_commitment(),
            VerkleNode::Inner247(n) => n.get_commitment(),
            VerkleNode::Inner248(n) => n.get_commitment(),
            VerkleNode::Inner249(n) => n.get_commitment(),
            VerkleNode::Inner250(n) => n.get_commitment(),
            VerkleNode::Inner251(n) => n.get_commitment(),
            VerkleNode::Inner252(n) => n.get_commitment(),
            VerkleNode::Inner253(n) => n.get_commitment(),
            VerkleNode::Inner254(n) => n.get_commitment(),
            VerkleNode::Inner255(n) => n.get_commitment(),
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
            VerkleNode::Leaf17(n) => n.get_commitment(),
            VerkleNode::Leaf18(n) => n.get_commitment(),
            VerkleNode::Leaf19(n) => n.get_commitment(),
            VerkleNode::Leaf20(n) => n.get_commitment(),
            VerkleNode::Leaf21(n) => n.get_commitment(),
            VerkleNode::Leaf22(n) => n.get_commitment(),
            VerkleNode::Leaf23(n) => n.get_commitment(),
            VerkleNode::Leaf24(n) => n.get_commitment(),
            VerkleNode::Leaf25(n) => n.get_commitment(),
            VerkleNode::Leaf26(n) => n.get_commitment(),
            VerkleNode::Leaf27(n) => n.get_commitment(),
            VerkleNode::Leaf28(n) => n.get_commitment(),
            VerkleNode::Leaf29(n) => n.get_commitment(),
            VerkleNode::Leaf30(n) => n.get_commitment(),
            VerkleNode::Leaf31(n) => n.get_commitment(),
            VerkleNode::Leaf32(n) => n.get_commitment(),
            VerkleNode::Leaf33(n) => n.get_commitment(),
            VerkleNode::Leaf34(n) => n.get_commitment(),
            VerkleNode::Leaf35(n) => n.get_commitment(),
            VerkleNode::Leaf36(n) => n.get_commitment(),
            VerkleNode::Leaf37(n) => n.get_commitment(),
            VerkleNode::Leaf38(n) => n.get_commitment(),
            VerkleNode::Leaf39(n) => n.get_commitment(),
            VerkleNode::Leaf40(n) => n.get_commitment(),
            VerkleNode::Leaf41(n) => n.get_commitment(),
            VerkleNode::Leaf42(n) => n.get_commitment(),
            VerkleNode::Leaf43(n) => n.get_commitment(),
            VerkleNode::Leaf44(n) => n.get_commitment(),
            VerkleNode::Leaf45(n) => n.get_commitment(),
            VerkleNode::Leaf46(n) => n.get_commitment(),
            VerkleNode::Leaf47(n) => n.get_commitment(),
            VerkleNode::Leaf48(n) => n.get_commitment(),
            VerkleNode::Leaf49(n) => n.get_commitment(),
            VerkleNode::Leaf50(n) => n.get_commitment(),
            VerkleNode::Leaf51(n) => n.get_commitment(),
            VerkleNode::Leaf52(n) => n.get_commitment(),
            VerkleNode::Leaf53(n) => n.get_commitment(),
            VerkleNode::Leaf54(n) => n.get_commitment(),
            VerkleNode::Leaf55(n) => n.get_commitment(),
            VerkleNode::Leaf56(n) => n.get_commitment(),
            VerkleNode::Leaf57(n) => n.get_commitment(),
            VerkleNode::Leaf58(n) => n.get_commitment(),
            VerkleNode::Leaf59(n) => n.get_commitment(),
            VerkleNode::Leaf60(n) => n.get_commitment(),
            VerkleNode::Leaf61(n) => n.get_commitment(),
            VerkleNode::Leaf62(n) => n.get_commitment(),
            VerkleNode::Leaf63(n) => n.get_commitment(),
            VerkleNode::Leaf64(n) => n.get_commitment(),
            VerkleNode::Leaf65(n) => n.get_commitment(),
            VerkleNode::Leaf66(n) => n.get_commitment(),
            VerkleNode::Leaf67(n) => n.get_commitment(),
            VerkleNode::Leaf68(n) => n.get_commitment(),
            VerkleNode::Leaf69(n) => n.get_commitment(),
            VerkleNode::Leaf70(n) => n.get_commitment(),
            VerkleNode::Leaf71(n) => n.get_commitment(),
            VerkleNode::Leaf72(n) => n.get_commitment(),
            VerkleNode::Leaf73(n) => n.get_commitment(),
            VerkleNode::Leaf74(n) => n.get_commitment(),
            VerkleNode::Leaf75(n) => n.get_commitment(),
            VerkleNode::Leaf76(n) => n.get_commitment(),
            VerkleNode::Leaf77(n) => n.get_commitment(),
            VerkleNode::Leaf78(n) => n.get_commitment(),
            VerkleNode::Leaf79(n) => n.get_commitment(),
            VerkleNode::Leaf80(n) => n.get_commitment(),
            VerkleNode::Leaf81(n) => n.get_commitment(),
            VerkleNode::Leaf82(n) => n.get_commitment(),
            VerkleNode::Leaf83(n) => n.get_commitment(),
            VerkleNode::Leaf84(n) => n.get_commitment(),
            VerkleNode::Leaf85(n) => n.get_commitment(),
            VerkleNode::Leaf86(n) => n.get_commitment(),
            VerkleNode::Leaf87(n) => n.get_commitment(),
            VerkleNode::Leaf88(n) => n.get_commitment(),
            VerkleNode::Leaf89(n) => n.get_commitment(),
            VerkleNode::Leaf90(n) => n.get_commitment(),
            VerkleNode::Leaf91(n) => n.get_commitment(),
            VerkleNode::Leaf92(n) => n.get_commitment(),
            VerkleNode::Leaf93(n) => n.get_commitment(),
            VerkleNode::Leaf94(n) => n.get_commitment(),
            VerkleNode::Leaf95(n) => n.get_commitment(),
            VerkleNode::Leaf96(n) => n.get_commitment(),
            VerkleNode::Leaf97(n) => n.get_commitment(),
            VerkleNode::Leaf98(n) => n.get_commitment(),
            VerkleNode::Leaf99(n) => n.get_commitment(),
            VerkleNode::Leaf100(n) => n.get_commitment(),
            VerkleNode::Leaf101(n) => n.get_commitment(),
            VerkleNode::Leaf102(n) => n.get_commitment(),
            VerkleNode::Leaf103(n) => n.get_commitment(),
            VerkleNode::Leaf104(n) => n.get_commitment(),
            VerkleNode::Leaf105(n) => n.get_commitment(),
            VerkleNode::Leaf106(n) => n.get_commitment(),
            VerkleNode::Leaf107(n) => n.get_commitment(),
            VerkleNode::Leaf108(n) => n.get_commitment(),
            VerkleNode::Leaf109(n) => n.get_commitment(),
            VerkleNode::Leaf110(n) => n.get_commitment(),
            VerkleNode::Leaf111(n) => n.get_commitment(),
            VerkleNode::Leaf112(n) => n.get_commitment(),
            VerkleNode::Leaf113(n) => n.get_commitment(),
            VerkleNode::Leaf114(n) => n.get_commitment(),
            VerkleNode::Leaf115(n) => n.get_commitment(),
            VerkleNode::Leaf116(n) => n.get_commitment(),
            VerkleNode::Leaf117(n) => n.get_commitment(),
            VerkleNode::Leaf118(n) => n.get_commitment(),
            VerkleNode::Leaf119(n) => n.get_commitment(),
            VerkleNode::Leaf120(n) => n.get_commitment(),
            VerkleNode::Leaf121(n) => n.get_commitment(),
            VerkleNode::Leaf122(n) => n.get_commitment(),
            VerkleNode::Leaf123(n) => n.get_commitment(),
            VerkleNode::Leaf124(n) => n.get_commitment(),
            VerkleNode::Leaf125(n) => n.get_commitment(),
            VerkleNode::Leaf126(n) => n.get_commitment(),
            VerkleNode::Leaf127(n) => n.get_commitment(),
            VerkleNode::Leaf128(n) => n.get_commitment(),
            VerkleNode::Leaf129(n) => n.get_commitment(),
            VerkleNode::Leaf130(n) => n.get_commitment(),
            VerkleNode::Leaf131(n) => n.get_commitment(),
            VerkleNode::Leaf132(n) => n.get_commitment(),
            VerkleNode::Leaf133(n) => n.get_commitment(),
            VerkleNode::Leaf134(n) => n.get_commitment(),
            VerkleNode::Leaf135(n) => n.get_commitment(),
            VerkleNode::Leaf136(n) => n.get_commitment(),
            VerkleNode::Leaf137(n) => n.get_commitment(),
            VerkleNode::Leaf138(n) => n.get_commitment(),
            VerkleNode::Leaf139(n) => n.get_commitment(),
            VerkleNode::Leaf140(n) => n.get_commitment(),
            VerkleNode::Leaf141(n) => n.get_commitment(),
            VerkleNode::Leaf142(n) => n.get_commitment(),
            VerkleNode::Leaf143(n) => n.get_commitment(),
            VerkleNode::Leaf144(n) => n.get_commitment(),
            VerkleNode::Leaf145(n) => n.get_commitment(),
            VerkleNode::Leaf146(n) => n.get_commitment(),
            VerkleNode::Leaf147(n) => n.get_commitment(),
            VerkleNode::Leaf148(n) => n.get_commitment(),
            VerkleNode::Leaf149(n) => n.get_commitment(),
            VerkleNode::Leaf150(n) => n.get_commitment(),
            VerkleNode::Leaf151(n) => n.get_commitment(),
            VerkleNode::Leaf152(n) => n.get_commitment(),
            VerkleNode::Leaf153(n) => n.get_commitment(),
            VerkleNode::Leaf154(n) => n.get_commitment(),
            VerkleNode::Leaf155(n) => n.get_commitment(),
            VerkleNode::Leaf156(n) => n.get_commitment(),
            VerkleNode::Leaf157(n) => n.get_commitment(),
            VerkleNode::Leaf158(n) => n.get_commitment(),
            VerkleNode::Leaf159(n) => n.get_commitment(),
            VerkleNode::Leaf160(n) => n.get_commitment(),
            VerkleNode::Leaf161(n) => n.get_commitment(),
            VerkleNode::Leaf162(n) => n.get_commitment(),
            VerkleNode::Leaf163(n) => n.get_commitment(),
            VerkleNode::Leaf164(n) => n.get_commitment(),
            VerkleNode::Leaf165(n) => n.get_commitment(),
            VerkleNode::Leaf166(n) => n.get_commitment(),
            VerkleNode::Leaf167(n) => n.get_commitment(),
            VerkleNode::Leaf168(n) => n.get_commitment(),
            VerkleNode::Leaf169(n) => n.get_commitment(),
            VerkleNode::Leaf170(n) => n.get_commitment(),
            VerkleNode::Leaf171(n) => n.get_commitment(),
            VerkleNode::Leaf172(n) => n.get_commitment(),
            VerkleNode::Leaf173(n) => n.get_commitment(),
            VerkleNode::Leaf174(n) => n.get_commitment(),
            VerkleNode::Leaf175(n) => n.get_commitment(),
            VerkleNode::Leaf176(n) => n.get_commitment(),
            VerkleNode::Leaf177(n) => n.get_commitment(),
            VerkleNode::Leaf178(n) => n.get_commitment(),
            VerkleNode::Leaf179(n) => n.get_commitment(),
            VerkleNode::Leaf180(n) => n.get_commitment(),
            VerkleNode::Leaf181(n) => n.get_commitment(),
            VerkleNode::Leaf182(n) => n.get_commitment(),
            VerkleNode::Leaf183(n) => n.get_commitment(),
            VerkleNode::Leaf184(n) => n.get_commitment(),
            VerkleNode::Leaf185(n) => n.get_commitment(),
            VerkleNode::Leaf186(n) => n.get_commitment(),
            VerkleNode::Leaf187(n) => n.get_commitment(),
            VerkleNode::Leaf188(n) => n.get_commitment(),
            VerkleNode::Leaf189(n) => n.get_commitment(),
            VerkleNode::Leaf190(n) => n.get_commitment(),
            VerkleNode::Leaf191(n) => n.get_commitment(),
            VerkleNode::Leaf192(n) => n.get_commitment(),
            VerkleNode::Leaf193(n) => n.get_commitment(),
            VerkleNode::Leaf194(n) => n.get_commitment(),
            VerkleNode::Leaf195(n) => n.get_commitment(),
            VerkleNode::Leaf196(n) => n.get_commitment(),
            VerkleNode::Leaf197(n) => n.get_commitment(),
            VerkleNode::Leaf198(n) => n.get_commitment(),
            VerkleNode::Leaf199(n) => n.get_commitment(),
            VerkleNode::Leaf200(n) => n.get_commitment(),
            VerkleNode::Leaf201(n) => n.get_commitment(),
            VerkleNode::Leaf202(n) => n.get_commitment(),
            VerkleNode::Leaf203(n) => n.get_commitment(),
            VerkleNode::Leaf204(n) => n.get_commitment(),
            VerkleNode::Leaf205(n) => n.get_commitment(),
            VerkleNode::Leaf206(n) => n.get_commitment(),
            VerkleNode::Leaf207(n) => n.get_commitment(),
            VerkleNode::Leaf208(n) => n.get_commitment(),
            VerkleNode::Leaf209(n) => n.get_commitment(),
            VerkleNode::Leaf210(n) => n.get_commitment(),
            VerkleNode::Leaf211(n) => n.get_commitment(),
            VerkleNode::Leaf212(n) => n.get_commitment(),
            VerkleNode::Leaf213(n) => n.get_commitment(),
            VerkleNode::Leaf214(n) => n.get_commitment(),
            VerkleNode::Leaf215(n) => n.get_commitment(),
            VerkleNode::Leaf216(n) => n.get_commitment(),
            VerkleNode::Leaf217(n) => n.get_commitment(),
            VerkleNode::Leaf218(n) => n.get_commitment(),
            VerkleNode::Leaf219(n) => n.get_commitment(),
            VerkleNode::Leaf220(n) => n.get_commitment(),
            VerkleNode::Leaf221(n) => n.get_commitment(),
            VerkleNode::Leaf222(n) => n.get_commitment(),
            VerkleNode::Leaf223(n) => n.get_commitment(),
            VerkleNode::Leaf224(n) => n.get_commitment(),
            VerkleNode::Leaf225(n) => n.get_commitment(),
            VerkleNode::Leaf226(n) => n.get_commitment(),
            VerkleNode::Leaf227(n) => n.get_commitment(),
            VerkleNode::Leaf228(n) => n.get_commitment(),
            VerkleNode::Leaf229(n) => n.get_commitment(),
            VerkleNode::Leaf230(n) => n.get_commitment(),
            VerkleNode::Leaf231(n) => n.get_commitment(),
            VerkleNode::Leaf232(n) => n.get_commitment(),
            VerkleNode::Leaf233(n) => n.get_commitment(),
            VerkleNode::Leaf234(n) => n.get_commitment(),
            VerkleNode::Leaf235(n) => n.get_commitment(),
            VerkleNode::Leaf236(n) => n.get_commitment(),
            VerkleNode::Leaf237(n) => n.get_commitment(),
            VerkleNode::Leaf238(n) => n.get_commitment(),
            VerkleNode::Leaf239(n) => n.get_commitment(),
            VerkleNode::Leaf240(n) => n.get_commitment(),
            VerkleNode::Leaf241(n) => n.get_commitment(),
            VerkleNode::Leaf242(n) => n.get_commitment(),
            VerkleNode::Leaf243(n) => n.get_commitment(),
            VerkleNode::Leaf244(n) => n.get_commitment(),
            VerkleNode::Leaf245(n) => n.get_commitment(),
            VerkleNode::Leaf246(n) => n.get_commitment(),
            VerkleNode::Leaf247(n) => n.get_commitment(),
            VerkleNode::Leaf248(n) => n.get_commitment(),
            VerkleNode::Leaf249(n) => n.get_commitment(),
            VerkleNode::Leaf250(n) => n.get_commitment(),
            VerkleNode::Leaf251(n) => n.get_commitment(),
            VerkleNode::Leaf252(n) => n.get_commitment(),
            VerkleNode::Leaf253(n) => n.get_commitment(),
            VerkleNode::Leaf254(n) => n.get_commitment(),
            VerkleNode::Leaf255(n) => n.get_commitment(),
            VerkleNode::Leaf256(n) => n.get_commitment(),
            VerkleNode::LeafDelta(n) => n.get_commitment(),
        }
    }

    fn set_commitment(&mut self, cache: Self::Commitment) -> BTResult<(), Error> {
        match self {
            VerkleNode::Empty(n) => n.set_commitment(cache),
            VerkleNode::Inner1(n) => n.set_commitment(cache),
            VerkleNode::Inner2(n) => n.set_commitment(cache),
            VerkleNode::Inner3(n) => n.set_commitment(cache),
            VerkleNode::Inner4(n) => n.set_commitment(cache),
            VerkleNode::Inner5(n) => n.set_commitment(cache),
            VerkleNode::Inner6(n) => n.set_commitment(cache),
            VerkleNode::Inner7(n) => n.set_commitment(cache),
            VerkleNode::Inner8(n) => n.set_commitment(cache),
            VerkleNode::Inner9(n) => n.set_commitment(cache),
            VerkleNode::Inner10(n) => n.set_commitment(cache),
            VerkleNode::Inner11(n) => n.set_commitment(cache),
            VerkleNode::Inner12(n) => n.set_commitment(cache),
            VerkleNode::Inner13(n) => n.set_commitment(cache),
            VerkleNode::Inner14(n) => n.set_commitment(cache),
            VerkleNode::Inner15(n) => n.set_commitment(cache),
            VerkleNode::Inner16(n) => n.set_commitment(cache),
            VerkleNode::Inner17(n) => n.set_commitment(cache),
            VerkleNode::Inner18(n) => n.set_commitment(cache),
            VerkleNode::Inner19(n) => n.set_commitment(cache),
            VerkleNode::Inner20(n) => n.set_commitment(cache),
            VerkleNode::Inner21(n) => n.set_commitment(cache),
            VerkleNode::Inner22(n) => n.set_commitment(cache),
            VerkleNode::Inner23(n) => n.set_commitment(cache),
            VerkleNode::Inner24(n) => n.set_commitment(cache),
            VerkleNode::Inner25(n) => n.set_commitment(cache),
            VerkleNode::Inner26(n) => n.set_commitment(cache),
            VerkleNode::Inner27(n) => n.set_commitment(cache),
            VerkleNode::Inner28(n) => n.set_commitment(cache),
            VerkleNode::Inner29(n) => n.set_commitment(cache),
            VerkleNode::Inner30(n) => n.set_commitment(cache),
            VerkleNode::Inner31(n) => n.set_commitment(cache),
            VerkleNode::Inner32(n) => n.set_commitment(cache),
            VerkleNode::Inner33(n) => n.set_commitment(cache),
            VerkleNode::Inner34(n) => n.set_commitment(cache),
            VerkleNode::Inner35(n) => n.set_commitment(cache),
            VerkleNode::Inner36(n) => n.set_commitment(cache),
            VerkleNode::Inner37(n) => n.set_commitment(cache),
            VerkleNode::Inner38(n) => n.set_commitment(cache),
            VerkleNode::Inner39(n) => n.set_commitment(cache),
            VerkleNode::Inner40(n) => n.set_commitment(cache),
            VerkleNode::Inner41(n) => n.set_commitment(cache),
            VerkleNode::Inner42(n) => n.set_commitment(cache),
            VerkleNode::Inner43(n) => n.set_commitment(cache),
            VerkleNode::Inner44(n) => n.set_commitment(cache),
            VerkleNode::Inner45(n) => n.set_commitment(cache),
            VerkleNode::Inner46(n) => n.set_commitment(cache),
            VerkleNode::Inner47(n) => n.set_commitment(cache),
            VerkleNode::Inner48(n) => n.set_commitment(cache),
            VerkleNode::Inner49(n) => n.set_commitment(cache),
            VerkleNode::Inner50(n) => n.set_commitment(cache),
            VerkleNode::Inner51(n) => n.set_commitment(cache),
            VerkleNode::Inner52(n) => n.set_commitment(cache),
            VerkleNode::Inner53(n) => n.set_commitment(cache),
            VerkleNode::Inner54(n) => n.set_commitment(cache),
            VerkleNode::Inner55(n) => n.set_commitment(cache),
            VerkleNode::Inner56(n) => n.set_commitment(cache),
            VerkleNode::Inner57(n) => n.set_commitment(cache),
            VerkleNode::Inner58(n) => n.set_commitment(cache),
            VerkleNode::Inner59(n) => n.set_commitment(cache),
            VerkleNode::Inner60(n) => n.set_commitment(cache),
            VerkleNode::Inner61(n) => n.set_commitment(cache),
            VerkleNode::Inner62(n) => n.set_commitment(cache),
            VerkleNode::Inner63(n) => n.set_commitment(cache),
            VerkleNode::Inner64(n) => n.set_commitment(cache),
            VerkleNode::Inner65(n) => n.set_commitment(cache),
            VerkleNode::Inner66(n) => n.set_commitment(cache),
            VerkleNode::Inner67(n) => n.set_commitment(cache),
            VerkleNode::Inner68(n) => n.set_commitment(cache),
            VerkleNode::Inner69(n) => n.set_commitment(cache),
            VerkleNode::Inner70(n) => n.set_commitment(cache),
            VerkleNode::Inner71(n) => n.set_commitment(cache),
            VerkleNode::Inner72(n) => n.set_commitment(cache),
            VerkleNode::Inner73(n) => n.set_commitment(cache),
            VerkleNode::Inner74(n) => n.set_commitment(cache),
            VerkleNode::Inner75(n) => n.set_commitment(cache),
            VerkleNode::Inner76(n) => n.set_commitment(cache),
            VerkleNode::Inner77(n) => n.set_commitment(cache),
            VerkleNode::Inner78(n) => n.set_commitment(cache),
            VerkleNode::Inner79(n) => n.set_commitment(cache),
            VerkleNode::Inner80(n) => n.set_commitment(cache),
            VerkleNode::Inner81(n) => n.set_commitment(cache),
            VerkleNode::Inner82(n) => n.set_commitment(cache),
            VerkleNode::Inner83(n) => n.set_commitment(cache),
            VerkleNode::Inner84(n) => n.set_commitment(cache),
            VerkleNode::Inner85(n) => n.set_commitment(cache),
            VerkleNode::Inner86(n) => n.set_commitment(cache),
            VerkleNode::Inner87(n) => n.set_commitment(cache),
            VerkleNode::Inner88(n) => n.set_commitment(cache),
            VerkleNode::Inner89(n) => n.set_commitment(cache),
            VerkleNode::Inner90(n) => n.set_commitment(cache),
            VerkleNode::Inner91(n) => n.set_commitment(cache),
            VerkleNode::Inner92(n) => n.set_commitment(cache),
            VerkleNode::Inner93(n) => n.set_commitment(cache),
            VerkleNode::Inner94(n) => n.set_commitment(cache),
            VerkleNode::Inner95(n) => n.set_commitment(cache),
            VerkleNode::Inner96(n) => n.set_commitment(cache),
            VerkleNode::Inner97(n) => n.set_commitment(cache),
            VerkleNode::Inner98(n) => n.set_commitment(cache),
            VerkleNode::Inner99(n) => n.set_commitment(cache),
            VerkleNode::Inner100(n) => n.set_commitment(cache),
            VerkleNode::Inner101(n) => n.set_commitment(cache),
            VerkleNode::Inner102(n) => n.set_commitment(cache),
            VerkleNode::Inner103(n) => n.set_commitment(cache),
            VerkleNode::Inner104(n) => n.set_commitment(cache),
            VerkleNode::Inner105(n) => n.set_commitment(cache),
            VerkleNode::Inner106(n) => n.set_commitment(cache),
            VerkleNode::Inner107(n) => n.set_commitment(cache),
            VerkleNode::Inner108(n) => n.set_commitment(cache),
            VerkleNode::Inner109(n) => n.set_commitment(cache),
            VerkleNode::Inner110(n) => n.set_commitment(cache),
            VerkleNode::Inner111(n) => n.set_commitment(cache),
            VerkleNode::Inner112(n) => n.set_commitment(cache),
            VerkleNode::Inner113(n) => n.set_commitment(cache),
            VerkleNode::Inner114(n) => n.set_commitment(cache),
            VerkleNode::Inner115(n) => n.set_commitment(cache),
            VerkleNode::Inner116(n) => n.set_commitment(cache),
            VerkleNode::Inner117(n) => n.set_commitment(cache),
            VerkleNode::Inner118(n) => n.set_commitment(cache),
            VerkleNode::Inner119(n) => n.set_commitment(cache),
            VerkleNode::Inner120(n) => n.set_commitment(cache),
            VerkleNode::Inner121(n) => n.set_commitment(cache),
            VerkleNode::Inner122(n) => n.set_commitment(cache),
            VerkleNode::Inner123(n) => n.set_commitment(cache),
            VerkleNode::Inner124(n) => n.set_commitment(cache),
            VerkleNode::Inner125(n) => n.set_commitment(cache),
            VerkleNode::Inner126(n) => n.set_commitment(cache),
            VerkleNode::Inner127(n) => n.set_commitment(cache),
            VerkleNode::Inner128(n) => n.set_commitment(cache),
            VerkleNode::Inner129(n) => n.set_commitment(cache),
            VerkleNode::Inner130(n) => n.set_commitment(cache),
            VerkleNode::Inner131(n) => n.set_commitment(cache),
            VerkleNode::Inner132(n) => n.set_commitment(cache),
            VerkleNode::Inner133(n) => n.set_commitment(cache),
            VerkleNode::Inner134(n) => n.set_commitment(cache),
            VerkleNode::Inner135(n) => n.set_commitment(cache),
            VerkleNode::Inner136(n) => n.set_commitment(cache),
            VerkleNode::Inner137(n) => n.set_commitment(cache),
            VerkleNode::Inner138(n) => n.set_commitment(cache),
            VerkleNode::Inner139(n) => n.set_commitment(cache),
            VerkleNode::Inner140(n) => n.set_commitment(cache),
            VerkleNode::Inner141(n) => n.set_commitment(cache),
            VerkleNode::Inner142(n) => n.set_commitment(cache),
            VerkleNode::Inner143(n) => n.set_commitment(cache),
            VerkleNode::Inner144(n) => n.set_commitment(cache),
            VerkleNode::Inner145(n) => n.set_commitment(cache),
            VerkleNode::Inner146(n) => n.set_commitment(cache),
            VerkleNode::Inner147(n) => n.set_commitment(cache),
            VerkleNode::Inner148(n) => n.set_commitment(cache),
            VerkleNode::Inner149(n) => n.set_commitment(cache),
            VerkleNode::Inner150(n) => n.set_commitment(cache),
            VerkleNode::Inner151(n) => n.set_commitment(cache),
            VerkleNode::Inner152(n) => n.set_commitment(cache),
            VerkleNode::Inner153(n) => n.set_commitment(cache),
            VerkleNode::Inner154(n) => n.set_commitment(cache),
            VerkleNode::Inner155(n) => n.set_commitment(cache),
            VerkleNode::Inner156(n) => n.set_commitment(cache),
            VerkleNode::Inner157(n) => n.set_commitment(cache),
            VerkleNode::Inner158(n) => n.set_commitment(cache),
            VerkleNode::Inner159(n) => n.set_commitment(cache),
            VerkleNode::Inner160(n) => n.set_commitment(cache),
            VerkleNode::Inner161(n) => n.set_commitment(cache),
            VerkleNode::Inner162(n) => n.set_commitment(cache),
            VerkleNode::Inner163(n) => n.set_commitment(cache),
            VerkleNode::Inner164(n) => n.set_commitment(cache),
            VerkleNode::Inner165(n) => n.set_commitment(cache),
            VerkleNode::Inner166(n) => n.set_commitment(cache),
            VerkleNode::Inner167(n) => n.set_commitment(cache),
            VerkleNode::Inner168(n) => n.set_commitment(cache),
            VerkleNode::Inner169(n) => n.set_commitment(cache),
            VerkleNode::Inner170(n) => n.set_commitment(cache),
            VerkleNode::Inner171(n) => n.set_commitment(cache),
            VerkleNode::Inner172(n) => n.set_commitment(cache),
            VerkleNode::Inner173(n) => n.set_commitment(cache),
            VerkleNode::Inner174(n) => n.set_commitment(cache),
            VerkleNode::Inner175(n) => n.set_commitment(cache),
            VerkleNode::Inner176(n) => n.set_commitment(cache),
            VerkleNode::Inner177(n) => n.set_commitment(cache),
            VerkleNode::Inner178(n) => n.set_commitment(cache),
            VerkleNode::Inner179(n) => n.set_commitment(cache),
            VerkleNode::Inner180(n) => n.set_commitment(cache),
            VerkleNode::Inner181(n) => n.set_commitment(cache),
            VerkleNode::Inner182(n) => n.set_commitment(cache),
            VerkleNode::Inner183(n) => n.set_commitment(cache),
            VerkleNode::Inner184(n) => n.set_commitment(cache),
            VerkleNode::Inner185(n) => n.set_commitment(cache),
            VerkleNode::Inner186(n) => n.set_commitment(cache),
            VerkleNode::Inner187(n) => n.set_commitment(cache),
            VerkleNode::Inner188(n) => n.set_commitment(cache),
            VerkleNode::Inner189(n) => n.set_commitment(cache),
            VerkleNode::Inner190(n) => n.set_commitment(cache),
            VerkleNode::Inner191(n) => n.set_commitment(cache),
            VerkleNode::Inner192(n) => n.set_commitment(cache),
            VerkleNode::Inner193(n) => n.set_commitment(cache),
            VerkleNode::Inner194(n) => n.set_commitment(cache),
            VerkleNode::Inner195(n) => n.set_commitment(cache),
            VerkleNode::Inner196(n) => n.set_commitment(cache),
            VerkleNode::Inner197(n) => n.set_commitment(cache),
            VerkleNode::Inner198(n) => n.set_commitment(cache),
            VerkleNode::Inner199(n) => n.set_commitment(cache),
            VerkleNode::Inner200(n) => n.set_commitment(cache),
            VerkleNode::Inner201(n) => n.set_commitment(cache),
            VerkleNode::Inner202(n) => n.set_commitment(cache),
            VerkleNode::Inner203(n) => n.set_commitment(cache),
            VerkleNode::Inner204(n) => n.set_commitment(cache),
            VerkleNode::Inner205(n) => n.set_commitment(cache),
            VerkleNode::Inner206(n) => n.set_commitment(cache),
            VerkleNode::Inner207(n) => n.set_commitment(cache),
            VerkleNode::Inner208(n) => n.set_commitment(cache),
            VerkleNode::Inner209(n) => n.set_commitment(cache),
            VerkleNode::Inner210(n) => n.set_commitment(cache),
            VerkleNode::Inner211(n) => n.set_commitment(cache),
            VerkleNode::Inner212(n) => n.set_commitment(cache),
            VerkleNode::Inner213(n) => n.set_commitment(cache),
            VerkleNode::Inner214(n) => n.set_commitment(cache),
            VerkleNode::Inner215(n) => n.set_commitment(cache),
            VerkleNode::Inner216(n) => n.set_commitment(cache),
            VerkleNode::Inner217(n) => n.set_commitment(cache),
            VerkleNode::Inner218(n) => n.set_commitment(cache),
            VerkleNode::Inner219(n) => n.set_commitment(cache),
            VerkleNode::Inner220(n) => n.set_commitment(cache),
            VerkleNode::Inner221(n) => n.set_commitment(cache),
            VerkleNode::Inner222(n) => n.set_commitment(cache),
            VerkleNode::Inner223(n) => n.set_commitment(cache),
            VerkleNode::Inner224(n) => n.set_commitment(cache),
            VerkleNode::Inner225(n) => n.set_commitment(cache),
            VerkleNode::Inner226(n) => n.set_commitment(cache),
            VerkleNode::Inner227(n) => n.set_commitment(cache),
            VerkleNode::Inner228(n) => n.set_commitment(cache),
            VerkleNode::Inner229(n) => n.set_commitment(cache),
            VerkleNode::Inner230(n) => n.set_commitment(cache),
            VerkleNode::Inner231(n) => n.set_commitment(cache),
            VerkleNode::Inner232(n) => n.set_commitment(cache),
            VerkleNode::Inner233(n) => n.set_commitment(cache),
            VerkleNode::Inner234(n) => n.set_commitment(cache),
            VerkleNode::Inner235(n) => n.set_commitment(cache),
            VerkleNode::Inner236(n) => n.set_commitment(cache),
            VerkleNode::Inner237(n) => n.set_commitment(cache),
            VerkleNode::Inner238(n) => n.set_commitment(cache),
            VerkleNode::Inner239(n) => n.set_commitment(cache),
            VerkleNode::Inner240(n) => n.set_commitment(cache),
            VerkleNode::Inner241(n) => n.set_commitment(cache),
            VerkleNode::Inner242(n) => n.set_commitment(cache),
            VerkleNode::Inner243(n) => n.set_commitment(cache),
            VerkleNode::Inner244(n) => n.set_commitment(cache),
            VerkleNode::Inner245(n) => n.set_commitment(cache),
            VerkleNode::Inner246(n) => n.set_commitment(cache),
            VerkleNode::Inner247(n) => n.set_commitment(cache),
            VerkleNode::Inner248(n) => n.set_commitment(cache),
            VerkleNode::Inner249(n) => n.set_commitment(cache),
            VerkleNode::Inner250(n) => n.set_commitment(cache),
            VerkleNode::Inner251(n) => n.set_commitment(cache),
            VerkleNode::Inner252(n) => n.set_commitment(cache),
            VerkleNode::Inner253(n) => n.set_commitment(cache),
            VerkleNode::Inner254(n) => n.set_commitment(cache),
            VerkleNode::Inner255(n) => n.set_commitment(cache),
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
            VerkleNode::Leaf17(n) => n.set_commitment(cache),
            VerkleNode::Leaf18(n) => n.set_commitment(cache),
            VerkleNode::Leaf19(n) => n.set_commitment(cache),
            VerkleNode::Leaf20(n) => n.set_commitment(cache),
            VerkleNode::Leaf21(n) => n.set_commitment(cache),
            VerkleNode::Leaf22(n) => n.set_commitment(cache),
            VerkleNode::Leaf23(n) => n.set_commitment(cache),
            VerkleNode::Leaf24(n) => n.set_commitment(cache),
            VerkleNode::Leaf25(n) => n.set_commitment(cache),
            VerkleNode::Leaf26(n) => n.set_commitment(cache),
            VerkleNode::Leaf27(n) => n.set_commitment(cache),
            VerkleNode::Leaf28(n) => n.set_commitment(cache),
            VerkleNode::Leaf29(n) => n.set_commitment(cache),
            VerkleNode::Leaf30(n) => n.set_commitment(cache),
            VerkleNode::Leaf31(n) => n.set_commitment(cache),
            VerkleNode::Leaf32(n) => n.set_commitment(cache),
            VerkleNode::Leaf33(n) => n.set_commitment(cache),
            VerkleNode::Leaf34(n) => n.set_commitment(cache),
            VerkleNode::Leaf35(n) => n.set_commitment(cache),
            VerkleNode::Leaf36(n) => n.set_commitment(cache),
            VerkleNode::Leaf37(n) => n.set_commitment(cache),
            VerkleNode::Leaf38(n) => n.set_commitment(cache),
            VerkleNode::Leaf39(n) => n.set_commitment(cache),
            VerkleNode::Leaf40(n) => n.set_commitment(cache),
            VerkleNode::Leaf41(n) => n.set_commitment(cache),
            VerkleNode::Leaf42(n) => n.set_commitment(cache),
            VerkleNode::Leaf43(n) => n.set_commitment(cache),
            VerkleNode::Leaf44(n) => n.set_commitment(cache),
            VerkleNode::Leaf45(n) => n.set_commitment(cache),
            VerkleNode::Leaf46(n) => n.set_commitment(cache),
            VerkleNode::Leaf47(n) => n.set_commitment(cache),
            VerkleNode::Leaf48(n) => n.set_commitment(cache),
            VerkleNode::Leaf49(n) => n.set_commitment(cache),
            VerkleNode::Leaf50(n) => n.set_commitment(cache),
            VerkleNode::Leaf51(n) => n.set_commitment(cache),
            VerkleNode::Leaf52(n) => n.set_commitment(cache),
            VerkleNode::Leaf53(n) => n.set_commitment(cache),
            VerkleNode::Leaf54(n) => n.set_commitment(cache),
            VerkleNode::Leaf55(n) => n.set_commitment(cache),
            VerkleNode::Leaf56(n) => n.set_commitment(cache),
            VerkleNode::Leaf57(n) => n.set_commitment(cache),
            VerkleNode::Leaf58(n) => n.set_commitment(cache),
            VerkleNode::Leaf59(n) => n.set_commitment(cache),
            VerkleNode::Leaf60(n) => n.set_commitment(cache),
            VerkleNode::Leaf61(n) => n.set_commitment(cache),
            VerkleNode::Leaf62(n) => n.set_commitment(cache),
            VerkleNode::Leaf63(n) => n.set_commitment(cache),
            VerkleNode::Leaf64(n) => n.set_commitment(cache),
            VerkleNode::Leaf65(n) => n.set_commitment(cache),
            VerkleNode::Leaf66(n) => n.set_commitment(cache),
            VerkleNode::Leaf67(n) => n.set_commitment(cache),
            VerkleNode::Leaf68(n) => n.set_commitment(cache),
            VerkleNode::Leaf69(n) => n.set_commitment(cache),
            VerkleNode::Leaf70(n) => n.set_commitment(cache),
            VerkleNode::Leaf71(n) => n.set_commitment(cache),
            VerkleNode::Leaf72(n) => n.set_commitment(cache),
            VerkleNode::Leaf73(n) => n.set_commitment(cache),
            VerkleNode::Leaf74(n) => n.set_commitment(cache),
            VerkleNode::Leaf75(n) => n.set_commitment(cache),
            VerkleNode::Leaf76(n) => n.set_commitment(cache),
            VerkleNode::Leaf77(n) => n.set_commitment(cache),
            VerkleNode::Leaf78(n) => n.set_commitment(cache),
            VerkleNode::Leaf79(n) => n.set_commitment(cache),
            VerkleNode::Leaf80(n) => n.set_commitment(cache),
            VerkleNode::Leaf81(n) => n.set_commitment(cache),
            VerkleNode::Leaf82(n) => n.set_commitment(cache),
            VerkleNode::Leaf83(n) => n.set_commitment(cache),
            VerkleNode::Leaf84(n) => n.set_commitment(cache),
            VerkleNode::Leaf85(n) => n.set_commitment(cache),
            VerkleNode::Leaf86(n) => n.set_commitment(cache),
            VerkleNode::Leaf87(n) => n.set_commitment(cache),
            VerkleNode::Leaf88(n) => n.set_commitment(cache),
            VerkleNode::Leaf89(n) => n.set_commitment(cache),
            VerkleNode::Leaf90(n) => n.set_commitment(cache),
            VerkleNode::Leaf91(n) => n.set_commitment(cache),
            VerkleNode::Leaf92(n) => n.set_commitment(cache),
            VerkleNode::Leaf93(n) => n.set_commitment(cache),
            VerkleNode::Leaf94(n) => n.set_commitment(cache),
            VerkleNode::Leaf95(n) => n.set_commitment(cache),
            VerkleNode::Leaf96(n) => n.set_commitment(cache),
            VerkleNode::Leaf97(n) => n.set_commitment(cache),
            VerkleNode::Leaf98(n) => n.set_commitment(cache),
            VerkleNode::Leaf99(n) => n.set_commitment(cache),
            VerkleNode::Leaf100(n) => n.set_commitment(cache),
            VerkleNode::Leaf101(n) => n.set_commitment(cache),
            VerkleNode::Leaf102(n) => n.set_commitment(cache),
            VerkleNode::Leaf103(n) => n.set_commitment(cache),
            VerkleNode::Leaf104(n) => n.set_commitment(cache),
            VerkleNode::Leaf105(n) => n.set_commitment(cache),
            VerkleNode::Leaf106(n) => n.set_commitment(cache),
            VerkleNode::Leaf107(n) => n.set_commitment(cache),
            VerkleNode::Leaf108(n) => n.set_commitment(cache),
            VerkleNode::Leaf109(n) => n.set_commitment(cache),
            VerkleNode::Leaf110(n) => n.set_commitment(cache),
            VerkleNode::Leaf111(n) => n.set_commitment(cache),
            VerkleNode::Leaf112(n) => n.set_commitment(cache),
            VerkleNode::Leaf113(n) => n.set_commitment(cache),
            VerkleNode::Leaf114(n) => n.set_commitment(cache),
            VerkleNode::Leaf115(n) => n.set_commitment(cache),
            VerkleNode::Leaf116(n) => n.set_commitment(cache),
            VerkleNode::Leaf117(n) => n.set_commitment(cache),
            VerkleNode::Leaf118(n) => n.set_commitment(cache),
            VerkleNode::Leaf119(n) => n.set_commitment(cache),
            VerkleNode::Leaf120(n) => n.set_commitment(cache),
            VerkleNode::Leaf121(n) => n.set_commitment(cache),
            VerkleNode::Leaf122(n) => n.set_commitment(cache),
            VerkleNode::Leaf123(n) => n.set_commitment(cache),
            VerkleNode::Leaf124(n) => n.set_commitment(cache),
            VerkleNode::Leaf125(n) => n.set_commitment(cache),
            VerkleNode::Leaf126(n) => n.set_commitment(cache),
            VerkleNode::Leaf127(n) => n.set_commitment(cache),
            VerkleNode::Leaf128(n) => n.set_commitment(cache),
            VerkleNode::Leaf129(n) => n.set_commitment(cache),
            VerkleNode::Leaf130(n) => n.set_commitment(cache),
            VerkleNode::Leaf131(n) => n.set_commitment(cache),
            VerkleNode::Leaf132(n) => n.set_commitment(cache),
            VerkleNode::Leaf133(n) => n.set_commitment(cache),
            VerkleNode::Leaf134(n) => n.set_commitment(cache),
            VerkleNode::Leaf135(n) => n.set_commitment(cache),
            VerkleNode::Leaf136(n) => n.set_commitment(cache),
            VerkleNode::Leaf137(n) => n.set_commitment(cache),
            VerkleNode::Leaf138(n) => n.set_commitment(cache),
            VerkleNode::Leaf139(n) => n.set_commitment(cache),
            VerkleNode::Leaf140(n) => n.set_commitment(cache),
            VerkleNode::Leaf141(n) => n.set_commitment(cache),
            VerkleNode::Leaf142(n) => n.set_commitment(cache),
            VerkleNode::Leaf143(n) => n.set_commitment(cache),
            VerkleNode::Leaf144(n) => n.set_commitment(cache),
            VerkleNode::Leaf145(n) => n.set_commitment(cache),
            VerkleNode::Leaf146(n) => n.set_commitment(cache),
            VerkleNode::Leaf147(n) => n.set_commitment(cache),
            VerkleNode::Leaf148(n) => n.set_commitment(cache),
            VerkleNode::Leaf149(n) => n.set_commitment(cache),
            VerkleNode::Leaf150(n) => n.set_commitment(cache),
            VerkleNode::Leaf151(n) => n.set_commitment(cache),
            VerkleNode::Leaf152(n) => n.set_commitment(cache),
            VerkleNode::Leaf153(n) => n.set_commitment(cache),
            VerkleNode::Leaf154(n) => n.set_commitment(cache),
            VerkleNode::Leaf155(n) => n.set_commitment(cache),
            VerkleNode::Leaf156(n) => n.set_commitment(cache),
            VerkleNode::Leaf157(n) => n.set_commitment(cache),
            VerkleNode::Leaf158(n) => n.set_commitment(cache),
            VerkleNode::Leaf159(n) => n.set_commitment(cache),
            VerkleNode::Leaf160(n) => n.set_commitment(cache),
            VerkleNode::Leaf161(n) => n.set_commitment(cache),
            VerkleNode::Leaf162(n) => n.set_commitment(cache),
            VerkleNode::Leaf163(n) => n.set_commitment(cache),
            VerkleNode::Leaf164(n) => n.set_commitment(cache),
            VerkleNode::Leaf165(n) => n.set_commitment(cache),
            VerkleNode::Leaf166(n) => n.set_commitment(cache),
            VerkleNode::Leaf167(n) => n.set_commitment(cache),
            VerkleNode::Leaf168(n) => n.set_commitment(cache),
            VerkleNode::Leaf169(n) => n.set_commitment(cache),
            VerkleNode::Leaf170(n) => n.set_commitment(cache),
            VerkleNode::Leaf171(n) => n.set_commitment(cache),
            VerkleNode::Leaf172(n) => n.set_commitment(cache),
            VerkleNode::Leaf173(n) => n.set_commitment(cache),
            VerkleNode::Leaf174(n) => n.set_commitment(cache),
            VerkleNode::Leaf175(n) => n.set_commitment(cache),
            VerkleNode::Leaf176(n) => n.set_commitment(cache),
            VerkleNode::Leaf177(n) => n.set_commitment(cache),
            VerkleNode::Leaf178(n) => n.set_commitment(cache),
            VerkleNode::Leaf179(n) => n.set_commitment(cache),
            VerkleNode::Leaf180(n) => n.set_commitment(cache),
            VerkleNode::Leaf181(n) => n.set_commitment(cache),
            VerkleNode::Leaf182(n) => n.set_commitment(cache),
            VerkleNode::Leaf183(n) => n.set_commitment(cache),
            VerkleNode::Leaf184(n) => n.set_commitment(cache),
            VerkleNode::Leaf185(n) => n.set_commitment(cache),
            VerkleNode::Leaf186(n) => n.set_commitment(cache),
            VerkleNode::Leaf187(n) => n.set_commitment(cache),
            VerkleNode::Leaf188(n) => n.set_commitment(cache),
            VerkleNode::Leaf189(n) => n.set_commitment(cache),
            VerkleNode::Leaf190(n) => n.set_commitment(cache),
            VerkleNode::Leaf191(n) => n.set_commitment(cache),
            VerkleNode::Leaf192(n) => n.set_commitment(cache),
            VerkleNode::Leaf193(n) => n.set_commitment(cache),
            VerkleNode::Leaf194(n) => n.set_commitment(cache),
            VerkleNode::Leaf195(n) => n.set_commitment(cache),
            VerkleNode::Leaf196(n) => n.set_commitment(cache),
            VerkleNode::Leaf197(n) => n.set_commitment(cache),
            VerkleNode::Leaf198(n) => n.set_commitment(cache),
            VerkleNode::Leaf199(n) => n.set_commitment(cache),
            VerkleNode::Leaf200(n) => n.set_commitment(cache),
            VerkleNode::Leaf201(n) => n.set_commitment(cache),
            VerkleNode::Leaf202(n) => n.set_commitment(cache),
            VerkleNode::Leaf203(n) => n.set_commitment(cache),
            VerkleNode::Leaf204(n) => n.set_commitment(cache),
            VerkleNode::Leaf205(n) => n.set_commitment(cache),
            VerkleNode::Leaf206(n) => n.set_commitment(cache),
            VerkleNode::Leaf207(n) => n.set_commitment(cache),
            VerkleNode::Leaf208(n) => n.set_commitment(cache),
            VerkleNode::Leaf209(n) => n.set_commitment(cache),
            VerkleNode::Leaf210(n) => n.set_commitment(cache),
            VerkleNode::Leaf211(n) => n.set_commitment(cache),
            VerkleNode::Leaf212(n) => n.set_commitment(cache),
            VerkleNode::Leaf213(n) => n.set_commitment(cache),
            VerkleNode::Leaf214(n) => n.set_commitment(cache),
            VerkleNode::Leaf215(n) => n.set_commitment(cache),
            VerkleNode::Leaf216(n) => n.set_commitment(cache),
            VerkleNode::Leaf217(n) => n.set_commitment(cache),
            VerkleNode::Leaf218(n) => n.set_commitment(cache),
            VerkleNode::Leaf219(n) => n.set_commitment(cache),
            VerkleNode::Leaf220(n) => n.set_commitment(cache),
            VerkleNode::Leaf221(n) => n.set_commitment(cache),
            VerkleNode::Leaf222(n) => n.set_commitment(cache),
            VerkleNode::Leaf223(n) => n.set_commitment(cache),
            VerkleNode::Leaf224(n) => n.set_commitment(cache),
            VerkleNode::Leaf225(n) => n.set_commitment(cache),
            VerkleNode::Leaf226(n) => n.set_commitment(cache),
            VerkleNode::Leaf227(n) => n.set_commitment(cache),
            VerkleNode::Leaf228(n) => n.set_commitment(cache),
            VerkleNode::Leaf229(n) => n.set_commitment(cache),
            VerkleNode::Leaf230(n) => n.set_commitment(cache),
            VerkleNode::Leaf231(n) => n.set_commitment(cache),
            VerkleNode::Leaf232(n) => n.set_commitment(cache),
            VerkleNode::Leaf233(n) => n.set_commitment(cache),
            VerkleNode::Leaf234(n) => n.set_commitment(cache),
            VerkleNode::Leaf235(n) => n.set_commitment(cache),
            VerkleNode::Leaf236(n) => n.set_commitment(cache),
            VerkleNode::Leaf237(n) => n.set_commitment(cache),
            VerkleNode::Leaf238(n) => n.set_commitment(cache),
            VerkleNode::Leaf239(n) => n.set_commitment(cache),
            VerkleNode::Leaf240(n) => n.set_commitment(cache),
            VerkleNode::Leaf241(n) => n.set_commitment(cache),
            VerkleNode::Leaf242(n) => n.set_commitment(cache),
            VerkleNode::Leaf243(n) => n.set_commitment(cache),
            VerkleNode::Leaf244(n) => n.set_commitment(cache),
            VerkleNode::Leaf245(n) => n.set_commitment(cache),
            VerkleNode::Leaf246(n) => n.set_commitment(cache),
            VerkleNode::Leaf247(n) => n.set_commitment(cache),
            VerkleNode::Leaf248(n) => n.set_commitment(cache),
            VerkleNode::Leaf249(n) => n.set_commitment(cache),
            VerkleNode::Leaf250(n) => n.set_commitment(cache),
            VerkleNode::Leaf251(n) => n.set_commitment(cache),
            VerkleNode::Leaf252(n) => n.set_commitment(cache),
            VerkleNode::Leaf253(n) => n.set_commitment(cache),
            VerkleNode::Leaf254(n) => n.set_commitment(cache),
            VerkleNode::Leaf255(n) => n.set_commitment(cache),
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
    Inner1,
    Inner2,
    Inner3,
    Inner4,
    Inner5,
    Inner6,
    Inner7,
    Inner8,
    Inner9,
    Inner10,
    Inner11,
    Inner12,
    Inner13,
    Inner14,
    Inner15,
    Inner16,
    Inner17,
    Inner18,
    Inner19,
    Inner20,
    Inner21,
    Inner22,
    Inner23,
    Inner24,
    Inner25,
    Inner26,
    Inner27,
    Inner28,
    Inner29,
    Inner30,
    Inner31,
    Inner32,
    Inner33,
    Inner34,
    Inner35,
    Inner36,
    Inner37,
    Inner38,
    Inner39,
    Inner40,
    Inner41,
    Inner42,
    Inner43,
    Inner44,
    Inner45,
    Inner46,
    Inner47,
    Inner48,
    Inner49,
    Inner50,
    Inner51,
    Inner52,
    Inner53,
    Inner54,
    Inner55,
    Inner56,
    Inner57,
    Inner58,
    Inner59,
    Inner60,
    Inner61,
    Inner62,
    Inner63,
    Inner64,
    Inner65,
    Inner66,
    Inner67,
    Inner68,
    Inner69,
    Inner70,
    Inner71,
    Inner72,
    Inner73,
    Inner74,
    Inner75,
    Inner76,
    Inner77,
    Inner78,
    Inner79,
    Inner80,
    Inner81,
    Inner82,
    Inner83,
    Inner84,
    Inner85,
    Inner86,
    Inner87,
    Inner88,
    Inner89,
    Inner90,
    Inner91,
    Inner92,
    Inner93,
    Inner94,
    Inner95,
    Inner96,
    Inner97,
    Inner98,
    Inner99,
    Inner100,
    Inner101,
    Inner102,
    Inner103,
    Inner104,
    Inner105,
    Inner106,
    Inner107,
    Inner108,
    Inner109,
    Inner110,
    Inner111,
    Inner112,
    Inner113,
    Inner114,
    Inner115,
    Inner116,
    Inner117,
    Inner118,
    Inner119,
    Inner120,
    Inner121,
    Inner122,
    Inner123,
    Inner124,
    Inner125,
    Inner126,
    Inner127,
    Inner128,
    Inner129,
    Inner130,
    Inner131,
    Inner132,
    Inner133,
    Inner134,
    Inner135,
    Inner136,
    Inner137,
    Inner138,
    Inner139,
    Inner140,
    Inner141,
    Inner142,
    Inner143,
    Inner144,
    Inner145,
    Inner146,
    Inner147,
    Inner148,
    Inner149,
    Inner150,
    Inner151,
    Inner152,
    Inner153,
    Inner154,
    Inner155,
    Inner156,
    Inner157,
    Inner158,
    Inner159,
    Inner160,
    Inner161,
    Inner162,
    Inner163,
    Inner164,
    Inner165,
    Inner166,
    Inner167,
    Inner168,
    Inner169,
    Inner170,
    Inner171,
    Inner172,
    Inner173,
    Inner174,
    Inner175,
    Inner176,
    Inner177,
    Inner178,
    Inner179,
    Inner180,
    Inner181,
    Inner182,
    Inner183,
    Inner184,
    Inner185,
    Inner186,
    Inner187,
    Inner188,
    Inner189,
    Inner190,
    Inner191,
    Inner192,
    Inner193,
    Inner194,
    Inner195,
    Inner196,
    Inner197,
    Inner198,
    Inner199,
    Inner200,
    Inner201,
    Inner202,
    Inner203,
    Inner204,
    Inner205,
    Inner206,
    Inner207,
    Inner208,
    Inner209,
    Inner210,
    Inner211,
    Inner212,
    Inner213,
    Inner214,
    Inner215,
    Inner216,
    Inner217,
    Inner218,
    Inner219,
    Inner220,
    Inner221,
    Inner222,
    Inner223,
    Inner224,
    Inner225,
    Inner226,
    Inner227,
    Inner228,
    Inner229,
    Inner230,
    Inner231,
    Inner232,
    Inner233,
    Inner234,
    Inner235,
    Inner236,
    Inner237,
    Inner238,
    Inner239,
    Inner240,
    Inner241,
    Inner242,
    Inner243,
    Inner244,
    Inner245,
    Inner246,
    Inner247,
    Inner248,
    Inner249,
    Inner250,
    Inner251,
    Inner252,
    Inner253,
    Inner254,
    Inner255,
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
    Leaf17,
    Leaf18,
    Leaf19,
    Leaf20,
    Leaf21,
    Leaf22,
    Leaf23,
    Leaf24,
    Leaf25,
    Leaf26,
    Leaf27,
    Leaf28,
    Leaf29,
    Leaf30,
    Leaf31,
    Leaf32,
    Leaf33,
    Leaf34,
    Leaf35,
    Leaf36,
    Leaf37,
    Leaf38,
    Leaf39,
    Leaf40,
    Leaf41,
    Leaf42,
    Leaf43,
    Leaf44,
    Leaf45,
    Leaf46,
    Leaf47,
    Leaf48,
    Leaf49,
    Leaf50,
    Leaf51,
    Leaf52,
    Leaf53,
    Leaf54,
    Leaf55,
    Leaf56,
    Leaf57,
    Leaf58,
    Leaf59,
    Leaf60,
    Leaf61,
    Leaf62,
    Leaf63,
    Leaf64,
    Leaf65,
    Leaf66,
    Leaf67,
    Leaf68,
    Leaf69,
    Leaf70,
    Leaf71,
    Leaf72,
    Leaf73,
    Leaf74,
    Leaf75,
    Leaf76,
    Leaf77,
    Leaf78,
    Leaf79,
    Leaf80,
    Leaf81,
    Leaf82,
    Leaf83,
    Leaf84,
    Leaf85,
    Leaf86,
    Leaf87,
    Leaf88,
    Leaf89,
    Leaf90,
    Leaf91,
    Leaf92,
    Leaf93,
    Leaf94,
    Leaf95,
    Leaf96,
    Leaf97,
    Leaf98,
    Leaf99,
    Leaf100,
    Leaf101,
    Leaf102,
    Leaf103,
    Leaf104,
    Leaf105,
    Leaf106,
    Leaf107,
    Leaf108,
    Leaf109,
    Leaf110,
    Leaf111,
    Leaf112,
    Leaf113,
    Leaf114,
    Leaf115,
    Leaf116,
    Leaf117,
    Leaf118,
    Leaf119,
    Leaf120,
    Leaf121,
    Leaf122,
    Leaf123,
    Leaf124,
    Leaf125,
    Leaf126,
    Leaf127,
    Leaf128,
    Leaf129,
    Leaf130,
    Leaf131,
    Leaf132,
    Leaf133,
    Leaf134,
    Leaf135,
    Leaf136,
    Leaf137,
    Leaf138,
    Leaf139,
    Leaf140,
    Leaf141,
    Leaf142,
    Leaf143,
    Leaf144,
    Leaf145,
    Leaf146,
    Leaf147,
    Leaf148,
    Leaf149,
    Leaf150,
    Leaf151,
    Leaf152,
    Leaf153,
    Leaf154,
    Leaf155,
    Leaf156,
    Leaf157,
    Leaf158,
    Leaf159,
    Leaf160,
    Leaf161,
    Leaf162,
    Leaf163,
    Leaf164,
    Leaf165,
    Leaf166,
    Leaf167,
    Leaf168,
    Leaf169,
    Leaf170,
    Leaf171,
    Leaf172,
    Leaf173,
    Leaf174,
    Leaf175,
    Leaf176,
    Leaf177,
    Leaf178,
    Leaf179,
    Leaf180,
    Leaf181,
    Leaf182,
    Leaf183,
    Leaf184,
    Leaf185,
    Leaf186,
    Leaf187,
    Leaf188,
    Leaf189,
    Leaf190,
    Leaf191,
    Leaf192,
    Leaf193,
    Leaf194,
    Leaf195,
    Leaf196,
    Leaf197,
    Leaf198,
    Leaf199,
    Leaf200,
    Leaf201,
    Leaf202,
    Leaf203,
    Leaf204,
    Leaf205,
    Leaf206,
    Leaf207,
    Leaf208,
    Leaf209,
    Leaf210,
    Leaf211,
    Leaf212,
    Leaf213,
    Leaf214,
    Leaf215,
    Leaf216,
    Leaf217,
    Leaf218,
    Leaf219,
    Leaf220,
    Leaf221,
    Leaf222,
    Leaf223,
    Leaf224,
    Leaf225,
    Leaf226,
    Leaf227,
    Leaf228,
    Leaf229,
    Leaf230,
    Leaf231,
    Leaf232,
    Leaf233,
    Leaf234,
    Leaf235,
    Leaf236,
    Leaf237,
    Leaf238,
    Leaf239,
    Leaf240,
    Leaf241,
    Leaf242,
    Leaf243,
    Leaf244,
    Leaf245,
    Leaf246,
    Leaf247,
    Leaf248,
    Leaf249,
    Leaf250,
    Leaf251,
    Leaf252,
    Leaf253,
    Leaf254,
    Leaf255,
    Leaf256,
    LeafDelta,
}

impl NodeSize for VerkleNodeKind {
    fn node_byte_size(&self) -> usize {
        let inner_size = match self {
            VerkleNodeKind::Empty => std::mem::size_of::<VerkleNode>(),
            VerkleNodeKind::Inner1 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<1>>>()
                    + std::mem::size_of::<SparseInnerNode<1>>()
            }
            VerkleNodeKind::Inner2 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<2>>>()
                    + std::mem::size_of::<SparseInnerNode<2>>()
            }
            VerkleNodeKind::Inner3 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<3>>>()
                    + std::mem::size_of::<SparseInnerNode<3>>()
            }
            VerkleNodeKind::Inner4 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<4>>>()
                    + std::mem::size_of::<SparseInnerNode<4>>()
            }
            VerkleNodeKind::Inner5 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<5>>>()
                    + std::mem::size_of::<SparseInnerNode<5>>()
            }
            VerkleNodeKind::Inner6 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<6>>>()
                    + std::mem::size_of::<SparseInnerNode<6>>()
            }
            VerkleNodeKind::Inner7 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<7>>>()
                    + std::mem::size_of::<SparseInnerNode<7>>()
            }
            VerkleNodeKind::Inner8 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<8>>>()
                    + std::mem::size_of::<SparseInnerNode<8>>()
            }
            VerkleNodeKind::Inner9 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<9>>>()
                    + std::mem::size_of::<SparseInnerNode<9>>()
            }
            VerkleNodeKind::Inner10 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<10>>>()
                    + std::mem::size_of::<SparseInnerNode<10>>()
            }
            VerkleNodeKind::Inner11 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<11>>>()
                    + std::mem::size_of::<SparseInnerNode<11>>()
            }
            VerkleNodeKind::Inner12 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<12>>>()
                    + std::mem::size_of::<SparseInnerNode<12>>()
            }
            VerkleNodeKind::Inner13 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<13>>>()
                    + std::mem::size_of::<SparseInnerNode<13>>()
            }
            VerkleNodeKind::Inner14 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<14>>>()
                    + std::mem::size_of::<SparseInnerNode<14>>()
            }
            VerkleNodeKind::Inner15 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<15>>>()
                    + std::mem::size_of::<SparseInnerNode<15>>()
            }
            VerkleNodeKind::Inner16 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<16>>>()
                    + std::mem::size_of::<SparseInnerNode<16>>()
            }
            VerkleNodeKind::Inner17 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<17>>>()
                    + std::mem::size_of::<SparseInnerNode<17>>()
            }
            VerkleNodeKind::Inner18 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<18>>>()
                    + std::mem::size_of::<SparseInnerNode<18>>()
            }
            VerkleNodeKind::Inner19 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<19>>>()
                    + std::mem::size_of::<SparseInnerNode<19>>()
            }
            VerkleNodeKind::Inner20 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<20>>>()
                    + std::mem::size_of::<SparseInnerNode<20>>()
            }
            VerkleNodeKind::Inner21 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<21>>>()
                    + std::mem::size_of::<SparseInnerNode<21>>()
            }
            VerkleNodeKind::Inner22 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<22>>>()
                    + std::mem::size_of::<SparseInnerNode<22>>()
            }
            VerkleNodeKind::Inner23 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<23>>>()
                    + std::mem::size_of::<SparseInnerNode<23>>()
            }
            VerkleNodeKind::Inner24 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<24>>>()
                    + std::mem::size_of::<SparseInnerNode<24>>()
            }
            VerkleNodeKind::Inner25 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<25>>>()
                    + std::mem::size_of::<SparseInnerNode<25>>()
            }
            VerkleNodeKind::Inner26 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<26>>>()
                    + std::mem::size_of::<SparseInnerNode<26>>()
            }
            VerkleNodeKind::Inner27 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<27>>>()
                    + std::mem::size_of::<SparseInnerNode<27>>()
            }
            VerkleNodeKind::Inner28 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<28>>>()
                    + std::mem::size_of::<SparseInnerNode<28>>()
            }
            VerkleNodeKind::Inner29 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<29>>>()
                    + std::mem::size_of::<SparseInnerNode<29>>()
            }
            VerkleNodeKind::Inner30 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<30>>>()
                    + std::mem::size_of::<SparseInnerNode<30>>()
            }
            VerkleNodeKind::Inner31 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<31>>>()
                    + std::mem::size_of::<SparseInnerNode<31>>()
            }
            VerkleNodeKind::Inner32 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<32>>>()
                    + std::mem::size_of::<SparseInnerNode<32>>()
            }
            VerkleNodeKind::Inner33 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<33>>>()
                    + std::mem::size_of::<SparseInnerNode<33>>()
            }
            VerkleNodeKind::Inner34 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<34>>>()
                    + std::mem::size_of::<SparseInnerNode<34>>()
            }
            VerkleNodeKind::Inner35 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<35>>>()
                    + std::mem::size_of::<SparseInnerNode<35>>()
            }
            VerkleNodeKind::Inner36 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<36>>>()
                    + std::mem::size_of::<SparseInnerNode<36>>()
            }
            VerkleNodeKind::Inner37 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<37>>>()
                    + std::mem::size_of::<SparseInnerNode<37>>()
            }
            VerkleNodeKind::Inner38 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<38>>>()
                    + std::mem::size_of::<SparseInnerNode<38>>()
            }
            VerkleNodeKind::Inner39 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<39>>>()
                    + std::mem::size_of::<SparseInnerNode<39>>()
            }
            VerkleNodeKind::Inner40 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<40>>>()
                    + std::mem::size_of::<SparseInnerNode<40>>()
            }
            VerkleNodeKind::Inner41 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<41>>>()
                    + std::mem::size_of::<SparseInnerNode<41>>()
            }
            VerkleNodeKind::Inner42 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<42>>>()
                    + std::mem::size_of::<SparseInnerNode<42>>()
            }
            VerkleNodeKind::Inner43 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<43>>>()
                    + std::mem::size_of::<SparseInnerNode<43>>()
            }
            VerkleNodeKind::Inner44 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<44>>>()
                    + std::mem::size_of::<SparseInnerNode<44>>()
            }
            VerkleNodeKind::Inner45 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<45>>>()
                    + std::mem::size_of::<SparseInnerNode<45>>()
            }
            VerkleNodeKind::Inner46 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<46>>>()
                    + std::mem::size_of::<SparseInnerNode<46>>()
            }
            VerkleNodeKind::Inner47 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<47>>>()
                    + std::mem::size_of::<SparseInnerNode<47>>()
            }
            VerkleNodeKind::Inner48 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<48>>>()
                    + std::mem::size_of::<SparseInnerNode<48>>()
            }
            VerkleNodeKind::Inner49 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<49>>>()
                    + std::mem::size_of::<SparseInnerNode<49>>()
            }
            VerkleNodeKind::Inner50 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<50>>>()
                    + std::mem::size_of::<SparseInnerNode<50>>()
            }
            VerkleNodeKind::Inner51 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<51>>>()
                    + std::mem::size_of::<SparseInnerNode<51>>()
            }
            VerkleNodeKind::Inner52 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<52>>>()
                    + std::mem::size_of::<SparseInnerNode<52>>()
            }
            VerkleNodeKind::Inner53 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<53>>>()
                    + std::mem::size_of::<SparseInnerNode<53>>()
            }
            VerkleNodeKind::Inner54 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<54>>>()
                    + std::mem::size_of::<SparseInnerNode<54>>()
            }
            VerkleNodeKind::Inner55 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<55>>>()
                    + std::mem::size_of::<SparseInnerNode<55>>()
            }
            VerkleNodeKind::Inner56 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<56>>>()
                    + std::mem::size_of::<SparseInnerNode<56>>()
            }
            VerkleNodeKind::Inner57 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<57>>>()
                    + std::mem::size_of::<SparseInnerNode<57>>()
            }
            VerkleNodeKind::Inner58 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<58>>>()
                    + std::mem::size_of::<SparseInnerNode<58>>()
            }
            VerkleNodeKind::Inner59 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<59>>>()
                    + std::mem::size_of::<SparseInnerNode<59>>()
            }
            VerkleNodeKind::Inner60 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<60>>>()
                    + std::mem::size_of::<SparseInnerNode<60>>()
            }
            VerkleNodeKind::Inner61 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<61>>>()
                    + std::mem::size_of::<SparseInnerNode<61>>()
            }
            VerkleNodeKind::Inner62 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<62>>>()
                    + std::mem::size_of::<SparseInnerNode<62>>()
            }
            VerkleNodeKind::Inner63 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<63>>>()
                    + std::mem::size_of::<SparseInnerNode<63>>()
            }
            VerkleNodeKind::Inner64 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<64>>>()
                    + std::mem::size_of::<SparseInnerNode<64>>()
            }
            VerkleNodeKind::Inner65 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<65>>>()
                    + std::mem::size_of::<SparseInnerNode<65>>()
            }
            VerkleNodeKind::Inner66 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<66>>>()
                    + std::mem::size_of::<SparseInnerNode<66>>()
            }
            VerkleNodeKind::Inner67 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<67>>>()
                    + std::mem::size_of::<SparseInnerNode<67>>()
            }
            VerkleNodeKind::Inner68 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<68>>>()
                    + std::mem::size_of::<SparseInnerNode<68>>()
            }
            VerkleNodeKind::Inner69 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<69>>>()
                    + std::mem::size_of::<SparseInnerNode<69>>()
            }
            VerkleNodeKind::Inner70 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<70>>>()
                    + std::mem::size_of::<SparseInnerNode<70>>()
            }
            VerkleNodeKind::Inner71 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<71>>>()
                    + std::mem::size_of::<SparseInnerNode<71>>()
            }
            VerkleNodeKind::Inner72 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<72>>>()
                    + std::mem::size_of::<SparseInnerNode<72>>()
            }
            VerkleNodeKind::Inner73 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<73>>>()
                    + std::mem::size_of::<SparseInnerNode<73>>()
            }
            VerkleNodeKind::Inner74 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<74>>>()
                    + std::mem::size_of::<SparseInnerNode<74>>()
            }
            VerkleNodeKind::Inner75 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<75>>>()
                    + std::mem::size_of::<SparseInnerNode<75>>()
            }
            VerkleNodeKind::Inner76 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<76>>>()
                    + std::mem::size_of::<SparseInnerNode<76>>()
            }
            VerkleNodeKind::Inner77 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<77>>>()
                    + std::mem::size_of::<SparseInnerNode<77>>()
            }
            VerkleNodeKind::Inner78 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<78>>>()
                    + std::mem::size_of::<SparseInnerNode<78>>()
            }
            VerkleNodeKind::Inner79 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<79>>>()
                    + std::mem::size_of::<SparseInnerNode<79>>()
            }
            VerkleNodeKind::Inner80 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<80>>>()
                    + std::mem::size_of::<SparseInnerNode<80>>()
            }
            VerkleNodeKind::Inner81 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<81>>>()
                    + std::mem::size_of::<SparseInnerNode<81>>()
            }
            VerkleNodeKind::Inner82 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<82>>>()
                    + std::mem::size_of::<SparseInnerNode<82>>()
            }
            VerkleNodeKind::Inner83 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<83>>>()
                    + std::mem::size_of::<SparseInnerNode<83>>()
            }
            VerkleNodeKind::Inner84 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<84>>>()
                    + std::mem::size_of::<SparseInnerNode<84>>()
            }
            VerkleNodeKind::Inner85 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<85>>>()
                    + std::mem::size_of::<SparseInnerNode<85>>()
            }
            VerkleNodeKind::Inner86 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<86>>>()
                    + std::mem::size_of::<SparseInnerNode<86>>()
            }
            VerkleNodeKind::Inner87 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<87>>>()
                    + std::mem::size_of::<SparseInnerNode<87>>()
            }
            VerkleNodeKind::Inner88 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<88>>>()
                    + std::mem::size_of::<SparseInnerNode<88>>()
            }
            VerkleNodeKind::Inner89 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<89>>>()
                    + std::mem::size_of::<SparseInnerNode<89>>()
            }
            VerkleNodeKind::Inner90 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<90>>>()
                    + std::mem::size_of::<SparseInnerNode<90>>()
            }
            VerkleNodeKind::Inner91 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<91>>>()
                    + std::mem::size_of::<SparseInnerNode<91>>()
            }
            VerkleNodeKind::Inner92 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<92>>>()
                    + std::mem::size_of::<SparseInnerNode<92>>()
            }
            VerkleNodeKind::Inner93 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<93>>>()
                    + std::mem::size_of::<SparseInnerNode<93>>()
            }
            VerkleNodeKind::Inner94 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<94>>>()
                    + std::mem::size_of::<SparseInnerNode<94>>()
            }
            VerkleNodeKind::Inner95 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<95>>>()
                    + std::mem::size_of::<SparseInnerNode<95>>()
            }
            VerkleNodeKind::Inner96 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<96>>>()
                    + std::mem::size_of::<SparseInnerNode<96>>()
            }
            VerkleNodeKind::Inner97 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<97>>>()
                    + std::mem::size_of::<SparseInnerNode<97>>()
            }
            VerkleNodeKind::Inner98 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<98>>>()
                    + std::mem::size_of::<SparseInnerNode<98>>()
            }
            VerkleNodeKind::Inner99 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<99>>>()
                    + std::mem::size_of::<SparseInnerNode<99>>()
            }
            VerkleNodeKind::Inner100 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<100>>>()
                    + std::mem::size_of::<SparseInnerNode<100>>()
            }
            VerkleNodeKind::Inner101 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<101>>>()
                    + std::mem::size_of::<SparseInnerNode<101>>()
            }
            VerkleNodeKind::Inner102 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<102>>>()
                    + std::mem::size_of::<SparseInnerNode<102>>()
            }
            VerkleNodeKind::Inner103 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<103>>>()
                    + std::mem::size_of::<SparseInnerNode<103>>()
            }
            VerkleNodeKind::Inner104 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<104>>>()
                    + std::mem::size_of::<SparseInnerNode<104>>()
            }
            VerkleNodeKind::Inner105 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<105>>>()
                    + std::mem::size_of::<SparseInnerNode<105>>()
            }
            VerkleNodeKind::Inner106 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<106>>>()
                    + std::mem::size_of::<SparseInnerNode<106>>()
            }
            VerkleNodeKind::Inner107 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<107>>>()
                    + std::mem::size_of::<SparseInnerNode<107>>()
            }
            VerkleNodeKind::Inner108 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<108>>>()
                    + std::mem::size_of::<SparseInnerNode<108>>()
            }
            VerkleNodeKind::Inner109 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<109>>>()
                    + std::mem::size_of::<SparseInnerNode<109>>()
            }
            VerkleNodeKind::Inner110 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<110>>>()
                    + std::mem::size_of::<SparseInnerNode<110>>()
            }
            VerkleNodeKind::Inner111 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<111>>>()
                    + std::mem::size_of::<SparseInnerNode<111>>()
            }
            VerkleNodeKind::Inner112 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<112>>>()
                    + std::mem::size_of::<SparseInnerNode<112>>()
            }
            VerkleNodeKind::Inner113 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<113>>>()
                    + std::mem::size_of::<SparseInnerNode<113>>()
            }
            VerkleNodeKind::Inner114 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<114>>>()
                    + std::mem::size_of::<SparseInnerNode<114>>()
            }
            VerkleNodeKind::Inner115 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<115>>>()
                    + std::mem::size_of::<SparseInnerNode<115>>()
            }
            VerkleNodeKind::Inner116 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<116>>>()
                    + std::mem::size_of::<SparseInnerNode<116>>()
            }
            VerkleNodeKind::Inner117 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<117>>>()
                    + std::mem::size_of::<SparseInnerNode<117>>()
            }
            VerkleNodeKind::Inner118 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<118>>>()
                    + std::mem::size_of::<SparseInnerNode<118>>()
            }
            VerkleNodeKind::Inner119 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<119>>>()
                    + std::mem::size_of::<SparseInnerNode<119>>()
            }
            VerkleNodeKind::Inner120 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<120>>>()
                    + std::mem::size_of::<SparseInnerNode<120>>()
            }
            VerkleNodeKind::Inner121 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<121>>>()
                    + std::mem::size_of::<SparseInnerNode<121>>()
            }
            VerkleNodeKind::Inner122 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<122>>>()
                    + std::mem::size_of::<SparseInnerNode<122>>()
            }
            VerkleNodeKind::Inner123 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<123>>>()
                    + std::mem::size_of::<SparseInnerNode<123>>()
            }
            VerkleNodeKind::Inner124 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<124>>>()
                    + std::mem::size_of::<SparseInnerNode<124>>()
            }
            VerkleNodeKind::Inner125 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<125>>>()
                    + std::mem::size_of::<SparseInnerNode<125>>()
            }
            VerkleNodeKind::Inner126 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<126>>>()
                    + std::mem::size_of::<SparseInnerNode<126>>()
            }
            VerkleNodeKind::Inner127 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<127>>>()
                    + std::mem::size_of::<SparseInnerNode<127>>()
            }
            VerkleNodeKind::Inner128 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<128>>>()
                    + std::mem::size_of::<SparseInnerNode<128>>()
            }
            VerkleNodeKind::Inner129 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<129>>>()
                    + std::mem::size_of::<SparseInnerNode<129>>()
            }
            VerkleNodeKind::Inner130 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<130>>>()
                    + std::mem::size_of::<SparseInnerNode<130>>()
            }
            VerkleNodeKind::Inner131 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<131>>>()
                    + std::mem::size_of::<SparseInnerNode<131>>()
            }
            VerkleNodeKind::Inner132 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<132>>>()
                    + std::mem::size_of::<SparseInnerNode<132>>()
            }
            VerkleNodeKind::Inner133 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<133>>>()
                    + std::mem::size_of::<SparseInnerNode<133>>()
            }
            VerkleNodeKind::Inner134 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<134>>>()
                    + std::mem::size_of::<SparseInnerNode<134>>()
            }
            VerkleNodeKind::Inner135 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<135>>>()
                    + std::mem::size_of::<SparseInnerNode<135>>()
            }
            VerkleNodeKind::Inner136 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<136>>>()
                    + std::mem::size_of::<SparseInnerNode<136>>()
            }
            VerkleNodeKind::Inner137 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<137>>>()
                    + std::mem::size_of::<SparseInnerNode<137>>()
            }
            VerkleNodeKind::Inner138 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<138>>>()
                    + std::mem::size_of::<SparseInnerNode<138>>()
            }
            VerkleNodeKind::Inner139 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<139>>>()
                    + std::mem::size_of::<SparseInnerNode<139>>()
            }
            VerkleNodeKind::Inner140 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<140>>>()
                    + std::mem::size_of::<SparseInnerNode<140>>()
            }
            VerkleNodeKind::Inner141 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<141>>>()
                    + std::mem::size_of::<SparseInnerNode<141>>()
            }
            VerkleNodeKind::Inner142 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<142>>>()
                    + std::mem::size_of::<SparseInnerNode<142>>()
            }
            VerkleNodeKind::Inner143 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<143>>>()
                    + std::mem::size_of::<SparseInnerNode<143>>()
            }
            VerkleNodeKind::Inner144 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<144>>>()
                    + std::mem::size_of::<SparseInnerNode<144>>()
            }
            VerkleNodeKind::Inner145 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<145>>>()
                    + std::mem::size_of::<SparseInnerNode<145>>()
            }
            VerkleNodeKind::Inner146 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<146>>>()
                    + std::mem::size_of::<SparseInnerNode<146>>()
            }
            VerkleNodeKind::Inner147 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<147>>>()
                    + std::mem::size_of::<SparseInnerNode<147>>()
            }
            VerkleNodeKind::Inner148 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<148>>>()
                    + std::mem::size_of::<SparseInnerNode<148>>()
            }
            VerkleNodeKind::Inner149 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<149>>>()
                    + std::mem::size_of::<SparseInnerNode<149>>()
            }
            VerkleNodeKind::Inner150 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<150>>>()
                    + std::mem::size_of::<SparseInnerNode<150>>()
            }
            VerkleNodeKind::Inner151 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<151>>>()
                    + std::mem::size_of::<SparseInnerNode<151>>()
            }
            VerkleNodeKind::Inner152 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<152>>>()
                    + std::mem::size_of::<SparseInnerNode<152>>()
            }
            VerkleNodeKind::Inner153 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<153>>>()
                    + std::mem::size_of::<SparseInnerNode<153>>()
            }
            VerkleNodeKind::Inner154 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<154>>>()
                    + std::mem::size_of::<SparseInnerNode<154>>()
            }
            VerkleNodeKind::Inner155 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<155>>>()
                    + std::mem::size_of::<SparseInnerNode<155>>()
            }
            VerkleNodeKind::Inner156 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<156>>>()
                    + std::mem::size_of::<SparseInnerNode<156>>()
            }
            VerkleNodeKind::Inner157 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<157>>>()
                    + std::mem::size_of::<SparseInnerNode<157>>()
            }
            VerkleNodeKind::Inner158 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<158>>>()
                    + std::mem::size_of::<SparseInnerNode<158>>()
            }
            VerkleNodeKind::Inner159 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<159>>>()
                    + std::mem::size_of::<SparseInnerNode<159>>()
            }
            VerkleNodeKind::Inner160 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<160>>>()
                    + std::mem::size_of::<SparseInnerNode<160>>()
            }
            VerkleNodeKind::Inner161 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<161>>>()
                    + std::mem::size_of::<SparseInnerNode<161>>()
            }
            VerkleNodeKind::Inner162 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<162>>>()
                    + std::mem::size_of::<SparseInnerNode<162>>()
            }
            VerkleNodeKind::Inner163 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<163>>>()
                    + std::mem::size_of::<SparseInnerNode<163>>()
            }
            VerkleNodeKind::Inner164 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<164>>>()
                    + std::mem::size_of::<SparseInnerNode<164>>()
            }
            VerkleNodeKind::Inner165 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<165>>>()
                    + std::mem::size_of::<SparseInnerNode<165>>()
            }
            VerkleNodeKind::Inner166 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<166>>>()
                    + std::mem::size_of::<SparseInnerNode<166>>()
            }
            VerkleNodeKind::Inner167 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<167>>>()
                    + std::mem::size_of::<SparseInnerNode<167>>()
            }
            VerkleNodeKind::Inner168 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<168>>>()
                    + std::mem::size_of::<SparseInnerNode<168>>()
            }
            VerkleNodeKind::Inner169 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<169>>>()
                    + std::mem::size_of::<SparseInnerNode<169>>()
            }
            VerkleNodeKind::Inner170 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<170>>>()
                    + std::mem::size_of::<SparseInnerNode<170>>()
            }
            VerkleNodeKind::Inner171 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<171>>>()
                    + std::mem::size_of::<SparseInnerNode<171>>()
            }
            VerkleNodeKind::Inner172 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<172>>>()
                    + std::mem::size_of::<SparseInnerNode<172>>()
            }
            VerkleNodeKind::Inner173 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<173>>>()
                    + std::mem::size_of::<SparseInnerNode<173>>()
            }
            VerkleNodeKind::Inner174 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<174>>>()
                    + std::mem::size_of::<SparseInnerNode<174>>()
            }
            VerkleNodeKind::Inner175 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<175>>>()
                    + std::mem::size_of::<SparseInnerNode<175>>()
            }
            VerkleNodeKind::Inner176 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<176>>>()
                    + std::mem::size_of::<SparseInnerNode<176>>()
            }
            VerkleNodeKind::Inner177 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<177>>>()
                    + std::mem::size_of::<SparseInnerNode<177>>()
            }
            VerkleNodeKind::Inner178 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<178>>>()
                    + std::mem::size_of::<SparseInnerNode<178>>()
            }
            VerkleNodeKind::Inner179 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<179>>>()
                    + std::mem::size_of::<SparseInnerNode<179>>()
            }
            VerkleNodeKind::Inner180 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<180>>>()
                    + std::mem::size_of::<SparseInnerNode<180>>()
            }
            VerkleNodeKind::Inner181 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<181>>>()
                    + std::mem::size_of::<SparseInnerNode<181>>()
            }
            VerkleNodeKind::Inner182 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<182>>>()
                    + std::mem::size_of::<SparseInnerNode<182>>()
            }
            VerkleNodeKind::Inner183 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<183>>>()
                    + std::mem::size_of::<SparseInnerNode<183>>()
            }
            VerkleNodeKind::Inner184 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<184>>>()
                    + std::mem::size_of::<SparseInnerNode<184>>()
            }
            VerkleNodeKind::Inner185 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<185>>>()
                    + std::mem::size_of::<SparseInnerNode<185>>()
            }
            VerkleNodeKind::Inner186 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<186>>>()
                    + std::mem::size_of::<SparseInnerNode<186>>()
            }
            VerkleNodeKind::Inner187 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<187>>>()
                    + std::mem::size_of::<SparseInnerNode<187>>()
            }
            VerkleNodeKind::Inner188 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<188>>>()
                    + std::mem::size_of::<SparseInnerNode<188>>()
            }
            VerkleNodeKind::Inner189 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<189>>>()
                    + std::mem::size_of::<SparseInnerNode<189>>()
            }
            VerkleNodeKind::Inner190 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<190>>>()
                    + std::mem::size_of::<SparseInnerNode<190>>()
            }
            VerkleNodeKind::Inner191 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<191>>>()
                    + std::mem::size_of::<SparseInnerNode<191>>()
            }
            VerkleNodeKind::Inner192 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<192>>>()
                    + std::mem::size_of::<SparseInnerNode<192>>()
            }
            VerkleNodeKind::Inner193 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<193>>>()
                    + std::mem::size_of::<SparseInnerNode<193>>()
            }
            VerkleNodeKind::Inner194 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<194>>>()
                    + std::mem::size_of::<SparseInnerNode<194>>()
            }
            VerkleNodeKind::Inner195 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<195>>>()
                    + std::mem::size_of::<SparseInnerNode<195>>()
            }
            VerkleNodeKind::Inner196 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<196>>>()
                    + std::mem::size_of::<SparseInnerNode<196>>()
            }
            VerkleNodeKind::Inner197 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<197>>>()
                    + std::mem::size_of::<SparseInnerNode<197>>()
            }
            VerkleNodeKind::Inner198 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<198>>>()
                    + std::mem::size_of::<SparseInnerNode<198>>()
            }
            VerkleNodeKind::Inner199 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<199>>>()
                    + std::mem::size_of::<SparseInnerNode<199>>()
            }
            VerkleNodeKind::Inner200 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<200>>>()
                    + std::mem::size_of::<SparseInnerNode<200>>()
            }
            VerkleNodeKind::Inner201 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<201>>>()
                    + std::mem::size_of::<SparseInnerNode<201>>()
            }
            VerkleNodeKind::Inner202 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<202>>>()
                    + std::mem::size_of::<SparseInnerNode<202>>()
            }
            VerkleNodeKind::Inner203 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<203>>>()
                    + std::mem::size_of::<SparseInnerNode<203>>()
            }
            VerkleNodeKind::Inner204 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<204>>>()
                    + std::mem::size_of::<SparseInnerNode<204>>()
            }
            VerkleNodeKind::Inner205 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<205>>>()
                    + std::mem::size_of::<SparseInnerNode<205>>()
            }
            VerkleNodeKind::Inner206 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<206>>>()
                    + std::mem::size_of::<SparseInnerNode<206>>()
            }
            VerkleNodeKind::Inner207 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<207>>>()
                    + std::mem::size_of::<SparseInnerNode<207>>()
            }
            VerkleNodeKind::Inner208 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<208>>>()
                    + std::mem::size_of::<SparseInnerNode<208>>()
            }
            VerkleNodeKind::Inner209 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<209>>>()
                    + std::mem::size_of::<SparseInnerNode<209>>()
            }
            VerkleNodeKind::Inner210 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<210>>>()
                    + std::mem::size_of::<SparseInnerNode<210>>()
            }
            VerkleNodeKind::Inner211 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<211>>>()
                    + std::mem::size_of::<SparseInnerNode<211>>()
            }
            VerkleNodeKind::Inner212 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<212>>>()
                    + std::mem::size_of::<SparseInnerNode<212>>()
            }
            VerkleNodeKind::Inner213 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<213>>>()
                    + std::mem::size_of::<SparseInnerNode<213>>()
            }
            VerkleNodeKind::Inner214 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<214>>>()
                    + std::mem::size_of::<SparseInnerNode<214>>()
            }
            VerkleNodeKind::Inner215 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<215>>>()
                    + std::mem::size_of::<SparseInnerNode<215>>()
            }
            VerkleNodeKind::Inner216 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<216>>>()
                    + std::mem::size_of::<SparseInnerNode<216>>()
            }
            VerkleNodeKind::Inner217 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<217>>>()
                    + std::mem::size_of::<SparseInnerNode<217>>()
            }
            VerkleNodeKind::Inner218 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<218>>>()
                    + std::mem::size_of::<SparseInnerNode<218>>()
            }
            VerkleNodeKind::Inner219 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<219>>>()
                    + std::mem::size_of::<SparseInnerNode<219>>()
            }
            VerkleNodeKind::Inner220 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<220>>>()
                    + std::mem::size_of::<SparseInnerNode<220>>()
            }
            VerkleNodeKind::Inner221 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<221>>>()
                    + std::mem::size_of::<SparseInnerNode<221>>()
            }
            VerkleNodeKind::Inner222 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<222>>>()
                    + std::mem::size_of::<SparseInnerNode<222>>()
            }
            VerkleNodeKind::Inner223 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<223>>>()
                    + std::mem::size_of::<SparseInnerNode<223>>()
            }
            VerkleNodeKind::Inner224 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<224>>>()
                    + std::mem::size_of::<SparseInnerNode<224>>()
            }
            VerkleNodeKind::Inner225 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<225>>>()
                    + std::mem::size_of::<SparseInnerNode<225>>()
            }
            VerkleNodeKind::Inner226 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<226>>>()
                    + std::mem::size_of::<SparseInnerNode<226>>()
            }
            VerkleNodeKind::Inner227 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<227>>>()
                    + std::mem::size_of::<SparseInnerNode<227>>()
            }
            VerkleNodeKind::Inner228 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<228>>>()
                    + std::mem::size_of::<SparseInnerNode<228>>()
            }
            VerkleNodeKind::Inner229 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<229>>>()
                    + std::mem::size_of::<SparseInnerNode<229>>()
            }
            VerkleNodeKind::Inner230 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<230>>>()
                    + std::mem::size_of::<SparseInnerNode<230>>()
            }
            VerkleNodeKind::Inner231 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<231>>>()
                    + std::mem::size_of::<SparseInnerNode<231>>()
            }
            VerkleNodeKind::Inner232 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<232>>>()
                    + std::mem::size_of::<SparseInnerNode<232>>()
            }
            VerkleNodeKind::Inner233 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<233>>>()
                    + std::mem::size_of::<SparseInnerNode<233>>()
            }
            VerkleNodeKind::Inner234 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<234>>>()
                    + std::mem::size_of::<SparseInnerNode<234>>()
            }
            VerkleNodeKind::Inner235 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<235>>>()
                    + std::mem::size_of::<SparseInnerNode<235>>()
            }
            VerkleNodeKind::Inner236 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<236>>>()
                    + std::mem::size_of::<SparseInnerNode<236>>()
            }
            VerkleNodeKind::Inner237 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<237>>>()
                    + std::mem::size_of::<SparseInnerNode<237>>()
            }
            VerkleNodeKind::Inner238 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<238>>>()
                    + std::mem::size_of::<SparseInnerNode<238>>()
            }
            VerkleNodeKind::Inner239 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<239>>>()
                    + std::mem::size_of::<SparseInnerNode<239>>()
            }
            VerkleNodeKind::Inner240 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<240>>>()
                    + std::mem::size_of::<SparseInnerNode<240>>()
            }
            VerkleNodeKind::Inner241 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<241>>>()
                    + std::mem::size_of::<SparseInnerNode<241>>()
            }
            VerkleNodeKind::Inner242 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<242>>>()
                    + std::mem::size_of::<SparseInnerNode<242>>()
            }
            VerkleNodeKind::Inner243 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<243>>>()
                    + std::mem::size_of::<SparseInnerNode<243>>()
            }
            VerkleNodeKind::Inner244 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<244>>>()
                    + std::mem::size_of::<SparseInnerNode<244>>()
            }
            VerkleNodeKind::Inner245 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<245>>>()
                    + std::mem::size_of::<SparseInnerNode<245>>()
            }
            VerkleNodeKind::Inner246 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<246>>>()
                    + std::mem::size_of::<SparseInnerNode<246>>()
            }
            VerkleNodeKind::Inner247 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<247>>>()
                    + std::mem::size_of::<SparseInnerNode<247>>()
            }
            VerkleNodeKind::Inner248 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<248>>>()
                    + std::mem::size_of::<SparseInnerNode<248>>()
            }
            VerkleNodeKind::Inner249 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<249>>>()
                    + std::mem::size_of::<SparseInnerNode<249>>()
            }
            VerkleNodeKind::Inner250 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<250>>>()
                    + std::mem::size_of::<SparseInnerNode<250>>()
            }
            VerkleNodeKind::Inner251 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<251>>>()
                    + std::mem::size_of::<SparseInnerNode<251>>()
            }
            VerkleNodeKind::Inner252 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<252>>>()
                    + std::mem::size_of::<SparseInnerNode<252>>()
            }
            VerkleNodeKind::Inner253 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<253>>>()
                    + std::mem::size_of::<SparseInnerNode<253>>()
            }
            VerkleNodeKind::Inner254 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<254>>>()
                    + std::mem::size_of::<SparseInnerNode<254>>()
            }
            VerkleNodeKind::Inner255 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseInnerNode<255>>>()
                    + std::mem::size_of::<SparseInnerNode<255>>()
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
            VerkleNodeKind::Leaf17 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<17>>>()
                    + std::mem::size_of::<SparseLeafNode<17>>()
            }
            VerkleNodeKind::Leaf18 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<18>>>()
                    + std::mem::size_of::<SparseLeafNode<18>>()
            }
            VerkleNodeKind::Leaf19 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<19>>>()
                    + std::mem::size_of::<SparseLeafNode<19>>()
            }
            VerkleNodeKind::Leaf20 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<20>>>()
                    + std::mem::size_of::<SparseLeafNode<20>>()
            }
            VerkleNodeKind::Leaf21 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<21>>>()
                    + std::mem::size_of::<SparseLeafNode<21>>()
            }
            VerkleNodeKind::Leaf22 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<22>>>()
                    + std::mem::size_of::<SparseLeafNode<22>>()
            }
            VerkleNodeKind::Leaf23 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<23>>>()
                    + std::mem::size_of::<SparseLeafNode<23>>()
            }
            VerkleNodeKind::Leaf24 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<24>>>()
                    + std::mem::size_of::<SparseLeafNode<24>>()
            }
            VerkleNodeKind::Leaf25 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<25>>>()
                    + std::mem::size_of::<SparseLeafNode<25>>()
            }
            VerkleNodeKind::Leaf26 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<26>>>()
                    + std::mem::size_of::<SparseLeafNode<26>>()
            }
            VerkleNodeKind::Leaf27 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<27>>>()
                    + std::mem::size_of::<SparseLeafNode<27>>()
            }
            VerkleNodeKind::Leaf28 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<28>>>()
                    + std::mem::size_of::<SparseLeafNode<28>>()
            }
            VerkleNodeKind::Leaf29 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<29>>>()
                    + std::mem::size_of::<SparseLeafNode<29>>()
            }
            VerkleNodeKind::Leaf30 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<30>>>()
                    + std::mem::size_of::<SparseLeafNode<30>>()
            }
            VerkleNodeKind::Leaf31 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<31>>>()
                    + std::mem::size_of::<SparseLeafNode<31>>()
            }
            VerkleNodeKind::Leaf32 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<32>>>()
                    + std::mem::size_of::<SparseLeafNode<32>>()
            }
            VerkleNodeKind::Leaf33 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<33>>>()
                    + std::mem::size_of::<SparseLeafNode<33>>()
            }
            VerkleNodeKind::Leaf34 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<34>>>()
                    + std::mem::size_of::<SparseLeafNode<34>>()
            }
            VerkleNodeKind::Leaf35 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<35>>>()
                    + std::mem::size_of::<SparseLeafNode<35>>()
            }
            VerkleNodeKind::Leaf36 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<36>>>()
                    + std::mem::size_of::<SparseLeafNode<36>>()
            }
            VerkleNodeKind::Leaf37 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<37>>>()
                    + std::mem::size_of::<SparseLeafNode<37>>()
            }
            VerkleNodeKind::Leaf38 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<38>>>()
                    + std::mem::size_of::<SparseLeafNode<38>>()
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
            VerkleNodeKind::Leaf42 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<42>>>()
                    + std::mem::size_of::<SparseLeafNode<42>>()
            }
            VerkleNodeKind::Leaf43 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<43>>>()
                    + std::mem::size_of::<SparseLeafNode<43>>()
            }
            VerkleNodeKind::Leaf44 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<44>>>()
                    + std::mem::size_of::<SparseLeafNode<44>>()
            }
            VerkleNodeKind::Leaf45 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<45>>>()
                    + std::mem::size_of::<SparseLeafNode<45>>()
            }
            VerkleNodeKind::Leaf46 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<46>>>()
                    + std::mem::size_of::<SparseLeafNode<46>>()
            }
            VerkleNodeKind::Leaf47 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<47>>>()
                    + std::mem::size_of::<SparseLeafNode<47>>()
            }
            VerkleNodeKind::Leaf48 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<48>>>()
                    + std::mem::size_of::<SparseLeafNode<48>>()
            }
            VerkleNodeKind::Leaf49 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<49>>>()
                    + std::mem::size_of::<SparseLeafNode<49>>()
            }
            VerkleNodeKind::Leaf50 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<50>>>()
                    + std::mem::size_of::<SparseLeafNode<50>>()
            }
            VerkleNodeKind::Leaf51 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<51>>>()
                    + std::mem::size_of::<SparseLeafNode<51>>()
            }
            VerkleNodeKind::Leaf52 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<52>>>()
                    + std::mem::size_of::<SparseLeafNode<52>>()
            }
            VerkleNodeKind::Leaf53 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<53>>>()
                    + std::mem::size_of::<SparseLeafNode<53>>()
            }
            VerkleNodeKind::Leaf54 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<54>>>()
                    + std::mem::size_of::<SparseLeafNode<54>>()
            }
            VerkleNodeKind::Leaf55 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<55>>>()
                    + std::mem::size_of::<SparseLeafNode<55>>()
            }
            VerkleNodeKind::Leaf56 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<56>>>()
                    + std::mem::size_of::<SparseLeafNode<56>>()
            }
            VerkleNodeKind::Leaf57 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<57>>>()
                    + std::mem::size_of::<SparseLeafNode<57>>()
            }
            VerkleNodeKind::Leaf58 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<58>>>()
                    + std::mem::size_of::<SparseLeafNode<58>>()
            }
            VerkleNodeKind::Leaf59 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<59>>>()
                    + std::mem::size_of::<SparseLeafNode<59>>()
            }
            VerkleNodeKind::Leaf60 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<60>>>()
                    + std::mem::size_of::<SparseLeafNode<60>>()
            }
            VerkleNodeKind::Leaf61 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<61>>>()
                    + std::mem::size_of::<SparseLeafNode<61>>()
            }
            VerkleNodeKind::Leaf62 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<62>>>()
                    + std::mem::size_of::<SparseLeafNode<62>>()
            }
            VerkleNodeKind::Leaf63 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<63>>>()
                    + std::mem::size_of::<SparseLeafNode<63>>()
            }
            VerkleNodeKind::Leaf64 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<64>>>()
                    + std::mem::size_of::<SparseLeafNode<64>>()
            }
            VerkleNodeKind::Leaf65 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<65>>>()
                    + std::mem::size_of::<SparseLeafNode<65>>()
            }
            VerkleNodeKind::Leaf66 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<66>>>()
                    + std::mem::size_of::<SparseLeafNode<66>>()
            }
            VerkleNodeKind::Leaf67 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<67>>>()
                    + std::mem::size_of::<SparseLeafNode<67>>()
            }
            VerkleNodeKind::Leaf68 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<68>>>()
                    + std::mem::size_of::<SparseLeafNode<68>>()
            }
            VerkleNodeKind::Leaf69 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<69>>>()
                    + std::mem::size_of::<SparseLeafNode<69>>()
            }
            VerkleNodeKind::Leaf70 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<70>>>()
                    + std::mem::size_of::<SparseLeafNode<70>>()
            }
            VerkleNodeKind::Leaf71 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<71>>>()
                    + std::mem::size_of::<SparseLeafNode<71>>()
            }
            VerkleNodeKind::Leaf72 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<72>>>()
                    + std::mem::size_of::<SparseLeafNode<72>>()
            }
            VerkleNodeKind::Leaf73 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<73>>>()
                    + std::mem::size_of::<SparseLeafNode<73>>()
            }
            VerkleNodeKind::Leaf74 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<74>>>()
                    + std::mem::size_of::<SparseLeafNode<74>>()
            }
            VerkleNodeKind::Leaf75 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<75>>>()
                    + std::mem::size_of::<SparseLeafNode<75>>()
            }
            VerkleNodeKind::Leaf76 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<76>>>()
                    + std::mem::size_of::<SparseLeafNode<76>>()
            }
            VerkleNodeKind::Leaf77 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<77>>>()
                    + std::mem::size_of::<SparseLeafNode<77>>()
            }
            VerkleNodeKind::Leaf78 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<78>>>()
                    + std::mem::size_of::<SparseLeafNode<78>>()
            }
            VerkleNodeKind::Leaf79 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<79>>>()
                    + std::mem::size_of::<SparseLeafNode<79>>()
            }
            VerkleNodeKind::Leaf80 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<80>>>()
                    + std::mem::size_of::<SparseLeafNode<80>>()
            }
            VerkleNodeKind::Leaf81 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<81>>>()
                    + std::mem::size_of::<SparseLeafNode<81>>()
            }
            VerkleNodeKind::Leaf82 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<82>>>()
                    + std::mem::size_of::<SparseLeafNode<82>>()
            }
            VerkleNodeKind::Leaf83 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<83>>>()
                    + std::mem::size_of::<SparseLeafNode<83>>()
            }
            VerkleNodeKind::Leaf84 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<84>>>()
                    + std::mem::size_of::<SparseLeafNode<84>>()
            }
            VerkleNodeKind::Leaf85 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<85>>>()
                    + std::mem::size_of::<SparseLeafNode<85>>()
            }
            VerkleNodeKind::Leaf86 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<86>>>()
                    + std::mem::size_of::<SparseLeafNode<86>>()
            }
            VerkleNodeKind::Leaf87 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<87>>>()
                    + std::mem::size_of::<SparseLeafNode<87>>()
            }
            VerkleNodeKind::Leaf88 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<88>>>()
                    + std::mem::size_of::<SparseLeafNode<88>>()
            }
            VerkleNodeKind::Leaf89 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<89>>>()
                    + std::mem::size_of::<SparseLeafNode<89>>()
            }
            VerkleNodeKind::Leaf90 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<90>>>()
                    + std::mem::size_of::<SparseLeafNode<90>>()
            }
            VerkleNodeKind::Leaf91 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<91>>>()
                    + std::mem::size_of::<SparseLeafNode<91>>()
            }
            VerkleNodeKind::Leaf92 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<92>>>()
                    + std::mem::size_of::<SparseLeafNode<92>>()
            }
            VerkleNodeKind::Leaf93 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<93>>>()
                    + std::mem::size_of::<SparseLeafNode<93>>()
            }
            VerkleNodeKind::Leaf94 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<94>>>()
                    + std::mem::size_of::<SparseLeafNode<94>>()
            }
            VerkleNodeKind::Leaf95 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<95>>>()
                    + std::mem::size_of::<SparseLeafNode<95>>()
            }
            VerkleNodeKind::Leaf96 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<96>>>()
                    + std::mem::size_of::<SparseLeafNode<96>>()
            }
            VerkleNodeKind::Leaf97 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<97>>>()
                    + std::mem::size_of::<SparseLeafNode<97>>()
            }
            VerkleNodeKind::Leaf98 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<98>>>()
                    + std::mem::size_of::<SparseLeafNode<98>>()
            }
            VerkleNodeKind::Leaf99 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<99>>>()
                    + std::mem::size_of::<SparseLeafNode<99>>()
            }
            VerkleNodeKind::Leaf100 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<100>>>()
                    + std::mem::size_of::<SparseLeafNode<100>>()
            }
            VerkleNodeKind::Leaf101 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<101>>>()
                    + std::mem::size_of::<SparseLeafNode<101>>()
            }
            VerkleNodeKind::Leaf102 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<102>>>()
                    + std::mem::size_of::<SparseLeafNode<102>>()
            }
            VerkleNodeKind::Leaf103 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<103>>>()
                    + std::mem::size_of::<SparseLeafNode<103>>()
            }
            VerkleNodeKind::Leaf104 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<104>>>()
                    + std::mem::size_of::<SparseLeafNode<104>>()
            }
            VerkleNodeKind::Leaf105 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<105>>>()
                    + std::mem::size_of::<SparseLeafNode<105>>()
            }
            VerkleNodeKind::Leaf106 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<106>>>()
                    + std::mem::size_of::<SparseLeafNode<106>>()
            }
            VerkleNodeKind::Leaf107 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<107>>>()
                    + std::mem::size_of::<SparseLeafNode<107>>()
            }
            VerkleNodeKind::Leaf108 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<108>>>()
                    + std::mem::size_of::<SparseLeafNode<108>>()
            }
            VerkleNodeKind::Leaf109 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<109>>>()
                    + std::mem::size_of::<SparseLeafNode<109>>()
            }
            VerkleNodeKind::Leaf110 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<110>>>()
                    + std::mem::size_of::<SparseLeafNode<110>>()
            }
            VerkleNodeKind::Leaf111 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<111>>>()
                    + std::mem::size_of::<SparseLeafNode<111>>()
            }
            VerkleNodeKind::Leaf112 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<112>>>()
                    + std::mem::size_of::<SparseLeafNode<112>>()
            }
            VerkleNodeKind::Leaf113 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<113>>>()
                    + std::mem::size_of::<SparseLeafNode<113>>()
            }
            VerkleNodeKind::Leaf114 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<114>>>()
                    + std::mem::size_of::<SparseLeafNode<114>>()
            }
            VerkleNodeKind::Leaf115 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<115>>>()
                    + std::mem::size_of::<SparseLeafNode<115>>()
            }
            VerkleNodeKind::Leaf116 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<116>>>()
                    + std::mem::size_of::<SparseLeafNode<116>>()
            }
            VerkleNodeKind::Leaf117 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<117>>>()
                    + std::mem::size_of::<SparseLeafNode<117>>()
            }
            VerkleNodeKind::Leaf118 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<118>>>()
                    + std::mem::size_of::<SparseLeafNode<118>>()
            }
            VerkleNodeKind::Leaf119 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<119>>>()
                    + std::mem::size_of::<SparseLeafNode<119>>()
            }
            VerkleNodeKind::Leaf120 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<120>>>()
                    + std::mem::size_of::<SparseLeafNode<120>>()
            }
            VerkleNodeKind::Leaf121 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<121>>>()
                    + std::mem::size_of::<SparseLeafNode<121>>()
            }
            VerkleNodeKind::Leaf122 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<122>>>()
                    + std::mem::size_of::<SparseLeafNode<122>>()
            }
            VerkleNodeKind::Leaf123 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<123>>>()
                    + std::mem::size_of::<SparseLeafNode<123>>()
            }
            VerkleNodeKind::Leaf124 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<124>>>()
                    + std::mem::size_of::<SparseLeafNode<124>>()
            }
            VerkleNodeKind::Leaf125 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<125>>>()
                    + std::mem::size_of::<SparseLeafNode<125>>()
            }
            VerkleNodeKind::Leaf126 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<126>>>()
                    + std::mem::size_of::<SparseLeafNode<126>>()
            }
            VerkleNodeKind::Leaf127 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<127>>>()
                    + std::mem::size_of::<SparseLeafNode<127>>()
            }
            VerkleNodeKind::Leaf128 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<128>>>()
                    + std::mem::size_of::<SparseLeafNode<128>>()
            }
            VerkleNodeKind::Leaf129 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<129>>>()
                    + std::mem::size_of::<SparseLeafNode<129>>()
            }
            VerkleNodeKind::Leaf130 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<130>>>()
                    + std::mem::size_of::<SparseLeafNode<130>>()
            }
            VerkleNodeKind::Leaf131 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<131>>>()
                    + std::mem::size_of::<SparseLeafNode<131>>()
            }
            VerkleNodeKind::Leaf132 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<132>>>()
                    + std::mem::size_of::<SparseLeafNode<132>>()
            }
            VerkleNodeKind::Leaf133 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<133>>>()
                    + std::mem::size_of::<SparseLeafNode<133>>()
            }
            VerkleNodeKind::Leaf134 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<134>>>()
                    + std::mem::size_of::<SparseLeafNode<134>>()
            }
            VerkleNodeKind::Leaf135 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<135>>>()
                    + std::mem::size_of::<SparseLeafNode<135>>()
            }
            VerkleNodeKind::Leaf136 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<136>>>()
                    + std::mem::size_of::<SparseLeafNode<136>>()
            }
            VerkleNodeKind::Leaf137 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<137>>>()
                    + std::mem::size_of::<SparseLeafNode<137>>()
            }
            VerkleNodeKind::Leaf138 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<138>>>()
                    + std::mem::size_of::<SparseLeafNode<138>>()
            }
            VerkleNodeKind::Leaf139 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<139>>>()
                    + std::mem::size_of::<SparseLeafNode<139>>()
            }
            VerkleNodeKind::Leaf140 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<140>>>()
                    + std::mem::size_of::<SparseLeafNode<140>>()
            }
            VerkleNodeKind::Leaf141 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<141>>>()
                    + std::mem::size_of::<SparseLeafNode<141>>()
            }
            VerkleNodeKind::Leaf142 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<142>>>()
                    + std::mem::size_of::<SparseLeafNode<142>>()
            }
            VerkleNodeKind::Leaf143 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<143>>>()
                    + std::mem::size_of::<SparseLeafNode<143>>()
            }
            VerkleNodeKind::Leaf144 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<144>>>()
                    + std::mem::size_of::<SparseLeafNode<144>>()
            }
            VerkleNodeKind::Leaf145 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<145>>>()
                    + std::mem::size_of::<SparseLeafNode<145>>()
            }
            VerkleNodeKind::Leaf146 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<146>>>()
                    + std::mem::size_of::<SparseLeafNode<146>>()
            }
            VerkleNodeKind::Leaf147 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<147>>>()
                    + std::mem::size_of::<SparseLeafNode<147>>()
            }
            VerkleNodeKind::Leaf148 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<148>>>()
                    + std::mem::size_of::<SparseLeafNode<148>>()
            }
            VerkleNodeKind::Leaf149 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<149>>>()
                    + std::mem::size_of::<SparseLeafNode<149>>()
            }
            VerkleNodeKind::Leaf150 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<150>>>()
                    + std::mem::size_of::<SparseLeafNode<150>>()
            }
            VerkleNodeKind::Leaf151 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<151>>>()
                    + std::mem::size_of::<SparseLeafNode<151>>()
            }
            VerkleNodeKind::Leaf152 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<152>>>()
                    + std::mem::size_of::<SparseLeafNode<152>>()
            }
            VerkleNodeKind::Leaf153 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<153>>>()
                    + std::mem::size_of::<SparseLeafNode<153>>()
            }
            VerkleNodeKind::Leaf154 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<154>>>()
                    + std::mem::size_of::<SparseLeafNode<154>>()
            }
            VerkleNodeKind::Leaf155 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<155>>>()
                    + std::mem::size_of::<SparseLeafNode<155>>()
            }
            VerkleNodeKind::Leaf156 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<156>>>()
                    + std::mem::size_of::<SparseLeafNode<156>>()
            }
            VerkleNodeKind::Leaf157 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<157>>>()
                    + std::mem::size_of::<SparseLeafNode<157>>()
            }
            VerkleNodeKind::Leaf158 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<158>>>()
                    + std::mem::size_of::<SparseLeafNode<158>>()
            }
            VerkleNodeKind::Leaf159 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<159>>>()
                    + std::mem::size_of::<SparseLeafNode<159>>()
            }
            VerkleNodeKind::Leaf160 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<160>>>()
                    + std::mem::size_of::<SparseLeafNode<160>>()
            }
            VerkleNodeKind::Leaf161 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<161>>>()
                    + std::mem::size_of::<SparseLeafNode<161>>()
            }
            VerkleNodeKind::Leaf162 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<162>>>()
                    + std::mem::size_of::<SparseLeafNode<162>>()
            }
            VerkleNodeKind::Leaf163 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<163>>>()
                    + std::mem::size_of::<SparseLeafNode<163>>()
            }
            VerkleNodeKind::Leaf164 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<164>>>()
                    + std::mem::size_of::<SparseLeafNode<164>>()
            }
            VerkleNodeKind::Leaf165 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<165>>>()
                    + std::mem::size_of::<SparseLeafNode<165>>()
            }
            VerkleNodeKind::Leaf166 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<166>>>()
                    + std::mem::size_of::<SparseLeafNode<166>>()
            }
            VerkleNodeKind::Leaf167 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<167>>>()
                    + std::mem::size_of::<SparseLeafNode<167>>()
            }
            VerkleNodeKind::Leaf168 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<168>>>()
                    + std::mem::size_of::<SparseLeafNode<168>>()
            }
            VerkleNodeKind::Leaf169 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<169>>>()
                    + std::mem::size_of::<SparseLeafNode<169>>()
            }
            VerkleNodeKind::Leaf170 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<170>>>()
                    + std::mem::size_of::<SparseLeafNode<170>>()
            }
            VerkleNodeKind::Leaf171 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<171>>>()
                    + std::mem::size_of::<SparseLeafNode<171>>()
            }
            VerkleNodeKind::Leaf172 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<172>>>()
                    + std::mem::size_of::<SparseLeafNode<172>>()
            }
            VerkleNodeKind::Leaf173 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<173>>>()
                    + std::mem::size_of::<SparseLeafNode<173>>()
            }
            VerkleNodeKind::Leaf174 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<174>>>()
                    + std::mem::size_of::<SparseLeafNode<174>>()
            }
            VerkleNodeKind::Leaf175 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<175>>>()
                    + std::mem::size_of::<SparseLeafNode<175>>()
            }
            VerkleNodeKind::Leaf176 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<176>>>()
                    + std::mem::size_of::<SparseLeafNode<176>>()
            }
            VerkleNodeKind::Leaf177 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<177>>>()
                    + std::mem::size_of::<SparseLeafNode<177>>()
            }
            VerkleNodeKind::Leaf178 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<178>>>()
                    + std::mem::size_of::<SparseLeafNode<178>>()
            }
            VerkleNodeKind::Leaf179 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<179>>>()
                    + std::mem::size_of::<SparseLeafNode<179>>()
            }
            VerkleNodeKind::Leaf180 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<180>>>()
                    + std::mem::size_of::<SparseLeafNode<180>>()
            }
            VerkleNodeKind::Leaf181 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<181>>>()
                    + std::mem::size_of::<SparseLeafNode<181>>()
            }
            VerkleNodeKind::Leaf182 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<182>>>()
                    + std::mem::size_of::<SparseLeafNode<182>>()
            }
            VerkleNodeKind::Leaf183 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<183>>>()
                    + std::mem::size_of::<SparseLeafNode<183>>()
            }
            VerkleNodeKind::Leaf184 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<184>>>()
                    + std::mem::size_of::<SparseLeafNode<184>>()
            }
            VerkleNodeKind::Leaf185 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<185>>>()
                    + std::mem::size_of::<SparseLeafNode<185>>()
            }
            VerkleNodeKind::Leaf186 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<186>>>()
                    + std::mem::size_of::<SparseLeafNode<186>>()
            }
            VerkleNodeKind::Leaf187 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<187>>>()
                    + std::mem::size_of::<SparseLeafNode<187>>()
            }
            VerkleNodeKind::Leaf188 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<188>>>()
                    + std::mem::size_of::<SparseLeafNode<188>>()
            }
            VerkleNodeKind::Leaf189 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<189>>>()
                    + std::mem::size_of::<SparseLeafNode<189>>()
            }
            VerkleNodeKind::Leaf190 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<190>>>()
                    + std::mem::size_of::<SparseLeafNode<190>>()
            }
            VerkleNodeKind::Leaf191 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<191>>>()
                    + std::mem::size_of::<SparseLeafNode<191>>()
            }
            VerkleNodeKind::Leaf192 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<192>>>()
                    + std::mem::size_of::<SparseLeafNode<192>>()
            }
            VerkleNodeKind::Leaf193 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<193>>>()
                    + std::mem::size_of::<SparseLeafNode<193>>()
            }
            VerkleNodeKind::Leaf194 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<194>>>()
                    + std::mem::size_of::<SparseLeafNode<194>>()
            }
            VerkleNodeKind::Leaf195 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<195>>>()
                    + std::mem::size_of::<SparseLeafNode<195>>()
            }
            VerkleNodeKind::Leaf196 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<196>>>()
                    + std::mem::size_of::<SparseLeafNode<196>>()
            }
            VerkleNodeKind::Leaf197 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<197>>>()
                    + std::mem::size_of::<SparseLeafNode<197>>()
            }
            VerkleNodeKind::Leaf198 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<198>>>()
                    + std::mem::size_of::<SparseLeafNode<198>>()
            }
            VerkleNodeKind::Leaf199 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<199>>>()
                    + std::mem::size_of::<SparseLeafNode<199>>()
            }
            VerkleNodeKind::Leaf200 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<200>>>()
                    + std::mem::size_of::<SparseLeafNode<200>>()
            }
            VerkleNodeKind::Leaf201 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<201>>>()
                    + std::mem::size_of::<SparseLeafNode<201>>()
            }
            VerkleNodeKind::Leaf202 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<202>>>()
                    + std::mem::size_of::<SparseLeafNode<202>>()
            }
            VerkleNodeKind::Leaf203 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<203>>>()
                    + std::mem::size_of::<SparseLeafNode<203>>()
            }
            VerkleNodeKind::Leaf204 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<204>>>()
                    + std::mem::size_of::<SparseLeafNode<204>>()
            }
            VerkleNodeKind::Leaf205 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<205>>>()
                    + std::mem::size_of::<SparseLeafNode<205>>()
            }
            VerkleNodeKind::Leaf206 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<206>>>()
                    + std::mem::size_of::<SparseLeafNode<206>>()
            }
            VerkleNodeKind::Leaf207 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<207>>>()
                    + std::mem::size_of::<SparseLeafNode<207>>()
            }
            VerkleNodeKind::Leaf208 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<208>>>()
                    + std::mem::size_of::<SparseLeafNode<208>>()
            }
            VerkleNodeKind::Leaf209 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<209>>>()
                    + std::mem::size_of::<SparseLeafNode<209>>()
            }
            VerkleNodeKind::Leaf210 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<210>>>()
                    + std::mem::size_of::<SparseLeafNode<210>>()
            }
            VerkleNodeKind::Leaf211 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<211>>>()
                    + std::mem::size_of::<SparseLeafNode<211>>()
            }
            VerkleNodeKind::Leaf212 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<212>>>()
                    + std::mem::size_of::<SparseLeafNode<212>>()
            }
            VerkleNodeKind::Leaf213 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<213>>>()
                    + std::mem::size_of::<SparseLeafNode<213>>()
            }
            VerkleNodeKind::Leaf214 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<214>>>()
                    + std::mem::size_of::<SparseLeafNode<214>>()
            }
            VerkleNodeKind::Leaf215 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<215>>>()
                    + std::mem::size_of::<SparseLeafNode<215>>()
            }
            VerkleNodeKind::Leaf216 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<216>>>()
                    + std::mem::size_of::<SparseLeafNode<216>>()
            }
            VerkleNodeKind::Leaf217 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<217>>>()
                    + std::mem::size_of::<SparseLeafNode<217>>()
            }
            VerkleNodeKind::Leaf218 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<218>>>()
                    + std::mem::size_of::<SparseLeafNode<218>>()
            }
            VerkleNodeKind::Leaf219 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<219>>>()
                    + std::mem::size_of::<SparseLeafNode<219>>()
            }
            VerkleNodeKind::Leaf220 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<220>>>()
                    + std::mem::size_of::<SparseLeafNode<220>>()
            }
            VerkleNodeKind::Leaf221 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<221>>>()
                    + std::mem::size_of::<SparseLeafNode<221>>()
            }
            VerkleNodeKind::Leaf222 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<222>>>()
                    + std::mem::size_of::<SparseLeafNode<222>>()
            }
            VerkleNodeKind::Leaf223 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<223>>>()
                    + std::mem::size_of::<SparseLeafNode<223>>()
            }
            VerkleNodeKind::Leaf224 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<224>>>()
                    + std::mem::size_of::<SparseLeafNode<224>>()
            }
            VerkleNodeKind::Leaf225 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<225>>>()
                    + std::mem::size_of::<SparseLeafNode<225>>()
            }
            VerkleNodeKind::Leaf226 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<226>>>()
                    + std::mem::size_of::<SparseLeafNode<226>>()
            }
            VerkleNodeKind::Leaf227 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<227>>>()
                    + std::mem::size_of::<SparseLeafNode<227>>()
            }
            VerkleNodeKind::Leaf228 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<228>>>()
                    + std::mem::size_of::<SparseLeafNode<228>>()
            }
            VerkleNodeKind::Leaf229 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<229>>>()
                    + std::mem::size_of::<SparseLeafNode<229>>()
            }
            VerkleNodeKind::Leaf230 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<230>>>()
                    + std::mem::size_of::<SparseLeafNode<230>>()
            }
            VerkleNodeKind::Leaf231 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<231>>>()
                    + std::mem::size_of::<SparseLeafNode<231>>()
            }
            VerkleNodeKind::Leaf232 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<232>>>()
                    + std::mem::size_of::<SparseLeafNode<232>>()
            }
            VerkleNodeKind::Leaf233 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<233>>>()
                    + std::mem::size_of::<SparseLeafNode<233>>()
            }
            VerkleNodeKind::Leaf234 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<234>>>()
                    + std::mem::size_of::<SparseLeafNode<234>>()
            }
            VerkleNodeKind::Leaf235 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<235>>>()
                    + std::mem::size_of::<SparseLeafNode<235>>()
            }
            VerkleNodeKind::Leaf236 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<236>>>()
                    + std::mem::size_of::<SparseLeafNode<236>>()
            }
            VerkleNodeKind::Leaf237 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<237>>>()
                    + std::mem::size_of::<SparseLeafNode<237>>()
            }
            VerkleNodeKind::Leaf238 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<238>>>()
                    + std::mem::size_of::<SparseLeafNode<238>>()
            }
            VerkleNodeKind::Leaf239 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<239>>>()
                    + std::mem::size_of::<SparseLeafNode<239>>()
            }
            VerkleNodeKind::Leaf240 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<240>>>()
                    + std::mem::size_of::<SparseLeafNode<240>>()
            }
            VerkleNodeKind::Leaf241 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<241>>>()
                    + std::mem::size_of::<SparseLeafNode<241>>()
            }
            VerkleNodeKind::Leaf242 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<242>>>()
                    + std::mem::size_of::<SparseLeafNode<242>>()
            }
            VerkleNodeKind::Leaf243 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<243>>>()
                    + std::mem::size_of::<SparseLeafNode<243>>()
            }
            VerkleNodeKind::Leaf244 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<244>>>()
                    + std::mem::size_of::<SparseLeafNode<244>>()
            }
            VerkleNodeKind::Leaf245 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<245>>>()
                    + std::mem::size_of::<SparseLeafNode<245>>()
            }
            VerkleNodeKind::Leaf246 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<246>>>()
                    + std::mem::size_of::<SparseLeafNode<246>>()
            }
            VerkleNodeKind::Leaf247 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<247>>>()
                    + std::mem::size_of::<SparseLeafNode<247>>()
            }
            VerkleNodeKind::Leaf248 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<248>>>()
                    + std::mem::size_of::<SparseLeafNode<248>>()
            }
            VerkleNodeKind::Leaf249 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<249>>>()
                    + std::mem::size_of::<SparseLeafNode<249>>()
            }
            VerkleNodeKind::Leaf250 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<250>>>()
                    + std::mem::size_of::<SparseLeafNode<250>>()
            }
            VerkleNodeKind::Leaf251 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<251>>>()
                    + std::mem::size_of::<SparseLeafNode<251>>()
            }
            VerkleNodeKind::Leaf252 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<252>>>()
                    + std::mem::size_of::<SparseLeafNode<252>>()
            }
            VerkleNodeKind::Leaf253 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<253>>>()
                    + std::mem::size_of::<SparseLeafNode<253>>()
            }
            VerkleNodeKind::Leaf254 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<254>>>()
                    + std::mem::size_of::<SparseLeafNode<254>>()
            }
            VerkleNodeKind::Leaf255 => {
                std::mem::size_of::<VerkleNode>()
                    + std::mem::size_of::<Box<SparseLeafNode<255>>>()
                    + std::mem::size_of::<SparseLeafNode<255>>()
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
        VerkleNodeKind::Leaf17 => Ok(VerkleNode::Leaf17(Box::new(
            SparseLeafNode::<17>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf18 => Ok(VerkleNode::Leaf18(Box::new(
            SparseLeafNode::<18>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf19 => Ok(VerkleNode::Leaf19(Box::new(
            SparseLeafNode::<19>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf20 => Ok(VerkleNode::Leaf20(Box::new(
            SparseLeafNode::<20>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf21 => Ok(VerkleNode::Leaf21(Box::new(
            SparseLeafNode::<21>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf22 => Ok(VerkleNode::Leaf22(Box::new(
            SparseLeafNode::<22>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf23 => Ok(VerkleNode::Leaf23(Box::new(
            SparseLeafNode::<23>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf24 => Ok(VerkleNode::Leaf24(Box::new(
            SparseLeafNode::<24>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf25 => Ok(VerkleNode::Leaf25(Box::new(
            SparseLeafNode::<25>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf26 => Ok(VerkleNode::Leaf26(Box::new(
            SparseLeafNode::<26>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf27 => Ok(VerkleNode::Leaf27(Box::new(
            SparseLeafNode::<27>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf28 => Ok(VerkleNode::Leaf28(Box::new(
            SparseLeafNode::<28>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf29 => Ok(VerkleNode::Leaf29(Box::new(
            SparseLeafNode::<29>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf30 => Ok(VerkleNode::Leaf30(Box::new(
            SparseLeafNode::<30>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf31 => Ok(VerkleNode::Leaf31(Box::new(
            SparseLeafNode::<31>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf32 => Ok(VerkleNode::Leaf32(Box::new(
            SparseLeafNode::<32>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf33 => Ok(VerkleNode::Leaf33(Box::new(
            SparseLeafNode::<33>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf34 => Ok(VerkleNode::Leaf34(Box::new(
            SparseLeafNode::<34>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf35 => Ok(VerkleNode::Leaf35(Box::new(
            SparseLeafNode::<35>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf36 => Ok(VerkleNode::Leaf36(Box::new(
            SparseLeafNode::<36>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf37 => Ok(VerkleNode::Leaf37(Box::new(
            SparseLeafNode::<37>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf38 => Ok(VerkleNode::Leaf38(Box::new(
            SparseLeafNode::<38>::from_existing(stem, values, commitment)?,
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
        VerkleNodeKind::Leaf42 => Ok(VerkleNode::Leaf42(Box::new(
            SparseLeafNode::<42>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf43 => Ok(VerkleNode::Leaf43(Box::new(
            SparseLeafNode::<43>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf44 => Ok(VerkleNode::Leaf44(Box::new(
            SparseLeafNode::<44>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf45 => Ok(VerkleNode::Leaf45(Box::new(
            SparseLeafNode::<45>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf46 => Ok(VerkleNode::Leaf46(Box::new(
            SparseLeafNode::<46>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf47 => Ok(VerkleNode::Leaf47(Box::new(
            SparseLeafNode::<47>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf48 => Ok(VerkleNode::Leaf48(Box::new(
            SparseLeafNode::<48>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf49 => Ok(VerkleNode::Leaf49(Box::new(
            SparseLeafNode::<49>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf50 => Ok(VerkleNode::Leaf50(Box::new(
            SparseLeafNode::<50>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf51 => Ok(VerkleNode::Leaf51(Box::new(
            SparseLeafNode::<51>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf52 => Ok(VerkleNode::Leaf52(Box::new(
            SparseLeafNode::<52>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf53 => Ok(VerkleNode::Leaf53(Box::new(
            SparseLeafNode::<53>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf54 => Ok(VerkleNode::Leaf54(Box::new(
            SparseLeafNode::<54>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf55 => Ok(VerkleNode::Leaf55(Box::new(
            SparseLeafNode::<55>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf56 => Ok(VerkleNode::Leaf56(Box::new(
            SparseLeafNode::<56>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf57 => Ok(VerkleNode::Leaf57(Box::new(
            SparseLeafNode::<57>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf58 => Ok(VerkleNode::Leaf58(Box::new(
            SparseLeafNode::<58>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf59 => Ok(VerkleNode::Leaf59(Box::new(
            SparseLeafNode::<59>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf60 => Ok(VerkleNode::Leaf60(Box::new(
            SparseLeafNode::<60>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf61 => Ok(VerkleNode::Leaf61(Box::new(
            SparseLeafNode::<61>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf62 => Ok(VerkleNode::Leaf62(Box::new(
            SparseLeafNode::<62>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf63 => Ok(VerkleNode::Leaf63(Box::new(
            SparseLeafNode::<63>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf64 => Ok(VerkleNode::Leaf64(Box::new(
            SparseLeafNode::<64>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf65 => Ok(VerkleNode::Leaf65(Box::new(
            SparseLeafNode::<65>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf66 => Ok(VerkleNode::Leaf66(Box::new(
            SparseLeafNode::<66>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf67 => Ok(VerkleNode::Leaf67(Box::new(
            SparseLeafNode::<67>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf68 => Ok(VerkleNode::Leaf68(Box::new(
            SparseLeafNode::<68>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf69 => Ok(VerkleNode::Leaf69(Box::new(
            SparseLeafNode::<69>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf70 => Ok(VerkleNode::Leaf70(Box::new(
            SparseLeafNode::<70>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf71 => Ok(VerkleNode::Leaf71(Box::new(
            SparseLeafNode::<71>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf72 => Ok(VerkleNode::Leaf72(Box::new(
            SparseLeafNode::<72>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf73 => Ok(VerkleNode::Leaf73(Box::new(
            SparseLeafNode::<73>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf74 => Ok(VerkleNode::Leaf74(Box::new(
            SparseLeafNode::<74>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf75 => Ok(VerkleNode::Leaf75(Box::new(
            SparseLeafNode::<75>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf76 => Ok(VerkleNode::Leaf76(Box::new(
            SparseLeafNode::<76>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf77 => Ok(VerkleNode::Leaf77(Box::new(
            SparseLeafNode::<77>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf78 => Ok(VerkleNode::Leaf78(Box::new(
            SparseLeafNode::<78>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf79 => Ok(VerkleNode::Leaf79(Box::new(
            SparseLeafNode::<79>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf80 => Ok(VerkleNode::Leaf80(Box::new(
            SparseLeafNode::<80>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf81 => Ok(VerkleNode::Leaf81(Box::new(
            SparseLeafNode::<81>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf82 => Ok(VerkleNode::Leaf82(Box::new(
            SparseLeafNode::<82>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf83 => Ok(VerkleNode::Leaf83(Box::new(
            SparseLeafNode::<83>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf84 => Ok(VerkleNode::Leaf84(Box::new(
            SparseLeafNode::<84>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf85 => Ok(VerkleNode::Leaf85(Box::new(
            SparseLeafNode::<85>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf86 => Ok(VerkleNode::Leaf86(Box::new(
            SparseLeafNode::<86>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf87 => Ok(VerkleNode::Leaf87(Box::new(
            SparseLeafNode::<87>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf88 => Ok(VerkleNode::Leaf88(Box::new(
            SparseLeafNode::<88>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf89 => Ok(VerkleNode::Leaf89(Box::new(
            SparseLeafNode::<89>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf90 => Ok(VerkleNode::Leaf90(Box::new(
            SparseLeafNode::<90>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf91 => Ok(VerkleNode::Leaf91(Box::new(
            SparseLeafNode::<91>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf92 => Ok(VerkleNode::Leaf92(Box::new(
            SparseLeafNode::<92>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf93 => Ok(VerkleNode::Leaf93(Box::new(
            SparseLeafNode::<93>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf94 => Ok(VerkleNode::Leaf94(Box::new(
            SparseLeafNode::<94>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf95 => Ok(VerkleNode::Leaf95(Box::new(
            SparseLeafNode::<95>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf96 => Ok(VerkleNode::Leaf96(Box::new(
            SparseLeafNode::<96>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf97 => Ok(VerkleNode::Leaf97(Box::new(
            SparseLeafNode::<97>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf98 => Ok(VerkleNode::Leaf98(Box::new(
            SparseLeafNode::<98>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf99 => Ok(VerkleNode::Leaf99(Box::new(
            SparseLeafNode::<99>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf100 => Ok(VerkleNode::Leaf100(Box::new(
            SparseLeafNode::<100>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf101 => Ok(VerkleNode::Leaf101(Box::new(
            SparseLeafNode::<101>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf102 => Ok(VerkleNode::Leaf102(Box::new(
            SparseLeafNode::<102>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf103 => Ok(VerkleNode::Leaf103(Box::new(
            SparseLeafNode::<103>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf104 => Ok(VerkleNode::Leaf104(Box::new(
            SparseLeafNode::<104>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf105 => Ok(VerkleNode::Leaf105(Box::new(
            SparseLeafNode::<105>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf106 => Ok(VerkleNode::Leaf106(Box::new(
            SparseLeafNode::<106>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf107 => Ok(VerkleNode::Leaf107(Box::new(
            SparseLeafNode::<107>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf108 => Ok(VerkleNode::Leaf108(Box::new(
            SparseLeafNode::<108>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf109 => Ok(VerkleNode::Leaf109(Box::new(
            SparseLeafNode::<109>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf110 => Ok(VerkleNode::Leaf110(Box::new(
            SparseLeafNode::<110>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf111 => Ok(VerkleNode::Leaf111(Box::new(
            SparseLeafNode::<111>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf112 => Ok(VerkleNode::Leaf112(Box::new(
            SparseLeafNode::<112>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf113 => Ok(VerkleNode::Leaf113(Box::new(
            SparseLeafNode::<113>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf114 => Ok(VerkleNode::Leaf114(Box::new(
            SparseLeafNode::<114>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf115 => Ok(VerkleNode::Leaf115(Box::new(
            SparseLeafNode::<115>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf116 => Ok(VerkleNode::Leaf116(Box::new(
            SparseLeafNode::<116>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf117 => Ok(VerkleNode::Leaf117(Box::new(
            SparseLeafNode::<117>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf118 => Ok(VerkleNode::Leaf118(Box::new(
            SparseLeafNode::<118>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf119 => Ok(VerkleNode::Leaf119(Box::new(
            SparseLeafNode::<119>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf120 => Ok(VerkleNode::Leaf120(Box::new(
            SparseLeafNode::<120>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf121 => Ok(VerkleNode::Leaf121(Box::new(
            SparseLeafNode::<121>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf122 => Ok(VerkleNode::Leaf122(Box::new(
            SparseLeafNode::<122>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf123 => Ok(VerkleNode::Leaf123(Box::new(
            SparseLeafNode::<123>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf124 => Ok(VerkleNode::Leaf124(Box::new(
            SparseLeafNode::<124>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf125 => Ok(VerkleNode::Leaf125(Box::new(
            SparseLeafNode::<125>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf126 => Ok(VerkleNode::Leaf126(Box::new(
            SparseLeafNode::<126>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf127 => Ok(VerkleNode::Leaf127(Box::new(
            SparseLeafNode::<127>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf128 => Ok(VerkleNode::Leaf128(Box::new(
            SparseLeafNode::<128>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf129 => Ok(VerkleNode::Leaf129(Box::new(
            SparseLeafNode::<129>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf130 => Ok(VerkleNode::Leaf130(Box::new(
            SparseLeafNode::<130>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf131 => Ok(VerkleNode::Leaf131(Box::new(
            SparseLeafNode::<131>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf132 => Ok(VerkleNode::Leaf132(Box::new(
            SparseLeafNode::<132>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf133 => Ok(VerkleNode::Leaf133(Box::new(
            SparseLeafNode::<133>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf134 => Ok(VerkleNode::Leaf134(Box::new(
            SparseLeafNode::<134>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf135 => Ok(VerkleNode::Leaf135(Box::new(
            SparseLeafNode::<135>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf136 => Ok(VerkleNode::Leaf136(Box::new(
            SparseLeafNode::<136>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf137 => Ok(VerkleNode::Leaf137(Box::new(
            SparseLeafNode::<137>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf138 => Ok(VerkleNode::Leaf138(Box::new(
            SparseLeafNode::<138>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf139 => Ok(VerkleNode::Leaf139(Box::new(
            SparseLeafNode::<139>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf140 => Ok(VerkleNode::Leaf140(Box::new(
            SparseLeafNode::<140>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf141 => Ok(VerkleNode::Leaf141(Box::new(
            SparseLeafNode::<141>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf142 => Ok(VerkleNode::Leaf142(Box::new(
            SparseLeafNode::<142>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf143 => Ok(VerkleNode::Leaf143(Box::new(
            SparseLeafNode::<143>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf144 => Ok(VerkleNode::Leaf144(Box::new(
            SparseLeafNode::<144>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf145 => Ok(VerkleNode::Leaf145(Box::new(
            SparseLeafNode::<145>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf146 => Ok(VerkleNode::Leaf146(Box::new(
            SparseLeafNode::<146>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf147 => Ok(VerkleNode::Leaf147(Box::new(
            SparseLeafNode::<147>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf148 => Ok(VerkleNode::Leaf148(Box::new(
            SparseLeafNode::<148>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf149 => Ok(VerkleNode::Leaf149(Box::new(
            SparseLeafNode::<149>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf150 => Ok(VerkleNode::Leaf150(Box::new(
            SparseLeafNode::<150>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf151 => Ok(VerkleNode::Leaf151(Box::new(
            SparseLeafNode::<151>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf152 => Ok(VerkleNode::Leaf152(Box::new(
            SparseLeafNode::<152>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf153 => Ok(VerkleNode::Leaf153(Box::new(
            SparseLeafNode::<153>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf154 => Ok(VerkleNode::Leaf154(Box::new(
            SparseLeafNode::<154>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf155 => Ok(VerkleNode::Leaf155(Box::new(
            SparseLeafNode::<155>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf156 => Ok(VerkleNode::Leaf156(Box::new(
            SparseLeafNode::<156>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf157 => Ok(VerkleNode::Leaf157(Box::new(
            SparseLeafNode::<157>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf158 => Ok(VerkleNode::Leaf158(Box::new(
            SparseLeafNode::<158>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf159 => Ok(VerkleNode::Leaf159(Box::new(
            SparseLeafNode::<159>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf160 => Ok(VerkleNode::Leaf160(Box::new(
            SparseLeafNode::<160>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf161 => Ok(VerkleNode::Leaf161(Box::new(
            SparseLeafNode::<161>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf162 => Ok(VerkleNode::Leaf162(Box::new(
            SparseLeafNode::<162>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf163 => Ok(VerkleNode::Leaf163(Box::new(
            SparseLeafNode::<163>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf164 => Ok(VerkleNode::Leaf164(Box::new(
            SparseLeafNode::<164>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf165 => Ok(VerkleNode::Leaf165(Box::new(
            SparseLeafNode::<165>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf166 => Ok(VerkleNode::Leaf166(Box::new(
            SparseLeafNode::<166>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf167 => Ok(VerkleNode::Leaf167(Box::new(
            SparseLeafNode::<167>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf168 => Ok(VerkleNode::Leaf168(Box::new(
            SparseLeafNode::<168>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf169 => Ok(VerkleNode::Leaf169(Box::new(
            SparseLeafNode::<169>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf170 => Ok(VerkleNode::Leaf170(Box::new(
            SparseLeafNode::<170>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf171 => Ok(VerkleNode::Leaf171(Box::new(
            SparseLeafNode::<171>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf172 => Ok(VerkleNode::Leaf172(Box::new(
            SparseLeafNode::<172>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf173 => Ok(VerkleNode::Leaf173(Box::new(
            SparseLeafNode::<173>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf174 => Ok(VerkleNode::Leaf174(Box::new(
            SparseLeafNode::<174>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf175 => Ok(VerkleNode::Leaf175(Box::new(
            SparseLeafNode::<175>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf176 => Ok(VerkleNode::Leaf176(Box::new(
            SparseLeafNode::<176>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf177 => Ok(VerkleNode::Leaf177(Box::new(
            SparseLeafNode::<177>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf178 => Ok(VerkleNode::Leaf178(Box::new(
            SparseLeafNode::<178>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf179 => Ok(VerkleNode::Leaf179(Box::new(
            SparseLeafNode::<179>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf180 => Ok(VerkleNode::Leaf180(Box::new(
            SparseLeafNode::<180>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf181 => Ok(VerkleNode::Leaf181(Box::new(
            SparseLeafNode::<181>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf182 => Ok(VerkleNode::Leaf182(Box::new(
            SparseLeafNode::<182>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf183 => Ok(VerkleNode::Leaf183(Box::new(
            SparseLeafNode::<183>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf184 => Ok(VerkleNode::Leaf184(Box::new(
            SparseLeafNode::<184>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf185 => Ok(VerkleNode::Leaf185(Box::new(
            SparseLeafNode::<185>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf186 => Ok(VerkleNode::Leaf186(Box::new(
            SparseLeafNode::<186>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf187 => Ok(VerkleNode::Leaf187(Box::new(
            SparseLeafNode::<187>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf188 => Ok(VerkleNode::Leaf188(Box::new(
            SparseLeafNode::<188>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf189 => Ok(VerkleNode::Leaf189(Box::new(
            SparseLeafNode::<189>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf190 => Ok(VerkleNode::Leaf190(Box::new(
            SparseLeafNode::<190>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf191 => Ok(VerkleNode::Leaf191(Box::new(
            SparseLeafNode::<191>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf192 => Ok(VerkleNode::Leaf192(Box::new(
            SparseLeafNode::<192>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf193 => Ok(VerkleNode::Leaf193(Box::new(
            SparseLeafNode::<193>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf194 => Ok(VerkleNode::Leaf194(Box::new(
            SparseLeafNode::<194>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf195 => Ok(VerkleNode::Leaf195(Box::new(
            SparseLeafNode::<195>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf196 => Ok(VerkleNode::Leaf196(Box::new(
            SparseLeafNode::<196>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf197 => Ok(VerkleNode::Leaf197(Box::new(
            SparseLeafNode::<197>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf198 => Ok(VerkleNode::Leaf198(Box::new(
            SparseLeafNode::<198>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf199 => Ok(VerkleNode::Leaf199(Box::new(
            SparseLeafNode::<199>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf200 => Ok(VerkleNode::Leaf200(Box::new(
            SparseLeafNode::<200>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf201 => Ok(VerkleNode::Leaf201(Box::new(
            SparseLeafNode::<201>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf202 => Ok(VerkleNode::Leaf202(Box::new(
            SparseLeafNode::<202>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf203 => Ok(VerkleNode::Leaf203(Box::new(
            SparseLeafNode::<203>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf204 => Ok(VerkleNode::Leaf204(Box::new(
            SparseLeafNode::<204>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf205 => Ok(VerkleNode::Leaf205(Box::new(
            SparseLeafNode::<205>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf206 => Ok(VerkleNode::Leaf206(Box::new(
            SparseLeafNode::<206>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf207 => Ok(VerkleNode::Leaf207(Box::new(
            SparseLeafNode::<207>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf208 => Ok(VerkleNode::Leaf208(Box::new(
            SparseLeafNode::<208>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf209 => Ok(VerkleNode::Leaf209(Box::new(
            SparseLeafNode::<209>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf210 => Ok(VerkleNode::Leaf210(Box::new(
            SparseLeafNode::<210>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf211 => Ok(VerkleNode::Leaf211(Box::new(
            SparseLeafNode::<211>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf212 => Ok(VerkleNode::Leaf212(Box::new(
            SparseLeafNode::<212>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf213 => Ok(VerkleNode::Leaf213(Box::new(
            SparseLeafNode::<213>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf214 => Ok(VerkleNode::Leaf214(Box::new(
            SparseLeafNode::<214>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf215 => Ok(VerkleNode::Leaf215(Box::new(
            SparseLeafNode::<215>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf216 => Ok(VerkleNode::Leaf216(Box::new(
            SparseLeafNode::<216>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf217 => Ok(VerkleNode::Leaf217(Box::new(
            SparseLeafNode::<217>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf218 => Ok(VerkleNode::Leaf218(Box::new(
            SparseLeafNode::<218>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf219 => Ok(VerkleNode::Leaf219(Box::new(
            SparseLeafNode::<219>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf220 => Ok(VerkleNode::Leaf220(Box::new(
            SparseLeafNode::<220>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf221 => Ok(VerkleNode::Leaf221(Box::new(
            SparseLeafNode::<221>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf222 => Ok(VerkleNode::Leaf222(Box::new(
            SparseLeafNode::<222>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf223 => Ok(VerkleNode::Leaf223(Box::new(
            SparseLeafNode::<223>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf224 => Ok(VerkleNode::Leaf224(Box::new(
            SparseLeafNode::<224>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf225 => Ok(VerkleNode::Leaf225(Box::new(
            SparseLeafNode::<225>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf226 => Ok(VerkleNode::Leaf226(Box::new(
            SparseLeafNode::<226>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf227 => Ok(VerkleNode::Leaf227(Box::new(
            SparseLeafNode::<227>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf228 => Ok(VerkleNode::Leaf228(Box::new(
            SparseLeafNode::<228>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf229 => Ok(VerkleNode::Leaf229(Box::new(
            SparseLeafNode::<229>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf230 => Ok(VerkleNode::Leaf230(Box::new(
            SparseLeafNode::<230>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf231 => Ok(VerkleNode::Leaf231(Box::new(
            SparseLeafNode::<231>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf232 => Ok(VerkleNode::Leaf232(Box::new(
            SparseLeafNode::<232>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf233 => Ok(VerkleNode::Leaf233(Box::new(
            SparseLeafNode::<233>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf234 => Ok(VerkleNode::Leaf234(Box::new(
            SparseLeafNode::<234>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf235 => Ok(VerkleNode::Leaf235(Box::new(
            SparseLeafNode::<235>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf236 => Ok(VerkleNode::Leaf236(Box::new(
            SparseLeafNode::<236>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf237 => Ok(VerkleNode::Leaf237(Box::new(
            SparseLeafNode::<237>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf238 => Ok(VerkleNode::Leaf238(Box::new(
            SparseLeafNode::<238>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf239 => Ok(VerkleNode::Leaf239(Box::new(
            SparseLeafNode::<239>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf240 => Ok(VerkleNode::Leaf240(Box::new(
            SparseLeafNode::<240>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf241 => Ok(VerkleNode::Leaf241(Box::new(
            SparseLeafNode::<241>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf242 => Ok(VerkleNode::Leaf242(Box::new(
            SparseLeafNode::<242>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf243 => Ok(VerkleNode::Leaf243(Box::new(
            SparseLeafNode::<243>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf244 => Ok(VerkleNode::Leaf244(Box::new(
            SparseLeafNode::<244>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf245 => Ok(VerkleNode::Leaf245(Box::new(
            SparseLeafNode::<245>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf246 => Ok(VerkleNode::Leaf246(Box::new(
            SparseLeafNode::<246>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf247 => Ok(VerkleNode::Leaf247(Box::new(
            SparseLeafNode::<247>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf248 => Ok(VerkleNode::Leaf248(Box::new(
            SparseLeafNode::<248>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf249 => Ok(VerkleNode::Leaf249(Box::new(
            SparseLeafNode::<249>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf250 => Ok(VerkleNode::Leaf250(Box::new(
            SparseLeafNode::<250>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf251 => Ok(VerkleNode::Leaf251(Box::new(
            SparseLeafNode::<251>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf252 => Ok(VerkleNode::Leaf252(Box::new(
            SparseLeafNode::<252>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf253 => Ok(VerkleNode::Leaf253(Box::new(
            SparseLeafNode::<253>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf254 => Ok(VerkleNode::Leaf254(Box::new(
            SparseLeafNode::<254>::from_existing(stem, values, commitment)?,
        ))),
        VerkleNodeKind::Leaf255 => Ok(VerkleNode::Leaf255(Box::new(
            SparseLeafNode::<255>::from_existing(stem, values, commitment)?,
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
        VerkleNodeKind::Inner1 => Ok(VerkleNode::Inner1(Box::new(
            SparseInnerNode::<1>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner2 => Ok(VerkleNode::Inner2(Box::new(
            SparseInnerNode::<2>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner3 => Ok(VerkleNode::Inner3(Box::new(
            SparseInnerNode::<3>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner4 => Ok(VerkleNode::Inner4(Box::new(
            SparseInnerNode::<4>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner5 => Ok(VerkleNode::Inner5(Box::new(
            SparseInnerNode::<5>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner6 => Ok(VerkleNode::Inner6(Box::new(
            SparseInnerNode::<6>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner7 => Ok(VerkleNode::Inner7(Box::new(
            SparseInnerNode::<7>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner8 => Ok(VerkleNode::Inner8(Box::new(
            SparseInnerNode::<8>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner9 => Ok(VerkleNode::Inner9(Box::new(
            SparseInnerNode::<9>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner10 => Ok(VerkleNode::Inner10(Box::new(
            SparseInnerNode::<10>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner11 => Ok(VerkleNode::Inner11(Box::new(
            SparseInnerNode::<11>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner12 => Ok(VerkleNode::Inner12(Box::new(
            SparseInnerNode::<12>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner13 => Ok(VerkleNode::Inner13(Box::new(
            SparseInnerNode::<13>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner14 => Ok(VerkleNode::Inner14(Box::new(
            SparseInnerNode::<14>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner15 => Ok(VerkleNode::Inner15(Box::new(
            SparseInnerNode::<15>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner16 => Ok(VerkleNode::Inner16(Box::new(
            SparseInnerNode::<16>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner17 => Ok(VerkleNode::Inner17(Box::new(
            SparseInnerNode::<17>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner18 => Ok(VerkleNode::Inner18(Box::new(
            SparseInnerNode::<18>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner19 => Ok(VerkleNode::Inner19(Box::new(
            SparseInnerNode::<19>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner20 => Ok(VerkleNode::Inner20(Box::new(
            SparseInnerNode::<20>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner21 => Ok(VerkleNode::Inner21(Box::new(
            SparseInnerNode::<21>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner22 => Ok(VerkleNode::Inner22(Box::new(
            SparseInnerNode::<22>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner23 => Ok(VerkleNode::Inner23(Box::new(
            SparseInnerNode::<23>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner24 => Ok(VerkleNode::Inner24(Box::new(
            SparseInnerNode::<24>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner25 => Ok(VerkleNode::Inner25(Box::new(
            SparseInnerNode::<25>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner26 => Ok(VerkleNode::Inner26(Box::new(
            SparseInnerNode::<26>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner27 => Ok(VerkleNode::Inner27(Box::new(
            SparseInnerNode::<27>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner28 => Ok(VerkleNode::Inner28(Box::new(
            SparseInnerNode::<28>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner29 => Ok(VerkleNode::Inner29(Box::new(
            SparseInnerNode::<29>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner30 => Ok(VerkleNode::Inner30(Box::new(
            SparseInnerNode::<30>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner31 => Ok(VerkleNode::Inner31(Box::new(
            SparseInnerNode::<31>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner32 => Ok(VerkleNode::Inner32(Box::new(
            SparseInnerNode::<32>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner33 => Ok(VerkleNode::Inner33(Box::new(
            SparseInnerNode::<33>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner34 => Ok(VerkleNode::Inner34(Box::new(
            SparseInnerNode::<34>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner35 => Ok(VerkleNode::Inner35(Box::new(
            SparseInnerNode::<35>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner36 => Ok(VerkleNode::Inner36(Box::new(
            SparseInnerNode::<36>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner37 => Ok(VerkleNode::Inner37(Box::new(
            SparseInnerNode::<37>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner38 => Ok(VerkleNode::Inner38(Box::new(
            SparseInnerNode::<38>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner39 => Ok(VerkleNode::Inner39(Box::new(
            SparseInnerNode::<39>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner40 => Ok(VerkleNode::Inner40(Box::new(
            SparseInnerNode::<40>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner41 => Ok(VerkleNode::Inner41(Box::new(
            SparseInnerNode::<41>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner42 => Ok(VerkleNode::Inner42(Box::new(
            SparseInnerNode::<42>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner43 => Ok(VerkleNode::Inner43(Box::new(
            SparseInnerNode::<43>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner44 => Ok(VerkleNode::Inner44(Box::new(
            SparseInnerNode::<44>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner45 => Ok(VerkleNode::Inner45(Box::new(
            SparseInnerNode::<45>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner46 => Ok(VerkleNode::Inner46(Box::new(
            SparseInnerNode::<46>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner47 => Ok(VerkleNode::Inner47(Box::new(
            SparseInnerNode::<47>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner48 => Ok(VerkleNode::Inner48(Box::new(
            SparseInnerNode::<48>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner49 => Ok(VerkleNode::Inner49(Box::new(
            SparseInnerNode::<49>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner50 => Ok(VerkleNode::Inner50(Box::new(
            SparseInnerNode::<50>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner51 => Ok(VerkleNode::Inner51(Box::new(
            SparseInnerNode::<51>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner52 => Ok(VerkleNode::Inner52(Box::new(
            SparseInnerNode::<52>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner53 => Ok(VerkleNode::Inner53(Box::new(
            SparseInnerNode::<53>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner54 => Ok(VerkleNode::Inner54(Box::new(
            SparseInnerNode::<54>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner55 => Ok(VerkleNode::Inner55(Box::new(
            SparseInnerNode::<55>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner56 => Ok(VerkleNode::Inner56(Box::new(
            SparseInnerNode::<56>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner57 => Ok(VerkleNode::Inner57(Box::new(
            SparseInnerNode::<57>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner58 => Ok(VerkleNode::Inner58(Box::new(
            SparseInnerNode::<58>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner59 => Ok(VerkleNode::Inner59(Box::new(
            SparseInnerNode::<59>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner60 => Ok(VerkleNode::Inner60(Box::new(
            SparseInnerNode::<60>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner61 => Ok(VerkleNode::Inner61(Box::new(
            SparseInnerNode::<61>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner62 => Ok(VerkleNode::Inner62(Box::new(
            SparseInnerNode::<62>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner63 => Ok(VerkleNode::Inner63(Box::new(
            SparseInnerNode::<63>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner64 => Ok(VerkleNode::Inner64(Box::new(
            SparseInnerNode::<64>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner65 => Ok(VerkleNode::Inner65(Box::new(
            SparseInnerNode::<65>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner66 => Ok(VerkleNode::Inner66(Box::new(
            SparseInnerNode::<66>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner67 => Ok(VerkleNode::Inner67(Box::new(
            SparseInnerNode::<67>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner68 => Ok(VerkleNode::Inner68(Box::new(
            SparseInnerNode::<68>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner69 => Ok(VerkleNode::Inner69(Box::new(
            SparseInnerNode::<69>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner70 => Ok(VerkleNode::Inner70(Box::new(
            SparseInnerNode::<70>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner71 => Ok(VerkleNode::Inner71(Box::new(
            SparseInnerNode::<71>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner72 => Ok(VerkleNode::Inner72(Box::new(
            SparseInnerNode::<72>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner73 => Ok(VerkleNode::Inner73(Box::new(
            SparseInnerNode::<73>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner74 => Ok(VerkleNode::Inner74(Box::new(
            SparseInnerNode::<74>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner75 => Ok(VerkleNode::Inner75(Box::new(
            SparseInnerNode::<75>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner76 => Ok(VerkleNode::Inner76(Box::new(
            SparseInnerNode::<76>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner77 => Ok(VerkleNode::Inner77(Box::new(
            SparseInnerNode::<77>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner78 => Ok(VerkleNode::Inner78(Box::new(
            SparseInnerNode::<78>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner79 => Ok(VerkleNode::Inner79(Box::new(
            SparseInnerNode::<79>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner80 => Ok(VerkleNode::Inner80(Box::new(
            SparseInnerNode::<80>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner81 => Ok(VerkleNode::Inner81(Box::new(
            SparseInnerNode::<81>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner82 => Ok(VerkleNode::Inner82(Box::new(
            SparseInnerNode::<82>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner83 => Ok(VerkleNode::Inner83(Box::new(
            SparseInnerNode::<83>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner84 => Ok(VerkleNode::Inner84(Box::new(
            SparseInnerNode::<84>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner85 => Ok(VerkleNode::Inner85(Box::new(
            SparseInnerNode::<85>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner86 => Ok(VerkleNode::Inner86(Box::new(
            SparseInnerNode::<86>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner87 => Ok(VerkleNode::Inner87(Box::new(
            SparseInnerNode::<87>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner88 => Ok(VerkleNode::Inner88(Box::new(
            SparseInnerNode::<88>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner89 => Ok(VerkleNode::Inner89(Box::new(
            SparseInnerNode::<89>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner90 => Ok(VerkleNode::Inner90(Box::new(
            SparseInnerNode::<90>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner91 => Ok(VerkleNode::Inner91(Box::new(
            SparseInnerNode::<91>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner92 => Ok(VerkleNode::Inner92(Box::new(
            SparseInnerNode::<92>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner93 => Ok(VerkleNode::Inner93(Box::new(
            SparseInnerNode::<93>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner94 => Ok(VerkleNode::Inner94(Box::new(
            SparseInnerNode::<94>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner95 => Ok(VerkleNode::Inner95(Box::new(
            SparseInnerNode::<95>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner96 => Ok(VerkleNode::Inner96(Box::new(
            SparseInnerNode::<96>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner97 => Ok(VerkleNode::Inner97(Box::new(
            SparseInnerNode::<97>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner98 => Ok(VerkleNode::Inner98(Box::new(
            SparseInnerNode::<98>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner99 => Ok(VerkleNode::Inner99(Box::new(
            SparseInnerNode::<99>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner100 => Ok(VerkleNode::Inner100(Box::new(
            SparseInnerNode::<100>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner101 => Ok(VerkleNode::Inner101(Box::new(
            SparseInnerNode::<101>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner102 => Ok(VerkleNode::Inner102(Box::new(
            SparseInnerNode::<102>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner103 => Ok(VerkleNode::Inner103(Box::new(
            SparseInnerNode::<103>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner104 => Ok(VerkleNode::Inner104(Box::new(
            SparseInnerNode::<104>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner105 => Ok(VerkleNode::Inner105(Box::new(
            SparseInnerNode::<105>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner106 => Ok(VerkleNode::Inner106(Box::new(
            SparseInnerNode::<106>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner107 => Ok(VerkleNode::Inner107(Box::new(
            SparseInnerNode::<107>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner108 => Ok(VerkleNode::Inner108(Box::new(
            SparseInnerNode::<108>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner109 => Ok(VerkleNode::Inner109(Box::new(
            SparseInnerNode::<109>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner110 => Ok(VerkleNode::Inner110(Box::new(
            SparseInnerNode::<110>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner111 => Ok(VerkleNode::Inner111(Box::new(
            SparseInnerNode::<111>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner112 => Ok(VerkleNode::Inner112(Box::new(
            SparseInnerNode::<112>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner113 => Ok(VerkleNode::Inner113(Box::new(
            SparseInnerNode::<113>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner114 => Ok(VerkleNode::Inner114(Box::new(
            SparseInnerNode::<114>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner115 => Ok(VerkleNode::Inner115(Box::new(
            SparseInnerNode::<115>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner116 => Ok(VerkleNode::Inner116(Box::new(
            SparseInnerNode::<116>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner117 => Ok(VerkleNode::Inner117(Box::new(
            SparseInnerNode::<117>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner118 => Ok(VerkleNode::Inner118(Box::new(
            SparseInnerNode::<118>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner119 => Ok(VerkleNode::Inner119(Box::new(
            SparseInnerNode::<119>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner120 => Ok(VerkleNode::Inner120(Box::new(
            SparseInnerNode::<120>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner121 => Ok(VerkleNode::Inner121(Box::new(
            SparseInnerNode::<121>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner122 => Ok(VerkleNode::Inner122(Box::new(
            SparseInnerNode::<122>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner123 => Ok(VerkleNode::Inner123(Box::new(
            SparseInnerNode::<123>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner124 => Ok(VerkleNode::Inner124(Box::new(
            SparseInnerNode::<124>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner125 => Ok(VerkleNode::Inner125(Box::new(
            SparseInnerNode::<125>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner126 => Ok(VerkleNode::Inner126(Box::new(
            SparseInnerNode::<126>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner127 => Ok(VerkleNode::Inner127(Box::new(
            SparseInnerNode::<127>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner128 => Ok(VerkleNode::Inner128(Box::new(
            SparseInnerNode::<128>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner129 => Ok(VerkleNode::Inner129(Box::new(
            SparseInnerNode::<129>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner130 => Ok(VerkleNode::Inner130(Box::new(
            SparseInnerNode::<130>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner131 => Ok(VerkleNode::Inner131(Box::new(
            SparseInnerNode::<131>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner132 => Ok(VerkleNode::Inner132(Box::new(
            SparseInnerNode::<132>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner133 => Ok(VerkleNode::Inner133(Box::new(
            SparseInnerNode::<133>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner134 => Ok(VerkleNode::Inner134(Box::new(
            SparseInnerNode::<134>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner135 => Ok(VerkleNode::Inner135(Box::new(
            SparseInnerNode::<135>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner136 => Ok(VerkleNode::Inner136(Box::new(
            SparseInnerNode::<136>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner137 => Ok(VerkleNode::Inner137(Box::new(
            SparseInnerNode::<137>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner138 => Ok(VerkleNode::Inner138(Box::new(
            SparseInnerNode::<138>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner139 => Ok(VerkleNode::Inner139(Box::new(
            SparseInnerNode::<139>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner140 => Ok(VerkleNode::Inner140(Box::new(
            SparseInnerNode::<140>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner141 => Ok(VerkleNode::Inner141(Box::new(
            SparseInnerNode::<141>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner142 => Ok(VerkleNode::Inner142(Box::new(
            SparseInnerNode::<142>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner143 => Ok(VerkleNode::Inner143(Box::new(
            SparseInnerNode::<143>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner144 => Ok(VerkleNode::Inner144(Box::new(
            SparseInnerNode::<144>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner145 => Ok(VerkleNode::Inner145(Box::new(
            SparseInnerNode::<145>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner146 => Ok(VerkleNode::Inner146(Box::new(
            SparseInnerNode::<146>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner147 => Ok(VerkleNode::Inner147(Box::new(
            SparseInnerNode::<147>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner148 => Ok(VerkleNode::Inner148(Box::new(
            SparseInnerNode::<148>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner149 => Ok(VerkleNode::Inner149(Box::new(
            SparseInnerNode::<149>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner150 => Ok(VerkleNode::Inner150(Box::new(
            SparseInnerNode::<150>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner151 => Ok(VerkleNode::Inner151(Box::new(
            SparseInnerNode::<151>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner152 => Ok(VerkleNode::Inner152(Box::new(
            SparseInnerNode::<152>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner153 => Ok(VerkleNode::Inner153(Box::new(
            SparseInnerNode::<153>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner154 => Ok(VerkleNode::Inner154(Box::new(
            SparseInnerNode::<154>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner155 => Ok(VerkleNode::Inner155(Box::new(
            SparseInnerNode::<155>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner156 => Ok(VerkleNode::Inner156(Box::new(
            SparseInnerNode::<156>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner157 => Ok(VerkleNode::Inner157(Box::new(
            SparseInnerNode::<157>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner158 => Ok(VerkleNode::Inner158(Box::new(
            SparseInnerNode::<158>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner159 => Ok(VerkleNode::Inner159(Box::new(
            SparseInnerNode::<159>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner160 => Ok(VerkleNode::Inner160(Box::new(
            SparseInnerNode::<160>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner161 => Ok(VerkleNode::Inner161(Box::new(
            SparseInnerNode::<161>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner162 => Ok(VerkleNode::Inner162(Box::new(
            SparseInnerNode::<162>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner163 => Ok(VerkleNode::Inner163(Box::new(
            SparseInnerNode::<163>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner164 => Ok(VerkleNode::Inner164(Box::new(
            SparseInnerNode::<164>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner165 => Ok(VerkleNode::Inner165(Box::new(
            SparseInnerNode::<165>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner166 => Ok(VerkleNode::Inner166(Box::new(
            SparseInnerNode::<166>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner167 => Ok(VerkleNode::Inner167(Box::new(
            SparseInnerNode::<167>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner168 => Ok(VerkleNode::Inner168(Box::new(
            SparseInnerNode::<168>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner169 => Ok(VerkleNode::Inner169(Box::new(
            SparseInnerNode::<169>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner170 => Ok(VerkleNode::Inner170(Box::new(
            SparseInnerNode::<170>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner171 => Ok(VerkleNode::Inner171(Box::new(
            SparseInnerNode::<171>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner172 => Ok(VerkleNode::Inner172(Box::new(
            SparseInnerNode::<172>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner173 => Ok(VerkleNode::Inner173(Box::new(
            SparseInnerNode::<173>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner174 => Ok(VerkleNode::Inner174(Box::new(
            SparseInnerNode::<174>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner175 => Ok(VerkleNode::Inner175(Box::new(
            SparseInnerNode::<175>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner176 => Ok(VerkleNode::Inner176(Box::new(
            SparseInnerNode::<176>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner177 => Ok(VerkleNode::Inner177(Box::new(
            SparseInnerNode::<177>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner178 => Ok(VerkleNode::Inner178(Box::new(
            SparseInnerNode::<178>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner179 => Ok(VerkleNode::Inner179(Box::new(
            SparseInnerNode::<179>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner180 => Ok(VerkleNode::Inner180(Box::new(
            SparseInnerNode::<180>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner181 => Ok(VerkleNode::Inner181(Box::new(
            SparseInnerNode::<181>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner182 => Ok(VerkleNode::Inner182(Box::new(
            SparseInnerNode::<182>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner183 => Ok(VerkleNode::Inner183(Box::new(
            SparseInnerNode::<183>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner184 => Ok(VerkleNode::Inner184(Box::new(
            SparseInnerNode::<184>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner185 => Ok(VerkleNode::Inner185(Box::new(
            SparseInnerNode::<185>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner186 => Ok(VerkleNode::Inner186(Box::new(
            SparseInnerNode::<186>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner187 => Ok(VerkleNode::Inner187(Box::new(
            SparseInnerNode::<187>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner188 => Ok(VerkleNode::Inner188(Box::new(
            SparseInnerNode::<188>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner189 => Ok(VerkleNode::Inner189(Box::new(
            SparseInnerNode::<189>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner190 => Ok(VerkleNode::Inner190(Box::new(
            SparseInnerNode::<190>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner191 => Ok(VerkleNode::Inner191(Box::new(
            SparseInnerNode::<191>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner192 => Ok(VerkleNode::Inner192(Box::new(
            SparseInnerNode::<192>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner193 => Ok(VerkleNode::Inner193(Box::new(
            SparseInnerNode::<193>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner194 => Ok(VerkleNode::Inner194(Box::new(
            SparseInnerNode::<194>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner195 => Ok(VerkleNode::Inner195(Box::new(
            SparseInnerNode::<195>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner196 => Ok(VerkleNode::Inner196(Box::new(
            SparseInnerNode::<196>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner197 => Ok(VerkleNode::Inner197(Box::new(
            SparseInnerNode::<197>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner198 => Ok(VerkleNode::Inner198(Box::new(
            SparseInnerNode::<198>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner199 => Ok(VerkleNode::Inner199(Box::new(
            SparseInnerNode::<199>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner200 => Ok(VerkleNode::Inner200(Box::new(
            SparseInnerNode::<200>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner201 => Ok(VerkleNode::Inner201(Box::new(
            SparseInnerNode::<201>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner202 => Ok(VerkleNode::Inner202(Box::new(
            SparseInnerNode::<202>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner203 => Ok(VerkleNode::Inner203(Box::new(
            SparseInnerNode::<203>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner204 => Ok(VerkleNode::Inner204(Box::new(
            SparseInnerNode::<204>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner205 => Ok(VerkleNode::Inner205(Box::new(
            SparseInnerNode::<205>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner206 => Ok(VerkleNode::Inner206(Box::new(
            SparseInnerNode::<206>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner207 => Ok(VerkleNode::Inner207(Box::new(
            SparseInnerNode::<207>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner208 => Ok(VerkleNode::Inner208(Box::new(
            SparseInnerNode::<208>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner209 => Ok(VerkleNode::Inner209(Box::new(
            SparseInnerNode::<209>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner210 => Ok(VerkleNode::Inner210(Box::new(
            SparseInnerNode::<210>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner211 => Ok(VerkleNode::Inner211(Box::new(
            SparseInnerNode::<211>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner212 => Ok(VerkleNode::Inner212(Box::new(
            SparseInnerNode::<212>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner213 => Ok(VerkleNode::Inner213(Box::new(
            SparseInnerNode::<213>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner214 => Ok(VerkleNode::Inner214(Box::new(
            SparseInnerNode::<214>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner215 => Ok(VerkleNode::Inner215(Box::new(
            SparseInnerNode::<215>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner216 => Ok(VerkleNode::Inner216(Box::new(
            SparseInnerNode::<216>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner217 => Ok(VerkleNode::Inner217(Box::new(
            SparseInnerNode::<217>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner218 => Ok(VerkleNode::Inner218(Box::new(
            SparseInnerNode::<218>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner219 => Ok(VerkleNode::Inner219(Box::new(
            SparseInnerNode::<219>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner220 => Ok(VerkleNode::Inner220(Box::new(
            SparseInnerNode::<220>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner221 => Ok(VerkleNode::Inner221(Box::new(
            SparseInnerNode::<221>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner222 => Ok(VerkleNode::Inner222(Box::new(
            SparseInnerNode::<222>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner223 => Ok(VerkleNode::Inner223(Box::new(
            SparseInnerNode::<223>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner224 => Ok(VerkleNode::Inner224(Box::new(
            SparseInnerNode::<224>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner225 => Ok(VerkleNode::Inner225(Box::new(
            SparseInnerNode::<225>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner226 => Ok(VerkleNode::Inner226(Box::new(
            SparseInnerNode::<226>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner227 => Ok(VerkleNode::Inner227(Box::new(
            SparseInnerNode::<227>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner228 => Ok(VerkleNode::Inner228(Box::new(
            SparseInnerNode::<228>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner229 => Ok(VerkleNode::Inner229(Box::new(
            SparseInnerNode::<229>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner230 => Ok(VerkleNode::Inner230(Box::new(
            SparseInnerNode::<230>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner231 => Ok(VerkleNode::Inner231(Box::new(
            SparseInnerNode::<231>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner232 => Ok(VerkleNode::Inner232(Box::new(
            SparseInnerNode::<232>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner233 => Ok(VerkleNode::Inner233(Box::new(
            SparseInnerNode::<233>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner234 => Ok(VerkleNode::Inner234(Box::new(
            SparseInnerNode::<234>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner235 => Ok(VerkleNode::Inner235(Box::new(
            SparseInnerNode::<235>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner236 => Ok(VerkleNode::Inner236(Box::new(
            SparseInnerNode::<236>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner237 => Ok(VerkleNode::Inner237(Box::new(
            SparseInnerNode::<237>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner238 => Ok(VerkleNode::Inner238(Box::new(
            SparseInnerNode::<238>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner239 => Ok(VerkleNode::Inner239(Box::new(
            SparseInnerNode::<239>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner240 => Ok(VerkleNode::Inner240(Box::new(
            SparseInnerNode::<240>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner241 => Ok(VerkleNode::Inner241(Box::new(
            SparseInnerNode::<241>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner242 => Ok(VerkleNode::Inner242(Box::new(
            SparseInnerNode::<242>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner243 => Ok(VerkleNode::Inner243(Box::new(
            SparseInnerNode::<243>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner244 => Ok(VerkleNode::Inner244(Box::new(
            SparseInnerNode::<244>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner245 => Ok(VerkleNode::Inner245(Box::new(
            SparseInnerNode::<245>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner246 => Ok(VerkleNode::Inner246(Box::new(
            SparseInnerNode::<246>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner247 => Ok(VerkleNode::Inner247(Box::new(
            SparseInnerNode::<247>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner248 => Ok(VerkleNode::Inner248(Box::new(
            SparseInnerNode::<248>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner249 => Ok(VerkleNode::Inner249(Box::new(
            SparseInnerNode::<249>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner250 => Ok(VerkleNode::Inner250(Box::new(
            SparseInnerNode::<250>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner251 => Ok(VerkleNode::Inner251(Box::new(
            SparseInnerNode::<251>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner252 => Ok(VerkleNode::Inner252(Box::new(
            SparseInnerNode::<252>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner253 => Ok(VerkleNode::Inner253(Box::new(
            SparseInnerNode::<253>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner254 => Ok(VerkleNode::Inner254(Box::new(
            SparseInnerNode::<254>::from_existing(children, commitment)?,
        ))),
        VerkleNodeKind::Inner255 => Ok(VerkleNode::Inner255(Box::new(
            SparseInnerNode::<255>::from_existing(children, commitment)?,
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
