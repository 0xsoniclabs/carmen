// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::fmt::Debug;

use zerocopy::{FromBytes, Immutable, IntoBytes, Unaligned};

use crate::{
    database::verkle::variants::managed::nodes::VerkleNodeKind,
    types::{HasEmptyId, NodeSize, ToNodeKind, TreeId},
};

/// An identifier for a node in a managed Verkle trie.
// NOTE: Changing the layout of this struct will break backwards compatibility of the
// serialization format.
#[derive(
    Clone, Copy, PartialEq, Eq, Hash, FromBytes, IntoBytes, Immutable, Unaligned, PartialOrd, Ord,
)]
#[repr(transparent)]
pub struct VerkleNodeId([u8; 6]);

impl VerkleNodeId {
    pub const PREFIX_MASK: u64 = 0x0000_FFC0_0000_0000;
    pub const INDEX_MASK: u64 = 0x0000_003F_FFFF_FFFF;

    // General constants
    pub const EMPTY: u64 = 0x000 << 38;

    // Inner node constants (1-256)
    pub const INNER_1_NODE_ID: u64 = 0x001 << 38;
    pub const INNER_2_NODE_ID: u64 = 0x002 << 38;
    pub const INNER_3_NODE_ID: u64 = 0x003 << 38;
    pub const INNER_4_NODE_ID: u64 = 0x004 << 38;
    pub const INNER_5_NODE_ID: u64 = 0x005 << 38;
    pub const INNER_6_NODE_ID: u64 = 0x006 << 38;
    pub const INNER_7_NODE_ID: u64 = 0x007 << 38;
    pub const INNER_8_NODE_ID: u64 = 0x008 << 38;
    pub const INNER_9_NODE_ID: u64 = 0x009 << 38;
    pub const INNER_10_NODE_ID: u64 = 0x00A << 38;
    pub const INNER_11_NODE_ID: u64 = 0x00B << 38;
    pub const INNER_12_NODE_ID: u64 = 0x00C << 38;
    pub const INNER_13_NODE_ID: u64 = 0x00D << 38;
    pub const INNER_14_NODE_ID: u64 = 0x00E << 38;
    pub const INNER_15_NODE_ID: u64 = 0x00F << 38;
    pub const INNER_16_NODE_ID: u64 = 0x010 << 38;
    pub const INNER_17_NODE_ID: u64 = 0x011 << 38;
    pub const INNER_18_NODE_ID: u64 = 0x012 << 38;
    pub const INNER_19_NODE_ID: u64 = 0x013 << 38;
    pub const INNER_20_NODE_ID: u64 = 0x014 << 38;
    pub const INNER_21_NODE_ID: u64 = 0x015 << 38;
    pub const INNER_22_NODE_ID: u64 = 0x016 << 38;
    pub const INNER_23_NODE_ID: u64 = 0x017 << 38;
    pub const INNER_24_NODE_ID: u64 = 0x018 << 38;
    pub const INNER_25_NODE_ID: u64 = 0x019 << 38;
    pub const INNER_26_NODE_ID: u64 = 0x01A << 38;
    pub const INNER_27_NODE_ID: u64 = 0x01B << 38;
    pub const INNER_28_NODE_ID: u64 = 0x01C << 38;
    pub const INNER_29_NODE_ID: u64 = 0x01D << 38;
    pub const INNER_30_NODE_ID: u64 = 0x01E << 38;
    pub const INNER_31_NODE_ID: u64 = 0x01F << 38;
    pub const INNER_32_NODE_ID: u64 = 0x020 << 38;
    pub const INNER_33_NODE_ID: u64 = 0x021 << 38;
    pub const INNER_34_NODE_ID: u64 = 0x022 << 38;
    pub const INNER_35_NODE_ID: u64 = 0x023 << 38;
    pub const INNER_36_NODE_ID: u64 = 0x024 << 38;
    pub const INNER_37_NODE_ID: u64 = 0x025 << 38;
    pub const INNER_38_NODE_ID: u64 = 0x026 << 38;
    pub const INNER_39_NODE_ID: u64 = 0x027 << 38;
    pub const INNER_40_NODE_ID: u64 = 0x028 << 38;
    pub const INNER_41_NODE_ID: u64 = 0x029 << 38;
    pub const INNER_42_NODE_ID: u64 = 0x02A << 38;
    pub const INNER_43_NODE_ID: u64 = 0x02B << 38;
    pub const INNER_44_NODE_ID: u64 = 0x02C << 38;
    pub const INNER_45_NODE_ID: u64 = 0x02D << 38;
    pub const INNER_46_NODE_ID: u64 = 0x02E << 38;
    pub const INNER_47_NODE_ID: u64 = 0x02F << 38;
    pub const INNER_48_NODE_ID: u64 = 0x030 << 38;
    pub const INNER_49_NODE_ID: u64 = 0x031 << 38;
    pub const INNER_50_NODE_ID: u64 = 0x032 << 38;
    pub const INNER_51_NODE_ID: u64 = 0x033 << 38;
    pub const INNER_52_NODE_ID: u64 = 0x034 << 38;
    pub const INNER_53_NODE_ID: u64 = 0x035 << 38;
    pub const INNER_54_NODE_ID: u64 = 0x036 << 38;
    pub const INNER_55_NODE_ID: u64 = 0x037 << 38;
    pub const INNER_56_NODE_ID: u64 = 0x038 << 38;
    pub const INNER_57_NODE_ID: u64 = 0x039 << 38;
    pub const INNER_58_NODE_ID: u64 = 0x03A << 38;
    pub const INNER_59_NODE_ID: u64 = 0x03B << 38;
    pub const INNER_60_NODE_ID: u64 = 0x03C << 38;
    pub const INNER_61_NODE_ID: u64 = 0x03D << 38;
    pub const INNER_62_NODE_ID: u64 = 0x03E << 38;
    pub const INNER_63_NODE_ID: u64 = 0x03F << 38;
    pub const INNER_64_NODE_ID: u64 = 0x040 << 38;
    pub const INNER_65_NODE_ID: u64 = 0x041 << 38;
    pub const INNER_66_NODE_ID: u64 = 0x042 << 38;
    pub const INNER_67_NODE_ID: u64 = 0x043 << 38;
    pub const INNER_68_NODE_ID: u64 = 0x044 << 38;
    pub const INNER_69_NODE_ID: u64 = 0x045 << 38;
    pub const INNER_70_NODE_ID: u64 = 0x046 << 38;
    pub const INNER_71_NODE_ID: u64 = 0x047 << 38;
    pub const INNER_72_NODE_ID: u64 = 0x048 << 38;
    pub const INNER_73_NODE_ID: u64 = 0x049 << 38;
    pub const INNER_74_NODE_ID: u64 = 0x04A << 38;
    pub const INNER_75_NODE_ID: u64 = 0x04B << 38;
    pub const INNER_76_NODE_ID: u64 = 0x04C << 38;
    pub const INNER_77_NODE_ID: u64 = 0x04D << 38;
    pub const INNER_78_NODE_ID: u64 = 0x04E << 38;
    pub const INNER_79_NODE_ID: u64 = 0x04F << 38;
    pub const INNER_80_NODE_ID: u64 = 0x050 << 38;
    pub const INNER_81_NODE_ID: u64 = 0x051 << 38;
    pub const INNER_82_NODE_ID: u64 = 0x052 << 38;
    pub const INNER_83_NODE_ID: u64 = 0x053 << 38;
    pub const INNER_84_NODE_ID: u64 = 0x054 << 38;
    pub const INNER_85_NODE_ID: u64 = 0x055 << 38;
    pub const INNER_86_NODE_ID: u64 = 0x056 << 38;
    pub const INNER_87_NODE_ID: u64 = 0x057 << 38;
    pub const INNER_88_NODE_ID: u64 = 0x058 << 38;
    pub const INNER_89_NODE_ID: u64 = 0x059 << 38;
    pub const INNER_90_NODE_ID: u64 = 0x05A << 38;
    pub const INNER_91_NODE_ID: u64 = 0x05B << 38;
    pub const INNER_92_NODE_ID: u64 = 0x05C << 38;
    pub const INNER_93_NODE_ID: u64 = 0x05D << 38;
    pub const INNER_94_NODE_ID: u64 = 0x05E << 38;
    pub const INNER_95_NODE_ID: u64 = 0x05F << 38;
    pub const INNER_96_NODE_ID: u64 = 0x060 << 38;
    pub const INNER_97_NODE_ID: u64 = 0x061 << 38;
    pub const INNER_98_NODE_ID: u64 = 0x062 << 38;
    pub const INNER_99_NODE_ID: u64 = 0x063 << 38;
    pub const INNER_100_NODE_ID: u64 = 0x064 << 38;
    pub const INNER_101_NODE_ID: u64 = 0x065 << 38;
    pub const INNER_102_NODE_ID: u64 = 0x066 << 38;
    pub const INNER_103_NODE_ID: u64 = 0x067 << 38;
    pub const INNER_104_NODE_ID: u64 = 0x068 << 38;
    pub const INNER_105_NODE_ID: u64 = 0x069 << 38;
    pub const INNER_106_NODE_ID: u64 = 0x06A << 38;
    pub const INNER_107_NODE_ID: u64 = 0x06B << 38;
    pub const INNER_108_NODE_ID: u64 = 0x06C << 38;
    pub const INNER_109_NODE_ID: u64 = 0x06D << 38;
    pub const INNER_110_NODE_ID: u64 = 0x06E << 38;
    pub const INNER_111_NODE_ID: u64 = 0x06F << 38;
    pub const INNER_112_NODE_ID: u64 = 0x070 << 38;
    pub const INNER_113_NODE_ID: u64 = 0x071 << 38;
    pub const INNER_114_NODE_ID: u64 = 0x072 << 38;
    pub const INNER_115_NODE_ID: u64 = 0x073 << 38;
    pub const INNER_116_NODE_ID: u64 = 0x074 << 38;
    pub const INNER_117_NODE_ID: u64 = 0x075 << 38;
    pub const INNER_118_NODE_ID: u64 = 0x076 << 38;
    pub const INNER_119_NODE_ID: u64 = 0x077 << 38;
    pub const INNER_120_NODE_ID: u64 = 0x078 << 38;
    pub const INNER_121_NODE_ID: u64 = 0x079 << 38;
    pub const INNER_122_NODE_ID: u64 = 0x07A << 38;
    pub const INNER_123_NODE_ID: u64 = 0x07B << 38;
    pub const INNER_124_NODE_ID: u64 = 0x07C << 38;
    pub const INNER_125_NODE_ID: u64 = 0x07D << 38;
    pub const INNER_126_NODE_ID: u64 = 0x07E << 38;
    pub const INNER_127_NODE_ID: u64 = 0x07F << 38;
    pub const INNER_128_NODE_ID: u64 = 0x080 << 38;
    pub const INNER_129_NODE_ID: u64 = 0x081 << 38;
    pub const INNER_130_NODE_ID: u64 = 0x082 << 38;
    pub const INNER_131_NODE_ID: u64 = 0x083 << 38;
    pub const INNER_132_NODE_ID: u64 = 0x084 << 38;
    pub const INNER_133_NODE_ID: u64 = 0x085 << 38;
    pub const INNER_134_NODE_ID: u64 = 0x086 << 38;
    pub const INNER_135_NODE_ID: u64 = 0x087 << 38;
    pub const INNER_136_NODE_ID: u64 = 0x088 << 38;
    pub const INNER_137_NODE_ID: u64 = 0x089 << 38;
    pub const INNER_138_NODE_ID: u64 = 0x08A << 38;
    pub const INNER_139_NODE_ID: u64 = 0x08B << 38;
    pub const INNER_140_NODE_ID: u64 = 0x08C << 38;
    pub const INNER_141_NODE_ID: u64 = 0x08D << 38;
    pub const INNER_142_NODE_ID: u64 = 0x08E << 38;
    pub const INNER_143_NODE_ID: u64 = 0x08F << 38;
    pub const INNER_144_NODE_ID: u64 = 0x090 << 38;
    pub const INNER_145_NODE_ID: u64 = 0x091 << 38;
    pub const INNER_146_NODE_ID: u64 = 0x092 << 38;
    pub const INNER_147_NODE_ID: u64 = 0x093 << 38;
    pub const INNER_148_NODE_ID: u64 = 0x094 << 38;
    pub const INNER_149_NODE_ID: u64 = 0x095 << 38;
    pub const INNER_150_NODE_ID: u64 = 0x096 << 38;
    pub const INNER_151_NODE_ID: u64 = 0x097 << 38;
    pub const INNER_152_NODE_ID: u64 = 0x098 << 38;
    pub const INNER_153_NODE_ID: u64 = 0x099 << 38;
    pub const INNER_154_NODE_ID: u64 = 0x09A << 38;
    pub const INNER_155_NODE_ID: u64 = 0x09B << 38;
    pub const INNER_156_NODE_ID: u64 = 0x09C << 38;
    pub const INNER_157_NODE_ID: u64 = 0x09D << 38;
    pub const INNER_158_NODE_ID: u64 = 0x09E << 38;
    pub const INNER_159_NODE_ID: u64 = 0x09F << 38;
    pub const INNER_160_NODE_ID: u64 = 0x0A0 << 38;
    pub const INNER_161_NODE_ID: u64 = 0x0A1 << 38;
    pub const INNER_162_NODE_ID: u64 = 0x0A2 << 38;
    pub const INNER_163_NODE_ID: u64 = 0x0A3 << 38;
    pub const INNER_164_NODE_ID: u64 = 0x0A4 << 38;
    pub const INNER_165_NODE_ID: u64 = 0x0A5 << 38;
    pub const INNER_166_NODE_ID: u64 = 0x0A6 << 38;
    pub const INNER_167_NODE_ID: u64 = 0x0A7 << 38;
    pub const INNER_168_NODE_ID: u64 = 0x0A8 << 38;
    pub const INNER_169_NODE_ID: u64 = 0x0A9 << 38;
    pub const INNER_170_NODE_ID: u64 = 0x0AA << 38;
    pub const INNER_171_NODE_ID: u64 = 0x0AB << 38;
    pub const INNER_172_NODE_ID: u64 = 0x0AC << 38;
    pub const INNER_173_NODE_ID: u64 = 0x0AD << 38;
    pub const INNER_174_NODE_ID: u64 = 0x0AE << 38;
    pub const INNER_175_NODE_ID: u64 = 0x0AF << 38;
    pub const INNER_176_NODE_ID: u64 = 0x0B0 << 38;
    pub const INNER_177_NODE_ID: u64 = 0x0B1 << 38;
    pub const INNER_178_NODE_ID: u64 = 0x0B2 << 38;
    pub const INNER_179_NODE_ID: u64 = 0x0B3 << 38;
    pub const INNER_180_NODE_ID: u64 = 0x0B4 << 38;
    pub const INNER_181_NODE_ID: u64 = 0x0B5 << 38;
    pub const INNER_182_NODE_ID: u64 = 0x0B6 << 38;
    pub const INNER_183_NODE_ID: u64 = 0x0B7 << 38;
    pub const INNER_184_NODE_ID: u64 = 0x0B8 << 38;
    pub const INNER_185_NODE_ID: u64 = 0x0B9 << 38;
    pub const INNER_186_NODE_ID: u64 = 0x0BA << 38;
    pub const INNER_187_NODE_ID: u64 = 0x0BB << 38;
    pub const INNER_188_NODE_ID: u64 = 0x0BC << 38;
    pub const INNER_189_NODE_ID: u64 = 0x0BD << 38;
    pub const INNER_190_NODE_ID: u64 = 0x0BE << 38;
    pub const INNER_191_NODE_ID: u64 = 0x0BF << 38;
    pub const INNER_192_NODE_ID: u64 = 0x0C0 << 38;
    pub const INNER_193_NODE_ID: u64 = 0x0C1 << 38;
    pub const INNER_194_NODE_ID: u64 = 0x0C2 << 38;
    pub const INNER_195_NODE_ID: u64 = 0x0C3 << 38;
    pub const INNER_196_NODE_ID: u64 = 0x0C4 << 38;
    pub const INNER_197_NODE_ID: u64 = 0x0C5 << 38;
    pub const INNER_198_NODE_ID: u64 = 0x0C6 << 38;
    pub const INNER_199_NODE_ID: u64 = 0x0C7 << 38;
    pub const INNER_200_NODE_ID: u64 = 0x0C8 << 38;
    pub const INNER_201_NODE_ID: u64 = 0x0C9 << 38;
    pub const INNER_202_NODE_ID: u64 = 0x0CA << 38;
    pub const INNER_203_NODE_ID: u64 = 0x0CB << 38;
    pub const INNER_204_NODE_ID: u64 = 0x0CC << 38;
    pub const INNER_205_NODE_ID: u64 = 0x0CD << 38;
    pub const INNER_206_NODE_ID: u64 = 0x0CE << 38;
    pub const INNER_207_NODE_ID: u64 = 0x0CF << 38;
    pub const INNER_208_NODE_ID: u64 = 0x0D0 << 38;
    pub const INNER_209_NODE_ID: u64 = 0x0D1 << 38;
    pub const INNER_210_NODE_ID: u64 = 0x0D2 << 38;
    pub const INNER_211_NODE_ID: u64 = 0x0D3 << 38;
    pub const INNER_212_NODE_ID: u64 = 0x0D4 << 38;
    pub const INNER_213_NODE_ID: u64 = 0x0D5 << 38;
    pub const INNER_214_NODE_ID: u64 = 0x0D6 << 38;
    pub const INNER_215_NODE_ID: u64 = 0x0D7 << 38;
    pub const INNER_216_NODE_ID: u64 = 0x0D8 << 38;
    pub const INNER_217_NODE_ID: u64 = 0x0D9 << 38;
    pub const INNER_218_NODE_ID: u64 = 0x0DA << 38;
    pub const INNER_219_NODE_ID: u64 = 0x0DB << 38;
    pub const INNER_220_NODE_ID: u64 = 0x0DC << 38;
    pub const INNER_221_NODE_ID: u64 = 0x0DD << 38;
    pub const INNER_222_NODE_ID: u64 = 0x0DE << 38;
    pub const INNER_223_NODE_ID: u64 = 0x0DF << 38;
    pub const INNER_224_NODE_ID: u64 = 0x0E0 << 38;
    pub const INNER_225_NODE_ID: u64 = 0x0E1 << 38;
    pub const INNER_226_NODE_ID: u64 = 0x0E2 << 38;
    pub const INNER_227_NODE_ID: u64 = 0x0E3 << 38;
    pub const INNER_228_NODE_ID: u64 = 0x0E4 << 38;
    pub const INNER_229_NODE_ID: u64 = 0x0E5 << 38;
    pub const INNER_230_NODE_ID: u64 = 0x0E6 << 38;
    pub const INNER_231_NODE_ID: u64 = 0x0E7 << 38;
    pub const INNER_232_NODE_ID: u64 = 0x0E8 << 38;
    pub const INNER_233_NODE_ID: u64 = 0x0E9 << 38;
    pub const INNER_234_NODE_ID: u64 = 0x0EA << 38;
    pub const INNER_235_NODE_ID: u64 = 0x0EB << 38;
    pub const INNER_236_NODE_ID: u64 = 0x0EC << 38;
    pub const INNER_237_NODE_ID: u64 = 0x0ED << 38;
    pub const INNER_238_NODE_ID: u64 = 0x0EE << 38;
    pub const INNER_239_NODE_ID: u64 = 0x0EF << 38;
    pub const INNER_240_NODE_ID: u64 = 0x0F0 << 38;
    pub const INNER_241_NODE_ID: u64 = 0x0F1 << 38;
    pub const INNER_242_NODE_ID: u64 = 0x0F2 << 38;
    pub const INNER_243_NODE_ID: u64 = 0x0F3 << 38;
    pub const INNER_244_NODE_ID: u64 = 0x0F4 << 38;
    pub const INNER_245_NODE_ID: u64 = 0x0F5 << 38;
    pub const INNER_246_NODE_ID: u64 = 0x0F6 << 38;
    pub const INNER_247_NODE_ID: u64 = 0x0F7 << 38;
    pub const INNER_248_NODE_ID: u64 = 0x0F8 << 38;
    pub const INNER_249_NODE_ID: u64 = 0x0F9 << 38;
    pub const INNER_250_NODE_ID: u64 = 0x0FA << 38;
    pub const INNER_251_NODE_ID: u64 = 0x0FB << 38;
    pub const INNER_252_NODE_ID: u64 = 0x0FC << 38;
    pub const INNER_253_NODE_ID: u64 = 0x0FD << 38;
    pub const INNER_254_NODE_ID: u64 = 0x0FE << 38;
    pub const INNER_255_NODE_ID: u64 = 0x0FF << 38;
    pub const INNER_256_NODE_ID: u64 = 0x100 << 38;

    // Leaf node constants (257-512)
    pub const LEAF_1_NODE_ID: u64 = 0x101 << 38;
    pub const LEAF_2_NODE_ID: u64 = 0x102 << 38;
    pub const LEAF_3_NODE_ID: u64 = 0x103 << 38;
    pub const LEAF_4_NODE_ID: u64 = 0x104 << 38;
    pub const LEAF_5_NODE_ID: u64 = 0x105 << 38;
    pub const LEAF_6_NODE_ID: u64 = 0x106 << 38;
    pub const LEAF_7_NODE_ID: u64 = 0x107 << 38;
    pub const LEAF_8_NODE_ID: u64 = 0x108 << 38;
    pub const LEAF_9_NODE_ID: u64 = 0x109 << 38;
    pub const LEAF_10_NODE_ID: u64 = 0x10A << 38;
    pub const LEAF_11_NODE_ID: u64 = 0x10B << 38;
    pub const LEAF_12_NODE_ID: u64 = 0x10C << 38;
    pub const LEAF_13_NODE_ID: u64 = 0x10D << 38;
    pub const LEAF_14_NODE_ID: u64 = 0x10E << 38;
    pub const LEAF_15_NODE_ID: u64 = 0x10F << 38;
    pub const LEAF_16_NODE_ID: u64 = 0x110 << 38;
    pub const LEAF_17_NODE_ID: u64 = 0x111 << 38;
    pub const LEAF_18_NODE_ID: u64 = 0x112 << 38;
    pub const LEAF_19_NODE_ID: u64 = 0x113 << 38;
    pub const LEAF_20_NODE_ID: u64 = 0x114 << 38;
    pub const LEAF_21_NODE_ID: u64 = 0x115 << 38;
    pub const LEAF_22_NODE_ID: u64 = 0x116 << 38;
    pub const LEAF_23_NODE_ID: u64 = 0x117 << 38;
    pub const LEAF_24_NODE_ID: u64 = 0x118 << 38;
    pub const LEAF_25_NODE_ID: u64 = 0x119 << 38;
    pub const LEAF_26_NODE_ID: u64 = 0x11A << 38;
    pub const LEAF_27_NODE_ID: u64 = 0x11B << 38;
    pub const LEAF_28_NODE_ID: u64 = 0x11C << 38;
    pub const LEAF_29_NODE_ID: u64 = 0x11D << 38;
    pub const LEAF_30_NODE_ID: u64 = 0x11E << 38;
    pub const LEAF_31_NODE_ID: u64 = 0x11F << 38;
    pub const LEAF_32_NODE_ID: u64 = 0x120 << 38;
    pub const LEAF_33_NODE_ID: u64 = 0x121 << 38;
    pub const LEAF_34_NODE_ID: u64 = 0x122 << 38;
    pub const LEAF_35_NODE_ID: u64 = 0x123 << 38;
    pub const LEAF_36_NODE_ID: u64 = 0x124 << 38;
    pub const LEAF_37_NODE_ID: u64 = 0x125 << 38;
    pub const LEAF_38_NODE_ID: u64 = 0x126 << 38;
    pub const LEAF_39_NODE_ID: u64 = 0x127 << 38;
    pub const LEAF_40_NODE_ID: u64 = 0x128 << 38;
    pub const LEAF_41_NODE_ID: u64 = 0x129 << 38;
    pub const LEAF_42_NODE_ID: u64 = 0x12A << 38;
    pub const LEAF_43_NODE_ID: u64 = 0x12B << 38;
    pub const LEAF_44_NODE_ID: u64 = 0x12C << 38;
    pub const LEAF_45_NODE_ID: u64 = 0x12D << 38;
    pub const LEAF_46_NODE_ID: u64 = 0x12E << 38;
    pub const LEAF_47_NODE_ID: u64 = 0x12F << 38;
    pub const LEAF_48_NODE_ID: u64 = 0x130 << 38;
    pub const LEAF_49_NODE_ID: u64 = 0x131 << 38;
    pub const LEAF_50_NODE_ID: u64 = 0x132 << 38;
    pub const LEAF_51_NODE_ID: u64 = 0x133 << 38;
    pub const LEAF_52_NODE_ID: u64 = 0x134 << 38;
    pub const LEAF_53_NODE_ID: u64 = 0x135 << 38;
    pub const LEAF_54_NODE_ID: u64 = 0x136 << 38;
    pub const LEAF_55_NODE_ID: u64 = 0x137 << 38;
    pub const LEAF_56_NODE_ID: u64 = 0x138 << 38;
    pub const LEAF_57_NODE_ID: u64 = 0x139 << 38;
    pub const LEAF_58_NODE_ID: u64 = 0x13A << 38;
    pub const LEAF_59_NODE_ID: u64 = 0x13B << 38;
    pub const LEAF_60_NODE_ID: u64 = 0x13C << 38;
    pub const LEAF_61_NODE_ID: u64 = 0x13D << 38;
    pub const LEAF_62_NODE_ID: u64 = 0x13E << 38;
    pub const LEAF_63_NODE_ID: u64 = 0x13F << 38;
    pub const LEAF_64_NODE_ID: u64 = 0x140 << 38;
    pub const LEAF_65_NODE_ID: u64 = 0x141 << 38;
    pub const LEAF_66_NODE_ID: u64 = 0x142 << 38;
    pub const LEAF_67_NODE_ID: u64 = 0x143 << 38;
    pub const LEAF_68_NODE_ID: u64 = 0x144 << 38;
    pub const LEAF_69_NODE_ID: u64 = 0x145 << 38;
    pub const LEAF_70_NODE_ID: u64 = 0x146 << 38;
    pub const LEAF_71_NODE_ID: u64 = 0x147 << 38;
    pub const LEAF_72_NODE_ID: u64 = 0x148 << 38;
    pub const LEAF_73_NODE_ID: u64 = 0x149 << 38;
    pub const LEAF_74_NODE_ID: u64 = 0x14A << 38;
    pub const LEAF_75_NODE_ID: u64 = 0x14B << 38;
    pub const LEAF_76_NODE_ID: u64 = 0x14C << 38;
    pub const LEAF_77_NODE_ID: u64 = 0x14D << 38;
    pub const LEAF_78_NODE_ID: u64 = 0x14E << 38;
    pub const LEAF_79_NODE_ID: u64 = 0x14F << 38;
    pub const LEAF_80_NODE_ID: u64 = 0x150 << 38;
    pub const LEAF_81_NODE_ID: u64 = 0x151 << 38;
    pub const LEAF_82_NODE_ID: u64 = 0x152 << 38;
    pub const LEAF_83_NODE_ID: u64 = 0x153 << 38;
    pub const LEAF_84_NODE_ID: u64 = 0x154 << 38;
    pub const LEAF_85_NODE_ID: u64 = 0x155 << 38;
    pub const LEAF_86_NODE_ID: u64 = 0x156 << 38;
    pub const LEAF_87_NODE_ID: u64 = 0x157 << 38;
    pub const LEAF_88_NODE_ID: u64 = 0x158 << 38;
    pub const LEAF_89_NODE_ID: u64 = 0x159 << 38;
    pub const LEAF_90_NODE_ID: u64 = 0x15A << 38;
    pub const LEAF_91_NODE_ID: u64 = 0x15B << 38;
    pub const LEAF_92_NODE_ID: u64 = 0x15C << 38;
    pub const LEAF_93_NODE_ID: u64 = 0x15D << 38;
    pub const LEAF_94_NODE_ID: u64 = 0x15E << 38;
    pub const LEAF_95_NODE_ID: u64 = 0x15F << 38;
    pub const LEAF_96_NODE_ID: u64 = 0x160 << 38;
    pub const LEAF_97_NODE_ID: u64 = 0x161 << 38;
    pub const LEAF_98_NODE_ID: u64 = 0x162 << 38;
    pub const LEAF_99_NODE_ID: u64 = 0x163 << 38;
    pub const LEAF_100_NODE_ID: u64 = 0x164 << 38;
    pub const LEAF_101_NODE_ID: u64 = 0x165 << 38;
    pub const LEAF_102_NODE_ID: u64 = 0x166 << 38;
    pub const LEAF_103_NODE_ID: u64 = 0x167 << 38;
    pub const LEAF_104_NODE_ID: u64 = 0x168 << 38;
    pub const LEAF_105_NODE_ID: u64 = 0x169 << 38;
    pub const LEAF_106_NODE_ID: u64 = 0x16A << 38;
    pub const LEAF_107_NODE_ID: u64 = 0x16B << 38;
    pub const LEAF_108_NODE_ID: u64 = 0x16C << 38;
    pub const LEAF_109_NODE_ID: u64 = 0x16D << 38;
    pub const LEAF_110_NODE_ID: u64 = 0x16E << 38;
    pub const LEAF_111_NODE_ID: u64 = 0x16F << 38;
    pub const LEAF_112_NODE_ID: u64 = 0x170 << 38;
    pub const LEAF_113_NODE_ID: u64 = 0x171 << 38;
    pub const LEAF_114_NODE_ID: u64 = 0x172 << 38;
    pub const LEAF_115_NODE_ID: u64 = 0x173 << 38;
    pub const LEAF_116_NODE_ID: u64 = 0x174 << 38;
    pub const LEAF_117_NODE_ID: u64 = 0x175 << 38;
    pub const LEAF_118_NODE_ID: u64 = 0x176 << 38;
    pub const LEAF_119_NODE_ID: u64 = 0x177 << 38;
    pub const LEAF_120_NODE_ID: u64 = 0x178 << 38;
    pub const LEAF_121_NODE_ID: u64 = 0x179 << 38;
    pub const LEAF_122_NODE_ID: u64 = 0x17A << 38;
    pub const LEAF_123_NODE_ID: u64 = 0x17B << 38;
    pub const LEAF_124_NODE_ID: u64 = 0x17C << 38;
    pub const LEAF_125_NODE_ID: u64 = 0x17D << 38;
    pub const LEAF_126_NODE_ID: u64 = 0x17E << 38;
    pub const LEAF_127_NODE_ID: u64 = 0x17F << 38;
    pub const LEAF_128_NODE_ID: u64 = 0x180 << 38;
    pub const LEAF_129_NODE_ID: u64 = 0x181 << 38;
    pub const LEAF_130_NODE_ID: u64 = 0x182 << 38;
    pub const LEAF_131_NODE_ID: u64 = 0x183 << 38;
    pub const LEAF_132_NODE_ID: u64 = 0x184 << 38;
    pub const LEAF_133_NODE_ID: u64 = 0x185 << 38;
    pub const LEAF_134_NODE_ID: u64 = 0x186 << 38;
    pub const LEAF_135_NODE_ID: u64 = 0x187 << 38;
    pub const LEAF_136_NODE_ID: u64 = 0x188 << 38;
    pub const LEAF_137_NODE_ID: u64 = 0x189 << 38;
    pub const LEAF_138_NODE_ID: u64 = 0x18A << 38;
    pub const LEAF_139_NODE_ID: u64 = 0x18B << 38;
    pub const LEAF_140_NODE_ID: u64 = 0x18C << 38;
    pub const LEAF_141_NODE_ID: u64 = 0x18D << 38;
    pub const LEAF_142_NODE_ID: u64 = 0x18E << 38;
    pub const LEAF_143_NODE_ID: u64 = 0x18F << 38;
    pub const LEAF_144_NODE_ID: u64 = 0x190 << 38;
    pub const LEAF_145_NODE_ID: u64 = 0x191 << 38;
    pub const LEAF_146_NODE_ID: u64 = 0x192 << 38;
    pub const LEAF_147_NODE_ID: u64 = 0x193 << 38;
    pub const LEAF_148_NODE_ID: u64 = 0x194 << 38;
    pub const LEAF_149_NODE_ID: u64 = 0x195 << 38;
    pub const LEAF_150_NODE_ID: u64 = 0x196 << 38;
    pub const LEAF_151_NODE_ID: u64 = 0x197 << 38;
    pub const LEAF_152_NODE_ID: u64 = 0x198 << 38;
    pub const LEAF_153_NODE_ID: u64 = 0x199 << 38;
    pub const LEAF_154_NODE_ID: u64 = 0x19A << 38;
    pub const LEAF_155_NODE_ID: u64 = 0x19B << 38;
    pub const LEAF_156_NODE_ID: u64 = 0x19C << 38;
    pub const LEAF_157_NODE_ID: u64 = 0x19D << 38;
    pub const LEAF_158_NODE_ID: u64 = 0x19E << 38;
    pub const LEAF_159_NODE_ID: u64 = 0x19F << 38;
    pub const LEAF_160_NODE_ID: u64 = 0x1A0 << 38;
    pub const LEAF_161_NODE_ID: u64 = 0x1A1 << 38;
    pub const LEAF_162_NODE_ID: u64 = 0x1A2 << 38;
    pub const LEAF_163_NODE_ID: u64 = 0x1A3 << 38;
    pub const LEAF_164_NODE_ID: u64 = 0x1A4 << 38;
    pub const LEAF_165_NODE_ID: u64 = 0x1A5 << 38;
    pub const LEAF_166_NODE_ID: u64 = 0x1A6 << 38;
    pub const LEAF_167_NODE_ID: u64 = 0x1A7 << 38;
    pub const LEAF_168_NODE_ID: u64 = 0x1A8 << 38;
    pub const LEAF_169_NODE_ID: u64 = 0x1A9 << 38;
    pub const LEAF_170_NODE_ID: u64 = 0x1AA << 38;
    pub const LEAF_171_NODE_ID: u64 = 0x1AB << 38;
    pub const LEAF_172_NODE_ID: u64 = 0x1AC << 38;
    pub const LEAF_173_NODE_ID: u64 = 0x1AD << 38;
    pub const LEAF_174_NODE_ID: u64 = 0x1AE << 38;
    pub const LEAF_175_NODE_ID: u64 = 0x1AF << 38;
    pub const LEAF_176_NODE_ID: u64 = 0x1B0 << 38;
    pub const LEAF_177_NODE_ID: u64 = 0x1B1 << 38;
    pub const LEAF_178_NODE_ID: u64 = 0x1B2 << 38;
    pub const LEAF_179_NODE_ID: u64 = 0x1B3 << 38;
    pub const LEAF_180_NODE_ID: u64 = 0x1B4 << 38;
    pub const LEAF_181_NODE_ID: u64 = 0x1B5 << 38;
    pub const LEAF_182_NODE_ID: u64 = 0x1B6 << 38;
    pub const LEAF_183_NODE_ID: u64 = 0x1B7 << 38;
    pub const LEAF_184_NODE_ID: u64 = 0x1B8 << 38;
    pub const LEAF_185_NODE_ID: u64 = 0x1B9 << 38;
    pub const LEAF_186_NODE_ID: u64 = 0x1BA << 38;
    pub const LEAF_187_NODE_ID: u64 = 0x1BB << 38;
    pub const LEAF_188_NODE_ID: u64 = 0x1BC << 38;
    pub const LEAF_189_NODE_ID: u64 = 0x1BD << 38;
    pub const LEAF_190_NODE_ID: u64 = 0x1BE << 38;
    pub const LEAF_191_NODE_ID: u64 = 0x1BF << 38;
    pub const LEAF_192_NODE_ID: u64 = 0x1C0 << 38;
    pub const LEAF_193_NODE_ID: u64 = 0x1C1 << 38;
    pub const LEAF_194_NODE_ID: u64 = 0x1C2 << 38;
    pub const LEAF_195_NODE_ID: u64 = 0x1C3 << 38;
    pub const LEAF_196_NODE_ID: u64 = 0x1C4 << 38;
    pub const LEAF_197_NODE_ID: u64 = 0x1C5 << 38;
    pub const LEAF_198_NODE_ID: u64 = 0x1C6 << 38;
    pub const LEAF_199_NODE_ID: u64 = 0x1C7 << 38;
    pub const LEAF_200_NODE_ID: u64 = 0x1C8 << 38;
    pub const LEAF_201_NODE_ID: u64 = 0x1C9 << 38;
    pub const LEAF_202_NODE_ID: u64 = 0x1CA << 38;
    pub const LEAF_203_NODE_ID: u64 = 0x1CB << 38;
    pub const LEAF_204_NODE_ID: u64 = 0x1CC << 38;
    pub const LEAF_205_NODE_ID: u64 = 0x1CD << 38;
    pub const LEAF_206_NODE_ID: u64 = 0x1CE << 38;
    pub const LEAF_207_NODE_ID: u64 = 0x1CF << 38;
    pub const LEAF_208_NODE_ID: u64 = 0x1D0 << 38;
    pub const LEAF_209_NODE_ID: u64 = 0x1D1 << 38;
    pub const LEAF_210_NODE_ID: u64 = 0x1D2 << 38;
    pub const LEAF_211_NODE_ID: u64 = 0x1D3 << 38;
    pub const LEAF_212_NODE_ID: u64 = 0x1D4 << 38;
    pub const LEAF_213_NODE_ID: u64 = 0x1D5 << 38;
    pub const LEAF_214_NODE_ID: u64 = 0x1D6 << 38;
    pub const LEAF_215_NODE_ID: u64 = 0x1D7 << 38;
    pub const LEAF_216_NODE_ID: u64 = 0x1D8 << 38;
    pub const LEAF_217_NODE_ID: u64 = 0x1D9 << 38;
    pub const LEAF_218_NODE_ID: u64 = 0x1DA << 38;
    pub const LEAF_219_NODE_ID: u64 = 0x1DB << 38;
    pub const LEAF_220_NODE_ID: u64 = 0x1DC << 38;
    pub const LEAF_221_NODE_ID: u64 = 0x1DD << 38;
    pub const LEAF_222_NODE_ID: u64 = 0x1DE << 38;
    pub const LEAF_223_NODE_ID: u64 = 0x1DF << 38;
    pub const LEAF_224_NODE_ID: u64 = 0x1E0 << 38;
    pub const LEAF_225_NODE_ID: u64 = 0x1E1 << 38;
    pub const LEAF_226_NODE_ID: u64 = 0x1E2 << 38;
    pub const LEAF_227_NODE_ID: u64 = 0x1E3 << 38;
    pub const LEAF_228_NODE_ID: u64 = 0x1E4 << 38;
    pub const LEAF_229_NODE_ID: u64 = 0x1E5 << 38;
    pub const LEAF_230_NODE_ID: u64 = 0x1E6 << 38;
    pub const LEAF_231_NODE_ID: u64 = 0x1E7 << 38;
    pub const LEAF_232_NODE_ID: u64 = 0x1E8 << 38;
    pub const LEAF_233_NODE_ID: u64 = 0x1E9 << 38;
    pub const LEAF_234_NODE_ID: u64 = 0x1EA << 38;
    pub const LEAF_235_NODE_ID: u64 = 0x1EB << 38;
    pub const LEAF_236_NODE_ID: u64 = 0x1EC << 38;
    pub const LEAF_237_NODE_ID: u64 = 0x1ED << 38;
    pub const LEAF_238_NODE_ID: u64 = 0x1EE << 38;
    pub const LEAF_239_NODE_ID: u64 = 0x1EF << 38;
    pub const LEAF_240_NODE_ID: u64 = 0x1F0 << 38;
    pub const LEAF_241_NODE_ID: u64 = 0x1F1 << 38;
    pub const LEAF_242_NODE_ID: u64 = 0x1F2 << 38;
    pub const LEAF_243_NODE_ID: u64 = 0x1F3 << 38;
    pub const LEAF_244_NODE_ID: u64 = 0x1F4 << 38;
    pub const LEAF_245_NODE_ID: u64 = 0x1F5 << 38;
    pub const LEAF_246_NODE_ID: u64 = 0x1F6 << 38;
    pub const LEAF_247_NODE_ID: u64 = 0x1F7 << 38;
    pub const LEAF_248_NODE_ID: u64 = 0x1F8 << 38;
    pub const LEAF_249_NODE_ID: u64 = 0x1F9 << 38;
    pub const LEAF_250_NODE_ID: u64 = 0x1FA << 38;
    pub const LEAF_251_NODE_ID: u64 = 0x1FB << 38;
    pub const LEAF_252_NODE_ID: u64 = 0x1FC << 38;
    pub const LEAF_253_NODE_ID: u64 = 0x1FD << 38;
    pub const LEAF_254_NODE_ID: u64 = 0x1FE << 38;
    pub const LEAF_255_NODE_ID: u64 = 0x1FF << 38;
    pub const LEAF_256_NODE_ID: u64 = 0x200 << 38;

    // Delta node constants (now safely away from the limit)
    pub const INNER_DELTA: u64 = 0x201 << 38;
    pub const LEAF_DELTA: u64 = 0x202 << 38;

    pub fn from_u64(value: u64) -> Self {
        let mut bytes = [0; 6];
        bytes[0..6].copy_from_slice(&value.to_be_bytes()[2..8]);
        VerkleNodeId(bytes)
    }

    /// Correct: Places the 6 bytes into the lower 48 bits of the u64.
    pub fn to_u64(self) -> u64 {
        let mut bytes = [0; 8];
        bytes[2..8].copy_from_slice(&self.0);
        u64::from_be_bytes(bytes)
    }
}

impl Default for VerkleNodeId {
    fn default() -> Self {
        Self::empty_id()
    }
}

impl ToNodeKind for VerkleNodeId {
    type Target = VerkleNodeKind;

    fn to_node_kind(&self) -> Option<VerkleNodeKind> {
        match self.to_u64() & Self::PREFIX_MASK {
            Self::EMPTY => Some(VerkleNodeKind::Empty),
            Self::INNER_1_NODE_ID => Some(VerkleNodeKind::Inner1),
            Self::INNER_2_NODE_ID => Some(VerkleNodeKind::Inner2),
            Self::INNER_3_NODE_ID => Some(VerkleNodeKind::Inner3),
            Self::INNER_4_NODE_ID => Some(VerkleNodeKind::Inner4),
            Self::INNER_5_NODE_ID => Some(VerkleNodeKind::Inner5),
            Self::INNER_6_NODE_ID => Some(VerkleNodeKind::Inner6),
            Self::INNER_7_NODE_ID => Some(VerkleNodeKind::Inner7),
            Self::INNER_8_NODE_ID => Some(VerkleNodeKind::Inner8),
            Self::INNER_9_NODE_ID => Some(VerkleNodeKind::Inner9),
            Self::INNER_10_NODE_ID => Some(VerkleNodeKind::Inner10),
            Self::INNER_11_NODE_ID => Some(VerkleNodeKind::Inner11),
            Self::INNER_12_NODE_ID => Some(VerkleNodeKind::Inner12),
            Self::INNER_13_NODE_ID => Some(VerkleNodeKind::Inner13),
            Self::INNER_14_NODE_ID => Some(VerkleNodeKind::Inner14),
            Self::INNER_15_NODE_ID => Some(VerkleNodeKind::Inner15),
            Self::INNER_16_NODE_ID => Some(VerkleNodeKind::Inner16),
            Self::INNER_17_NODE_ID => Some(VerkleNodeKind::Inner17),
            Self::INNER_18_NODE_ID => Some(VerkleNodeKind::Inner18),
            Self::INNER_19_NODE_ID => Some(VerkleNodeKind::Inner19),
            Self::INNER_20_NODE_ID => Some(VerkleNodeKind::Inner20),
            Self::INNER_21_NODE_ID => Some(VerkleNodeKind::Inner21),
            Self::INNER_22_NODE_ID => Some(VerkleNodeKind::Inner22),
            Self::INNER_23_NODE_ID => Some(VerkleNodeKind::Inner23),
            Self::INNER_24_NODE_ID => Some(VerkleNodeKind::Inner24),
            Self::INNER_25_NODE_ID => Some(VerkleNodeKind::Inner25),
            Self::INNER_26_NODE_ID => Some(VerkleNodeKind::Inner26),
            Self::INNER_27_NODE_ID => Some(VerkleNodeKind::Inner27),
            Self::INNER_28_NODE_ID => Some(VerkleNodeKind::Inner28),
            Self::INNER_29_NODE_ID => Some(VerkleNodeKind::Inner29),
            Self::INNER_30_NODE_ID => Some(VerkleNodeKind::Inner30),
            Self::INNER_31_NODE_ID => Some(VerkleNodeKind::Inner31),
            Self::INNER_32_NODE_ID => Some(VerkleNodeKind::Inner32),
            Self::INNER_33_NODE_ID => Some(VerkleNodeKind::Inner33),
            Self::INNER_34_NODE_ID => Some(VerkleNodeKind::Inner34),
            Self::INNER_35_NODE_ID => Some(VerkleNodeKind::Inner35),
            Self::INNER_36_NODE_ID => Some(VerkleNodeKind::Inner36),
            Self::INNER_37_NODE_ID => Some(VerkleNodeKind::Inner37),
            Self::INNER_38_NODE_ID => Some(VerkleNodeKind::Inner38),
            Self::INNER_39_NODE_ID => Some(VerkleNodeKind::Inner39),
            Self::INNER_40_NODE_ID => Some(VerkleNodeKind::Inner40),
            Self::INNER_41_NODE_ID => Some(VerkleNodeKind::Inner41),
            Self::INNER_42_NODE_ID => Some(VerkleNodeKind::Inner42),
            Self::INNER_43_NODE_ID => Some(VerkleNodeKind::Inner43),
            Self::INNER_44_NODE_ID => Some(VerkleNodeKind::Inner44),
            Self::INNER_45_NODE_ID => Some(VerkleNodeKind::Inner45),
            Self::INNER_46_NODE_ID => Some(VerkleNodeKind::Inner46),
            Self::INNER_47_NODE_ID => Some(VerkleNodeKind::Inner47),
            Self::INNER_48_NODE_ID => Some(VerkleNodeKind::Inner48),
            Self::INNER_49_NODE_ID => Some(VerkleNodeKind::Inner49),
            Self::INNER_50_NODE_ID => Some(VerkleNodeKind::Inner50),
            Self::INNER_51_NODE_ID => Some(VerkleNodeKind::Inner51),
            Self::INNER_52_NODE_ID => Some(VerkleNodeKind::Inner52),
            Self::INNER_53_NODE_ID => Some(VerkleNodeKind::Inner53),
            Self::INNER_54_NODE_ID => Some(VerkleNodeKind::Inner54),
            Self::INNER_55_NODE_ID => Some(VerkleNodeKind::Inner55),
            Self::INNER_56_NODE_ID => Some(VerkleNodeKind::Inner56),
            Self::INNER_57_NODE_ID => Some(VerkleNodeKind::Inner57),
            Self::INNER_58_NODE_ID => Some(VerkleNodeKind::Inner58),
            Self::INNER_59_NODE_ID => Some(VerkleNodeKind::Inner59),
            Self::INNER_60_NODE_ID => Some(VerkleNodeKind::Inner60),
            Self::INNER_61_NODE_ID => Some(VerkleNodeKind::Inner61),
            Self::INNER_62_NODE_ID => Some(VerkleNodeKind::Inner62),
            Self::INNER_63_NODE_ID => Some(VerkleNodeKind::Inner63),
            Self::INNER_64_NODE_ID => Some(VerkleNodeKind::Inner64),
            Self::INNER_65_NODE_ID => Some(VerkleNodeKind::Inner65),
            Self::INNER_66_NODE_ID => Some(VerkleNodeKind::Inner66),
            Self::INNER_67_NODE_ID => Some(VerkleNodeKind::Inner67),
            Self::INNER_68_NODE_ID => Some(VerkleNodeKind::Inner68),
            Self::INNER_69_NODE_ID => Some(VerkleNodeKind::Inner69),
            Self::INNER_70_NODE_ID => Some(VerkleNodeKind::Inner70),
            Self::INNER_71_NODE_ID => Some(VerkleNodeKind::Inner71),
            Self::INNER_72_NODE_ID => Some(VerkleNodeKind::Inner72),
            Self::INNER_73_NODE_ID => Some(VerkleNodeKind::Inner73),
            Self::INNER_74_NODE_ID => Some(VerkleNodeKind::Inner74),
            Self::INNER_75_NODE_ID => Some(VerkleNodeKind::Inner75),
            Self::INNER_76_NODE_ID => Some(VerkleNodeKind::Inner76),
            Self::INNER_77_NODE_ID => Some(VerkleNodeKind::Inner77),
            Self::INNER_78_NODE_ID => Some(VerkleNodeKind::Inner78),
            Self::INNER_79_NODE_ID => Some(VerkleNodeKind::Inner79),
            Self::INNER_80_NODE_ID => Some(VerkleNodeKind::Inner80),
            Self::INNER_81_NODE_ID => Some(VerkleNodeKind::Inner81),
            Self::INNER_82_NODE_ID => Some(VerkleNodeKind::Inner82),
            Self::INNER_83_NODE_ID => Some(VerkleNodeKind::Inner83),
            Self::INNER_84_NODE_ID => Some(VerkleNodeKind::Inner84),
            Self::INNER_85_NODE_ID => Some(VerkleNodeKind::Inner85),
            Self::INNER_86_NODE_ID => Some(VerkleNodeKind::Inner86),
            Self::INNER_87_NODE_ID => Some(VerkleNodeKind::Inner87),
            Self::INNER_88_NODE_ID => Some(VerkleNodeKind::Inner88),
            Self::INNER_89_NODE_ID => Some(VerkleNodeKind::Inner89),
            Self::INNER_90_NODE_ID => Some(VerkleNodeKind::Inner90),
            Self::INNER_91_NODE_ID => Some(VerkleNodeKind::Inner91),
            Self::INNER_92_NODE_ID => Some(VerkleNodeKind::Inner92),
            Self::INNER_93_NODE_ID => Some(VerkleNodeKind::Inner93),
            Self::INNER_94_NODE_ID => Some(VerkleNodeKind::Inner94),
            Self::INNER_95_NODE_ID => Some(VerkleNodeKind::Inner95),
            Self::INNER_96_NODE_ID => Some(VerkleNodeKind::Inner96),
            Self::INNER_97_NODE_ID => Some(VerkleNodeKind::Inner97),
            Self::INNER_98_NODE_ID => Some(VerkleNodeKind::Inner98),
            Self::INNER_99_NODE_ID => Some(VerkleNodeKind::Inner99),
            Self::INNER_100_NODE_ID => Some(VerkleNodeKind::Inner100),
            Self::INNER_101_NODE_ID => Some(VerkleNodeKind::Inner101),
            Self::INNER_102_NODE_ID => Some(VerkleNodeKind::Inner102),
            Self::INNER_103_NODE_ID => Some(VerkleNodeKind::Inner103),
            Self::INNER_104_NODE_ID => Some(VerkleNodeKind::Inner104),
            Self::INNER_105_NODE_ID => Some(VerkleNodeKind::Inner105),
            Self::INNER_106_NODE_ID => Some(VerkleNodeKind::Inner106),
            Self::INNER_107_NODE_ID => Some(VerkleNodeKind::Inner107),
            Self::INNER_108_NODE_ID => Some(VerkleNodeKind::Inner108),
            Self::INNER_109_NODE_ID => Some(VerkleNodeKind::Inner109),
            Self::INNER_110_NODE_ID => Some(VerkleNodeKind::Inner110),
            Self::INNER_111_NODE_ID => Some(VerkleNodeKind::Inner111),
            Self::INNER_112_NODE_ID => Some(VerkleNodeKind::Inner112),
            Self::INNER_113_NODE_ID => Some(VerkleNodeKind::Inner113),
            Self::INNER_114_NODE_ID => Some(VerkleNodeKind::Inner114),
            Self::INNER_115_NODE_ID => Some(VerkleNodeKind::Inner115),
            Self::INNER_116_NODE_ID => Some(VerkleNodeKind::Inner116),
            Self::INNER_117_NODE_ID => Some(VerkleNodeKind::Inner117),
            Self::INNER_118_NODE_ID => Some(VerkleNodeKind::Inner118),
            Self::INNER_119_NODE_ID => Some(VerkleNodeKind::Inner119),
            Self::INNER_120_NODE_ID => Some(VerkleNodeKind::Inner120),
            Self::INNER_121_NODE_ID => Some(VerkleNodeKind::Inner121),
            Self::INNER_122_NODE_ID => Some(VerkleNodeKind::Inner122),
            Self::INNER_123_NODE_ID => Some(VerkleNodeKind::Inner123),
            Self::INNER_124_NODE_ID => Some(VerkleNodeKind::Inner124),
            Self::INNER_125_NODE_ID => Some(VerkleNodeKind::Inner125),
            Self::INNER_126_NODE_ID => Some(VerkleNodeKind::Inner126),
            Self::INNER_127_NODE_ID => Some(VerkleNodeKind::Inner127),
            Self::INNER_128_NODE_ID => Some(VerkleNodeKind::Inner128),
            Self::INNER_129_NODE_ID => Some(VerkleNodeKind::Inner129),
            Self::INNER_130_NODE_ID => Some(VerkleNodeKind::Inner130),
            Self::INNER_131_NODE_ID => Some(VerkleNodeKind::Inner131),
            Self::INNER_132_NODE_ID => Some(VerkleNodeKind::Inner132),
            Self::INNER_133_NODE_ID => Some(VerkleNodeKind::Inner133),
            Self::INNER_134_NODE_ID => Some(VerkleNodeKind::Inner134),
            Self::INNER_135_NODE_ID => Some(VerkleNodeKind::Inner135),
            Self::INNER_136_NODE_ID => Some(VerkleNodeKind::Inner136),
            Self::INNER_137_NODE_ID => Some(VerkleNodeKind::Inner137),
            Self::INNER_138_NODE_ID => Some(VerkleNodeKind::Inner138),
            Self::INNER_139_NODE_ID => Some(VerkleNodeKind::Inner139),
            Self::INNER_140_NODE_ID => Some(VerkleNodeKind::Inner140),
            Self::INNER_141_NODE_ID => Some(VerkleNodeKind::Inner141),
            Self::INNER_142_NODE_ID => Some(VerkleNodeKind::Inner142),
            Self::INNER_143_NODE_ID => Some(VerkleNodeKind::Inner143),
            Self::INNER_144_NODE_ID => Some(VerkleNodeKind::Inner144),
            Self::INNER_145_NODE_ID => Some(VerkleNodeKind::Inner145),
            Self::INNER_146_NODE_ID => Some(VerkleNodeKind::Inner146),
            Self::INNER_147_NODE_ID => Some(VerkleNodeKind::Inner147),
            Self::INNER_148_NODE_ID => Some(VerkleNodeKind::Inner148),
            Self::INNER_149_NODE_ID => Some(VerkleNodeKind::Inner149),
            Self::INNER_150_NODE_ID => Some(VerkleNodeKind::Inner150),
            Self::INNER_151_NODE_ID => Some(VerkleNodeKind::Inner151),
            Self::INNER_152_NODE_ID => Some(VerkleNodeKind::Inner152),
            Self::INNER_153_NODE_ID => Some(VerkleNodeKind::Inner153),
            Self::INNER_154_NODE_ID => Some(VerkleNodeKind::Inner154),
            Self::INNER_155_NODE_ID => Some(VerkleNodeKind::Inner155),
            Self::INNER_156_NODE_ID => Some(VerkleNodeKind::Inner156),
            Self::INNER_157_NODE_ID => Some(VerkleNodeKind::Inner157),
            Self::INNER_158_NODE_ID => Some(VerkleNodeKind::Inner158),
            Self::INNER_159_NODE_ID => Some(VerkleNodeKind::Inner159),
            Self::INNER_160_NODE_ID => Some(VerkleNodeKind::Inner160),
            Self::INNER_161_NODE_ID => Some(VerkleNodeKind::Inner161),
            Self::INNER_162_NODE_ID => Some(VerkleNodeKind::Inner162),
            Self::INNER_163_NODE_ID => Some(VerkleNodeKind::Inner163),
            Self::INNER_164_NODE_ID => Some(VerkleNodeKind::Inner164),
            Self::INNER_165_NODE_ID => Some(VerkleNodeKind::Inner165),
            Self::INNER_166_NODE_ID => Some(VerkleNodeKind::Inner166),
            Self::INNER_167_NODE_ID => Some(VerkleNodeKind::Inner167),
            Self::INNER_168_NODE_ID => Some(VerkleNodeKind::Inner168),
            Self::INNER_169_NODE_ID => Some(VerkleNodeKind::Inner169),
            Self::INNER_170_NODE_ID => Some(VerkleNodeKind::Inner170),
            Self::INNER_171_NODE_ID => Some(VerkleNodeKind::Inner171),
            Self::INNER_172_NODE_ID => Some(VerkleNodeKind::Inner172),
            Self::INNER_173_NODE_ID => Some(VerkleNodeKind::Inner173),
            Self::INNER_174_NODE_ID => Some(VerkleNodeKind::Inner174),
            Self::INNER_175_NODE_ID => Some(VerkleNodeKind::Inner175),
            Self::INNER_176_NODE_ID => Some(VerkleNodeKind::Inner176),
            Self::INNER_177_NODE_ID => Some(VerkleNodeKind::Inner177),
            Self::INNER_178_NODE_ID => Some(VerkleNodeKind::Inner178),
            Self::INNER_179_NODE_ID => Some(VerkleNodeKind::Inner179),
            Self::INNER_180_NODE_ID => Some(VerkleNodeKind::Inner180),
            Self::INNER_181_NODE_ID => Some(VerkleNodeKind::Inner181),
            Self::INNER_182_NODE_ID => Some(VerkleNodeKind::Inner182),
            Self::INNER_183_NODE_ID => Some(VerkleNodeKind::Inner183),
            Self::INNER_184_NODE_ID => Some(VerkleNodeKind::Inner184),
            Self::INNER_185_NODE_ID => Some(VerkleNodeKind::Inner185),
            Self::INNER_186_NODE_ID => Some(VerkleNodeKind::Inner186),
            Self::INNER_187_NODE_ID => Some(VerkleNodeKind::Inner187),
            Self::INNER_188_NODE_ID => Some(VerkleNodeKind::Inner188),
            Self::INNER_189_NODE_ID => Some(VerkleNodeKind::Inner189),
            Self::INNER_190_NODE_ID => Some(VerkleNodeKind::Inner190),
            Self::INNER_191_NODE_ID => Some(VerkleNodeKind::Inner191),
            Self::INNER_192_NODE_ID => Some(VerkleNodeKind::Inner192),
            Self::INNER_193_NODE_ID => Some(VerkleNodeKind::Inner193),
            Self::INNER_194_NODE_ID => Some(VerkleNodeKind::Inner194),
            Self::INNER_195_NODE_ID => Some(VerkleNodeKind::Inner195),
            Self::INNER_196_NODE_ID => Some(VerkleNodeKind::Inner196),
            Self::INNER_197_NODE_ID => Some(VerkleNodeKind::Inner197),
            Self::INNER_198_NODE_ID => Some(VerkleNodeKind::Inner198),
            Self::INNER_199_NODE_ID => Some(VerkleNodeKind::Inner199),
            Self::INNER_200_NODE_ID => Some(VerkleNodeKind::Inner200),
            Self::INNER_201_NODE_ID => Some(VerkleNodeKind::Inner201),
            Self::INNER_202_NODE_ID => Some(VerkleNodeKind::Inner202),
            Self::INNER_203_NODE_ID => Some(VerkleNodeKind::Inner203),
            Self::INNER_204_NODE_ID => Some(VerkleNodeKind::Inner204),
            Self::INNER_205_NODE_ID => Some(VerkleNodeKind::Inner205),
            Self::INNER_206_NODE_ID => Some(VerkleNodeKind::Inner206),
            Self::INNER_207_NODE_ID => Some(VerkleNodeKind::Inner207),
            Self::INNER_208_NODE_ID => Some(VerkleNodeKind::Inner208),
            Self::INNER_209_NODE_ID => Some(VerkleNodeKind::Inner209),
            Self::INNER_210_NODE_ID => Some(VerkleNodeKind::Inner210),
            Self::INNER_211_NODE_ID => Some(VerkleNodeKind::Inner211),
            Self::INNER_212_NODE_ID => Some(VerkleNodeKind::Inner212),
            Self::INNER_213_NODE_ID => Some(VerkleNodeKind::Inner213),
            Self::INNER_214_NODE_ID => Some(VerkleNodeKind::Inner214),
            Self::INNER_215_NODE_ID => Some(VerkleNodeKind::Inner215),
            Self::INNER_216_NODE_ID => Some(VerkleNodeKind::Inner216),
            Self::INNER_217_NODE_ID => Some(VerkleNodeKind::Inner217),
            Self::INNER_218_NODE_ID => Some(VerkleNodeKind::Inner218),
            Self::INNER_219_NODE_ID => Some(VerkleNodeKind::Inner219),
            Self::INNER_220_NODE_ID => Some(VerkleNodeKind::Inner220),
            Self::INNER_221_NODE_ID => Some(VerkleNodeKind::Inner221),
            Self::INNER_222_NODE_ID => Some(VerkleNodeKind::Inner222),
            Self::INNER_223_NODE_ID => Some(VerkleNodeKind::Inner223),
            Self::INNER_224_NODE_ID => Some(VerkleNodeKind::Inner224),
            Self::INNER_225_NODE_ID => Some(VerkleNodeKind::Inner225),
            Self::INNER_226_NODE_ID => Some(VerkleNodeKind::Inner226),
            Self::INNER_227_NODE_ID => Some(VerkleNodeKind::Inner227),
            Self::INNER_228_NODE_ID => Some(VerkleNodeKind::Inner228),
            Self::INNER_229_NODE_ID => Some(VerkleNodeKind::Inner229),
            Self::INNER_230_NODE_ID => Some(VerkleNodeKind::Inner230),
            Self::INNER_231_NODE_ID => Some(VerkleNodeKind::Inner231),
            Self::INNER_232_NODE_ID => Some(VerkleNodeKind::Inner232),
            Self::INNER_233_NODE_ID => Some(VerkleNodeKind::Inner233),
            Self::INNER_234_NODE_ID => Some(VerkleNodeKind::Inner234),
            Self::INNER_235_NODE_ID => Some(VerkleNodeKind::Inner235),
            Self::INNER_236_NODE_ID => Some(VerkleNodeKind::Inner236),
            Self::INNER_237_NODE_ID => Some(VerkleNodeKind::Inner237),
            Self::INNER_238_NODE_ID => Some(VerkleNodeKind::Inner238),
            Self::INNER_239_NODE_ID => Some(VerkleNodeKind::Inner239),
            Self::INNER_240_NODE_ID => Some(VerkleNodeKind::Inner240),
            Self::INNER_241_NODE_ID => Some(VerkleNodeKind::Inner241),
            Self::INNER_242_NODE_ID => Some(VerkleNodeKind::Inner242),
            Self::INNER_243_NODE_ID => Some(VerkleNodeKind::Inner243),
            Self::INNER_244_NODE_ID => Some(VerkleNodeKind::Inner244),
            Self::INNER_245_NODE_ID => Some(VerkleNodeKind::Inner245),
            Self::INNER_246_NODE_ID => Some(VerkleNodeKind::Inner246),
            Self::INNER_247_NODE_ID => Some(VerkleNodeKind::Inner247),
            Self::INNER_248_NODE_ID => Some(VerkleNodeKind::Inner248),
            Self::INNER_249_NODE_ID => Some(VerkleNodeKind::Inner249),
            Self::INNER_250_NODE_ID => Some(VerkleNodeKind::Inner250),
            Self::INNER_251_NODE_ID => Some(VerkleNodeKind::Inner251),
            Self::INNER_252_NODE_ID => Some(VerkleNodeKind::Inner252),
            Self::INNER_253_NODE_ID => Some(VerkleNodeKind::Inner253),
            Self::INNER_254_NODE_ID => Some(VerkleNodeKind::Inner254),
            Self::INNER_255_NODE_ID => Some(VerkleNodeKind::Inner255),
            Self::INNER_256_NODE_ID => Some(VerkleNodeKind::Inner256),
            Self::LEAF_1_NODE_ID => Some(VerkleNodeKind::Leaf1),
            Self::LEAF_2_NODE_ID => Some(VerkleNodeKind::Leaf2),
            Self::LEAF_3_NODE_ID => Some(VerkleNodeKind::Leaf3),
            Self::LEAF_4_NODE_ID => Some(VerkleNodeKind::Leaf4),
            Self::LEAF_5_NODE_ID => Some(VerkleNodeKind::Leaf5),
            Self::LEAF_6_NODE_ID => Some(VerkleNodeKind::Leaf6),
            Self::LEAF_7_NODE_ID => Some(VerkleNodeKind::Leaf7),
            Self::LEAF_8_NODE_ID => Some(VerkleNodeKind::Leaf8),
            Self::LEAF_9_NODE_ID => Some(VerkleNodeKind::Leaf9),
            Self::LEAF_10_NODE_ID => Some(VerkleNodeKind::Leaf10),
            Self::LEAF_11_NODE_ID => Some(VerkleNodeKind::Leaf11),
            Self::LEAF_12_NODE_ID => Some(VerkleNodeKind::Leaf12),
            Self::LEAF_13_NODE_ID => Some(VerkleNodeKind::Leaf13),
            Self::LEAF_14_NODE_ID => Some(VerkleNodeKind::Leaf14),
            Self::LEAF_15_NODE_ID => Some(VerkleNodeKind::Leaf15),
            Self::LEAF_16_NODE_ID => Some(VerkleNodeKind::Leaf16),
            Self::LEAF_17_NODE_ID => Some(VerkleNodeKind::Leaf17),
            Self::LEAF_18_NODE_ID => Some(VerkleNodeKind::Leaf18),
            Self::LEAF_19_NODE_ID => Some(VerkleNodeKind::Leaf19),
            Self::LEAF_20_NODE_ID => Some(VerkleNodeKind::Leaf20),
            Self::LEAF_21_NODE_ID => Some(VerkleNodeKind::Leaf21),
            Self::LEAF_22_NODE_ID => Some(VerkleNodeKind::Leaf22),
            Self::LEAF_23_NODE_ID => Some(VerkleNodeKind::Leaf23),
            Self::LEAF_24_NODE_ID => Some(VerkleNodeKind::Leaf24),
            Self::LEAF_25_NODE_ID => Some(VerkleNodeKind::Leaf25),
            Self::LEAF_26_NODE_ID => Some(VerkleNodeKind::Leaf26),
            Self::LEAF_27_NODE_ID => Some(VerkleNodeKind::Leaf27),
            Self::LEAF_28_NODE_ID => Some(VerkleNodeKind::Leaf28),
            Self::LEAF_29_NODE_ID => Some(VerkleNodeKind::Leaf29),
            Self::LEAF_30_NODE_ID => Some(VerkleNodeKind::Leaf30),
            Self::LEAF_31_NODE_ID => Some(VerkleNodeKind::Leaf31),
            Self::LEAF_32_NODE_ID => Some(VerkleNodeKind::Leaf32),
            Self::LEAF_33_NODE_ID => Some(VerkleNodeKind::Leaf33),
            Self::LEAF_34_NODE_ID => Some(VerkleNodeKind::Leaf34),
            Self::LEAF_35_NODE_ID => Some(VerkleNodeKind::Leaf35),
            Self::LEAF_36_NODE_ID => Some(VerkleNodeKind::Leaf36),
            Self::LEAF_37_NODE_ID => Some(VerkleNodeKind::Leaf37),
            Self::LEAF_38_NODE_ID => Some(VerkleNodeKind::Leaf38),
            Self::LEAF_39_NODE_ID => Some(VerkleNodeKind::Leaf39),
            Self::LEAF_40_NODE_ID => Some(VerkleNodeKind::Leaf40),
            Self::LEAF_41_NODE_ID => Some(VerkleNodeKind::Leaf41),
            Self::LEAF_42_NODE_ID => Some(VerkleNodeKind::Leaf42),
            Self::LEAF_43_NODE_ID => Some(VerkleNodeKind::Leaf43),
            Self::LEAF_44_NODE_ID => Some(VerkleNodeKind::Leaf44),
            Self::LEAF_45_NODE_ID => Some(VerkleNodeKind::Leaf45),
            Self::LEAF_46_NODE_ID => Some(VerkleNodeKind::Leaf46),
            Self::LEAF_47_NODE_ID => Some(VerkleNodeKind::Leaf47),
            Self::LEAF_48_NODE_ID => Some(VerkleNodeKind::Leaf48),
            Self::LEAF_49_NODE_ID => Some(VerkleNodeKind::Leaf49),
            Self::LEAF_50_NODE_ID => Some(VerkleNodeKind::Leaf50),
            Self::LEAF_51_NODE_ID => Some(VerkleNodeKind::Leaf51),
            Self::LEAF_52_NODE_ID => Some(VerkleNodeKind::Leaf52),
            Self::LEAF_53_NODE_ID => Some(VerkleNodeKind::Leaf53),
            Self::LEAF_54_NODE_ID => Some(VerkleNodeKind::Leaf54),
            Self::LEAF_55_NODE_ID => Some(VerkleNodeKind::Leaf55),
            Self::LEAF_56_NODE_ID => Some(VerkleNodeKind::Leaf56),
            Self::LEAF_57_NODE_ID => Some(VerkleNodeKind::Leaf57),
            Self::LEAF_58_NODE_ID => Some(VerkleNodeKind::Leaf58),
            Self::LEAF_59_NODE_ID => Some(VerkleNodeKind::Leaf59),
            Self::LEAF_60_NODE_ID => Some(VerkleNodeKind::Leaf60),
            Self::LEAF_61_NODE_ID => Some(VerkleNodeKind::Leaf61),
            Self::LEAF_62_NODE_ID => Some(VerkleNodeKind::Leaf62),
            Self::LEAF_63_NODE_ID => Some(VerkleNodeKind::Leaf63),
            Self::LEAF_64_NODE_ID => Some(VerkleNodeKind::Leaf64),
            Self::LEAF_65_NODE_ID => Some(VerkleNodeKind::Leaf65),
            Self::LEAF_66_NODE_ID => Some(VerkleNodeKind::Leaf66),
            Self::LEAF_67_NODE_ID => Some(VerkleNodeKind::Leaf67),
            Self::LEAF_68_NODE_ID => Some(VerkleNodeKind::Leaf68),
            Self::LEAF_69_NODE_ID => Some(VerkleNodeKind::Leaf69),
            Self::LEAF_70_NODE_ID => Some(VerkleNodeKind::Leaf70),
            Self::LEAF_71_NODE_ID => Some(VerkleNodeKind::Leaf71),
            Self::LEAF_72_NODE_ID => Some(VerkleNodeKind::Leaf72),
            Self::LEAF_73_NODE_ID => Some(VerkleNodeKind::Leaf73),
            Self::LEAF_74_NODE_ID => Some(VerkleNodeKind::Leaf74),
            Self::LEAF_75_NODE_ID => Some(VerkleNodeKind::Leaf75),
            Self::LEAF_76_NODE_ID => Some(VerkleNodeKind::Leaf76),
            Self::LEAF_77_NODE_ID => Some(VerkleNodeKind::Leaf77),
            Self::LEAF_78_NODE_ID => Some(VerkleNodeKind::Leaf78),
            Self::LEAF_79_NODE_ID => Some(VerkleNodeKind::Leaf79),
            Self::LEAF_80_NODE_ID => Some(VerkleNodeKind::Leaf80),
            Self::LEAF_81_NODE_ID => Some(VerkleNodeKind::Leaf81),
            Self::LEAF_82_NODE_ID => Some(VerkleNodeKind::Leaf82),
            Self::LEAF_83_NODE_ID => Some(VerkleNodeKind::Leaf83),
            Self::LEAF_84_NODE_ID => Some(VerkleNodeKind::Leaf84),
            Self::LEAF_85_NODE_ID => Some(VerkleNodeKind::Leaf85),
            Self::LEAF_86_NODE_ID => Some(VerkleNodeKind::Leaf86),
            Self::LEAF_87_NODE_ID => Some(VerkleNodeKind::Leaf87),
            Self::LEAF_88_NODE_ID => Some(VerkleNodeKind::Leaf88),
            Self::LEAF_89_NODE_ID => Some(VerkleNodeKind::Leaf89),
            Self::LEAF_90_NODE_ID => Some(VerkleNodeKind::Leaf90),
            Self::LEAF_91_NODE_ID => Some(VerkleNodeKind::Leaf91),
            Self::LEAF_92_NODE_ID => Some(VerkleNodeKind::Leaf92),
            Self::LEAF_93_NODE_ID => Some(VerkleNodeKind::Leaf93),
            Self::LEAF_94_NODE_ID => Some(VerkleNodeKind::Leaf94),
            Self::LEAF_95_NODE_ID => Some(VerkleNodeKind::Leaf95),
            Self::LEAF_96_NODE_ID => Some(VerkleNodeKind::Leaf96),
            Self::LEAF_97_NODE_ID => Some(VerkleNodeKind::Leaf97),
            Self::LEAF_98_NODE_ID => Some(VerkleNodeKind::Leaf98),
            Self::LEAF_99_NODE_ID => Some(VerkleNodeKind::Leaf99),
            Self::LEAF_100_NODE_ID => Some(VerkleNodeKind::Leaf100),
            Self::LEAF_101_NODE_ID => Some(VerkleNodeKind::Leaf101),
            Self::LEAF_102_NODE_ID => Some(VerkleNodeKind::Leaf102),
            Self::LEAF_103_NODE_ID => Some(VerkleNodeKind::Leaf103),
            Self::LEAF_104_NODE_ID => Some(VerkleNodeKind::Leaf104),
            Self::LEAF_105_NODE_ID => Some(VerkleNodeKind::Leaf105),
            Self::LEAF_106_NODE_ID => Some(VerkleNodeKind::Leaf106),
            Self::LEAF_107_NODE_ID => Some(VerkleNodeKind::Leaf107),
            Self::LEAF_108_NODE_ID => Some(VerkleNodeKind::Leaf108),
            Self::LEAF_109_NODE_ID => Some(VerkleNodeKind::Leaf109),
            Self::LEAF_110_NODE_ID => Some(VerkleNodeKind::Leaf110),
            Self::LEAF_111_NODE_ID => Some(VerkleNodeKind::Leaf111),
            Self::LEAF_112_NODE_ID => Some(VerkleNodeKind::Leaf112),
            Self::LEAF_113_NODE_ID => Some(VerkleNodeKind::Leaf113),
            Self::LEAF_114_NODE_ID => Some(VerkleNodeKind::Leaf114),
            Self::LEAF_115_NODE_ID => Some(VerkleNodeKind::Leaf115),
            Self::LEAF_116_NODE_ID => Some(VerkleNodeKind::Leaf116),
            Self::LEAF_117_NODE_ID => Some(VerkleNodeKind::Leaf117),
            Self::LEAF_118_NODE_ID => Some(VerkleNodeKind::Leaf118),
            Self::LEAF_119_NODE_ID => Some(VerkleNodeKind::Leaf119),
            Self::LEAF_120_NODE_ID => Some(VerkleNodeKind::Leaf120),
            Self::LEAF_121_NODE_ID => Some(VerkleNodeKind::Leaf121),
            Self::LEAF_122_NODE_ID => Some(VerkleNodeKind::Leaf122),
            Self::LEAF_123_NODE_ID => Some(VerkleNodeKind::Leaf123),
            Self::LEAF_124_NODE_ID => Some(VerkleNodeKind::Leaf124),
            Self::LEAF_125_NODE_ID => Some(VerkleNodeKind::Leaf125),
            Self::LEAF_126_NODE_ID => Some(VerkleNodeKind::Leaf126),
            Self::LEAF_127_NODE_ID => Some(VerkleNodeKind::Leaf127),
            Self::LEAF_128_NODE_ID => Some(VerkleNodeKind::Leaf128),
            Self::LEAF_129_NODE_ID => Some(VerkleNodeKind::Leaf129),
            Self::LEAF_130_NODE_ID => Some(VerkleNodeKind::Leaf130),
            Self::LEAF_131_NODE_ID => Some(VerkleNodeKind::Leaf131),
            Self::LEAF_132_NODE_ID => Some(VerkleNodeKind::Leaf132),
            Self::LEAF_133_NODE_ID => Some(VerkleNodeKind::Leaf133),
            Self::LEAF_134_NODE_ID => Some(VerkleNodeKind::Leaf134),
            Self::LEAF_135_NODE_ID => Some(VerkleNodeKind::Leaf135),
            Self::LEAF_136_NODE_ID => Some(VerkleNodeKind::Leaf136),
            Self::LEAF_137_NODE_ID => Some(VerkleNodeKind::Leaf137),
            Self::LEAF_138_NODE_ID => Some(VerkleNodeKind::Leaf138),
            Self::LEAF_139_NODE_ID => Some(VerkleNodeKind::Leaf139),
            Self::LEAF_140_NODE_ID => Some(VerkleNodeKind::Leaf140),
            Self::LEAF_141_NODE_ID => Some(VerkleNodeKind::Leaf141),
            Self::LEAF_142_NODE_ID => Some(VerkleNodeKind::Leaf142),
            Self::LEAF_143_NODE_ID => Some(VerkleNodeKind::Leaf143),
            Self::LEAF_144_NODE_ID => Some(VerkleNodeKind::Leaf144),
            Self::LEAF_145_NODE_ID => Some(VerkleNodeKind::Leaf145),
            Self::LEAF_146_NODE_ID => Some(VerkleNodeKind::Leaf146),
            Self::LEAF_147_NODE_ID => Some(VerkleNodeKind::Leaf147),
            Self::LEAF_148_NODE_ID => Some(VerkleNodeKind::Leaf148),
            Self::LEAF_149_NODE_ID => Some(VerkleNodeKind::Leaf149),
            Self::LEAF_150_NODE_ID => Some(VerkleNodeKind::Leaf150),
            Self::LEAF_151_NODE_ID => Some(VerkleNodeKind::Leaf151),
            Self::LEAF_152_NODE_ID => Some(VerkleNodeKind::Leaf152),
            Self::LEAF_153_NODE_ID => Some(VerkleNodeKind::Leaf153),
            Self::LEAF_154_NODE_ID => Some(VerkleNodeKind::Leaf154),
            Self::LEAF_155_NODE_ID => Some(VerkleNodeKind::Leaf155),
            Self::LEAF_156_NODE_ID => Some(VerkleNodeKind::Leaf156),
            Self::LEAF_157_NODE_ID => Some(VerkleNodeKind::Leaf157),
            Self::LEAF_158_NODE_ID => Some(VerkleNodeKind::Leaf158),
            Self::LEAF_159_NODE_ID => Some(VerkleNodeKind::Leaf159),
            Self::LEAF_160_NODE_ID => Some(VerkleNodeKind::Leaf160),
            Self::LEAF_161_NODE_ID => Some(VerkleNodeKind::Leaf161),
            Self::LEAF_162_NODE_ID => Some(VerkleNodeKind::Leaf162),
            Self::LEAF_163_NODE_ID => Some(VerkleNodeKind::Leaf163),
            Self::LEAF_164_NODE_ID => Some(VerkleNodeKind::Leaf164),
            Self::LEAF_165_NODE_ID => Some(VerkleNodeKind::Leaf165),
            Self::LEAF_166_NODE_ID => Some(VerkleNodeKind::Leaf166),
            Self::LEAF_167_NODE_ID => Some(VerkleNodeKind::Leaf167),
            Self::LEAF_168_NODE_ID => Some(VerkleNodeKind::Leaf168),
            Self::LEAF_169_NODE_ID => Some(VerkleNodeKind::Leaf169),
            Self::LEAF_170_NODE_ID => Some(VerkleNodeKind::Leaf170),
            Self::LEAF_171_NODE_ID => Some(VerkleNodeKind::Leaf171),
            Self::LEAF_172_NODE_ID => Some(VerkleNodeKind::Leaf172),
            Self::LEAF_173_NODE_ID => Some(VerkleNodeKind::Leaf173),
            Self::LEAF_174_NODE_ID => Some(VerkleNodeKind::Leaf174),
            Self::LEAF_175_NODE_ID => Some(VerkleNodeKind::Leaf175),
            Self::LEAF_176_NODE_ID => Some(VerkleNodeKind::Leaf176),
            Self::LEAF_177_NODE_ID => Some(VerkleNodeKind::Leaf177),
            Self::LEAF_178_NODE_ID => Some(VerkleNodeKind::Leaf178),
            Self::LEAF_179_NODE_ID => Some(VerkleNodeKind::Leaf179),
            Self::LEAF_180_NODE_ID => Some(VerkleNodeKind::Leaf180),
            Self::LEAF_181_NODE_ID => Some(VerkleNodeKind::Leaf181),
            Self::LEAF_182_NODE_ID => Some(VerkleNodeKind::Leaf182),
            Self::LEAF_183_NODE_ID => Some(VerkleNodeKind::Leaf183),
            Self::LEAF_184_NODE_ID => Some(VerkleNodeKind::Leaf184),
            Self::LEAF_185_NODE_ID => Some(VerkleNodeKind::Leaf185),
            Self::LEAF_186_NODE_ID => Some(VerkleNodeKind::Leaf186),
            Self::LEAF_187_NODE_ID => Some(VerkleNodeKind::Leaf187),
            Self::LEAF_188_NODE_ID => Some(VerkleNodeKind::Leaf188),
            Self::LEAF_189_NODE_ID => Some(VerkleNodeKind::Leaf189),
            Self::LEAF_190_NODE_ID => Some(VerkleNodeKind::Leaf190),
            Self::LEAF_191_NODE_ID => Some(VerkleNodeKind::Leaf191),
            Self::LEAF_192_NODE_ID => Some(VerkleNodeKind::Leaf192),
            Self::LEAF_193_NODE_ID => Some(VerkleNodeKind::Leaf193),
            Self::LEAF_194_NODE_ID => Some(VerkleNodeKind::Leaf194),
            Self::LEAF_195_NODE_ID => Some(VerkleNodeKind::Leaf195),
            Self::LEAF_196_NODE_ID => Some(VerkleNodeKind::Leaf196),
            Self::LEAF_197_NODE_ID => Some(VerkleNodeKind::Leaf197),
            Self::LEAF_198_NODE_ID => Some(VerkleNodeKind::Leaf198),
            Self::LEAF_199_NODE_ID => Some(VerkleNodeKind::Leaf199),
            Self::LEAF_200_NODE_ID => Some(VerkleNodeKind::Leaf200),
            Self::LEAF_201_NODE_ID => Some(VerkleNodeKind::Leaf201),
            Self::LEAF_202_NODE_ID => Some(VerkleNodeKind::Leaf202),
            Self::LEAF_203_NODE_ID => Some(VerkleNodeKind::Leaf203),
            Self::LEAF_204_NODE_ID => Some(VerkleNodeKind::Leaf204),
            Self::LEAF_205_NODE_ID => Some(VerkleNodeKind::Leaf205),
            Self::LEAF_206_NODE_ID => Some(VerkleNodeKind::Leaf206),
            Self::LEAF_207_NODE_ID => Some(VerkleNodeKind::Leaf207),
            Self::LEAF_208_NODE_ID => Some(VerkleNodeKind::Leaf208),
            Self::LEAF_209_NODE_ID => Some(VerkleNodeKind::Leaf209),
            Self::LEAF_210_NODE_ID => Some(VerkleNodeKind::Leaf210),
            Self::LEAF_211_NODE_ID => Some(VerkleNodeKind::Leaf211),
            Self::LEAF_212_NODE_ID => Some(VerkleNodeKind::Leaf212),
            Self::LEAF_213_NODE_ID => Some(VerkleNodeKind::Leaf213),
            Self::LEAF_214_NODE_ID => Some(VerkleNodeKind::Leaf214),
            Self::LEAF_215_NODE_ID => Some(VerkleNodeKind::Leaf215),
            Self::LEAF_216_NODE_ID => Some(VerkleNodeKind::Leaf216),
            Self::LEAF_217_NODE_ID => Some(VerkleNodeKind::Leaf217),
            Self::LEAF_218_NODE_ID => Some(VerkleNodeKind::Leaf218),
            Self::LEAF_219_NODE_ID => Some(VerkleNodeKind::Leaf219),
            Self::LEAF_220_NODE_ID => Some(VerkleNodeKind::Leaf220),
            Self::LEAF_221_NODE_ID => Some(VerkleNodeKind::Leaf221),
            Self::LEAF_222_NODE_ID => Some(VerkleNodeKind::Leaf222),
            Self::LEAF_223_NODE_ID => Some(VerkleNodeKind::Leaf223),
            Self::LEAF_224_NODE_ID => Some(VerkleNodeKind::Leaf224),
            Self::LEAF_225_NODE_ID => Some(VerkleNodeKind::Leaf225),
            Self::LEAF_226_NODE_ID => Some(VerkleNodeKind::Leaf226),
            Self::LEAF_227_NODE_ID => Some(VerkleNodeKind::Leaf227),
            Self::LEAF_228_NODE_ID => Some(VerkleNodeKind::Leaf228),
            Self::LEAF_229_NODE_ID => Some(VerkleNodeKind::Leaf229),
            Self::LEAF_230_NODE_ID => Some(VerkleNodeKind::Leaf230),
            Self::LEAF_231_NODE_ID => Some(VerkleNodeKind::Leaf231),
            Self::LEAF_232_NODE_ID => Some(VerkleNodeKind::Leaf232),
            Self::LEAF_233_NODE_ID => Some(VerkleNodeKind::Leaf233),
            Self::LEAF_234_NODE_ID => Some(VerkleNodeKind::Leaf234),
            Self::LEAF_235_NODE_ID => Some(VerkleNodeKind::Leaf235),
            Self::LEAF_236_NODE_ID => Some(VerkleNodeKind::Leaf236),
            Self::LEAF_237_NODE_ID => Some(VerkleNodeKind::Leaf237),
            Self::LEAF_238_NODE_ID => Some(VerkleNodeKind::Leaf238),
            Self::LEAF_239_NODE_ID => Some(VerkleNodeKind::Leaf239),
            Self::LEAF_240_NODE_ID => Some(VerkleNodeKind::Leaf240),
            Self::LEAF_241_NODE_ID => Some(VerkleNodeKind::Leaf241),
            Self::LEAF_242_NODE_ID => Some(VerkleNodeKind::Leaf242),
            Self::LEAF_243_NODE_ID => Some(VerkleNodeKind::Leaf243),
            Self::LEAF_244_NODE_ID => Some(VerkleNodeKind::Leaf244),
            Self::LEAF_245_NODE_ID => Some(VerkleNodeKind::Leaf245),
            Self::LEAF_246_NODE_ID => Some(VerkleNodeKind::Leaf246),
            Self::LEAF_247_NODE_ID => Some(VerkleNodeKind::Leaf247),
            Self::LEAF_248_NODE_ID => Some(VerkleNodeKind::Leaf248),
            Self::LEAF_249_NODE_ID => Some(VerkleNodeKind::Leaf249),
            Self::LEAF_250_NODE_ID => Some(VerkleNodeKind::Leaf250),
            Self::LEAF_251_NODE_ID => Some(VerkleNodeKind::Leaf251),
            Self::LEAF_252_NODE_ID => Some(VerkleNodeKind::Leaf252),
            Self::LEAF_253_NODE_ID => Some(VerkleNodeKind::Leaf253),
            Self::LEAF_254_NODE_ID => Some(VerkleNodeKind::Leaf254),
            Self::LEAF_255_NODE_ID => Some(VerkleNodeKind::Leaf255),
            Self::LEAF_256_NODE_ID => Some(VerkleNodeKind::Leaf256),
            Self::INNER_DELTA => Some(VerkleNodeKind::InnerDelta),
            Self::LEAF_DELTA => Some(VerkleNodeKind::LeafDelta),
            _ => None,
        }
    }
}

impl TreeId for VerkleNodeId {
    fn from_idx_and_node_kind(idx: u64, node_type: VerkleNodeKind) -> Self {
        // println!("Index: {}, Node Type: {:?}", idx, node_type);
        // const {
        //     assert!(
        //         (Self::INDEX_MASK + Self::PREFIX_MASK) == 0x00FF_FFFF_FFFF_FFFF,
        //         "index and prefix masks should not overlap"
        //     );
        // }
        assert!(
            (idx & !Self::INDEX_MASK) == 0,
            "indices cannot get this large, unless we have a bug somewhere"
        );
        match node_type {
            VerkleNodeKind::Empty => VerkleNodeId::from_u64(idx | Self::EMPTY),
            VerkleNodeKind::Inner1 => VerkleNodeId::from_u64(idx | Self::INNER_1_NODE_ID),
            VerkleNodeKind::Inner2 => VerkleNodeId::from_u64(idx | Self::INNER_2_NODE_ID),
            VerkleNodeKind::Inner3 => VerkleNodeId::from_u64(idx | Self::INNER_3_NODE_ID),
            VerkleNodeKind::Inner4 => VerkleNodeId::from_u64(idx | Self::INNER_4_NODE_ID),
            VerkleNodeKind::Inner5 => VerkleNodeId::from_u64(idx | Self::INNER_5_NODE_ID),
            VerkleNodeKind::Inner6 => VerkleNodeId::from_u64(idx | Self::INNER_6_NODE_ID),
            VerkleNodeKind::Inner7 => VerkleNodeId::from_u64(idx | Self::INNER_7_NODE_ID),
            VerkleNodeKind::Inner8 => VerkleNodeId::from_u64(idx | Self::INNER_8_NODE_ID),
            VerkleNodeKind::Inner9 => VerkleNodeId::from_u64(idx | Self::INNER_9_NODE_ID),
            VerkleNodeKind::Inner10 => VerkleNodeId::from_u64(idx | Self::INNER_10_NODE_ID),
            VerkleNodeKind::Inner11 => VerkleNodeId::from_u64(idx | Self::INNER_11_NODE_ID),
            VerkleNodeKind::Inner12 => VerkleNodeId::from_u64(idx | Self::INNER_12_NODE_ID),
            VerkleNodeKind::Inner13 => VerkleNodeId::from_u64(idx | Self::INNER_13_NODE_ID),
            VerkleNodeKind::Inner14 => VerkleNodeId::from_u64(idx | Self::INNER_14_NODE_ID),
            VerkleNodeKind::Inner15 => VerkleNodeId::from_u64(idx | Self::INNER_15_NODE_ID),
            VerkleNodeKind::Inner16 => VerkleNodeId::from_u64(idx | Self::INNER_16_NODE_ID),
            VerkleNodeKind::Inner17 => VerkleNodeId::from_u64(idx | Self::INNER_17_NODE_ID),
            VerkleNodeKind::Inner18 => VerkleNodeId::from_u64(idx | Self::INNER_18_NODE_ID),
            VerkleNodeKind::Inner19 => VerkleNodeId::from_u64(idx | Self::INNER_19_NODE_ID),
            VerkleNodeKind::Inner20 => VerkleNodeId::from_u64(idx | Self::INNER_20_NODE_ID),
            VerkleNodeKind::Inner21 => VerkleNodeId::from_u64(idx | Self::INNER_21_NODE_ID),
            VerkleNodeKind::Inner22 => VerkleNodeId::from_u64(idx | Self::INNER_22_NODE_ID),
            VerkleNodeKind::Inner23 => VerkleNodeId::from_u64(idx | Self::INNER_23_NODE_ID),
            VerkleNodeKind::Inner24 => VerkleNodeId::from_u64(idx | Self::INNER_24_NODE_ID),
            VerkleNodeKind::Inner25 => VerkleNodeId::from_u64(idx | Self::INNER_25_NODE_ID),
            VerkleNodeKind::Inner26 => VerkleNodeId::from_u64(idx | Self::INNER_26_NODE_ID),
            VerkleNodeKind::Inner27 => VerkleNodeId::from_u64(idx | Self::INNER_27_NODE_ID),
            VerkleNodeKind::Inner28 => VerkleNodeId::from_u64(idx | Self::INNER_28_NODE_ID),
            VerkleNodeKind::Inner29 => VerkleNodeId::from_u64(idx | Self::INNER_29_NODE_ID),
            VerkleNodeKind::Inner30 => VerkleNodeId::from_u64(idx | Self::INNER_30_NODE_ID),
            VerkleNodeKind::Inner31 => VerkleNodeId::from_u64(idx | Self::INNER_31_NODE_ID),
            VerkleNodeKind::Inner32 => VerkleNodeId::from_u64(idx | Self::INNER_32_NODE_ID),
            VerkleNodeKind::Inner33 => VerkleNodeId::from_u64(idx | Self::INNER_33_NODE_ID),
            VerkleNodeKind::Inner34 => VerkleNodeId::from_u64(idx | Self::INNER_34_NODE_ID),
            VerkleNodeKind::Inner35 => VerkleNodeId::from_u64(idx | Self::INNER_35_NODE_ID),
            VerkleNodeKind::Inner36 => VerkleNodeId::from_u64(idx | Self::INNER_36_NODE_ID),
            VerkleNodeKind::Inner37 => VerkleNodeId::from_u64(idx | Self::INNER_37_NODE_ID),
            VerkleNodeKind::Inner38 => VerkleNodeId::from_u64(idx | Self::INNER_38_NODE_ID),
            VerkleNodeKind::Inner39 => VerkleNodeId::from_u64(idx | Self::INNER_39_NODE_ID),
            VerkleNodeKind::Inner40 => VerkleNodeId::from_u64(idx | Self::INNER_40_NODE_ID),
            VerkleNodeKind::Inner41 => VerkleNodeId::from_u64(idx | Self::INNER_41_NODE_ID),
            VerkleNodeKind::Inner42 => VerkleNodeId::from_u64(idx | Self::INNER_42_NODE_ID),
            VerkleNodeKind::Inner43 => VerkleNodeId::from_u64(idx | Self::INNER_43_NODE_ID),
            VerkleNodeKind::Inner44 => VerkleNodeId::from_u64(idx | Self::INNER_44_NODE_ID),
            VerkleNodeKind::Inner45 => VerkleNodeId::from_u64(idx | Self::INNER_45_NODE_ID),
            VerkleNodeKind::Inner46 => VerkleNodeId::from_u64(idx | Self::INNER_46_NODE_ID),
            VerkleNodeKind::Inner47 => VerkleNodeId::from_u64(idx | Self::INNER_47_NODE_ID),
            VerkleNodeKind::Inner48 => VerkleNodeId::from_u64(idx | Self::INNER_48_NODE_ID),
            VerkleNodeKind::Inner49 => VerkleNodeId::from_u64(idx | Self::INNER_49_NODE_ID),
            VerkleNodeKind::Inner50 => VerkleNodeId::from_u64(idx | Self::INNER_50_NODE_ID),
            VerkleNodeKind::Inner51 => VerkleNodeId::from_u64(idx | Self::INNER_51_NODE_ID),
            VerkleNodeKind::Inner52 => VerkleNodeId::from_u64(idx | Self::INNER_52_NODE_ID),
            VerkleNodeKind::Inner53 => VerkleNodeId::from_u64(idx | Self::INNER_53_NODE_ID),
            VerkleNodeKind::Inner54 => VerkleNodeId::from_u64(idx | Self::INNER_54_NODE_ID),
            VerkleNodeKind::Inner55 => VerkleNodeId::from_u64(idx | Self::INNER_55_NODE_ID),
            VerkleNodeKind::Inner56 => VerkleNodeId::from_u64(idx | Self::INNER_56_NODE_ID),
            VerkleNodeKind::Inner57 => VerkleNodeId::from_u64(idx | Self::INNER_57_NODE_ID),
            VerkleNodeKind::Inner58 => VerkleNodeId::from_u64(idx | Self::INNER_58_NODE_ID),
            VerkleNodeKind::Inner59 => VerkleNodeId::from_u64(idx | Self::INNER_59_NODE_ID),
            VerkleNodeKind::Inner60 => VerkleNodeId::from_u64(idx | Self::INNER_60_NODE_ID),
            VerkleNodeKind::Inner61 => VerkleNodeId::from_u64(idx | Self::INNER_61_NODE_ID),
            VerkleNodeKind::Inner62 => VerkleNodeId::from_u64(idx | Self::INNER_62_NODE_ID),
            VerkleNodeKind::Inner63 => VerkleNodeId::from_u64(idx | Self::INNER_63_NODE_ID),
            VerkleNodeKind::Inner64 => VerkleNodeId::from_u64(idx | Self::INNER_64_NODE_ID),
            VerkleNodeKind::Inner65 => VerkleNodeId::from_u64(idx | Self::INNER_65_NODE_ID),
            VerkleNodeKind::Inner66 => VerkleNodeId::from_u64(idx | Self::INNER_66_NODE_ID),
            VerkleNodeKind::Inner67 => VerkleNodeId::from_u64(idx | Self::INNER_67_NODE_ID),
            VerkleNodeKind::Inner68 => VerkleNodeId::from_u64(idx | Self::INNER_68_NODE_ID),
            VerkleNodeKind::Inner69 => VerkleNodeId::from_u64(idx | Self::INNER_69_NODE_ID),
            VerkleNodeKind::Inner70 => VerkleNodeId::from_u64(idx | Self::INNER_70_NODE_ID),
            VerkleNodeKind::Inner71 => VerkleNodeId::from_u64(idx | Self::INNER_71_NODE_ID),
            VerkleNodeKind::Inner72 => VerkleNodeId::from_u64(idx | Self::INNER_72_NODE_ID),
            VerkleNodeKind::Inner73 => VerkleNodeId::from_u64(idx | Self::INNER_73_NODE_ID),
            VerkleNodeKind::Inner74 => VerkleNodeId::from_u64(idx | Self::INNER_74_NODE_ID),
            VerkleNodeKind::Inner75 => VerkleNodeId::from_u64(idx | Self::INNER_75_NODE_ID),
            VerkleNodeKind::Inner76 => VerkleNodeId::from_u64(idx | Self::INNER_76_NODE_ID),
            VerkleNodeKind::Inner77 => VerkleNodeId::from_u64(idx | Self::INNER_77_NODE_ID),
            VerkleNodeKind::Inner78 => VerkleNodeId::from_u64(idx | Self::INNER_78_NODE_ID),
            VerkleNodeKind::Inner79 => VerkleNodeId::from_u64(idx | Self::INNER_79_NODE_ID),
            VerkleNodeKind::Inner80 => VerkleNodeId::from_u64(idx | Self::INNER_80_NODE_ID),
            VerkleNodeKind::Inner81 => VerkleNodeId::from_u64(idx | Self::INNER_81_NODE_ID),
            VerkleNodeKind::Inner82 => VerkleNodeId::from_u64(idx | Self::INNER_82_NODE_ID),
            VerkleNodeKind::Inner83 => VerkleNodeId::from_u64(idx | Self::INNER_83_NODE_ID),
            VerkleNodeKind::Inner84 => VerkleNodeId::from_u64(idx | Self::INNER_84_NODE_ID),
            VerkleNodeKind::Inner85 => VerkleNodeId::from_u64(idx | Self::INNER_85_NODE_ID),
            VerkleNodeKind::Inner86 => VerkleNodeId::from_u64(idx | Self::INNER_86_NODE_ID),
            VerkleNodeKind::Inner87 => VerkleNodeId::from_u64(idx | Self::INNER_87_NODE_ID),
            VerkleNodeKind::Inner88 => VerkleNodeId::from_u64(idx | Self::INNER_88_NODE_ID),
            VerkleNodeKind::Inner89 => VerkleNodeId::from_u64(idx | Self::INNER_89_NODE_ID),
            VerkleNodeKind::Inner90 => VerkleNodeId::from_u64(idx | Self::INNER_90_NODE_ID),
            VerkleNodeKind::Inner91 => VerkleNodeId::from_u64(idx | Self::INNER_91_NODE_ID),
            VerkleNodeKind::Inner92 => VerkleNodeId::from_u64(idx | Self::INNER_92_NODE_ID),
            VerkleNodeKind::Inner93 => VerkleNodeId::from_u64(idx | Self::INNER_93_NODE_ID),
            VerkleNodeKind::Inner94 => VerkleNodeId::from_u64(idx | Self::INNER_94_NODE_ID),
            VerkleNodeKind::Inner95 => VerkleNodeId::from_u64(idx | Self::INNER_95_NODE_ID),
            VerkleNodeKind::Inner96 => VerkleNodeId::from_u64(idx | Self::INNER_96_NODE_ID),
            VerkleNodeKind::Inner97 => VerkleNodeId::from_u64(idx | Self::INNER_97_NODE_ID),
            VerkleNodeKind::Inner98 => VerkleNodeId::from_u64(idx | Self::INNER_98_NODE_ID),
            VerkleNodeKind::Inner99 => VerkleNodeId::from_u64(idx | Self::INNER_99_NODE_ID),
            VerkleNodeKind::Inner100 => VerkleNodeId::from_u64(idx | Self::INNER_100_NODE_ID),
            VerkleNodeKind::Inner101 => VerkleNodeId::from_u64(idx | Self::INNER_101_NODE_ID),
            VerkleNodeKind::Inner102 => VerkleNodeId::from_u64(idx | Self::INNER_102_NODE_ID),
            VerkleNodeKind::Inner103 => VerkleNodeId::from_u64(idx | Self::INNER_103_NODE_ID),
            VerkleNodeKind::Inner104 => VerkleNodeId::from_u64(idx | Self::INNER_104_NODE_ID),
            VerkleNodeKind::Inner105 => VerkleNodeId::from_u64(idx | Self::INNER_105_NODE_ID),
            VerkleNodeKind::Inner106 => VerkleNodeId::from_u64(idx | Self::INNER_106_NODE_ID),
            VerkleNodeKind::Inner107 => VerkleNodeId::from_u64(idx | Self::INNER_107_NODE_ID),
            VerkleNodeKind::Inner108 => VerkleNodeId::from_u64(idx | Self::INNER_108_NODE_ID),
            VerkleNodeKind::Inner109 => VerkleNodeId::from_u64(idx | Self::INNER_109_NODE_ID),
            VerkleNodeKind::Inner110 => VerkleNodeId::from_u64(idx | Self::INNER_110_NODE_ID),
            VerkleNodeKind::Inner111 => VerkleNodeId::from_u64(idx | Self::INNER_111_NODE_ID),
            VerkleNodeKind::Inner112 => VerkleNodeId::from_u64(idx | Self::INNER_112_NODE_ID),
            VerkleNodeKind::Inner113 => VerkleNodeId::from_u64(idx | Self::INNER_113_NODE_ID),
            VerkleNodeKind::Inner114 => VerkleNodeId::from_u64(idx | Self::INNER_114_NODE_ID),
            VerkleNodeKind::Inner115 => VerkleNodeId::from_u64(idx | Self::INNER_115_NODE_ID),
            VerkleNodeKind::Inner116 => VerkleNodeId::from_u64(idx | Self::INNER_116_NODE_ID),
            VerkleNodeKind::Inner117 => VerkleNodeId::from_u64(idx | Self::INNER_117_NODE_ID),
            VerkleNodeKind::Inner118 => VerkleNodeId::from_u64(idx | Self::INNER_118_NODE_ID),
            VerkleNodeKind::Inner119 => VerkleNodeId::from_u64(idx | Self::INNER_119_NODE_ID),
            VerkleNodeKind::Inner120 => VerkleNodeId::from_u64(idx | Self::INNER_120_NODE_ID),
            VerkleNodeKind::Inner121 => VerkleNodeId::from_u64(idx | Self::INNER_121_NODE_ID),
            VerkleNodeKind::Inner122 => VerkleNodeId::from_u64(idx | Self::INNER_122_NODE_ID),
            VerkleNodeKind::Inner123 => VerkleNodeId::from_u64(idx | Self::INNER_123_NODE_ID),
            VerkleNodeKind::Inner124 => VerkleNodeId::from_u64(idx | Self::INNER_124_NODE_ID),
            VerkleNodeKind::Inner125 => VerkleNodeId::from_u64(idx | Self::INNER_125_NODE_ID),
            VerkleNodeKind::Inner126 => VerkleNodeId::from_u64(idx | Self::INNER_126_NODE_ID),
            VerkleNodeKind::Inner127 => VerkleNodeId::from_u64(idx | Self::INNER_127_NODE_ID),
            VerkleNodeKind::Inner128 => VerkleNodeId::from_u64(idx | Self::INNER_128_NODE_ID),
            VerkleNodeKind::Inner129 => VerkleNodeId::from_u64(idx | Self::INNER_129_NODE_ID),
            VerkleNodeKind::Inner130 => VerkleNodeId::from_u64(idx | Self::INNER_130_NODE_ID),
            VerkleNodeKind::Inner131 => VerkleNodeId::from_u64(idx | Self::INNER_131_NODE_ID),
            VerkleNodeKind::Inner132 => VerkleNodeId::from_u64(idx | Self::INNER_132_NODE_ID),
            VerkleNodeKind::Inner133 => VerkleNodeId::from_u64(idx | Self::INNER_133_NODE_ID),
            VerkleNodeKind::Inner134 => VerkleNodeId::from_u64(idx | Self::INNER_134_NODE_ID),
            VerkleNodeKind::Inner135 => VerkleNodeId::from_u64(idx | Self::INNER_135_NODE_ID),
            VerkleNodeKind::Inner136 => VerkleNodeId::from_u64(idx | Self::INNER_136_NODE_ID),
            VerkleNodeKind::Inner137 => VerkleNodeId::from_u64(idx | Self::INNER_137_NODE_ID),
            VerkleNodeKind::Inner138 => VerkleNodeId::from_u64(idx | Self::INNER_138_NODE_ID),
            VerkleNodeKind::Inner139 => VerkleNodeId::from_u64(idx | Self::INNER_139_NODE_ID),
            VerkleNodeKind::Inner140 => VerkleNodeId::from_u64(idx | Self::INNER_140_NODE_ID),
            VerkleNodeKind::Inner141 => VerkleNodeId::from_u64(idx | Self::INNER_141_NODE_ID),
            VerkleNodeKind::Inner142 => VerkleNodeId::from_u64(idx | Self::INNER_142_NODE_ID),
            VerkleNodeKind::Inner143 => VerkleNodeId::from_u64(idx | Self::INNER_143_NODE_ID),
            VerkleNodeKind::Inner144 => VerkleNodeId::from_u64(idx | Self::INNER_144_NODE_ID),
            VerkleNodeKind::Inner145 => VerkleNodeId::from_u64(idx | Self::INNER_145_NODE_ID),
            VerkleNodeKind::Inner146 => VerkleNodeId::from_u64(idx | Self::INNER_146_NODE_ID),
            VerkleNodeKind::Inner147 => VerkleNodeId::from_u64(idx | Self::INNER_147_NODE_ID),
            VerkleNodeKind::Inner148 => VerkleNodeId::from_u64(idx | Self::INNER_148_NODE_ID),
            VerkleNodeKind::Inner149 => VerkleNodeId::from_u64(idx | Self::INNER_149_NODE_ID),
            VerkleNodeKind::Inner150 => VerkleNodeId::from_u64(idx | Self::INNER_150_NODE_ID),
            VerkleNodeKind::Inner151 => VerkleNodeId::from_u64(idx | Self::INNER_151_NODE_ID),
            VerkleNodeKind::Inner152 => VerkleNodeId::from_u64(idx | Self::INNER_152_NODE_ID),
            VerkleNodeKind::Inner153 => VerkleNodeId::from_u64(idx | Self::INNER_153_NODE_ID),
            VerkleNodeKind::Inner154 => VerkleNodeId::from_u64(idx | Self::INNER_154_NODE_ID),
            VerkleNodeKind::Inner155 => VerkleNodeId::from_u64(idx | Self::INNER_155_NODE_ID),
            VerkleNodeKind::Inner156 => VerkleNodeId::from_u64(idx | Self::INNER_156_NODE_ID),
            VerkleNodeKind::Inner157 => VerkleNodeId::from_u64(idx | Self::INNER_157_NODE_ID),
            VerkleNodeKind::Inner158 => VerkleNodeId::from_u64(idx | Self::INNER_158_NODE_ID),
            VerkleNodeKind::Inner159 => VerkleNodeId::from_u64(idx | Self::INNER_159_NODE_ID),
            VerkleNodeKind::Inner160 => VerkleNodeId::from_u64(idx | Self::INNER_160_NODE_ID),
            VerkleNodeKind::Inner161 => VerkleNodeId::from_u64(idx | Self::INNER_161_NODE_ID),
            VerkleNodeKind::Inner162 => VerkleNodeId::from_u64(idx | Self::INNER_162_NODE_ID),
            VerkleNodeKind::Inner163 => VerkleNodeId::from_u64(idx | Self::INNER_163_NODE_ID),
            VerkleNodeKind::Inner164 => VerkleNodeId::from_u64(idx | Self::INNER_164_NODE_ID),
            VerkleNodeKind::Inner165 => VerkleNodeId::from_u64(idx | Self::INNER_165_NODE_ID),
            VerkleNodeKind::Inner166 => VerkleNodeId::from_u64(idx | Self::INNER_166_NODE_ID),
            VerkleNodeKind::Inner167 => VerkleNodeId::from_u64(idx | Self::INNER_167_NODE_ID),
            VerkleNodeKind::Inner168 => VerkleNodeId::from_u64(idx | Self::INNER_168_NODE_ID),
            VerkleNodeKind::Inner169 => VerkleNodeId::from_u64(idx | Self::INNER_169_NODE_ID),
            VerkleNodeKind::Inner170 => VerkleNodeId::from_u64(idx | Self::INNER_170_NODE_ID),
            VerkleNodeKind::Inner171 => VerkleNodeId::from_u64(idx | Self::INNER_171_NODE_ID),
            VerkleNodeKind::Inner172 => VerkleNodeId::from_u64(idx | Self::INNER_172_NODE_ID),
            VerkleNodeKind::Inner173 => VerkleNodeId::from_u64(idx | Self::INNER_173_NODE_ID),
            VerkleNodeKind::Inner174 => VerkleNodeId::from_u64(idx | Self::INNER_174_NODE_ID),
            VerkleNodeKind::Inner175 => VerkleNodeId::from_u64(idx | Self::INNER_175_NODE_ID),
            VerkleNodeKind::Inner176 => VerkleNodeId::from_u64(idx | Self::INNER_176_NODE_ID),
            VerkleNodeKind::Inner177 => VerkleNodeId::from_u64(idx | Self::INNER_177_NODE_ID),
            VerkleNodeKind::Inner178 => VerkleNodeId::from_u64(idx | Self::INNER_178_NODE_ID),
            VerkleNodeKind::Inner179 => VerkleNodeId::from_u64(idx | Self::INNER_179_NODE_ID),
            VerkleNodeKind::Inner180 => VerkleNodeId::from_u64(idx | Self::INNER_180_NODE_ID),
            VerkleNodeKind::Inner181 => VerkleNodeId::from_u64(idx | Self::INNER_181_NODE_ID),
            VerkleNodeKind::Inner182 => VerkleNodeId::from_u64(idx | Self::INNER_182_NODE_ID),
            VerkleNodeKind::Inner183 => VerkleNodeId::from_u64(idx | Self::INNER_183_NODE_ID),
            VerkleNodeKind::Inner184 => VerkleNodeId::from_u64(idx | Self::INNER_184_NODE_ID),
            VerkleNodeKind::Inner185 => VerkleNodeId::from_u64(idx | Self::INNER_185_NODE_ID),
            VerkleNodeKind::Inner186 => VerkleNodeId::from_u64(idx | Self::INNER_186_NODE_ID),
            VerkleNodeKind::Inner187 => VerkleNodeId::from_u64(idx | Self::INNER_187_NODE_ID),
            VerkleNodeKind::Inner188 => VerkleNodeId::from_u64(idx | Self::INNER_188_NODE_ID),
            VerkleNodeKind::Inner189 => VerkleNodeId::from_u64(idx | Self::INNER_189_NODE_ID),
            VerkleNodeKind::Inner190 => VerkleNodeId::from_u64(idx | Self::INNER_190_NODE_ID),
            VerkleNodeKind::Inner191 => VerkleNodeId::from_u64(idx | Self::INNER_191_NODE_ID),
            VerkleNodeKind::Inner192 => VerkleNodeId::from_u64(idx | Self::INNER_192_NODE_ID),
            VerkleNodeKind::Inner193 => VerkleNodeId::from_u64(idx | Self::INNER_193_NODE_ID),
            VerkleNodeKind::Inner194 => VerkleNodeId::from_u64(idx | Self::INNER_194_NODE_ID),
            VerkleNodeKind::Inner195 => VerkleNodeId::from_u64(idx | Self::INNER_195_NODE_ID),
            VerkleNodeKind::Inner196 => VerkleNodeId::from_u64(idx | Self::INNER_196_NODE_ID),
            VerkleNodeKind::Inner197 => VerkleNodeId::from_u64(idx | Self::INNER_197_NODE_ID),
            VerkleNodeKind::Inner198 => VerkleNodeId::from_u64(idx | Self::INNER_198_NODE_ID),
            VerkleNodeKind::Inner199 => VerkleNodeId::from_u64(idx | Self::INNER_199_NODE_ID),
            VerkleNodeKind::Inner200 => VerkleNodeId::from_u64(idx | Self::INNER_200_NODE_ID),
            VerkleNodeKind::Inner201 => VerkleNodeId::from_u64(idx | Self::INNER_201_NODE_ID),
            VerkleNodeKind::Inner202 => VerkleNodeId::from_u64(idx | Self::INNER_202_NODE_ID),
            VerkleNodeKind::Inner203 => VerkleNodeId::from_u64(idx | Self::INNER_203_NODE_ID),
            VerkleNodeKind::Inner204 => VerkleNodeId::from_u64(idx | Self::INNER_204_NODE_ID),
            VerkleNodeKind::Inner205 => VerkleNodeId::from_u64(idx | Self::INNER_205_NODE_ID),
            VerkleNodeKind::Inner206 => VerkleNodeId::from_u64(idx | Self::INNER_206_NODE_ID),
            VerkleNodeKind::Inner207 => VerkleNodeId::from_u64(idx | Self::INNER_207_NODE_ID),
            VerkleNodeKind::Inner208 => VerkleNodeId::from_u64(idx | Self::INNER_208_NODE_ID),
            VerkleNodeKind::Inner209 => VerkleNodeId::from_u64(idx | Self::INNER_209_NODE_ID),
            VerkleNodeKind::Inner210 => VerkleNodeId::from_u64(idx | Self::INNER_210_NODE_ID),
            VerkleNodeKind::Inner211 => VerkleNodeId::from_u64(idx | Self::INNER_211_NODE_ID),
            VerkleNodeKind::Inner212 => VerkleNodeId::from_u64(idx | Self::INNER_212_NODE_ID),
            VerkleNodeKind::Inner213 => VerkleNodeId::from_u64(idx | Self::INNER_213_NODE_ID),
            VerkleNodeKind::Inner214 => VerkleNodeId::from_u64(idx | Self::INNER_214_NODE_ID),
            VerkleNodeKind::Inner215 => VerkleNodeId::from_u64(idx | Self::INNER_215_NODE_ID),
            VerkleNodeKind::Inner216 => VerkleNodeId::from_u64(idx | Self::INNER_216_NODE_ID),
            VerkleNodeKind::Inner217 => VerkleNodeId::from_u64(idx | Self::INNER_217_NODE_ID),
            VerkleNodeKind::Inner218 => VerkleNodeId::from_u64(idx | Self::INNER_218_NODE_ID),
            VerkleNodeKind::Inner219 => VerkleNodeId::from_u64(idx | Self::INNER_219_NODE_ID),
            VerkleNodeKind::Inner220 => VerkleNodeId::from_u64(idx | Self::INNER_220_NODE_ID),
            VerkleNodeKind::Inner221 => VerkleNodeId::from_u64(idx | Self::INNER_221_NODE_ID),
            VerkleNodeKind::Inner222 => VerkleNodeId::from_u64(idx | Self::INNER_222_NODE_ID),
            VerkleNodeKind::Inner223 => VerkleNodeId::from_u64(idx | Self::INNER_223_NODE_ID),
            VerkleNodeKind::Inner224 => VerkleNodeId::from_u64(idx | Self::INNER_224_NODE_ID),
            VerkleNodeKind::Inner225 => VerkleNodeId::from_u64(idx | Self::INNER_225_NODE_ID),
            VerkleNodeKind::Inner226 => VerkleNodeId::from_u64(idx | Self::INNER_226_NODE_ID),
            VerkleNodeKind::Inner227 => VerkleNodeId::from_u64(idx | Self::INNER_227_NODE_ID),
            VerkleNodeKind::Inner228 => VerkleNodeId::from_u64(idx | Self::INNER_228_NODE_ID),
            VerkleNodeKind::Inner229 => VerkleNodeId::from_u64(idx | Self::INNER_229_NODE_ID),
            VerkleNodeKind::Inner230 => VerkleNodeId::from_u64(idx | Self::INNER_230_NODE_ID),
            VerkleNodeKind::Inner231 => VerkleNodeId::from_u64(idx | Self::INNER_231_NODE_ID),
            VerkleNodeKind::Inner232 => VerkleNodeId::from_u64(idx | Self::INNER_232_NODE_ID),
            VerkleNodeKind::Inner233 => VerkleNodeId::from_u64(idx | Self::INNER_233_NODE_ID),
            VerkleNodeKind::Inner234 => VerkleNodeId::from_u64(idx | Self::INNER_234_NODE_ID),
            VerkleNodeKind::Inner235 => VerkleNodeId::from_u64(idx | Self::INNER_235_NODE_ID),
            VerkleNodeKind::Inner236 => VerkleNodeId::from_u64(idx | Self::INNER_236_NODE_ID),
            VerkleNodeKind::Inner237 => VerkleNodeId::from_u64(idx | Self::INNER_237_NODE_ID),
            VerkleNodeKind::Inner238 => VerkleNodeId::from_u64(idx | Self::INNER_238_NODE_ID),
            VerkleNodeKind::Inner239 => VerkleNodeId::from_u64(idx | Self::INNER_239_NODE_ID),
            VerkleNodeKind::Inner240 => VerkleNodeId::from_u64(idx | Self::INNER_240_NODE_ID),
            VerkleNodeKind::Inner241 => VerkleNodeId::from_u64(idx | Self::INNER_241_NODE_ID),
            VerkleNodeKind::Inner242 => VerkleNodeId::from_u64(idx | Self::INNER_242_NODE_ID),
            VerkleNodeKind::Inner243 => VerkleNodeId::from_u64(idx | Self::INNER_243_NODE_ID),
            VerkleNodeKind::Inner244 => VerkleNodeId::from_u64(idx | Self::INNER_244_NODE_ID),
            VerkleNodeKind::Inner245 => VerkleNodeId::from_u64(idx | Self::INNER_245_NODE_ID),
            VerkleNodeKind::Inner246 => VerkleNodeId::from_u64(idx | Self::INNER_246_NODE_ID),
            VerkleNodeKind::Inner247 => VerkleNodeId::from_u64(idx | Self::INNER_247_NODE_ID),
            VerkleNodeKind::Inner248 => VerkleNodeId::from_u64(idx | Self::INNER_248_NODE_ID),
            VerkleNodeKind::Inner249 => VerkleNodeId::from_u64(idx | Self::INNER_249_NODE_ID),
            VerkleNodeKind::Inner250 => VerkleNodeId::from_u64(idx | Self::INNER_250_NODE_ID),
            VerkleNodeKind::Inner251 => VerkleNodeId::from_u64(idx | Self::INNER_251_NODE_ID),
            VerkleNodeKind::Inner252 => VerkleNodeId::from_u64(idx | Self::INNER_252_NODE_ID),
            VerkleNodeKind::Inner253 => VerkleNodeId::from_u64(idx | Self::INNER_253_NODE_ID),
            VerkleNodeKind::Inner254 => VerkleNodeId::from_u64(idx | Self::INNER_254_NODE_ID),
            VerkleNodeKind::Inner255 => VerkleNodeId::from_u64(idx | Self::INNER_255_NODE_ID),
            VerkleNodeKind::Inner256 => VerkleNodeId::from_u64(idx | Self::INNER_256_NODE_ID),
            VerkleNodeKind::Leaf1 => VerkleNodeId::from_u64(idx | Self::LEAF_1_NODE_ID),
            VerkleNodeKind::Leaf2 => VerkleNodeId::from_u64(idx | Self::LEAF_2_NODE_ID),
            VerkleNodeKind::Leaf3 => VerkleNodeId::from_u64(idx | Self::LEAF_3_NODE_ID),
            VerkleNodeKind::Leaf4 => VerkleNodeId::from_u64(idx | Self::LEAF_4_NODE_ID),
            VerkleNodeKind::Leaf5 => VerkleNodeId::from_u64(idx | Self::LEAF_5_NODE_ID),
            VerkleNodeKind::Leaf6 => VerkleNodeId::from_u64(idx | Self::LEAF_6_NODE_ID),
            VerkleNodeKind::Leaf7 => VerkleNodeId::from_u64(idx | Self::LEAF_7_NODE_ID),
            VerkleNodeKind::Leaf8 => VerkleNodeId::from_u64(idx | Self::LEAF_8_NODE_ID),
            VerkleNodeKind::Leaf9 => VerkleNodeId::from_u64(idx | Self::LEAF_9_NODE_ID),
            VerkleNodeKind::Leaf10 => VerkleNodeId::from_u64(idx | Self::LEAF_10_NODE_ID),
            VerkleNodeKind::Leaf11 => VerkleNodeId::from_u64(idx | Self::LEAF_11_NODE_ID),
            VerkleNodeKind::Leaf12 => VerkleNodeId::from_u64(idx | Self::LEAF_12_NODE_ID),
            VerkleNodeKind::Leaf13 => VerkleNodeId::from_u64(idx | Self::LEAF_13_NODE_ID),
            VerkleNodeKind::Leaf14 => VerkleNodeId::from_u64(idx | Self::LEAF_14_NODE_ID),
            VerkleNodeKind::Leaf15 => VerkleNodeId::from_u64(idx | Self::LEAF_15_NODE_ID),
            VerkleNodeKind::Leaf16 => VerkleNodeId::from_u64(idx | Self::LEAF_16_NODE_ID),
            VerkleNodeKind::Leaf17 => VerkleNodeId::from_u64(idx | Self::LEAF_17_NODE_ID),
            VerkleNodeKind::Leaf18 => VerkleNodeId::from_u64(idx | Self::LEAF_18_NODE_ID),
            VerkleNodeKind::Leaf19 => VerkleNodeId::from_u64(idx | Self::LEAF_19_NODE_ID),
            VerkleNodeKind::Leaf20 => VerkleNodeId::from_u64(idx | Self::LEAF_20_NODE_ID),
            VerkleNodeKind::Leaf21 => VerkleNodeId::from_u64(idx | Self::LEAF_21_NODE_ID),
            VerkleNodeKind::Leaf22 => VerkleNodeId::from_u64(idx | Self::LEAF_22_NODE_ID),
            VerkleNodeKind::Leaf23 => VerkleNodeId::from_u64(idx | Self::LEAF_23_NODE_ID),
            VerkleNodeKind::Leaf24 => VerkleNodeId::from_u64(idx | Self::LEAF_24_NODE_ID),
            VerkleNodeKind::Leaf25 => VerkleNodeId::from_u64(idx | Self::LEAF_25_NODE_ID),
            VerkleNodeKind::Leaf26 => VerkleNodeId::from_u64(idx | Self::LEAF_26_NODE_ID),
            VerkleNodeKind::Leaf27 => VerkleNodeId::from_u64(idx | Self::LEAF_27_NODE_ID),
            VerkleNodeKind::Leaf28 => VerkleNodeId::from_u64(idx | Self::LEAF_28_NODE_ID),
            VerkleNodeKind::Leaf29 => VerkleNodeId::from_u64(idx | Self::LEAF_29_NODE_ID),
            VerkleNodeKind::Leaf30 => VerkleNodeId::from_u64(idx | Self::LEAF_30_NODE_ID),
            VerkleNodeKind::Leaf31 => VerkleNodeId::from_u64(idx | Self::LEAF_31_NODE_ID),
            VerkleNodeKind::Leaf32 => VerkleNodeId::from_u64(idx | Self::LEAF_32_NODE_ID),
            VerkleNodeKind::Leaf33 => VerkleNodeId::from_u64(idx | Self::LEAF_33_NODE_ID),
            VerkleNodeKind::Leaf34 => VerkleNodeId::from_u64(idx | Self::LEAF_34_NODE_ID),
            VerkleNodeKind::Leaf35 => VerkleNodeId::from_u64(idx | Self::LEAF_35_NODE_ID),
            VerkleNodeKind::Leaf36 => VerkleNodeId::from_u64(idx | Self::LEAF_36_NODE_ID),
            VerkleNodeKind::Leaf37 => VerkleNodeId::from_u64(idx | Self::LEAF_37_NODE_ID),
            VerkleNodeKind::Leaf38 => VerkleNodeId::from_u64(idx | Self::LEAF_38_NODE_ID),
            VerkleNodeKind::Leaf39 => VerkleNodeId::from_u64(idx | Self::LEAF_39_NODE_ID),
            VerkleNodeKind::Leaf40 => VerkleNodeId::from_u64(idx | Self::LEAF_40_NODE_ID),
            VerkleNodeKind::Leaf41 => VerkleNodeId::from_u64(idx | Self::LEAF_41_NODE_ID),
            VerkleNodeKind::Leaf42 => VerkleNodeId::from_u64(idx | Self::LEAF_42_NODE_ID),
            VerkleNodeKind::Leaf43 => VerkleNodeId::from_u64(idx | Self::LEAF_43_NODE_ID),
            VerkleNodeKind::Leaf44 => VerkleNodeId::from_u64(idx | Self::LEAF_44_NODE_ID),
            VerkleNodeKind::Leaf45 => VerkleNodeId::from_u64(idx | Self::LEAF_45_NODE_ID),
            VerkleNodeKind::Leaf46 => VerkleNodeId::from_u64(idx | Self::LEAF_46_NODE_ID),
            VerkleNodeKind::Leaf47 => VerkleNodeId::from_u64(idx | Self::LEAF_47_NODE_ID),
            VerkleNodeKind::Leaf48 => VerkleNodeId::from_u64(idx | Self::LEAF_48_NODE_ID),
            VerkleNodeKind::Leaf49 => VerkleNodeId::from_u64(idx | Self::LEAF_49_NODE_ID),
            VerkleNodeKind::Leaf50 => VerkleNodeId::from_u64(idx | Self::LEAF_50_NODE_ID),
            VerkleNodeKind::Leaf51 => VerkleNodeId::from_u64(idx | Self::LEAF_51_NODE_ID),
            VerkleNodeKind::Leaf52 => VerkleNodeId::from_u64(idx | Self::LEAF_52_NODE_ID),
            VerkleNodeKind::Leaf53 => VerkleNodeId::from_u64(idx | Self::LEAF_53_NODE_ID),
            VerkleNodeKind::Leaf54 => VerkleNodeId::from_u64(idx | Self::LEAF_54_NODE_ID),
            VerkleNodeKind::Leaf55 => VerkleNodeId::from_u64(idx | Self::LEAF_55_NODE_ID),
            VerkleNodeKind::Leaf56 => VerkleNodeId::from_u64(idx | Self::LEAF_56_NODE_ID),
            VerkleNodeKind::Leaf57 => VerkleNodeId::from_u64(idx | Self::LEAF_57_NODE_ID),
            VerkleNodeKind::Leaf58 => VerkleNodeId::from_u64(idx | Self::LEAF_58_NODE_ID),
            VerkleNodeKind::Leaf59 => VerkleNodeId::from_u64(idx | Self::LEAF_59_NODE_ID),
            VerkleNodeKind::Leaf60 => VerkleNodeId::from_u64(idx | Self::LEAF_60_NODE_ID),
            VerkleNodeKind::Leaf61 => VerkleNodeId::from_u64(idx | Self::LEAF_61_NODE_ID),
            VerkleNodeKind::Leaf62 => VerkleNodeId::from_u64(idx | Self::LEAF_62_NODE_ID),
            VerkleNodeKind::Leaf63 => VerkleNodeId::from_u64(idx | Self::LEAF_63_NODE_ID),
            VerkleNodeKind::Leaf64 => VerkleNodeId::from_u64(idx | Self::LEAF_64_NODE_ID),
            VerkleNodeKind::Leaf65 => VerkleNodeId::from_u64(idx | Self::LEAF_65_NODE_ID),
            VerkleNodeKind::Leaf66 => VerkleNodeId::from_u64(idx | Self::LEAF_66_NODE_ID),
            VerkleNodeKind::Leaf67 => VerkleNodeId::from_u64(idx | Self::LEAF_67_NODE_ID),
            VerkleNodeKind::Leaf68 => VerkleNodeId::from_u64(idx | Self::LEAF_68_NODE_ID),
            VerkleNodeKind::Leaf69 => VerkleNodeId::from_u64(idx | Self::LEAF_69_NODE_ID),
            VerkleNodeKind::Leaf70 => VerkleNodeId::from_u64(idx | Self::LEAF_70_NODE_ID),
            VerkleNodeKind::Leaf71 => VerkleNodeId::from_u64(idx | Self::LEAF_71_NODE_ID),
            VerkleNodeKind::Leaf72 => VerkleNodeId::from_u64(idx | Self::LEAF_72_NODE_ID),
            VerkleNodeKind::Leaf73 => VerkleNodeId::from_u64(idx | Self::LEAF_73_NODE_ID),
            VerkleNodeKind::Leaf74 => VerkleNodeId::from_u64(idx | Self::LEAF_74_NODE_ID),
            VerkleNodeKind::Leaf75 => VerkleNodeId::from_u64(idx | Self::LEAF_75_NODE_ID),
            VerkleNodeKind::Leaf76 => VerkleNodeId::from_u64(idx | Self::LEAF_76_NODE_ID),
            VerkleNodeKind::Leaf77 => VerkleNodeId::from_u64(idx | Self::LEAF_77_NODE_ID),
            VerkleNodeKind::Leaf78 => VerkleNodeId::from_u64(idx | Self::LEAF_78_NODE_ID),
            VerkleNodeKind::Leaf79 => VerkleNodeId::from_u64(idx | Self::LEAF_79_NODE_ID),
            VerkleNodeKind::Leaf80 => VerkleNodeId::from_u64(idx | Self::LEAF_80_NODE_ID),
            VerkleNodeKind::Leaf81 => VerkleNodeId::from_u64(idx | Self::LEAF_81_NODE_ID),
            VerkleNodeKind::Leaf82 => VerkleNodeId::from_u64(idx | Self::LEAF_82_NODE_ID),
            VerkleNodeKind::Leaf83 => VerkleNodeId::from_u64(idx | Self::LEAF_83_NODE_ID),
            VerkleNodeKind::Leaf84 => VerkleNodeId::from_u64(idx | Self::LEAF_84_NODE_ID),
            VerkleNodeKind::Leaf85 => VerkleNodeId::from_u64(idx | Self::LEAF_85_NODE_ID),
            VerkleNodeKind::Leaf86 => VerkleNodeId::from_u64(idx | Self::LEAF_86_NODE_ID),
            VerkleNodeKind::Leaf87 => VerkleNodeId::from_u64(idx | Self::LEAF_87_NODE_ID),
            VerkleNodeKind::Leaf88 => VerkleNodeId::from_u64(idx | Self::LEAF_88_NODE_ID),
            VerkleNodeKind::Leaf89 => VerkleNodeId::from_u64(idx | Self::LEAF_89_NODE_ID),
            VerkleNodeKind::Leaf90 => VerkleNodeId::from_u64(idx | Self::LEAF_90_NODE_ID),
            VerkleNodeKind::Leaf91 => VerkleNodeId::from_u64(idx | Self::LEAF_91_NODE_ID),
            VerkleNodeKind::Leaf92 => VerkleNodeId::from_u64(idx | Self::LEAF_92_NODE_ID),
            VerkleNodeKind::Leaf93 => VerkleNodeId::from_u64(idx | Self::LEAF_93_NODE_ID),
            VerkleNodeKind::Leaf94 => VerkleNodeId::from_u64(idx | Self::LEAF_94_NODE_ID),
            VerkleNodeKind::Leaf95 => VerkleNodeId::from_u64(idx | Self::LEAF_95_NODE_ID),
            VerkleNodeKind::Leaf96 => VerkleNodeId::from_u64(idx | Self::LEAF_96_NODE_ID),
            VerkleNodeKind::Leaf97 => VerkleNodeId::from_u64(idx | Self::LEAF_97_NODE_ID),
            VerkleNodeKind::Leaf98 => VerkleNodeId::from_u64(idx | Self::LEAF_98_NODE_ID),
            VerkleNodeKind::Leaf99 => VerkleNodeId::from_u64(idx | Self::LEAF_99_NODE_ID),
            VerkleNodeKind::Leaf100 => VerkleNodeId::from_u64(idx | Self::LEAF_100_NODE_ID),
            VerkleNodeKind::Leaf101 => VerkleNodeId::from_u64(idx | Self::LEAF_101_NODE_ID),
            VerkleNodeKind::Leaf102 => VerkleNodeId::from_u64(idx | Self::LEAF_102_NODE_ID),
            VerkleNodeKind::Leaf103 => VerkleNodeId::from_u64(idx | Self::LEAF_103_NODE_ID),
            VerkleNodeKind::Leaf104 => VerkleNodeId::from_u64(idx | Self::LEAF_104_NODE_ID),
            VerkleNodeKind::Leaf105 => VerkleNodeId::from_u64(idx | Self::LEAF_105_NODE_ID),
            VerkleNodeKind::Leaf106 => VerkleNodeId::from_u64(idx | Self::LEAF_106_NODE_ID),
            VerkleNodeKind::Leaf107 => VerkleNodeId::from_u64(idx | Self::LEAF_107_NODE_ID),
            VerkleNodeKind::Leaf108 => VerkleNodeId::from_u64(idx | Self::LEAF_108_NODE_ID),
            VerkleNodeKind::Leaf109 => VerkleNodeId::from_u64(idx | Self::LEAF_109_NODE_ID),
            VerkleNodeKind::Leaf110 => VerkleNodeId::from_u64(idx | Self::LEAF_110_NODE_ID),
            VerkleNodeKind::Leaf111 => VerkleNodeId::from_u64(idx | Self::LEAF_111_NODE_ID),
            VerkleNodeKind::Leaf112 => VerkleNodeId::from_u64(idx | Self::LEAF_112_NODE_ID),
            VerkleNodeKind::Leaf113 => VerkleNodeId::from_u64(idx | Self::LEAF_113_NODE_ID),
            VerkleNodeKind::Leaf114 => VerkleNodeId::from_u64(idx | Self::LEAF_114_NODE_ID),
            VerkleNodeKind::Leaf115 => VerkleNodeId::from_u64(idx | Self::LEAF_115_NODE_ID),
            VerkleNodeKind::Leaf116 => VerkleNodeId::from_u64(idx | Self::LEAF_116_NODE_ID),
            VerkleNodeKind::Leaf117 => VerkleNodeId::from_u64(idx | Self::LEAF_117_NODE_ID),
            VerkleNodeKind::Leaf118 => VerkleNodeId::from_u64(idx | Self::LEAF_118_NODE_ID),
            VerkleNodeKind::Leaf119 => VerkleNodeId::from_u64(idx | Self::LEAF_119_NODE_ID),
            VerkleNodeKind::Leaf120 => VerkleNodeId::from_u64(idx | Self::LEAF_120_NODE_ID),
            VerkleNodeKind::Leaf121 => VerkleNodeId::from_u64(idx | Self::LEAF_121_NODE_ID),
            VerkleNodeKind::Leaf122 => VerkleNodeId::from_u64(idx | Self::LEAF_122_NODE_ID),
            VerkleNodeKind::Leaf123 => VerkleNodeId::from_u64(idx | Self::LEAF_123_NODE_ID),
            VerkleNodeKind::Leaf124 => VerkleNodeId::from_u64(idx | Self::LEAF_124_NODE_ID),
            VerkleNodeKind::Leaf125 => VerkleNodeId::from_u64(idx | Self::LEAF_125_NODE_ID),
            VerkleNodeKind::Leaf126 => VerkleNodeId::from_u64(idx | Self::LEAF_126_NODE_ID),
            VerkleNodeKind::Leaf127 => VerkleNodeId::from_u64(idx | Self::LEAF_127_NODE_ID),
            VerkleNodeKind::Leaf128 => VerkleNodeId::from_u64(idx | Self::LEAF_128_NODE_ID),
            VerkleNodeKind::Leaf129 => VerkleNodeId::from_u64(idx | Self::LEAF_129_NODE_ID),
            VerkleNodeKind::Leaf130 => VerkleNodeId::from_u64(idx | Self::LEAF_130_NODE_ID),
            VerkleNodeKind::Leaf131 => VerkleNodeId::from_u64(idx | Self::LEAF_131_NODE_ID),
            VerkleNodeKind::Leaf132 => VerkleNodeId::from_u64(idx | Self::LEAF_132_NODE_ID),
            VerkleNodeKind::Leaf133 => VerkleNodeId::from_u64(idx | Self::LEAF_133_NODE_ID),
            VerkleNodeKind::Leaf134 => VerkleNodeId::from_u64(idx | Self::LEAF_134_NODE_ID),
            VerkleNodeKind::Leaf135 => VerkleNodeId::from_u64(idx | Self::LEAF_135_NODE_ID),
            VerkleNodeKind::Leaf136 => VerkleNodeId::from_u64(idx | Self::LEAF_136_NODE_ID),
            VerkleNodeKind::Leaf137 => VerkleNodeId::from_u64(idx | Self::LEAF_137_NODE_ID),
            VerkleNodeKind::Leaf138 => VerkleNodeId::from_u64(idx | Self::LEAF_138_NODE_ID),
            VerkleNodeKind::Leaf139 => VerkleNodeId::from_u64(idx | Self::LEAF_139_NODE_ID),
            VerkleNodeKind::Leaf140 => VerkleNodeId::from_u64(idx | Self::LEAF_140_NODE_ID),
            VerkleNodeKind::Leaf141 => VerkleNodeId::from_u64(idx | Self::LEAF_141_NODE_ID),
            VerkleNodeKind::Leaf142 => VerkleNodeId::from_u64(idx | Self::LEAF_142_NODE_ID),
            VerkleNodeKind::Leaf143 => VerkleNodeId::from_u64(idx | Self::LEAF_143_NODE_ID),
            VerkleNodeKind::Leaf144 => VerkleNodeId::from_u64(idx | Self::LEAF_144_NODE_ID),
            VerkleNodeKind::Leaf145 => VerkleNodeId::from_u64(idx | Self::LEAF_145_NODE_ID),
            VerkleNodeKind::Leaf146 => VerkleNodeId::from_u64(idx | Self::LEAF_146_NODE_ID),
            VerkleNodeKind::Leaf147 => VerkleNodeId::from_u64(idx | Self::LEAF_147_NODE_ID),
            VerkleNodeKind::Leaf148 => VerkleNodeId::from_u64(idx | Self::LEAF_148_NODE_ID),
            VerkleNodeKind::Leaf149 => VerkleNodeId::from_u64(idx | Self::LEAF_149_NODE_ID),
            VerkleNodeKind::Leaf150 => VerkleNodeId::from_u64(idx | Self::LEAF_150_NODE_ID),
            VerkleNodeKind::Leaf151 => VerkleNodeId::from_u64(idx | Self::LEAF_151_NODE_ID),
            VerkleNodeKind::Leaf152 => VerkleNodeId::from_u64(idx | Self::LEAF_152_NODE_ID),
            VerkleNodeKind::Leaf153 => VerkleNodeId::from_u64(idx | Self::LEAF_153_NODE_ID),
            VerkleNodeKind::Leaf154 => VerkleNodeId::from_u64(idx | Self::LEAF_154_NODE_ID),
            VerkleNodeKind::Leaf155 => VerkleNodeId::from_u64(idx | Self::LEAF_155_NODE_ID),
            VerkleNodeKind::Leaf156 => VerkleNodeId::from_u64(idx | Self::LEAF_156_NODE_ID),
            VerkleNodeKind::Leaf157 => VerkleNodeId::from_u64(idx | Self::LEAF_157_NODE_ID),
            VerkleNodeKind::Leaf158 => VerkleNodeId::from_u64(idx | Self::LEAF_158_NODE_ID),
            VerkleNodeKind::Leaf159 => VerkleNodeId::from_u64(idx | Self::LEAF_159_NODE_ID),
            VerkleNodeKind::Leaf160 => VerkleNodeId::from_u64(idx | Self::LEAF_160_NODE_ID),
            VerkleNodeKind::Leaf161 => VerkleNodeId::from_u64(idx | Self::LEAF_161_NODE_ID),
            VerkleNodeKind::Leaf162 => VerkleNodeId::from_u64(idx | Self::LEAF_162_NODE_ID),
            VerkleNodeKind::Leaf163 => VerkleNodeId::from_u64(idx | Self::LEAF_163_NODE_ID),
            VerkleNodeKind::Leaf164 => VerkleNodeId::from_u64(idx | Self::LEAF_164_NODE_ID),
            VerkleNodeKind::Leaf165 => VerkleNodeId::from_u64(idx | Self::LEAF_165_NODE_ID),
            VerkleNodeKind::Leaf166 => VerkleNodeId::from_u64(idx | Self::LEAF_166_NODE_ID),
            VerkleNodeKind::Leaf167 => VerkleNodeId::from_u64(idx | Self::LEAF_167_NODE_ID),
            VerkleNodeKind::Leaf168 => VerkleNodeId::from_u64(idx | Self::LEAF_168_NODE_ID),
            VerkleNodeKind::Leaf169 => VerkleNodeId::from_u64(idx | Self::LEAF_169_NODE_ID),
            VerkleNodeKind::Leaf170 => VerkleNodeId::from_u64(idx | Self::LEAF_170_NODE_ID),
            VerkleNodeKind::Leaf171 => VerkleNodeId::from_u64(idx | Self::LEAF_171_NODE_ID),
            VerkleNodeKind::Leaf172 => VerkleNodeId::from_u64(idx | Self::LEAF_172_NODE_ID),
            VerkleNodeKind::Leaf173 => VerkleNodeId::from_u64(idx | Self::LEAF_173_NODE_ID),
            VerkleNodeKind::Leaf174 => VerkleNodeId::from_u64(idx | Self::LEAF_174_NODE_ID),
            VerkleNodeKind::Leaf175 => VerkleNodeId::from_u64(idx | Self::LEAF_175_NODE_ID),
            VerkleNodeKind::Leaf176 => VerkleNodeId::from_u64(idx | Self::LEAF_176_NODE_ID),
            VerkleNodeKind::Leaf177 => VerkleNodeId::from_u64(idx | Self::LEAF_177_NODE_ID),
            VerkleNodeKind::Leaf178 => VerkleNodeId::from_u64(idx | Self::LEAF_178_NODE_ID),
            VerkleNodeKind::Leaf179 => VerkleNodeId::from_u64(idx | Self::LEAF_179_NODE_ID),
            VerkleNodeKind::Leaf180 => VerkleNodeId::from_u64(idx | Self::LEAF_180_NODE_ID),
            VerkleNodeKind::Leaf181 => VerkleNodeId::from_u64(idx | Self::LEAF_181_NODE_ID),
            VerkleNodeKind::Leaf182 => VerkleNodeId::from_u64(idx | Self::LEAF_182_NODE_ID),
            VerkleNodeKind::Leaf183 => VerkleNodeId::from_u64(idx | Self::LEAF_183_NODE_ID),
            VerkleNodeKind::Leaf184 => VerkleNodeId::from_u64(idx | Self::LEAF_184_NODE_ID),
            VerkleNodeKind::Leaf185 => VerkleNodeId::from_u64(idx | Self::LEAF_185_NODE_ID),
            VerkleNodeKind::Leaf186 => VerkleNodeId::from_u64(idx | Self::LEAF_186_NODE_ID),
            VerkleNodeKind::Leaf187 => VerkleNodeId::from_u64(idx | Self::LEAF_187_NODE_ID),
            VerkleNodeKind::Leaf188 => VerkleNodeId::from_u64(idx | Self::LEAF_188_NODE_ID),
            VerkleNodeKind::Leaf189 => VerkleNodeId::from_u64(idx | Self::LEAF_189_NODE_ID),
            VerkleNodeKind::Leaf190 => VerkleNodeId::from_u64(idx | Self::LEAF_190_NODE_ID),
            VerkleNodeKind::Leaf191 => VerkleNodeId::from_u64(idx | Self::LEAF_191_NODE_ID),
            VerkleNodeKind::Leaf192 => VerkleNodeId::from_u64(idx | Self::LEAF_192_NODE_ID),
            VerkleNodeKind::Leaf193 => VerkleNodeId::from_u64(idx | Self::LEAF_193_NODE_ID),
            VerkleNodeKind::Leaf194 => VerkleNodeId::from_u64(idx | Self::LEAF_194_NODE_ID),
            VerkleNodeKind::Leaf195 => VerkleNodeId::from_u64(idx | Self::LEAF_195_NODE_ID),
            VerkleNodeKind::Leaf196 => VerkleNodeId::from_u64(idx | Self::LEAF_196_NODE_ID),
            VerkleNodeKind::Leaf197 => VerkleNodeId::from_u64(idx | Self::LEAF_197_NODE_ID),
            VerkleNodeKind::Leaf198 => VerkleNodeId::from_u64(idx | Self::LEAF_198_NODE_ID),
            VerkleNodeKind::Leaf199 => VerkleNodeId::from_u64(idx | Self::LEAF_199_NODE_ID),
            VerkleNodeKind::Leaf200 => VerkleNodeId::from_u64(idx | Self::LEAF_200_NODE_ID),
            VerkleNodeKind::Leaf201 => VerkleNodeId::from_u64(idx | Self::LEAF_201_NODE_ID),
            VerkleNodeKind::Leaf202 => VerkleNodeId::from_u64(idx | Self::LEAF_202_NODE_ID),
            VerkleNodeKind::Leaf203 => VerkleNodeId::from_u64(idx | Self::LEAF_203_NODE_ID),
            VerkleNodeKind::Leaf204 => VerkleNodeId::from_u64(idx | Self::LEAF_204_NODE_ID),
            VerkleNodeKind::Leaf205 => VerkleNodeId::from_u64(idx | Self::LEAF_205_NODE_ID),
            VerkleNodeKind::Leaf206 => VerkleNodeId::from_u64(idx | Self::LEAF_206_NODE_ID),
            VerkleNodeKind::Leaf207 => VerkleNodeId::from_u64(idx | Self::LEAF_207_NODE_ID),
            VerkleNodeKind::Leaf208 => VerkleNodeId::from_u64(idx | Self::LEAF_208_NODE_ID),
            VerkleNodeKind::Leaf209 => VerkleNodeId::from_u64(idx | Self::LEAF_209_NODE_ID),
            VerkleNodeKind::Leaf210 => VerkleNodeId::from_u64(idx | Self::LEAF_210_NODE_ID),
            VerkleNodeKind::Leaf211 => VerkleNodeId::from_u64(idx | Self::LEAF_211_NODE_ID),
            VerkleNodeKind::Leaf212 => VerkleNodeId::from_u64(idx | Self::LEAF_212_NODE_ID),
            VerkleNodeKind::Leaf213 => VerkleNodeId::from_u64(idx | Self::LEAF_213_NODE_ID),
            VerkleNodeKind::Leaf214 => VerkleNodeId::from_u64(idx | Self::LEAF_214_NODE_ID),
            VerkleNodeKind::Leaf215 => VerkleNodeId::from_u64(idx | Self::LEAF_215_NODE_ID),
            VerkleNodeKind::Leaf216 => VerkleNodeId::from_u64(idx | Self::LEAF_216_NODE_ID),
            VerkleNodeKind::Leaf217 => VerkleNodeId::from_u64(idx | Self::LEAF_217_NODE_ID),
            VerkleNodeKind::Leaf218 => VerkleNodeId::from_u64(idx | Self::LEAF_218_NODE_ID),
            VerkleNodeKind::Leaf219 => VerkleNodeId::from_u64(idx | Self::LEAF_219_NODE_ID),
            VerkleNodeKind::Leaf220 => VerkleNodeId::from_u64(idx | Self::LEAF_220_NODE_ID),
            VerkleNodeKind::Leaf221 => VerkleNodeId::from_u64(idx | Self::LEAF_221_NODE_ID),
            VerkleNodeKind::Leaf222 => VerkleNodeId::from_u64(idx | Self::LEAF_222_NODE_ID),
            VerkleNodeKind::Leaf223 => VerkleNodeId::from_u64(idx | Self::LEAF_223_NODE_ID),
            VerkleNodeKind::Leaf224 => VerkleNodeId::from_u64(idx | Self::LEAF_224_NODE_ID),
            VerkleNodeKind::Leaf225 => VerkleNodeId::from_u64(idx | Self::LEAF_225_NODE_ID),
            VerkleNodeKind::Leaf226 => VerkleNodeId::from_u64(idx | Self::LEAF_226_NODE_ID),
            VerkleNodeKind::Leaf227 => VerkleNodeId::from_u64(idx | Self::LEAF_227_NODE_ID),
            VerkleNodeKind::Leaf228 => VerkleNodeId::from_u64(idx | Self::LEAF_228_NODE_ID),
            VerkleNodeKind::Leaf229 => VerkleNodeId::from_u64(idx | Self::LEAF_229_NODE_ID),
            VerkleNodeKind::Leaf230 => VerkleNodeId::from_u64(idx | Self::LEAF_230_NODE_ID),
            VerkleNodeKind::Leaf231 => VerkleNodeId::from_u64(idx | Self::LEAF_231_NODE_ID),
            VerkleNodeKind::Leaf232 => VerkleNodeId::from_u64(idx | Self::LEAF_232_NODE_ID),
            VerkleNodeKind::Leaf233 => VerkleNodeId::from_u64(idx | Self::LEAF_233_NODE_ID),
            VerkleNodeKind::Leaf234 => VerkleNodeId::from_u64(idx | Self::LEAF_234_NODE_ID),
            VerkleNodeKind::Leaf235 => VerkleNodeId::from_u64(idx | Self::LEAF_235_NODE_ID),
            VerkleNodeKind::Leaf236 => VerkleNodeId::from_u64(idx | Self::LEAF_236_NODE_ID),
            VerkleNodeKind::Leaf237 => VerkleNodeId::from_u64(idx | Self::LEAF_237_NODE_ID),
            VerkleNodeKind::Leaf238 => VerkleNodeId::from_u64(idx | Self::LEAF_238_NODE_ID),
            VerkleNodeKind::Leaf239 => VerkleNodeId::from_u64(idx | Self::LEAF_239_NODE_ID),
            VerkleNodeKind::Leaf240 => VerkleNodeId::from_u64(idx | Self::LEAF_240_NODE_ID),
            VerkleNodeKind::Leaf241 => VerkleNodeId::from_u64(idx | Self::LEAF_241_NODE_ID),
            VerkleNodeKind::Leaf242 => VerkleNodeId::from_u64(idx | Self::LEAF_242_NODE_ID),
            VerkleNodeKind::Leaf243 => VerkleNodeId::from_u64(idx | Self::LEAF_243_NODE_ID),
            VerkleNodeKind::Leaf244 => VerkleNodeId::from_u64(idx | Self::LEAF_244_NODE_ID),
            VerkleNodeKind::Leaf245 => VerkleNodeId::from_u64(idx | Self::LEAF_245_NODE_ID),
            VerkleNodeKind::Leaf246 => VerkleNodeId::from_u64(idx | Self::LEAF_246_NODE_ID),
            VerkleNodeKind::Leaf247 => VerkleNodeId::from_u64(idx | Self::LEAF_247_NODE_ID),
            VerkleNodeKind::Leaf248 => VerkleNodeId::from_u64(idx | Self::LEAF_248_NODE_ID),
            VerkleNodeKind::Leaf249 => VerkleNodeId::from_u64(idx | Self::LEAF_249_NODE_ID),
            VerkleNodeKind::Leaf250 => VerkleNodeId::from_u64(idx | Self::LEAF_250_NODE_ID),
            VerkleNodeKind::Leaf251 => VerkleNodeId::from_u64(idx | Self::LEAF_251_NODE_ID),
            VerkleNodeKind::Leaf252 => VerkleNodeId::from_u64(idx | Self::LEAF_252_NODE_ID),
            VerkleNodeKind::Leaf253 => VerkleNodeId::from_u64(idx | Self::LEAF_253_NODE_ID),
            VerkleNodeKind::Leaf254 => VerkleNodeId::from_u64(idx | Self::LEAF_254_NODE_ID),
            VerkleNodeKind::Leaf255 => VerkleNodeId::from_u64(idx | Self::LEAF_255_NODE_ID),
            VerkleNodeKind::Leaf256 => VerkleNodeId::from_u64(idx | Self::LEAF_256_NODE_ID),
            VerkleNodeKind::InnerDelta => VerkleNodeId::from_u64(idx | Self::INNER_DELTA),
            VerkleNodeKind::LeafDelta => VerkleNodeId::from_u64(idx | Self::LEAF_DELTA),
        }
    }

    fn to_index(self) -> u64 {
        self.to_u64() & Self::INDEX_MASK
    }
}

impl NodeSize for VerkleNodeId {
    /// Returns the byte size of the node variant it refers to.
    /// Panics if the ID does not refer to a valid node type.
    fn node_byte_size(&self) -> usize {
        self.to_node_kind().unwrap().node_byte_size()
    }

    /// Returns the size of the smallest non-empty node variant.
    fn min_non_empty_node_size() -> usize {
        VerkleNodeKind::min_non_empty_node_size()
    }
}

impl HasEmptyId for VerkleNodeId {
    fn is_empty_id(&self) -> bool {
        self.to_node_kind() == Some(VerkleNodeKind::Empty)
    }

    fn empty_id() -> Self {
        VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty)
    }
}

impl Debug for VerkleNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerkleNodeId")
            .field("kind", &self.to_node_kind().unwrap())
            .field("idx", &self.to_index())
            .field("raw", &self.0)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_idx_and_node_type_creates_id_from_lower_6_bytes_logic_or_node_type_prefix() {
        let idx = 0x0000_0123_4567_89ab;
        let cases = [
            (VerkleNodeKind::Empty, 0x0000_0000_0000_0000),
            (VerkleNodeKind::Inner9, 0x0000_1000_0000_0000),
            (VerkleNodeKind::Inner15, 0x0000_2000_0000_0000),
            (VerkleNodeKind::Inner21, 0x0000_3000_0000_0000),
            (VerkleNodeKind::Inner256, 0x0000_4000_0000_0000),
            (VerkleNodeKind::InnerDelta, 0x0000_B000_0000_0000),
            (VerkleNodeKind::Leaf1, 0x0000_5000_0000_0000),
            (VerkleNodeKind::Leaf2, 0x0000_6000_0000_0000),
            (VerkleNodeKind::Leaf5, 0x0000_7000_0000_0000),
            (VerkleNodeKind::Leaf18, 0x0000_8000_0000_0000),
            (VerkleNodeKind::Leaf146, 0x0000_9000_0000_0000),
            (VerkleNodeKind::Leaf256, 0x0000_A000_0000_0000),
            (VerkleNodeKind::LeafDelta, 0x0000_C000_0000_0000),
        ];

        for (node_type, prefix) in cases {
            let id = VerkleNodeId::from_idx_and_node_kind(idx, node_type);
            assert_eq!(id.to_u64(), idx | prefix);
        }
    }

    #[test]
    #[should_panic]
    fn from_idx_and_node_type_panics_if_index_too_large() {
        let idx = 0x0000_f000_0000_0000;

        VerkleNodeId::from_idx_and_node_kind(idx, VerkleNodeKind::Empty);
    }

    #[test]
    fn to_index_masks_out_node_type() {
        let id = VerkleNodeId([0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
        assert_eq!(id.to_index(), 0x0f_ff_ff_ff_ff_ff);
    }

    #[test]
    fn to_node_type_returns_node_type_for_valid_prefixes() {
        let cases = [
            (
                VerkleNodeId([0x00, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Empty),
            ),
            (
                VerkleNodeId([0x10, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Inner9),
            ),
            (
                VerkleNodeId([0x20, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Inner15),
            ),
            (
                VerkleNodeId([0x30, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Inner21),
            ),
            (
                VerkleNodeId([0x40, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Inner256),
            ),
            (
                VerkleNodeId([0xB0, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::InnerDelta),
            ),
            (
                VerkleNodeId([0x50, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Leaf1),
            ),
            (
                VerkleNodeId([0x60, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Leaf2),
            ),
            (
                VerkleNodeId([0x70, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Leaf5),
            ),
            (
                VerkleNodeId([0x80, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Leaf18),
            ),
            (
                VerkleNodeId([0x90, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Leaf146),
            ),
            (
                VerkleNodeId([0xA0, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::Leaf256),
            ),
            (
                VerkleNodeId([0xC0, 0x00, 0x00, 0x00, 0x00, 0x2a]),
                Some(VerkleNodeKind::LeafDelta),
            ),
        ];
        for (node_id, node_type) in cases {
            assert_eq!(node_id.to_node_kind(), node_type);
        }
    }

    #[test]
    fn node_id_to_node_type_returns_none_for_invalid_prefixes() {
        let id = VerkleNodeId([0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
        assert_eq!(id.to_node_kind(), None);
    }

    #[test]
    fn from_u64_constructs_integer_from_lower_6_bytes() {
        let id = VerkleNodeId::from_u64(0x1234_5678_90ab_cdef);
        assert_eq!(id.0, [0x56, 0x78, 0x90, 0xab, 0xcd, 0xef]);
    }

    #[test]
    fn to_u64_converts_node_id_to_integer_with_lower_6_bytes() {
        let id = VerkleNodeId([0x12, 0x34, 0x56, 0x78, 0x90, 0xab]);
        assert_eq!(id.to_u64(), 0x1234_5678_90ab);
    }

    #[test]
    fn node_id_byte_size_returns_byte_size_of_encoded_node_type() {
        let cases = [
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Empty),
                VerkleNodeKind::Empty,
            ),
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Inner9),
                VerkleNodeKind::Inner9,
            ),
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Inner15),
                VerkleNodeKind::Inner15,
            ),
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Inner21),
                VerkleNodeKind::Inner21,
            ),
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Inner256),
                VerkleNodeKind::Inner256,
            ),
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::InnerDelta),
                VerkleNodeKind::InnerDelta,
            ),
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Leaf1),
                VerkleNodeKind::Leaf1,
            ),
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Leaf1),
                VerkleNodeKind::Leaf1,
            ),
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Leaf2),
                VerkleNodeKind::Leaf2,
            ),
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Leaf5),
                VerkleNodeKind::Leaf5,
            ),
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Leaf18),
                VerkleNodeKind::Leaf18,
            ),
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Leaf146),
                VerkleNodeKind::Leaf146,
            ),
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::Leaf256),
                VerkleNodeKind::Leaf256,
            ),
            (
                VerkleNodeId::from_idx_and_node_kind(0, VerkleNodeKind::LeafDelta),
                VerkleNodeKind::LeafDelta,
            ),
        ];
        for (node_id, node_type) in cases {
            assert_eq!(node_id.node_byte_size(), node_type.node_byte_size());
        }
    }

    #[test]
    fn node_id_min_non_empty_node_size_returns_min_byte_size_of_node_type() {
        assert_eq!(
            VerkleNodeId::min_non_empty_node_size(),
            VerkleNodeKind::min_non_empty_node_size()
        );
    }

    #[test]
    fn debug_print_kind_index_and_raw_bytes() {
        assert_eq!(
            format!(
                "{:?}",
                VerkleNodeId::from_idx_and_node_kind(1, VerkleNodeKind::Inner256)
            ),
            "VerkleNodeId { kind: Inner256, idx: 1, raw: [64, 0, 0, 0, 0, 1] }"
        );
        assert_eq!(
            format!(
                "{:?}",
                VerkleNodeId::from_idx_and_node_kind(2, VerkleNodeKind::Leaf256)
            ),
            "VerkleNodeId { kind: Leaf256, idx: 2, raw: [160, 0, 0, 0, 0, 2] }"
        );
    }
}
