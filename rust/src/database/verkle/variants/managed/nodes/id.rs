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
            Self::INNER_9_NODE_ID => Some(VerkleNodeKind::Inner9),
            Self::INNER_15_NODE_ID => Some(VerkleNodeKind::Inner15),
            Self::INNER_21_NODE_ID => Some(VerkleNodeKind::Inner21),
            Self::INNER_256_NODE_ID => Some(VerkleNodeKind::Inner256),
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
            VerkleNodeKind::Inner9 => VerkleNodeId::from_u64(idx | Self::INNER_9_NODE_ID),
            VerkleNodeKind::Inner15 => VerkleNodeId::from_u64(idx | Self::INNER_15_NODE_ID),
            VerkleNodeKind::Inner21 => VerkleNodeId::from_u64(idx | Self::INNER_21_NODE_ID),
            VerkleNodeKind::Inner256 => VerkleNodeId::from_u64(idx | Self::INNER_256_NODE_ID),
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
