// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

/// An empty node in a managed Verkle trie.
/// This is a zero-sized type that only exists for implementing traits on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyNode;
