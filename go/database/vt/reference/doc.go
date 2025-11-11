// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

// Package reference implements a simple reference Verkle trie (S6) for the
// Carmen project providing
//   - a proof-of-concept implementation of the Verkle trie in Carmen
//   - an executable specification for developers
//   - a light-weight reference implementation for testing of other components
//
// The implementation is maintained to be suitable for testing purposes for all
// potential use cases, with the following limitations:
//   - it retains all data in memory, so it can not scale to large datasets
//   - there is no support for checkpointing to recover from crashes
//
// WARNING: This package is not intended for production use. It is a reference
// implementation and is not optimized for performance or memory usage. It is
// also lacking sufficient testing and error handling. It is only intended
// to be used as a reference for the Verkle trie implementation in Carmen.
package reference
