// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

// Package memory implements a simple in-memory Verkle trie (S6) for the Carmen
// project providing
//   - a performance-oriented implementation of the Verkle trie in Carmen
//   - a proving ground for optimizations on the Go side of Carmen
//   - a benchmark implementation for comparing with other implementations
//
// The implementation is maintained to be suitable for testing and benchmarking
// purposes for all potential use cases, with the following limitations:
//   - it retains all data in memory, so it can not scale to large datasets
//   - there is no support for checkpointing to recover from crashes
//
// WARNING: This package is not intended for production use. It is a prototype
// implementation and is not optimized for memory usage. It is also lacking
// sufficient testing and error handling. It is only intended to be used as a
// prototype for Verkle trie implementations in Carmen.
package memory
