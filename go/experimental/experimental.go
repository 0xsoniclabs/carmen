// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package experimental

// This package is used to enable experimental features in the Carmen project to
// be used for testing and development purposes of off-chain components.
// In particular, by importing this package, access to C++ and Rust based DB
// implementations is enabled.
//
// To enable those features, import this package into your project as follows:
//
//  import _ "github.com/0xsoniclabs/carmen/go/experimental"
//
// As a side-effect, additional packages implementing experimental features are
// imported into your project. Some of those, like the C++ and Rust based DB
// require additional library dependencies to be accessible on your system.
// Build steps may have to be adjusted to make sure that these required
// dependencies can be linked correctly.
//
// This package is intended for use in development and testing environments only.
// It is not recommended to use this package in production environments, as
// experimental features may be unstable and could lead to unexpected behavior.

import (
	_ "github.com/0xsoniclabs/carmen/go/state/externalstate"
	_ "github.com/0xsoniclabs/carmen/go/state/gostate/experimental"
)
