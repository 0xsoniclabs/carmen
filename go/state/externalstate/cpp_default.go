// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

//go:build !carmen_cpp

package externalstate

func initCpp() {
	// No C++ external state configurations are supported when the "cpp" build tag is not set.
}
