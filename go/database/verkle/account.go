// Copyright (c) 2024 Fantom Foundation
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at fantom.foundation/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package verkle

import (
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/holiman/uint256"
)

type AccountInfo struct {
	Nonce    uint64
	Balance  uint256.Int
	CodeHash common.Hash
}

// IsEmpty checks whether the account information is empty, and thus, the
// default value. All accounts not present in an MPT are implicitly empty. Also
// no empty accounts may be explicitly stored.
func (a *AccountInfo) IsEmpty() bool {
	return *a == AccountInfo{}
}
