// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package state

import (
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
)

// IsEmptyAccount checks if the account is empty in the state, i.e. if it has zero balance, zero nonce and empty code
func IsEmptyAccount(s State, addr common.Address) (bool, error) {
	balance, err := s.GetBalance(addr)
	if err != nil {
		return false, err
	}
	nonce, err := s.GetNonce(addr)
	if err != nil {
		return false, err
	}
	code, err := s.GetCode(addr)
	if err != nil {
		return false, err
	}
	return balance == (amount.Amount{}) && nonce == (common.Nonce{}) && len(code) == 0, nil
}
