// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package common

// Rules are the protocol rules of a transaction that affect the StateDB.
// The zero value selects the rules before Amsterdam.
type Rules struct {
	// IsAmsterdam enables EIP-8246: a self-destructed account keeps its balance
	// and remains as a balance-only account, unless the balance is zero.
	IsAmsterdam bool
}
