// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package mpt

import (
	"errors"

	"github.com/0xsoniclabs/carmen/go/common/amount"
)

// errTotalSupplyOverflow is reported if the sum of all balances exceeds the
// range of the amount type.
var errTotalSupplyOverflow = errors.New("total supply overflows the amount range")

// TotalSupply provides the sum of the balances of all accounts in the state.
// The value is maintained incrementally while balances are updated and is
// persisted with the trie meta-data. If it is not known yet, e.g. for states
// created before it was tracked, it is computed by traversing the whole
// account trie, which may take a considerable amount of time for large
// states, and maintained from then on.
func (s *MptState) TotalSupply() (amount.Amount, error) {
	if total, known := s.trie.getTotalSupply(); known {
		return total, nil
	}
	total, err := s.computeTotalSupply()
	if err != nil {
		return amount.New(), err
	}
	s.trie.setTotalSupply(total)
	return total, nil
}

// computeTotalSupply sums up the balances of all accounts by traversing the
// account trie. Storage tries are skipped.
func (s *MptState) computeTotalSupply() (amount.Amount, error) {
	var (
		total    amount.Amount
		overflow bool
	)
	err := s.Visit(ReadAccess{}, MakeVisitor(func(node Node, _ NodeInfo) VisitResponse {
		account, ok := node.(*AccountNode)
		if !ok {
			return VisitResponseContinue
		}
		total, overflow = amount.AddOverflow(total, account.Info().Balance)
		if overflow {
			return VisitResponseAbort
		}
		return VisitResponsePrune
	}))
	if err != nil {
		return amount.New(), err
	}
	if overflow {
		return amount.New(), errTotalSupplyOverflow
	}
	return total, nil
}
