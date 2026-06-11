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
