// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package carmen

import (
	"fmt"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/state"
)

type bulkLoad struct {
	nested state.BulkLoad
	db     *database
	block  int64
}

func (l *bulkLoad) CreateAccount(address Address) {
	if l.db != nil {
		l.nested.CreateAccount(common.Address(address))
	}
}

func (l *bulkLoad) SetBalance(address Address, balance Amount) {
	if l.db != nil {
		l.nested.SetBalance(common.Address(address), balance)
	}
}

func (l *bulkLoad) SetNonce(address Address, nonce uint64) {
	if l.db != nil {
		l.nested.SetNonce(common.Address(address), nonce)
	}
}

func (l *bulkLoad) SetState(address Address, key Key, value Value) {
	if l.db != nil {
		l.nested.SetState(common.Address(address), common.Key(key), common.Value(value))
	}
}

func (l *bulkLoad) SetCode(address Address, code []byte) {
	if l.db != nil {
		l.nested.SetCode(common.Address(address), code)
	}
}

func (l *bulkLoad) Finalize() error {
	if l.db == nil {
		return fmt.Errorf("bulk load already closed")
	}

	err := l.nested.Close()
	l.db.moveBlockAndReleaseHead(l.block)
	l.db = nil
	return err
}
