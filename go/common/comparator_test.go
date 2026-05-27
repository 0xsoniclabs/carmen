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

import "testing"

var (
	addressA = Address{0xA}
	addressB = Address{0xB}

	keyA = &Key{0xA}
	keyB = &Key{0xB}
)

func TestAddressComparator(t *testing.T) {

	if addressA.Compare(&addressA) != 0 {
		t.Errorf("Wrong comparator error")
	}
	if addressA.Compare(&addressB) > 0 {
		t.Errorf("Wrong comparator error")
	}
	if addressB.Compare(&addressA) < 0 {
		t.Errorf("Wrong comparator error")
	}
}
func TestKeyComparator(t *testing.T) {
	if keyA.Compare(keyA) != 0 {
		t.Errorf("Wrong comparator error")
	}
	if keyA.Compare(keyB) > 0 {
		t.Errorf("Wrong comparator error")
	}
	if keyB.Compare(keyA) < 0 {
		t.Errorf("Wrong comparator error")
	}
}
