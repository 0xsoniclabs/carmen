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

func Test_MapEntry_String(t *testing.T) {
	e := MapEntry[int, int]{10, 20}

	if got, want := e.String(), "Entry: 10 -> 20"; got != want {
		t.Errorf("provided string does not match: %s != %s", got, want)
	}
}
