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
	"strings"
	"testing"
)

func TestMemoryFootprint_Contains_Data(t *testing.T) {
	db, err := openTestDatabase(t)
	if err != nil {
		t.Fatalf("cannot open test database: %v", err)
	}
	fp := db.GetMemoryFootprint()

	if fp.Total() <= 0 {
		t.Error("memory footprint returned 0 for existing open database")
	}

	s := fmt.Sprintf("%s", fp)

	if !strings.Contains(s, "live") {
		t.Error("memory-footprint breakdown does not contain 'live' keyword even though database contains LiveDB")
	}

	if !strings.Contains(s, "archive") {
		t.Error("memory-footprint breakdown does not contain 'archive' keyword even though database contains Archive")
	}

	if !strings.Contains(s, "liveState") {
		t.Error("memory-footprint breakdown does not contain 'liveState' keyword for live database")
	}

	if !strings.Contains(s, "database") {
		t.Error("memory-footprint breakdown does not contain 'database' keyword for database")
	}

}
