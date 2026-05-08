// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package experimental_test

import (
	"slices"
	"testing"

	"github.com/0xsoniclabs/carmen/go/carmen"
	"github.com/0xsoniclabs/carmen/go/carmen/experimental"
)

func TestConfigurations_ConfigurationsAreRegisteredGlobally(t *testing.T) {
	registeredConfigs := carmen.GetAllConfigurations()
	for _, config := range experimental.GetDatabaseConfigurations() {
		if !slices.Contains(registeredConfigs, config) {
			t.Errorf("missing registration of configuration %v", config)
		}
	}
}

func TestConfiguration_RegisteredConfigurationsCanBeUsed(t *testing.T) {
	for _, config := range carmen.GetAllConfigurations() {
		config := config
		t.Run(config.String(), func(t *testing.T) {
			t.Parallel()
			db, err := carmen.OpenDatabase(t.TempDir(), config, nil)
			if err != nil {
				t.Fatalf("failed to open database: %v", err)
			}
			if err := db.Close(); err != nil {
				t.Fatalf("failed to close database: %v", err)
			}
		})
	}
}
