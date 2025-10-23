// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package experimental

import (
	"testing"

	"github.com/0xsoniclabs/carmen/go/state"
)

func TestConfigurations_ContainAllConfigurations(t *testing.T) {
	configsSeen := make(map[state.Configuration]struct{})
	for config := range state.GetAllRegisteredStateFactories() {
		configsSeen[config] = struct{}{}
	}

	for expected := range configurations {
		if _, found := configsSeen[expected]; !found {
			t.Errorf("missing configuration: %v", expected)
		}
	}

	if got, want := len(configsSeen), len(configurations); got != want {
		t.Errorf("unexpected number of configurations: got %d, want %d", got, want)
	}
}

func TestConfiguration_RegisteredConfigurationsCanBeUsed(t *testing.T) {
	for config, fact := range state.GetAllRegisteredStateFactories() {
		config := config
		t.Run(config.String(), func(t *testing.T) {
			t.Parallel()
			st, err := fact(state.Parameters{})
			if err != nil {
				t.Fatalf("failed to create state: %v", err)
			}
			if err := st.Close(); err != nil {
				t.Fatalf("failed to close database: %v", err)
			}
		})
	}
}
