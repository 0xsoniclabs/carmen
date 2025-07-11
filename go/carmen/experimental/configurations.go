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
	"github.com/0xsoniclabs/carmen/go/carmen"
	"github.com/0xsoniclabs/carmen/go/state"

	_ "github.com/0xsoniclabs/carmen/go/state/cppstate"
	_ "github.com/0xsoniclabs/carmen/go/state/gostate"
)

// GetDatabaseConfigurations returns a list of experimental database configurations
// which should not be used in productive settings but may be used for special
// purpose setups.
// WARNING: do not use those configurations without understanding the involved
// risks. Consult Carmen developers if you have questions.
func GetDatabaseConfigurations() []carmen.Configuration {
	// Register all configurations with a known factory.
	res := []carmen.Configuration{}
	for config := range state.GetAllRegisteredStateFactories() {
		res = append(res, carmen.Configuration{
			Variant: carmen.Variant(config.Variant),
			Schema:  carmen.Schema(config.Schema),
			Archive: carmen.Archive(config.Archive),
		})
	}
	return res
}

func init() {
	for _, config := range GetDatabaseConfigurations() {
		carmen.RegisterConfiguration(config)
	}
}
