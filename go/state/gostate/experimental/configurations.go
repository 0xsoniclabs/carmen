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
	vtgeth "github.com/0xsoniclabs/carmen/go/database/vt/geth"
	vtmemory "github.com/0xsoniclabs/carmen/go/database/vt/memory"
	"github.com/0xsoniclabs/carmen/go/state"
)

// configurations contains the experimental state configurations.
var configurations = map[state.Configuration]state.StateFactory{
	{
		Variant: "go-geth-memory",
		Schema:  6,
		Archive: state.NoArchive,
	}: vtgeth.NewState,
	{
		Variant: "go-memory",
		Schema:  6,
		Archive: state.NoArchive,
	}: vtmemory.NewState,
}

func init() {
	for config, fact := range configurations {
		state.RegisterStateFactory(config, wrapInSyncState(fact))
	}
}

// wrapInSyncState creates a state factory that ensures that the returned state is always wrapped in a synced state.
func wrapInSyncState(factory state.StateFactory) state.StateFactory {
	return func(params state.Parameters) (state.State, error) {
		res, err := factory(params)
		return state.WrapIntoSyncedState(res), err
	}
}
