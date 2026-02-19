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
	"github.com/0xsoniclabs/carmen/go/database/flat"
	vtgeth "github.com/0xsoniclabs/carmen/go/database/vt/geth"
	vtgeth2 "github.com/0xsoniclabs/carmen/go/database/vt/geth2"
	vtmemory "github.com/0xsoniclabs/carmen/go/database/vt/memory"
	vtref "github.com/0xsoniclabs/carmen/go/database/vt/reference"
	"github.com/0xsoniclabs/carmen/go/state"
)

// configurations contains the experimental state configurations.
var configurations = map[state.Configuration]state.StateFactory{
	{
		Variant: "go-memory",
		Schema:  0,
		Archive: state.NoArchive,
	}: func(params state.Parameters) (state.State, error) {
		return flat.NewState(params.Directory, nil)
	},
	{
		Variant: "go-geth-memory",
		Schema:  6,
		Archive: state.NoArchive,
	}: vtgeth.NewState,
	{
		Variant: "go-geth-memory-flat",
		Schema:  6,
		Archive: state.NoArchive,
	}: flat.WrapFactory(vtgeth.NewState),
	{
		Variant: "go-geth2-leveldb",
		Schema:  6,
		Archive: state.NoArchive,
	}: vtgeth2.NewState,
	{
		Variant: "go-geth2-leveldb",
		Schema:  6,
		Archive: state.S5Archive,
	}: vtgeth2.NewState,
	{
		Variant: "go-geth2-leveldb",
		Schema:  6,
		Archive: state.LevelDbArchive,
	}: vtgeth2.NewState,
	{
		Variant: "go-geth2-leveldb-flat",
		Schema:  6,
		Archive: state.NoArchive,
	}: flat.WrapFactory(vtgeth2.NewState),
	{
		Variant: "go-geth2-leveldb-flat",
		Schema:  6,
		Archive: state.LevelDbArchive,
	}: flat.WrapFactory(vtgeth2.NewState),
	{
		Variant: "go-memory",
		Schema:  6,
		Archive: state.NoArchive,
	}: vtmemory.NewState,
	{
		Variant: "go-memory-flat",
		Schema:  6,
		Archive: state.NoArchive,
	}: flat.WrapFactory(vtmemory.NewState),
	{
		Variant: "go-memory-seq",
		Schema:  6,
		Archive: state.NoArchive,
	}: vtmemory.NewState,
	{
		Variant: "go-memory-seq-flat",
		Schema:  6,
		Archive: state.NoArchive,
	}: flat.WrapFactory(vtmemory.NewState),
	{
		Variant: "go-reference",
		Schema:  6,
		Archive: state.NoArchive,
	}: vtref.NewState,
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
