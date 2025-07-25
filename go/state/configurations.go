// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package state

import (
	"fmt"
	"maps"
	"time"

	"github.com/0xsoniclabs/carmen/go/common"
)

// ----------------------------------------------------------------------------
//                        for Carmen users
// ----------------------------------------------------------------------------

// Parameters struct defining configuration parameters for state instances.
type Parameters struct {
	Variant               Variant
	Schema                Schema
	Archive               ArchiveType
	Directory             string
	LiveCache             int64 // bytes, approximate, supported only by S5 now
	ArchiveCache          int64 // bytes, approximate, supported only by S5 now
	CheckpointInterval    int   // in blocks
	CheckpointPeriod      time.Duration
	BackgroundFlushPeriod time.Duration // the time between background flushes; a default is chosen if zero, disabled if negative
}

const (
	// UnsupportedConfiguration is the error returned if unsupported configuration
	// parameters have been specified. The text may contain further details regarding the
	// unsupported feature.
	UnsupportedConfiguration = common.ConstError("unsupported configuration")

	// ExportNotSupported is returned by State implementation if it does not support Export.
	ExportNotSupported = common.ConstError("export not supported")
)

// NewState is the public interface for creating Carmen state instances. If for the
// given parameters a state can be constructed, the resulting state is returned. If
// construction fails, an error is reported. If the requested configuration is not
// supported, the error is an UnsupportedConfiguration error.
func NewState(params Parameters) (State, error) {
	config := Configuration{
		Variant: params.Variant,
		Schema:  Schema(params.Schema),
		Archive: params.Archive,
	}
	// Enforce default values.
	if config.Variant == "" {
		config.Variant = "go-file"
	}
	if config.Schema == 0 {
		config.Schema = 5
	}
	if config.Archive == "" {
		config.Archive = NoArchive
	}
	factory, found := stateFactoryRegistry[config]
	if !found {
		return nil, fmt.Errorf("%w: no registered implementation for %v", UnsupportedConfiguration, config)
	}
	return factory(params)
}

// ----------------------------------------------------------------------------
//                      for Carmen implementations
// ----------------------------------------------------------------------------

type Configuration struct {
	Variant Variant
	Schema  Schema
	Archive ArchiveType
}

func (c *Configuration) String() string {
	return fmt.Sprintf("%s_s%d_%v", c.Variant, c.Schema, c.Archive)
}

type Variant string

type Schema uint8

type ArchiveType string

const (
	NoArchive      ArchiveType = "none"
	LevelDbArchive ArchiveType = "ldb"
	SqliteArchive  ArchiveType = "sql"
	S4Archive      ArchiveType = "s4"
	S5Archive      ArchiveType = "s5"
)

type StateFactory func(params Parameters) (State, error)

var stateFactoryRegistry = map[Configuration]StateFactory{}

func RegisterStateFactory(config Configuration, factory StateFactory) {
	if _, found := stateFactoryRegistry[config]; found {
		panic(fmt.Sprintf("attempted to register multiple factories for %v", config))
	}
	stateFactoryRegistry[config] = factory
}

func GetAllRegisteredStateFactories() map[Configuration]StateFactory {
	return maps.Clone(stateFactoryRegistry)
}
