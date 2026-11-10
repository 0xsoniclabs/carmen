// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package io

import (
	"bytes"
	"context"
	"log"
	"strings"
	"testing"
	"time"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"github.com/stretchr/testify/require"
)

func TestIO_GetLiveDbTotalSupply(t *testing.T) {
	sourceDir := t.TempDir()
	db := createExampleLiveDB(t, sourceDir)
	if err := db.SetBalance(common.Address{}, amount.New(5)); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	logger := Log{logger: log.New(&buf, "", 0), start: time.Now()}

	err := CalculateLiveTotalSupply(context.Background(), &logger, sourceDir)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(buf.String(), "Total native token supply: 65\n") {
		t.Errorf("Reported total supply does not match:\n%s", buf.String())
	}
	if !strings.Contains(buf.String(), "Excluding zero address:    60\n") {
		t.Errorf("Reported total supply does not match:\n%s", buf.String())
	}
}

func TestIO_CalculateLiveTotalSupplyFromFiles_ReportsSupply(t *testing.T) {
	require := require.New(t)
	sourceDir := t.TempDir()
	db := createExampleLiveDB(t, sourceDir)
	require.NoError(db.SetBalance(common.Address{}, amount.New(5)))
	require.NoError(db.Close())

	var buf bytes.Buffer
	logger := Log{logger: log.New(&buf, "", 0), start: time.Now()}

	require.NoError(CalculateLiveTotalSupplyFromFiles(context.Background(), &logger, sourceDir))
	require.Contains(buf.String(), "Total native token supply: 65\n")
	require.Contains(buf.String(), "Excluding zero address:    60\n")
}

func TestIO_CalculateLiveTotalSupplyFromFiles_RejectsArchive(t *testing.T) {
	require := require.New(t)
	sourceDir := t.TempDir()
	createTestArchive(t, sourceDir)

	logger := Log{logger: log.New(&bytes.Buffer{}, "", 0), start: time.Now()}
	require.Error(CalculateLiveTotalSupplyFromFiles(context.Background(), &logger, sourceDir))
}

func TestIO_GetArchiveTotalSupply(t *testing.T) {
	sourceDir := t.TempDir()
	createTestArchive(t, sourceDir)

	var buf bytes.Buffer
	logger := Log{logger: log.New(&buf, "", 0), start: time.Now()}

	err := CalculateArchiveTotalSupply(context.Background(), &logger, sourceDir, 3)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(buf.String(), "Total native token supply: 3\n") {
		t.Errorf("Reported total supply does not match:\n%s", buf.String())
	}
	if !strings.Contains(buf.String(), "Excluding zero address:    3\n") {
		t.Errorf("Reported total supply does not match:\n%s", buf.String())
	}
}
