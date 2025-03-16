package io

import (
	"bytes"
	"context"
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/amount"
	"log"
	"strings"
	"testing"
	"time"
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

func TestIO_GetArchiveTotalSupply(t *testing.T) {
	sourceDir := t.TempDir()
	createTestArchive(t, sourceDir)

	var buf bytes.Buffer
	logger := Log{logger: log.New(&buf, "", 0), start: time.Now()}

	err := CalculateArchiveTotalSupply(context.Background(), &logger, sourceDir, 5)
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
