// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package main

import (
	"bufio"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"

	mptIo "github.com/0xsoniclabs/carmen/go/database/mpt/io"
	"github.com/urfave/cli/v2"
)

var ImportLiveDbCmd = cli.Command{
	Action:    addPerformanceDiagnoses(doLiveDbImport),
	Name:      "import-live-db",
	Usage:     "imports a LiveDB instance from a file",
	ArgsUsage: "<source-file> <target director>",
}

var ImportArchiveCmd = cli.Command{
	Action:    addPerformanceDiagnoses(doArchiveImport),
	Name:      "import-archive",
	Usage:     "imports an Archive instance from a file",
	ArgsUsage: "<source-file> <target director>",
}

var ImportLiveAndArchiveCmd = cli.Command{
	Action:    addPerformanceDiagnoses(doLiveAndArchiveImport),
	Name:      "import",
	Usage:     "imports both LiveDB and Archive instance from a file",
	ArgsUsage: "<source-file> <target director>",
}

func doLiveDbImport(context *cli.Context) error {
	return doImport(context, mptIo.ImportLiveDb)
}

func doArchiveImport(context *cli.Context) error {
	return doImport(context, mptIo.ImportArchive)
}

func doLiveAndArchiveImport(context *cli.Context) error {
	return doImport(context, mptIo.ImportLiveAndArchive)
}

func doImport(context *cli.Context, runImport func(logger *mptIo.Log, directory string, in io.Reader) error) error {
	if context.Args().Len() != 2 {
		return fmt.Errorf("missing source file and/or target directory parameter")
	}
	src := context.Args().Get(0)
	dir := context.Args().Get(1)

	if err := os.MkdirAll(dir, 0700); err != nil {
		return fmt.Errorf("error creating output directory: %v", err)
	}

	logger := mptIo.NewLog()
	logger.Print("import started")
	file, err := os.Open(src)
	if err != nil {
		return err
	}
	var in io.Reader = bufio.NewReader(file)
	if in, err = gzip.NewReader(in); err != nil {
		return err
	}
	defer func() {
		logger.Printf("import done")
	}()
	return errors.Join(
		runImport(logger, dir, in),
		file.Close(),
	)
}
