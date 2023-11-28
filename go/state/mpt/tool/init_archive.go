package main

import (
	"bufio"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"

	mptIo "github.com/Fantom-foundation/Carmen/go/state/mpt/io"

	"github.com/urfave/cli/v2"
)

var InitArchive = cli.Command{
	Action:    doArchiveInit,
	Name:      "init-archive",
	Usage:     "initializes an Archive instance from a file",
	ArgsUsage: "<source-file> <archive target director>",
	Flags: []cli.Flag{
		&blockHeightFlag,
	},
}

var (
	blockHeightFlag = cli.Uint64Flag{
		Name:  "block-height",
		Usage: "the block height the input file is describing",
	}
)

func doArchiveInit(context *cli.Context) error {
	if context.Args().Len() != 2 {
		return fmt.Errorf("missing source file and/or target directory parameter")
	}
	src := context.Args().Get(0)
	dir := context.Args().Get(1)

	if err := os.Mkdir(dir, 0700); err != nil {
		return fmt.Errorf("error creating output directory: %v", err)
	}

	height := context.Uint64(blockHeightFlag.Name)

	file, err := os.Open(src)
	if err != nil {
		return err
	}
	var in io.Reader = bufio.NewReader(file)
	if in, err = gzip.NewReader(in); err != nil {
		return err
	}
	return errors.Join(
		mptIo.InitializeArchive(dir, in, height),
		file.Close(),
	)
}