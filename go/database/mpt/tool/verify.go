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
	"fmt"
	"time"

	"github.com/0xsoniclabs/carmen/go/common/interrupt"
	"github.com/0xsoniclabs/carmen/go/database/mpt"
	"github.com/0xsoniclabs/carmen/go/database/mpt/io"
	"github.com/urfave/cli/v2"
)

var Verify = cli.Command{
	Action:    addPerformanceDiagnoses(verify),
	Name:      "verify",
	Usage:     "verifies the consistency of an MPT",
	ArgsUsage: "<director>",
}

func verify(context *cli.Context) error {
	// parse the directory argument
	if context.Args().Len() != 1 {
		return fmt.Errorf("missing directory storing state")
	}

	dir := context.Args().Get(0)

	// try to obtain information of the contained MPT
	info, err := io.CheckMptDirectoryAndGetInfo(dir)
	if err != nil {
		return err
	}

	// run forest verification
	observer := &verificationObserver{}

	ctx := interrupt.CancelOnInterrupt(context.Context)

	if info.Mode == mpt.Immutable {
		return mpt.VerifyArchiveTrie(ctx, dir, info.Config, observer)
	}
	return mpt.VerifyFileLiveTrie(ctx, dir, info.Config, observer)
}

type verificationObserver struct {
	start time.Time
}

func (o *verificationObserver) StartVerification() {
	o.start = time.Now()
	o.printHeader()
	fmt.Println("Starting verification ...")
}

func (o *verificationObserver) Progress(msg string) {
	o.printHeader()
	fmt.Println(msg)
}

func (o *verificationObserver) EndVerification(res error) {
	if res == nil {
		o.printHeader()
		fmt.Println("Verification successful!")
	}
}

func (o *verificationObserver) printHeader() {
	now := time.Now()
	t := uint64(now.Sub(o.start).Seconds())
	fmt.Printf("%s [t=%4d:%02d] - ", now.Format("15:04:05"), t/60, t%60)
}
