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
	"encoding/hex"
	"fmt"
	"slices"

	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/diagnostics"
	"github.com/0xsoniclabs/carmen/go/database/mpt"
	"github.com/0xsoniclabs/carmen/go/database/mpt/io"
	"github.com/urfave/cli/v2"
)

var AccountInfo = cli.Command{
	Action:    diagnostics.AddPerformanceDiagnosticsAction(accountInfo, &diagnosticsFlag, &cpuProfileFlag, &traceFlag),
	Name:      "account-info",
	Usage:     "lists information about a given account",
	ArgsUsage: "<directory> <account>",
}

func accountInfo(context *cli.Context) error {
	// parse the directory argument
	if context.Args().Len() != 2 {
		return fmt.Errorf("missing directory storing state")
	}
	dir := context.Args().Get(0)
	account := context.Args().Get(1)

	// try to obtain information of the contained MPT
	mptInfo, err := io.CheckMptDirectoryAndGetInfo(dir)
	if err != nil {
		return err
	}

	fmt.Printf("Directory contains an MPT State with the following properties:\n")
	fmt.Printf("\tMPT Configuration: %v\n", mptInfo.Config.Name)
	fmt.Printf("\tMode:              %v\n", mptInfo.Mode)

	// attempt to open the MPT
	// if mptInfo.Mode == mpt.Mutable {
	trie, err := mpt.OpenFileLiveTrie(dir, mptInfo.Config, mpt.NodeCacheConfig{})
	if err != nil {
		fmt.Printf("\tFailed to open:    %v\n", err)
		return nil
	} else {
		fmt.Printf("\tCan be opened:     Yes\n")
	}

	defer func() {
		if err := trie.Close(); err != nil {
			fmt.Printf("error closing forest: %v\n", err)
		}
	}()

	// Print account info
	addr := AddressFromString(account)
	accountInfo, exists, err := trie.GetAccountInfo(addr)

	if err != nil {
		fmt.Printf("\tError retrieving account info: %v\n", err)
		return nil
	}

	if !exists {
		fmt.Printf("\tAccount does not exist\n")
		return nil
	}

	fmt.Printf("\tAccount Info:      %v\n", accountInfo)

	// collect storage values
	storage := map[common.Key]common.Value{}
	trie.VisitAccountStorage(addr, mpt.ReadAccess{}, mpt.MakeVisitor(func(node mpt.Node, info mpt.NodeInfo) mpt.VisitResponse {
		n := node.(*mpt.ValueNode)
		storage[n.Key()] = n.Value()
		return mpt.VisitResponseContinue
	}))
	// sort the keys
	keys := make([]common.Key, 0, len(storage))
	for k := range storage {
		keys = append(keys, k)
	}
	slices.SortFunc(keys, func(a, b common.Key) int {
		return a.Compare(&b)
	})
	// print the storage values
	fmt.Printf("\tStorage Values:\n")
	for _, k := range keys {
		v := storage[k]
		fmt.Printf("\t\t%s: %x\n", k.String(), v[:])
	}

	return nil
}

func AddressFromString(str string) common.Address {
	if len(str) != 40 {
		panic(fmt.Sprintf("invalid address-string length, expected %d, got %d", 40, len(str)))
	}
	bytes, err := hex.DecodeString(str)
	if err != nil {
		panic(fmt.Sprintf("invalid hex string `%s`: %v", str, err))
	}
	res := common.Address{}
	copy(res[:], bytes)
	return res
}
