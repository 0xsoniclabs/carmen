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
	Flags:     []cli.Flag{&blockHeightFlag},
	Usage:     "lists information about a given account",
	ArgsUsage: "[--block-height <height>] <directory> <account>",
}

func accountInfo(context *cli.Context) error {
	if context.Args().Len() != 2 {
		return fmt.Errorf("expected 2 positional arguments, got %d; usage: %s",
			context.Args().Len(), context.Command.ArgsUsage)
	}
	dir := context.Args().Get(0)
	account := context.Args().Get(1)

	addr, err := addressFromString(account)
	if err != nil {
		return err
	}

	mptInfo, err := io.CheckMptDirectoryAndGetInfo(dir)
	if err != nil {
		return err
	}

	trie, err := openTrie(context, dir, mptInfo)
	if err != nil {
		return err
	}
	defer func() {
		if err := trie.Close(); err != nil {
			fmt.Printf("\tError closing MPT: %v\n", err)
		}
	}()

	accountInfo, exists, err := trie.GetAccountInfo(addr)
	if err != nil {
		return fmt.Errorf("failed to retrieve account info: %w", err)
	}
	if !exists {
		fmt.Printf("\tAccount does not exist\n")
		return nil
	}
	fmt.Printf("\tAccount Info for address %s:\n", addr.String())
	fmt.Printf("\t\tNonce:           %d\n", accountInfo.Nonce.ToUint64())
	fmt.Printf("\t\tBalance:         %s\n", accountInfo.Balance.String())
	fmt.Printf("\t\tCode Hash:       0x%x\n", accountInfo.CodeHash[:])

	// Collect the storage values.
	storage := map[common.Key]common.Value{}
	err = trie.VisitAccountStorage(addr, mpt.ReadAccess{}, mpt.MakeVisitor(func(node mpt.Node, info mpt.NodeInfo) mpt.VisitResponse {
		if n, ok := node.(*mpt.ValueNode); ok {
			storage[n.Key()] = n.Value()
		}
		return mpt.VisitResponseContinue
	}))
	if err != nil {
		return fmt.Errorf("failed to visit account storage: %w", err)
	}

	keys := make([]common.Key, 0, len(storage))
	for k := range storage {
		keys = append(keys, k)
	}
	slices.SortFunc(keys, func(a, b common.Key) int {
		return a.Compare(&b)
	})

	fmt.Printf("\t\tStorage Values:\n")
	for _, k := range keys {
		v := storage[k]
		fmt.Printf("\t\t\t0x%s: 0x%x\n", k.String(), v[:])
	}

	return nil
}

// openTrie opens either the live or the archive trie in the given directory,
// depending on the MPT mode and whether the --block-height flag is set. When
// the archive is opened without an explicit block height, the archive's head
// block is used.
func openTrie(context *cli.Context, dir string, mptInfo io.MptInfo) (*LiveOrArchiveTrie, error) {
	blockHeightSet := context.IsSet(blockHeightFlag.Name)

	if mptInfo.Mode == mpt.Mutable && !blockHeightSet {
		fmt.Printf("\tOpening live MPT\n")
		liveTrie, err := mpt.OpenFileLiveTrie(dir, mptInfo.Config, mpt.NodeCacheConfig{})
		if err != nil {
			return nil, fmt.Errorf("failed to open live MPT: %w", err)
		}
		return NewLiveTrie(liveTrie), nil
	}

	archive, err := mpt.OpenArchiveTrie(dir, mptInfo.Config, mpt.NodeCacheConfig{}, mpt.ArchiveConfig{})
	if err != nil {
		return nil, fmt.Errorf("failed to open archive MPT: %w", err)
	}

	var block *uint64
	if blockHeightSet {
		b := context.Uint64(blockHeightFlag.Name)
		block = &b
	}
	trie, err := NewArchiveTrie(archive, block)
	if err != nil {
		if closeErr := archive.Close(); closeErr != nil {
			err = fmt.Errorf("%w (also failed to close archive: %v)", err, closeErr)
		}
		return nil, err
	}
	fmt.Printf("\tOpening archive MPT at block height %d\n", trie.block)
	return trie, nil
}

// addressFromString decodes a 40-character hex string into a common.Address.
func addressFromString(str string) (common.Address, error) {
	var addr common.Address
	if len(str) != 2*len(addr) {
		return addr, fmt.Errorf("invalid address length, expected %d hex characters, got %d", 2*len(addr), len(str))
	}
	if _, err := hex.Decode(addr[:], []byte(str)); err != nil {
		return common.Address{}, fmt.Errorf("invalid address hex string %q: %w", str, err)
	}
	return addr, nil
}

// LiveOrArchiveTrie is a thin wrapper that dispatches account queries to
// either a live or an archive MPT. Instances must be constructed via
// NewLiveTrie or NewArchiveTrie.
type LiveOrArchiveTrie struct {
	liveTrie    *mpt.LiveTrie
	archiveTrie *mpt.ArchiveTrie
	block       uint64 // only meaningful when archiveTrie is set
}

// NewLiveTrie wraps a live MPT.
func NewLiveTrie(liveTrie *mpt.LiveTrie) *LiveOrArchiveTrie {
	return &LiveOrArchiveTrie{liveTrie: liveTrie}
}

// NewArchiveTrie wraps an archive MPT at the given block height. If block is
// nil, the archive's head block is used. An error is returned when the
// requested block is invalid for the given archive.
func NewArchiveTrie(archiveTrie *mpt.ArchiveTrie, block *uint64) (*LiveOrArchiveTrie, error) {
	head, empty, err := archiveTrie.GetBlockHeight()
	if err != nil {
		return nil, fmt.Errorf("failed to get block height from archive MPT: %w", err)
	}

	var target uint64
	switch {
	case block == nil:
		if empty {
			return nil, fmt.Errorf("archive MPT is empty")
		}
		target = head
	case empty && *block != 0:
		return nil, fmt.Errorf("archive MPT is empty, but block height %d was requested", *block)
	case !empty && *block > head:
		return nil, fmt.Errorf("requested block height %d exceeds archive head %d", *block, head)
	default:
		target = *block
	}

	return &LiveOrArchiveTrie{archiveTrie: archiveTrie, block: target}, nil
}

func (t *LiveOrArchiveTrie) GetAccountInfo(addr common.Address) (mpt.AccountInfo, bool, error) {
	if t.liveTrie != nil {
		return t.liveTrie.GetAccountInfo(addr)
	}
	return t.archiveTrie.GetAccountInfo(t.block, addr)
}

func (t *LiveOrArchiveTrie) VisitAccountStorage(addr common.Address, access mpt.AccessMode, visitor mpt.NodeVisitor) error {
	if t.liveTrie != nil {
		return t.liveTrie.VisitAccountStorage(addr, access, visitor)
	}
	return t.archiveTrie.VisitAccountStorage(t.block, addr, access, visitor)
}

func (t *LiveOrArchiveTrie) Close() error {
	if t.liveTrie != nil {
		return t.liveTrie.Close()
	}
	return t.archiveTrie.Close()
}
