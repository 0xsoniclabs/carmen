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
	"context"
	"fmt"
	"github.com/0xsoniclabs/carmen/go/common"
	"github.com/0xsoniclabs/carmen/go/common/interrupt"
	"github.com/0xsoniclabs/carmen/go/database/mpt"
	"github.com/holiman/uint256"
)

func CalculateLiveTotalSupply(ctx context.Context, logger *Log, directory string) error {
	info, err := CheckMptDirectoryAndGetInfo(directory)
	if err != nil {
		return fmt.Errorf("error in input directory: %v", err)
	}

	if info.Config.Name != mpt.S5LiveConfig.Name {
		return fmt.Errorf("can only calculate total supply on LiveDB instances, found %v in directory", info.Mode)
	}

	mptState, err := mpt.OpenGoFileState(directory, info.Config, mpt.NodeCacheConfig{Capacity: exportCacheCapacitySize})
	if err != nil {
		return fmt.Errorf("failed to open LiveDB: %v", err)
	}
	defer mptState.Close()

	hash, err := mptState.GetHash()
	if err != nil {
		return fmt.Errorf("failed to get state root: %v", err)
	}

	zeroBalanceAmount, err := mptState.GetBalance(common.Address{})
	if err != nil {
		return fmt.Errorf("failed to get balance of zero address: %w", err)
	}
	zeroBalance := zeroBalanceAmount.Uint256()

	logger.Printf("Calculating total supply...")
	progress := logger.NewProgressTracker("visited %d accounts, %.2f accounts/s", 1_000_000)
	db := &exportableLiveTrie{db: mptState, directory: directory}
	visitor := totalSupplyCalculatingVisitor{ctx: ctx, progress: progress}
	if err := db.Visit(&visitor, true); err != nil {
		return fmt.Errorf("failed visiting content: %v", err)
	}

	logger.Printf("Calculated for: %s", directory)
	logger.Printf("State Root: %x", hash)
	logger.Printf("Accounts count: %d", visitor.accounts)
	logger.Printf("Total native token supply: %s", visitor.totalSupply.String())
	diff := new(uint256.Int).Sub(&visitor.totalSupply, &zeroBalance)
	logger.Printf("Excluding zero address:    %s", diff.String())
	return nil
}

func CalculateArchiveTotalSupply(ctx context.Context, logger *Log, directory string, block uint64) error {
	info, err := CheckMptDirectoryAndGetInfo(directory)
	if err != nil {
		return fmt.Errorf("error in input directory: %v", err)
	}

	if info.Config.Name != mpt.S5ArchiveConfig.Name {
		return fmt.Errorf("can only support export of S5 Archive instances, found %v in directory", info.Config.Name)
	}

	archive, err := mpt.OpenArchiveTrie(directory, info.Config, mpt.NodeCacheConfig{Capacity: exportCacheCapacitySize}, mpt.ArchiveConfig{})
	if err != nil {
		return err
	}
	defer archive.Close()

	hash, err := archive.GetHash(block)
	if err != nil {
		return fmt.Errorf("failed to get state root: %w", err)
	}

	zeroBalanceAmount, err := archive.GetBalance(block, common.Address{})
	if err != nil {
		return fmt.Errorf("failed to get balance of zero address: %w", err)
	}
	zeroBalance := zeroBalanceAmount.Uint256()

	logger.Printf("Calculating total supply...")
	progress := logger.NewProgressTracker("visited %d accounts, %.2f accounts/s", 1_000_000)
	db := exportableArchiveTrie{trie: archive, block: block}
	visitor := totalSupplyCalculatingVisitor{ctx: ctx, progress: progress}
	if err := db.Visit(&visitor, true); err != nil {
		return fmt.Errorf("failed visiting content: %v", err)
	}

	logger.Printf("Calculated for: %s", directory)
	logger.Printf("Block Number: %d", block)
	logger.Printf("State Root: %x", hash)
	logger.Printf("Accounts count: %d", visitor.accounts)
	logger.Printf("Total native token supply: %s", visitor.totalSupply.String())
	diff := new(uint256.Int).Sub(&visitor.totalSupply, &zeroBalance)
	logger.Printf("Excluding zero address:    %s", diff.String())
	return err
}

type totalSupplyCalculatingVisitor struct {
	ctx         context.Context
	progress    *ProgressLogger
	totalSupply uint256.Int
	accounts    uint64
}

func (e *totalSupplyCalculatingVisitor) Visit(node mpt.Node, _ mpt.NodeInfo) error {
	// outside call to interrupt
	if interrupt.IsCancelled(e.ctx) {
		return interrupt.ErrCanceled
	}
	if n, ok := node.(*mpt.AccountNode); ok {
		balance := n.Info().Balance.Uint256()
		e.totalSupply.Add(&e.totalSupply, &balance)
		e.accounts++
		e.progress.Step(1)
	}
	return nil
}
