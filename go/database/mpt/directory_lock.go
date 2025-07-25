// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package mpt

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/0xsoniclabs/carmen/go/common"
)

const lockFileName = "~lock"

// LockDirectory acquires a lock on the given directory. If needed,
// the directory is implicitly created. The operation fails if the
// lock can not be acquired due to some other thread or process holding
// the lock or due to an IO error.
//
// Note: if successful, the acquired lock needs to be explicitly released.
// The lock is not automatically released when the process is terminated.
func LockDirectory(directory string) (common.LockFile, error) {
	if err := os.MkdirAll(directory, 0700); err != nil {
		return nil, err
	}
	lock, err := common.CreateLockFile(filepath.Join(directory, lockFileName))
	if err != nil {
		return nil, fmt.Errorf("unable to gain exclusive access to %s: %w", directory, err)
	}
	return lock, nil
}

// ForceUnlockDirectory removes the lock file from the given directory.
// Use this with care, as it may lead to multiple processes writing to
// the same directory.
func ForceUnlockDirectory(directory string) error {
	err := os.Remove(filepath.Join(directory, lockFileName))
	if os.IsNotExist(err) {
		return nil
	}
	return err
}
