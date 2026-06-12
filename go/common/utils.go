// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package common

import (
	"io/fs"
	"path/filepath"
)

// GetDirectorySize computes the size of all files in the given directory in bytes.
// Returns 0 if the directory does not exist.
func GetDirectorySize(directory string) (int64, error) {
	var sum int64 = 0
	err := filepath.Walk(directory, func(path string, info fs.FileInfo, retErr error) error {
		if retErr != nil {
			return nil
		}
		if !info.IsDir() {
			sum += info.Size()
		}
		return nil
	})
	return sum, err
}
