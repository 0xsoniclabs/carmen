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
