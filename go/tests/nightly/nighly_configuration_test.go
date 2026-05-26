// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package nightly

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_IsNightlyIsEnabledWhenFlagIsProvided(t *testing.T) {
	fmt.Printf("Nightly flag is %t\n", IsNightly())
}

func Test_IsNightly(t *testing.T) {
	require := require.New(t)
	stdout := execSubProcessTest(t, "Test_IsNightlyIsEnabledWhenFlagIsProvided", "CARMEN_NIGHTLY=true")
	// Use contains as test output includes PASS and statement coverage info
	require.Contains(stdout.String(), "Nightly flag is true")
	stdout = execSubProcessTest(t, "Test_IsNightlyIsEnabledWhenFlagIsProvided", "")
	require.Contains(stdout.String(), "Nightly flag is false")
}

// execSubProcessTest executes the test with the given name in a subprocess with the provided flag and returns the stdout buffer.
func execSubProcessTest(t *testing.T, execTestName string, envVar string) bytes.Buffer {
	path, err := os.Executable()
	if err != nil {
		t.Fatalf("failed to resolve path to test binary: %v", err)
	}

	cmd := exec.Command(path, "-test.run", execTestName)
	cmd.Env = []string{envVar}
	errBuf := new(bytes.Buffer)
	cmd.Stderr = errBuf
	stdBuf := new(bytes.Buffer)
	cmd.Stdout = stdBuf

	if err := cmd.Run(); err != nil {
		t.Errorf("Subprocess finished with error: %v\n stdout:\n%s stderr:\n%s", err, stdBuf.String(), errBuf.String())
	}

	return *stdBuf
}
