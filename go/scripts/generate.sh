#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

# go generate must run inside the Go module (repo-root/go).
(cd "$REPO_ROOT/go" && go generate --tags carmen_rust ./...)

# git apply must be run from the repo root: when invoked from a subdirectory it
# silently ignores paths outside that directory (see `git help apply`).
git -C "$REPO_ROOT" apply "$SCRIPT_DIR/patches/fix_external_state_mock.patch"
