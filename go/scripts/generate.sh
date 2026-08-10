#!/bin/bash
set -euox pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PATCH="$SCRIPT_DIR/patches/fix_external_state_mock.patch"

# go generate must run inside the Go module (repo-root/go).
# Restrict to mockgen directives only: other directives (e.g. `cargo build`)
# rebuild external artifacts and are unrelated to mock generation.
(cd "$REPO_ROOT/go" && go generate -run 'mockgen' --tags carmen_rust ./...)

# git apply must be run from the repo root: when invoked from a subdirectory it
# silently ignores paths outside that directory (see `git help apply`).
# Apply the patch idempotently: if it is already applied (reverse-check
# succeeds), skip re-applying it so the script can be re-run safely.
if git -C "$REPO_ROOT" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    echo "fix_external_state_mock.patch is already applied; skipping."
else
    git -C "$REPO_ROOT" apply "$PATCH"
fi
