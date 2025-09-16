#!/bin/sh

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
go run "$SCRIPT_DIR/../../go/scripts/license/add_license_header.go" "$@"