# Copyright (c) 2025 Sonic Operations Ltd
#
# Use of this software is governed by the Business Source License included
# in the LICENSE file and at soniclabs.com/bsl11.
#
# Change Date: 2028-4-16
#
# On the date above, in accordance with the Business Source License, use of
# this software will be governed by the GNU Lesser General Public License v3.

.PHONY: all clean

all: carmen-cpp carmen-rust

# this target builds the C++ library required by Go
carmen-cpp:
	@cd ./go/lib ; \
	./build_libcarmen.sh ;

# this target builds the Rust library required by Go
carmen-rust:
	@cd ./rust ; \
	cargo build --release

clean:
	cd ./go ; \
	rm -f lib/libcarmen.so ; \
	cd ../cpp ; \
	bazel clean ; \

.PHONY: license-check
license-check:
	./scripts/license/add_license_header.sh --check -dir ./

.PHONY: license-add
license-add:
	./scripts/license/add_license_header.sh -dir ./