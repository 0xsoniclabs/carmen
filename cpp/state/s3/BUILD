# Copyright (c) 2025 Sonic Operations Ltd
#
# Use of this software is governed by the Business Source License included
# in the LICENSE file and at soniclabs.com/bsl11.
#
# Change Date: 2028-4-16
#
# On the date above, in accordance with the Business Source License, use of
# this software will be governed by the GNU Lesser General Public License v3.

cc_library(
    name = "state",
    hdrs = ["state.h"],
    visibility = [
        "//state:__subpackages__",
    ],
    deps = [
        "//archive",
        "//backend:structure",
        "//common:type",
        "//state:schema",
        "//state:update",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "state_test",
    srcs = ["state_test.cc"],
    deps = [
        ":state",
        "//archive:test_util",
        "//archive/leveldb:archive",
        "//backend/depot:test_util",
        "//backend/depot/memory:depot",
        "//backend/index:test_util",
        "//backend/index/memory:index",
        "//backend/multimap:test_util",
        "//backend/multimap/memory:multimap",
        "//backend/store:test_util",
        "//backend/store/memory:store",
        "//common:status_test_util",
        "//common:type",
        "//state:configurations",
        "//state:state_test_suite",
        "//state:update",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)
