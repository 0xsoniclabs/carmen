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
    name = "store",
    hdrs = ["store.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//backend/common/leveldb",
        "//backend/store",
        "//backend/store:hash_tree",
        "//common:byte_util",
        "//common:hash",
        "//common:status_util",
        "//common:type",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "store_test",
    srcs = ["store_test.cc"],
    deps = [
        ":store",
        "//backend:structure",
        "//backend/store:store_test_suite",
        "//common:file_util",
        "//common:status_test_util",
        "//common:type",
        "@com_google_googletest//:gtest_main",
    ],
)
