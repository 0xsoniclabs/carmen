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
    name = "hash_page",
    hdrs = ["hash_page.h"],
    deps = [
        "//backend/common:page_id",
        "//common:type",
    ],
)

cc_test(
    name = "hash_page_test",
    srcs = ["hash_page_test.cc"],
    deps = [
        ":hash_page",
        "//backend/common:page",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "stable_hash",
    hdrs = ["stable_hash.h"],
    deps = [
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/numeric:int128",
    ],
)

cc_test(
    name = "stable_hash_test",
    srcs = ["stable_hash_test.cc"],
    deps = [
        ":stable_hash",
        "//backend/common:page",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "stable_hash_benchmark",
    srcs = ["stable_hash_benchmark.cc"],
    deps = [
        ":stable_hash",
        "//common:type",
        "//third_party/gperftools:profiler",
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_absl//absl/hash",
    ],
)

cc_library(
    name = "index",
    hdrs = ["index.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":hash_page",
        ":stable_hash",
        "//backend:structure",
        "//backend/common:file",
        "//backend/common:page_pool",
        "//common:fstream",
        "//common:hash",
        "//common:memory_usage",
        "//common:status_util",
        "//common:type",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "index_test",
    srcs = ["index_test.cc"],
    deps = [
        ":index",
        "//backend:structure",
        "//backend/index:index_test_suite",
        "//backend/index:test_util",
        "//common:file_util",
        "//common:status_test_util",
        "@com_google_googletest//:gtest_main",
    ],
)
