// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

#include "common/file_util.h"

#include <filesystem>
#include <fstream>
#include <type_traits>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace carmen {
namespace {

using ::testing::StartsWith;

TEST(TempFile, TypeTraits) {
  EXPECT_TRUE(std::is_default_constructible_v<TempFile>);
  EXPECT_TRUE(std::is_move_constructible_v<TempFile>);
  EXPECT_FALSE(std::is_copy_constructible_v<TempFile>);
  EXPECT_FALSE(std::is_copy_assignable_v<TempFile>);
  EXPECT_FALSE(std::is_move_assignable_v<TempFile>);
}

TEST(TempFile, MultipleTempFilesHaveDifferentPaths) {
  TempFile a;
  TempFile b;
  EXPECT_NE(a.GetPath(), b.GetPath());
}

TEST(TempFile, TempFileUsedDesiredPrefix) {
  TempFile a("my_prefix");
  EXPECT_THAT(a.GetPath().filename(), StartsWith("my_prefix"));
}

TEST(TempFile, TheTemporaryFileExistsAfterCreation) {
  TempFile a;
  EXPECT_TRUE(std::filesystem::exists(a.GetPath()));
}

TEST(TempFile, TheTemporaryFileIsAutomaticallyRemoved) {
  std::filesystem::path path;
  {
    TempFile a;
    path = a.GetPath();
    EXPECT_TRUE(std::filesystem::exists(path));
  }
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(TempFile, TheTemporaryFileCanBeRemovedManually) {
  TempFile a;
  EXPECT_TRUE(std::filesystem::exists(a.GetPath()));
  std::filesystem::remove(a.GetPath());
  EXPECT_FALSE(std::filesystem::exists(a.GetPath()));
}

TEST(TempFile, TheTemporaryFileCanBeRemovedAndRecreatedManually) {
  TempFile a;
  EXPECT_TRUE(std::filesystem::exists(a.GetPath()));
  std::filesystem::remove(a.GetPath());
  EXPECT_FALSE(std::filesystem::exists(a.GetPath()));
  { std::fstream out(a.GetPath(), std::ios::out); }
  EXPECT_TRUE(std::filesystem::exists(a.GetPath()));
}

TEST(TempFile, TempFilePersistsMoveConstruction) {
  std::filesystem::path path;
  {
    auto file = std::make_unique<TempFile>();
    path = file->GetPath();
    EXPECT_TRUE(std::filesystem::exists(path));
    TempFile new_owner(std::move(*file));
    EXPECT_EQ(path, new_owner.GetPath());
    file.reset(nullptr);
    EXPECT_TRUE(std::filesystem::exists(path));
  }
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(TempDir, TypeTraits) {
  EXPECT_TRUE(std::is_default_constructible_v<TempDir>);
  EXPECT_TRUE(std::is_move_constructible_v<TempDir>);
  EXPECT_FALSE(std::is_copy_constructible_v<TempDir>);
  EXPECT_FALSE(std::is_copy_assignable_v<TempDir>);
  EXPECT_FALSE(std::is_move_assignable_v<TempDir>);
}

TEST(TempDir, TheTemporaryDirectoryExistsAfterCreation) {
  TempDir a;
  EXPECT_TRUE(std::filesystem::exists(a));
  EXPECT_TRUE(std::filesystem::is_directory(a));
}

TEST(TempDir, TheTemporaryDirectoryIsEmpty) {
  TempDir a;
  EXPECT_TRUE(std::filesystem::exists(a));
  int num_entries = 0;
  for (auto _ : std::filesystem::directory_iterator(a)) {
    num_entries++;
  }
  EXPECT_EQ(0, num_entries);
}

TEST(TempDir, MultipleTempFilesHaveDifferentPaths) {
  TempDir a;
  TempDir b;
  EXPECT_NE(a.GetPath(), b.GetPath());
}

TEST(TempDir, TempFileUsedDesiredPrefix) {
  TempDir a("my_prefix");
  EXPECT_THAT(a.GetPath().filename(), StartsWith("my_prefix"));
}

TEST(TempDir, TheTemporaryDirectoryIsAutomaticallyRemoved) {
  std::filesystem::path path;
  {
    TempDir a;
    path = a.GetPath();
    EXPECT_TRUE(std::filesystem::exists(path));
  }
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(TempDir, ContentOfTemporaryDirectoryIsAutomaticallyRemoved) {
  std::filesystem::path path;
  std::filesystem::path file;
  {
    TempDir a;
    path = a.GetPath();
    EXPECT_TRUE(std::filesystem::exists(path));
    file = a.GetPath() / "file.dat";
    ASSERT_FALSE(std::filesystem::exists(file));
    { std::fstream out(file, std::ios::out); }
    EXPECT_TRUE(std::filesystem::exists(file));
  }
  EXPECT_FALSE(std::filesystem::exists(path));
  EXPECT_FALSE(std::filesystem::exists(file));
}

TEST(TempDir, TempDirPersistsMoveConstruction) {
  std::filesystem::path path;
  {
    auto dir = std::make_unique<TempDir>();
    path = dir->GetPath();
    EXPECT_TRUE(std::filesystem::exists(path));
    TempDir new_owner(std::move(*dir));
    EXPECT_EQ(path, new_owner.GetPath());
    dir.reset(nullptr);
    EXPECT_TRUE(std::filesystem::exists(path));
  }
  EXPECT_FALSE(std::filesystem::exists(path));
}

}  // namespace
}  // namespace carmen
