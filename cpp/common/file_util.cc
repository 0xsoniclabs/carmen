#include "common/file_util.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

namespace carmen {

namespace {

std::filesystem::path GetTempFilePath(std::string_view prefix, int i) {
  std::stringstream out;
  out << prefix << "_" << i << ".dat";
  return std::filesystem::temp_directory_path() / out.str();
}

}  // namespace

TempFile::TempFile(std::string_view prefix) {
  // Look for a file name that is not yet used.
  std::random_device rd;
  std::default_random_engine rnd(rd());
  std::uniform_int_distribution<int> dist;
  path_ = GetTempFilePath(prefix, dist(rnd));
  while (std::filesystem::exists(path_)) {
    path_ = GetTempFilePath(prefix, dist(rnd));
  }
  // Create that file to technically allocate that name.
  // Note: it may happen that a concurrent process is taking ownership of the
  // same file at the same time, but since this is a test utility, we accept
  // that risk.
  std::fstream(path_, std::ios::out);
}

TempFile::~TempFile() {
  // Delete the owned temporary file.
  std::filesystem::remove(path_);
}

const std::filesystem::path& TempFile::GetPath() const { return path_; }

}  // namespace carmen