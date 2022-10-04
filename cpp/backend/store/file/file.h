#pragma once

#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <span>

#include "backend/store/file/page.h"

namespace carmen::backend::store {

// ------------------------------- Declarations -------------------------------

// The File concept defines an interface for file implementations supporting the
// loading and storing of fixed length pages. Pages are expected to be numbered
// in the range [0..n-1], where n is the number of pages in the file.
//
// The type T is the type tested for satisfying the File concept and the page
// size parameter defines the size of each page in bytes.
template <typename T, std::size_t page_size>
concept File = requires(T a) {
  T::kPageSize;
  // Each file implementation must support the extraction of the number of
  // pages.
  { a.GetNumPages() } -> std::same_as<std::size_t>;
  // LoadPage is intended to be used for fetching a single page from the file.
  {
    a.LoadPage(PageId{}, std::declval<std::span<std::byte, page_size>>())
    } -> std::same_as<void>;
  // StorePage is intended to be used for fetching a single page from the file.
  {
    a.StorePage(PageId{}, std::declval<std::span<const std::byte, page_size>>())
    } -> std::same_as<void>;
  // Each file has to support a flush operation after which data previously
  // written must be persisted on disk.
  { a.Flush() } -> std::same_as<void>;
};

// An InMemoryFile implement is provided to for testing purposes, where actual
// file operations are not relevant. It may also serve as a reference
// implementation to compare other implementations to in unit testing.
template <typename std::size_t page_size>
class InMemoryFile {
 public:
  static constexpr std::size_t kPageSize = page_size;

  std::size_t GetNumPages() const { return data_.size() / page_size; }

  void LoadPage(PageId id, std::span<std::byte, page_size> trg) const;

  void StorePage(PageId id, std::span<const std::byte, page_size> src);

  void Flush() const {
    // Nothing to do.
  }

 private:
  std::deque<std::byte> data_;
};

namespace internal {

// A RawFile instance provides raw read/write access to a file. It provides a
// utility for implementing actual stricter typed File implementations.
// Note: a raw File is not satisfying any File concept.
class RawFile {
 public:
  // Opens the given file in read/write mode. If it does not exist, the file is
  // created. TODO(herbertjordan): add error handling.
  RawFile(std::filesystem::path file);
  // Flushes the content and closes the file.
  ~RawFile();

  // Provides the current file size in bytes.
  std::size_t GetFileSize();

  // Reads a range of bytes from the file to the given span. The provided
  // position is the starting position. The number of bytes to be read is taken
  // from the length of the provided span.
  void Read(std::size_t pos, std::span<std::byte> span);

  // Writes a span of bytes to the file at the given position. If needed, the
  // file is grown to fit all the data of the span. Additional bytes between the
  // current end and the starting position are initialized with zeros.
  void Write(std::size_t pos, std::span<const std::byte> span);

  // Flushes all pending/buffered writes to disk.
  void Flush();

 private:
  // Grows the underlying file to the given size.
  void GrowFileIfNeeded(std::size_t needed);

  std::size_t file_size_;
  std::fstream data_;
};

}  // namespace internal

// An implementation of the File concept using a single file as a persistent
// storage solution.
template <typename std::size_t page_size>
class SingleFile {
 public:
  static constexpr std::size_t kPageSize = page_size;

  SingleFile(std::filesystem::path file_path) : file_(file_path) {}

  std::size_t GetNumPages() const { return file_.GetFileSize() / page_size; }

  void LoadPage(PageId id, std::span<std::byte, page_size> trg) const {
    file_.Read(id * page_size, trg);
  }

  void StorePage(PageId id, std::span<const std::byte, page_size> src) {
    file_.Write(id * page_size, src);
  }

  void Flush() const { file_.Flush(); }

 private:
  mutable internal::RawFile file_;
};

// ------------------------------- Definitions --------------------------------

template <typename std::size_t page_size>
void InMemoryFile<page_size>::LoadPage(
    PageId id, std::span<std::byte, page_size> trg) const {
  const auto offset = id * page_size;
  std::size_t i = 0;
  for (; i < page_size && offset + i < data_.size(); i++) {
    trg[i] = data_[offset + i];
  }
  for (; i < page_size; i++) {
    trg[i] = std::byte{0};
  }
}

template <typename std::size_t page_size>
void InMemoryFile<page_size>::StorePage(
    PageId id, std::span<const std::byte, page_size> src) {
  const auto offset = id * page_size;
  if (data_.size() < offset + page_size) {
    data_.resize(offset + page_size);
  }
  for (std::size_t i = 0; i < page_size; i++) {
    data_[offset + i] = src[i];
  }
}

}  // namespace carmen::backend::store